# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from typing import Union, Any, Optional, List
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import transformers
import inspect

from trl import GRPOTrainer as _GRPOTrainer
from accelerate.utils import gather_object
from trl.data_utils import is_conversational
from trl.trainer.utils import pad, nanstd, nanmin, nanmax

# Import Triton-accelerated LoRA merge (with PyTorch fallback)
try:
    from lora_merge_triton import lora_merge_batch_triton, TRITON_AVAILABLE
except ImportError:
    print("⚠️  lora_merge_triton.py not found, using standard PyTorch LoRA merge")
    TRITON_AVAILABLE = False


class GRPOTrainer(_GRPOTrainer):
    def __init__(self, *args, enable_self_refine: bool = False, use_critique: bool = True, 
                 refine_log_file: str = "self_refine_log.jsonl", 
                 return_both_responses: bool = False,
                 micro_batch_size: int = None,
                 self_refine_rounds: int = 1,
                 refine_until_correct: bool = True,
                 use_ring_attention: bool = False,
                 sequence_parallel_size: int = None,
                 use_chunked_attention: bool = False,
                 chunk_size: int = 512,
                 quantize_kv: bool = True,
                 use_uvm: bool = True,
                 **kwargs):
        """
        Initialize the GRPO trainer with optional self-refine capability.
        
        Args:
            enable_self_refine: If True, will attempt to self-refine incorrect responses
            use_critique: If True, will first generate a critique before refining (two-stage refine)
            refine_log_file: Path to save self-refine logs (default: "self_refine_log.jsonl")
            return_both_responses: If True, return both original and final refined responses (doubles sample size)
            micro_batch_size: If set and using FSDP2, split large batches to avoid OOM
            self_refine_rounds: Number of refine rounds to perform (>=1). 1 = current behavior
            refine_until_correct: If True, stop refining early when all samples are correct
            use_ring_attention: If True, enable RingAttention for sequence parallelism
            sequence_parallel_size: Number of GPUs to use for sequence parallelism (default: all GPUs)
                                   Must be <= world_size and divide world_size evenly
            use_chunked_attention: If True, enable ChunkedAttention for memory optimization
            chunk_size: Chunk size for ChunkedAttention (default: 512)
            quantize_kv: Whether to quantize K/V in ChunkedAttention (default: True)
            use_uvm: Whether to use UVM for offloading in ChunkedAttention (default: True)
        """
        super().__init__(*args, **kwargs)
        self.enable_self_refine = enable_self_refine
        self.use_critique = use_critique
        self.refine_log_file = refine_log_file
        self.return_both_responses = return_both_responses
        self.micro_batch_size = micro_batch_size
        self.self_refine_rounds = max(1, int(self_refine_rounds))
        self.refine_until_correct = bool(refine_until_correct)
        self.batch_counter = 0
        self.use_ring_attention = use_ring_attention
        self.sequence_parallel_size = sequence_parallel_size
        self.use_chunked_attention = use_chunked_attention
        self.chunk_size = chunk_size
        self.quantize_kv = quantize_kv
        self.use_uvm = use_uvm
        
        # Apply attention optimizations if requested
        if self.use_ring_attention or self.use_chunked_attention:
            self._apply_attention_optimizations()
    
    def _apply_attention_optimizations(self):
        """
        Apply attention optimizations (RingAttention, ChunkedAttention, or both).
        
        This method patches the model's attention layers to use:
        - RingAttention: sequence parallelism across GPUs
        - ChunkedAttention: memory optimization via chunking + UVM offload
        - Hybrid: both optimizations combined
        """
        try:
            import torch.distributed as dist
            
            world_size = self.accelerator.num_processes
            
            # Determine operation mode
            if self.use_ring_attention and self.use_chunked_attention:
                mode = "Hybrid (Ring + Chunked)"
            elif self.use_ring_attention:
                mode = "RingAttention only"
            elif self.use_chunked_attention:
                mode = "ChunkedAttention only"
            else:
                return  # Nothing to do
            
            if self.accelerator.is_main_process:
                print(f"\n{'='*70}")
                print(f"Attention Optimization: {mode}")
                print(f"{'='*70}")
            
            # Setup RingAttention parameters if enabled
            if self.use_ring_attention:
                # Validate sequence_parallel_size
                if self.sequence_parallel_size is None:
                    sp_size = world_size
                else:
                    sp_size = self.sequence_parallel_size
                    
                    if sp_size > world_size:
                        raise ValueError(
                            f"sequence_parallel_size ({sp_size}) cannot exceed world_size ({world_size})"
                        )
                    
                    if world_size % sp_size != 0:
                        raise ValueError(
                            f"world_size ({world_size}) must be divisible by sequence_parallel_size ({sp_size})"
                        )
                
                if self.accelerator.is_main_process:
                    print(f"RingAttention:")
                    print(f"  World size: {world_size}")
                    print(f"  Sequence parallel size: {sp_size}")
                    print(f"  Data parallel size: {world_size // sp_size}")
                
                # Create process groups for sequence parallelism
                if dist.is_initialized() and sp_size < world_size:
                    self._setup_sequence_parallel_groups(sp_size)
                
                process_group = getattr(self, 'sp_group', None) if sp_size < world_size else None
            else:
                sp_size = None
                process_group = None
            
            # Setup ChunkedAttention parameters if enabled
            if self.use_chunked_attention:
                if self.accelerator.is_main_process:
                    print(f"ChunkedAttention:")
                    print(f"  Chunk size: {self.chunk_size}")
                    print(f"  K/V quantization: {self.quantize_kv}")
                    print(f"  UVM offloading: {self.use_uvm}")
            
            # Print memory savings estimate for hybrid mode
            if self.use_ring_attention and self.use_chunked_attention and self.accelerator.is_main_process:
                print(f"\nHybrid Mode Memory Savings:")
                print(f"  RingAttention: {sp_size}x (sequence split across GPUs)")
                print(f"  ChunkedAttention: ~1000x (chunking + quantization)")
                print(f"  Combined: ~{sp_size * 1000}x total memory savings")
                print(f"\nExample: 128K sequence on {sp_size} GPUs")
                print(f"  Per GPU: {128 // sp_size}K tokens (after Ring split)")
                num_chunks = (128 * 1024 // sp_size) // self.chunk_size
                print(f"  Per GPU: {num_chunks} chunks × {self.chunk_size} tokens (after Chunking)")
                print(f"  GPU peak: ~20MB (one chunk at a time)")
                print(f"  CPU/UVM: ~4MB (quantized K/V cache)")
            
            # Patch the model
            if self.use_ring_attention and self.use_chunked_attention:
                # Hybrid mode: RingAttention with ChunkedAttention for local computation
                from ring_attention import patch_model_for_ring_attention
                self.model = patch_model_for_ring_attention(
                    self.model,
                    sequence_parallel_size=sp_size,
                    process_group=process_group,
                    enable_chunking=True,
                    chunk_size=self.chunk_size,
                    quantize_kv=self.quantize_kv,
                    use_uvm=self.use_uvm,
                )
            elif self.use_ring_attention:
                # RingAttention only
                from ring_attention import patch_model_for_ring_attention
                self.model = patch_model_for_ring_attention(
                    self.model,
                    sequence_parallel_size=sp_size,
                    process_group=process_group,
                    enable_chunking=False,
                )
            elif self.use_chunked_attention:
                # ChunkedAttention only
                from chunked_attention import patch_model_for_chunked_attention
                self.model = patch_model_for_chunked_attention(
                    self.model,
                    chunk_size=self.chunk_size,
                    quantize_kv=self.quantize_kv,
                    use_uvm=self.use_uvm,
                )
            
            if self.accelerator.is_main_process:
                print(f"{'='*70}\n")
                
        except ImportError as e:
            if self.accelerator.is_main_process:
                print(f"Warning: Could not import required modules: {e}")
                print("Continuing without attention optimizations")
        except Exception as e:
            if self.accelerator.is_main_process:
                print(f"Warning: Failed to apply attention optimizations: {e}")
                print("Continuing without attention optimizations")
    
    def _setup_sequence_parallel_groups(self, sp_size: int):
        """
        Setup process groups for hybrid data + sequence parallelism.
        
        Example with 8 GPUs and sp_size=4:
        - SP Group 0: [0, 1, 2, 3]  (handles same data, split sequence)
        - SP Group 1: [4, 5, 6, 7]  (handles different data, split sequence)
        
        Args:
            sp_size: Sequence parallel size
        """
        import torch.distributed as dist
        
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        dp_size = world_size // sp_size
        
        # Create sequence parallel groups
        for i in range(dp_size):
            ranks = list(range(i * sp_size, (i + 1) * sp_size))
            group = dist.new_group(ranks)
            
            if rank in ranks:
                self.sp_group = group
                self.sp_rank = rank - i * sp_size
                self.dp_rank = i
        
        if self.accelerator.is_main_process:
            print(f"[RingAttention] Created {dp_size} sequence parallel groups of size {sp_size}")
            for i in range(dp_size):
                ranks = list(range(i * sp_size, (i + 1) * sp_size))
                print(f"[RingAttention]   SP Group {i}: {ranks}")
        
    def _log_refine_sample(self, sample_data: dict):
        """
        Log a single self-refine sample to file in JSONL format.
        Only saves on main process to avoid duplicate logs.
        
        Args:
            sample_data: Dictionary containing sample information
        """
        if not self.accelerator.is_main_process:
            return
        
        import json
        from pathlib import Path
        
        try:
            # Create log directory if needed
            log_path = Path(self.refine_log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Append to JSONL file
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(sample_data, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"[Warning] Failed to write to log file: {e}")
    
    def _extract_answer_for_critique(self, text: str) -> str:
        """
        Extract <answer> section for critique.
        Returns the CONTENT without tags.
        
        Args:
            text: Text that may contain <think> and <answer> tags
            
        Returns:
            Content of <answer> (without tags), or cleaned text
        """
        import re
        
        # Extract <answer> content (without the tags themselves)
        answer_match = re.search(r'<answer>(.*?)</answer>', text, flags=re.DOTALL)
        
        if answer_match:
            # Return answer content only
            return answer_match.group(1).strip()
        
        # Fallback: remove <think> tag only, keep remaining text
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()
    
    def _remove_think_tags(self, text: str) -> str:
        """
        Alias for _extract_answer_for_critique for backward compatibility.
        """
        return self._extract_answer_for_critique(text)
    
    def _clean_critique_output(self, text: str) -> str:
        """
        Normalize thinking tags in critique output.
        Converts incorrect tag forms like <thinking> to <think>.
        
        Args:
            text: Critique text that may contain various thinking tag forms
            
        Returns:
            Text with normalized <think></think> tags
        """
        import re
        
        # Convert <thinking> to <think> (keep the content)
        cleaned = text
        cleaned = re.sub(r'<thinking>', '<think>', cleaned)
        cleaned = re.sub(r'</thinking>', '</think>', cleaned)
        cleaned = re.sub(r'<thought>', '<think>', cleaned)
        cleaned = re.sub(r'</thought>', '</think>', cleaned)
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        return cleaned.strip()
    
    def _check_if_correct(self, completion: str, solution: str) -> bool:
        """
        Check if a completion is correct by comparing it with the ground truth solution.
        Uses the same logic as accuracy_reward function.
        
        Args:
            completion: Generated completion text
            solution: Ground truth solution
            
        Returns:
            bool: True if correct, False otherwise
        """
        try:
            gold_parsed = parse(solution, extraction_mode="first_match")
        except Exception:
            gold_parsed = []
        
        if len(gold_parsed) != 0:
            # Try parsing predicted answer too
            try:
                answer_parsed = parse(
                    completion,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                boxed="all",
                                units=True,
                            ),
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode="first_match",
                )
                is_correct = verify(gold_parsed, answer_parsed)
                return bool(is_correct)
            except Exception:
                return False
        else:
            # fallback to text match
            return completion.strip().lower() == solution.strip().lower()
    
    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        """
        Override the generation method to add self-refine capability.
        When a completion is incorrect and self-refine is enabled, 
        modify the prompts to include the first attempt and ask for refinement,
        then regenerate for the ENTIRE batch.
        """
        # First, generate completions normally
        result = super()._generate_and_score_completions(inputs)
        
        # If self-refine is not enabled, return the result as-is
        if not self.enable_self_refine:
            return result
        
        # Only apply self-refine during training
        if not self.model.training:
            return result
        
        # Check if we have solution data to verify correctness
        if "solution" not in inputs[0]:
            return result
        
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        solutions = [x["solution"] for x in inputs]
        
        # Decode completions to check correctness
        completion_ids = result["completion_ids"]
        completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Track which samples need refinement and their correctness at t1
        needs_refine_mask = []
        refine_indices = []
        correctness_at_t1 = []  # Store correctness for each sample
        
        for idx, (completion, solution) in enumerate(zip(completions_text, solutions)):
            is_correct = self._check_if_correct(completion, solution)
            correctness_at_t1.append(is_correct)
            
            if not is_correct:
                needs_refine_mask.append(True)
                refine_indices.append(idx)
            else:
                needs_refine_mask.append(False)
        
        # If some samples need refinement, run multi-round refinement with optional early stopping
        if refine_indices:
            current_batch_size = len(completions_text)
            self.batch_counter += 1

            if self.accelerator.is_main_process:
                batch_correct_t0 = sum(correctness_at_t1)
                batch_acc_t0 = 100 * batch_correct_t0 / current_batch_size
                print(f"\n{'='*70}")
                print(f"[Self-Refine] Batch {self.batch_counter} Statistics")
                print(f"{'='*70}")
                print(f"Batch size: {current_batch_size}")
                print(f"Batch Accuracy@r0: {batch_correct_t0}/{current_batch_size} = {batch_acc_t0:.2f}%")
                print(f"Samples needing refine (r0): {len(refine_indices)}")
                print(f"Log file: {self.refine_log_file}")
                print(f"{'-'*70}")

            original_result = result
            original_completions_text = completions_text
            original_correctness = correctness_at_t1

            # Collect results for each round including the original
            results_across_rounds = [result]

            current_result = result
            current_completions_text = completions_text
            current_correctness = correctness_at_t1
            last_critiques_text = None

            total_rounds = self.self_refine_rounds
            for round_idx in range(1, total_rounds + 1):
                needs_refine_mask = [not x for x in current_correctness]
                refine_indices = [i for i, need in enumerate(needs_refine_mask) if need]

                if self.accelerator.is_main_process:
                    print(f"[Self-Refine] Round {round_idx}/{total_rounds}: Generating critiques and refining ({len(refine_indices)} samples)")

                if not refine_indices:
                    if self.accelerator.is_main_process:
                        print(f"[Self-Refine] All samples correct before refine round {round_idx}. Early stop.")
                    break

                critiques_text = None
                if self.use_critique:
                    if self.accelerator.is_main_process:
                        print(f"[Self-Refine] Round {round_idx}: Generating critiques...")

                    critique_inputs = []
                    for idx, inp in enumerate(inputs):
                        if needs_refine_mask[idx]:
                            original_prompt = prompts[idx]
                            original_completion = current_completions_text[idx]
                            # Keep the full completion including <think> tags for critique
                            ground_truth = solutions[idx]

                            if is_conversational(inp):
                                if isinstance(original_prompt, list):
                                    critique_prompt = []
                                    for i, msg in enumerate(original_prompt):
                                        critique_msg = msg.copy()
                                        if msg["role"] == "system":
                                            critique_msg["content"] = (
                                                "You are an expert critic. You have access to:\n"
                                                "1. The original problem\n"
                                                "2. A student's full solution including their reasoning process\n"
                                                "3. The correct answer\n\n"
                                                "Your task:\n"
                                                "- Compare the student's reasoning and answer against the ground truth\n"
                                                "- Identify WHERE the student's reasoning went wrong (if incorrect)\n"
                                                "- Identify WHAT misconceptions or errors led to the mistake\n"
                                                "- If the student is correct, identify the key insights that made their solution work\n"
                                                "- Provide 2-3 specific, actionable suggestions for improvement\n"
                                                "- Be precise: reference specific steps, calculations, or logical jumps\n\n"
                                                "Format: Brief, focused critique (2-4 sentences max)"
                                            )
                                        is_last_user_msg = (
                                            msg["role"] == "user" and
                                            all(m["role"] != "user" for m in original_prompt[i+1:])
                                        )
                                        if is_last_user_msg:
                                            original_content = msg["content"]
                                            critique_instruction = (
                                                f"\n\n=== STUDENT'S SOLUTION (including reasoning) ===\n"
                                                f"{original_completion}\n\n"
                                                f"=== CORRECT ANSWER ===\n"
                                                f"{ground_truth}\n\n"
                                                f"Analyze the student's solution. Where did they go wrong? What should they focus on?"
                                            )
                                            if isinstance(original_content, str):
                                                critique_msg["content"] = f"{original_content}{critique_instruction}"
                                            elif isinstance(original_content, list):
                                                critique_msg["content"] = original_content.copy()
                                                critique_msg["content"].append({"type": "text", "text": critique_instruction})
                                        critique_prompt.append(critique_msg)
                                else:
                                    critique_prompt = original_prompt
                            else:
                                critique_prompt = (
                                    f"{original_prompt}\n\n"
                                    f"=== STUDENT'S SOLUTION ===\n{original_completion}\n\n"
                                    f"=== CORRECT ANSWER ===\n{ground_truth}\n\n"
                                    f"Critique: What went wrong and what should be improved?"
                                )

                            critique_input = inp.copy()
                            critique_input["prompt"] = critique_prompt
                            critique_inputs.append(critique_input)
                        else:
                            critique_inputs.append(inp)

                    critique_result = super()._generate_and_score_completions(critique_inputs)
                    critiques_text_raw = self.processing_class.batch_decode(critique_result["completion_ids"], skip_special_tokens=True)
                    critiques_text = [self._clean_critique_output(c) for c in critiques_text_raw]
                    last_critiques_text = critiques_text

                    if self.accelerator.is_main_process:
                        example_idx = refine_indices[0]
                        print(f"[Self-Refine] Round {round_idx}: Critique example:")
                        print(f"  Answer: {self._remove_think_tags(current_completions_text[example_idx])[:150]}...")
                        print(f"  Critique: {critiques_text[example_idx][:150]}...")

                if self.accelerator.is_main_process:
                    stage_name = "Refining based on critique..." if self.use_critique else "Refining directly..."
                    print(f"[Self-Refine] Round {round_idx}: {stage_name}")

                refine_inputs = []
                for idx, inp in enumerate(inputs):
                    if needs_refine_mask[idx]:
                        original_prompt = prompts[idx]
                        original_completion = current_completions_text[idx]
                        critique = critiques_text[idx] if critiques_text else None

                        if is_conversational(inp):
                            if isinstance(original_prompt, list):
                                refine_prompt = []
                                for i, msg in enumerate(original_prompt):
                                    refine_msg = msg.copy()
                                    # Keep original system prompt for improver - no changes needed
                                    is_last_user_msg = (
                                        msg["role"] == "user" and
                                        all(m["role"] != "user" for m in original_prompt[i+1:])
                                    )
                                    if is_last_user_msg:
                                        original_content = msg["content"]
                                        if self.use_critique and critique:
                                            refine_instruction = (
                                                f"\n\n=== PREVIOUS ATTEMPT ANALYSIS ===\n"
                                                f"Your earlier attempt had issues. Here's what an expert identified:\n\n"
                                                f"{critique}\n\n"
                                                f"=== YOUR TASK ===\n"
                                                f"Absorb the feedback above, then solve this problem FROM SCRATCH.\n"
                                                f"Do NOT try to patch the previous solution - start fresh with the insights you gained.\n"
                                                f"Think through the problem completely, applying what you learned about:\n"
                                                f"- Where the reasoning went wrong\n"
                                                f"- What approach would be correct\n"
                                                f"- Key concepts or calculations that need attention\n\n"
                                                f"Now solve it yourself, step by step."
                                            )
                                        else:
                                            refine_instruction = (
                                                f"\n\nYour previous attempt was incorrect. Reconsider the problem and solve it from scratch."
                                            )
                                        if isinstance(original_content, str):
                                            refine_msg["content"] = f"{original_content}{refine_instruction}"
                                        elif isinstance(original_content, list):
                                            refine_msg["content"] = original_content.copy()
                                            refine_msg["content"].append({"type": "text", "text": refine_instruction})
                                    refine_prompt.append(refine_msg)
                            else:
                                refine_prompt = original_prompt
                        else:
                            if self.use_critique and critique:
                                refine_prompt = (
                                    f"{original_prompt}\n\n"
                                    f"=== PREVIOUS ATTEMPT ANALYSIS ===\n{critique}\n\n"
                                    f"=== YOUR TASK ===\n"
                                    f"Learn from the feedback above, then solve FROM SCRATCH (do not patch the previous solution)."
                                )
                            else:
                                refine_prompt = (
                                    f"{original_prompt}\n\n"
                                    f"Your previous attempt was incorrect. Solve from scratch."
                                )

                        refine_input = inp.copy()
                        refine_input["prompt"] = refine_prompt
                        refine_inputs.append(refine_input)
                    else:
                        refine_inputs.append(inp)

                new_result = super()._generate_and_score_completions(refine_inputs)
                new_completions_text = self.processing_class.batch_decode(new_result["completion_ids"], skip_special_tokens=True)

                batch_transitions = {"i→c": 0, "i→i": 0, "c→c": 0, "c→i": 0}
                new_correctness = []
                for idx in range(len(current_completions_text)):
                    was_correct = current_correctness[idx]
                    is_correct_now = self._check_if_correct(new_completions_text[idx], solutions[idx])
                    new_correctness.append(is_correct_now)
                    if not was_correct and is_correct_now:
                        batch_transitions["i→c"] += 1
                    elif not was_correct and not is_correct_now:
                        batch_transitions["i→i"] += 1
                    elif was_correct and is_correct_now:
                        batch_transitions["c→c"] += 1
                    else:
                        batch_transitions["c→i"] += 1

                if self.accelerator.is_main_process:
                    prev_correct = sum(current_correctness)
                    curr_correct = sum(new_correctness)
                    acc_prev = 100 * prev_correct / current_batch_size
                    acc_curr = 100 * curr_correct / current_batch_size
                    delta = acc_curr - acc_prev
                    print(f"\n📊 Round {round_idx} Results (n={current_batch_size}):")
                    print(f"  Accuracy@r{round_idx-1}: {prev_correct}/{current_batch_size} = {acc_prev:.2f}%")
                    print(f"  Accuracy@r{round_idx}:   {curr_correct}/{current_batch_size} = {acc_curr:.2f}%")
                    print(f"  Δ(batch):      {delta:+.2f}% {'📈' if delta > 0 else '📉' if delta < 0 else '➡️'}")
                    print(f"\nRound {round_idx} Transitions:")
                    print(f"  i→c: {batch_transitions['i→c']:2d}  |  i→i: {batch_transitions['i→i']:2d}")
                    print(f"  c→c: {batch_transitions['c→c']:2d}  |  c→i: {batch_transitions['c→i']:2d}")
                    print(f"{'='*70}")

                current_result = new_result
                current_completions_text = new_completions_text
                current_correctness = new_correctness
                results_across_rounds.append(new_result)

                if self.refine_until_correct and all(current_correctness):
                    if self.accelerator.is_main_process:
                        print(f"[Self-Refine] All samples correct after round {round_idx}. Early stopping.")
                    break

            # Final logging for the last state vs original
            if self.accelerator.is_main_process:
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

                # Example trace
                example_idx = None
                for i in range(len(original_completions_text)):
                    if not original_correctness[i] or original_completions_text[i] != current_completions_text[i]:
                        example_idx = i
                        break
                if example_idx is None:
                    example_idx = 0
                print(f"\n[Self-Refine] Example trace (final):")
                print(f"  Original: {self._remove_think_tags(original_completions_text[example_idx])[:150]}...")
                if last_critiques_text:
                    print(f"  Last critique: {last_critiques_text[example_idx][:150]}...")
                print(f"  Final:    {self._remove_think_tags(current_completions_text[example_idx])[:150]}...")
                print(f"  Result: {'✓ Correct' if current_correctness[example_idx] else '✗ Incorrect'}")

                for idx in range(len(original_completions_text)):
                    log_entry = {
                        "batch": self.batch_counter,
                        "sample_idx": idx,
                        "timestamp": timestamp,
                        "was_correct_at_t1": original_correctness[idx],
                        "is_correct_at_t2": current_correctness[idx],
                        "transition": None,
                        "original_answer_full": original_completions_text[idx],
                        "original_answer_clean": self._remove_think_tags(original_completions_text[idx]),
                        "solution": solutions[idx],
                    }
                    if not original_correctness[idx] and current_correctness[idx]:
                        log_entry["transition"] = "i→c"
                    elif not original_correctness[idx] and not current_correctness[idx]:
                        log_entry["transition"] = "i→i"
                    elif original_correctness[idx] and current_correctness[idx]:
                        log_entry["transition"] = "c→c"
                    else:
                        log_entry["transition"] = "c→i"
                    if last_critiques_text:
                        log_entry["critique"] = last_critiques_text[idx]
                    log_entry["refined_answer_full"] = current_completions_text[idx]
                    log_entry["refined_answer_clean"] = self._remove_think_tags(current_completions_text[idx])
                    self._log_refine_sample(log_entry)

                print(f"✓ Logged {len(original_completions_text)} samples to {self.refine_log_file}")

            if self.return_both_responses:
                # Combine ALL rounds (original + every refine round)
                combined_result = results_across_rounds[0]
                for r in results_across_rounds[1:]:
                    combined_result = self._combine_results(combined_result, r)
                if self.accelerator.is_main_process:
                    expected = current_batch_size * len(results_across_rounds)
                    print(f"\n[Self-Refine] Returning ALL responses across {len(results_across_rounds)} rounds (including original)")
                    print(f"  Total samples returned: {combined_result['completion_ids'].shape[0]} (should be {expected})")
                return combined_result
            else:
                return current_result

        return result
    
    def _combine_results(self, result1: dict, result2: dict) -> dict:
        """
        Combine two batches by simple concatenation.
        Both batches must have same structure.
        """
        combined = {}
        batch_size1 = result1["completion_ids"].shape[0]
        batch_size2 = result2["completion_ids"].shape[0]
        device = result1["completion_ids"].device
        
        # Padded sequence fields
        padded_fields = {
            "completion_ids": (self.pad_token_id, "right"),
            "completion_mask": (0, "right"),
            "prompt_ids": (self.pad_token_id, "left"),
            "prompt_mask": (0, "left"),
            "policy_per_token_logps": (0.0, "right"),
            "sampling_per_token_logps": (0.0, "right"),
            "old_per_token_logps": (0.0, "right"),
            "ref_per_token_logps": (0.0, "right"),
            "importance_sampling_ratio": (0.0, "right"),
        }
        
        # FIRST PASS: Unpad all padded fields and find max length
        unpadded_data = {}
        max_seq_len = 0
        
        for key in padded_fields:
            if key in result1 and key in result2 and isinstance(result1[key], torch.Tensor):
                padding_value, _ = padded_fields[key]
                val1 = result1[key]
                val2 = result2[key]
                
                # Unpad respecting each batch size
                list1 = [val1[i][val1[i] != padding_value] if (val1[i] != padding_value).any() else val1[i] 
                         for i in range(batch_size1)]
                list2 = [val2[i][val2[i] != padding_value] if (val2[i] != padding_value).any() else val2[i]
                         for i in range(batch_size2)]
                
                # Store unpadded data
                unpadded_data[key] = list1 + list2
                
                # Track max length across all padded fields
                for seq in unpadded_data[key]:
                    max_seq_len = max(max_seq_len, len(seq))
        
        # SECOND PASS: Repad all fields to the same max length
        for key in result1.keys():
            if key not in result2:
                combined[key] = result1[key]
                continue
            
            val1 = result1[key]
            val2 = result2[key]
            
            # Handle padded fields - use unified max_seq_len
            if key in unpadded_data:
                padding_value, padding_side = padded_fields[key]
                # Manual padding to ensure all sequences have same length
                padded_seqs = []
                for seq in unpadded_data[key]:
                    if len(seq) < max_seq_len:
                        pad_len = max_seq_len - len(seq)
                        if padding_side == "right":
                            padded_seq = torch.cat([seq, torch.full((pad_len,), padding_value, dtype=seq.dtype, device=seq.device)])
                        else:  # left
                            padded_seq = torch.cat([torch.full((pad_len,), padding_value, dtype=seq.dtype, device=seq.device), seq])
                    else:
                        padded_seq = seq
                    padded_seqs.append(padded_seq)
                combined[key] = torch.stack(padded_seqs)
                
                # Debug: verify all sequences have same length
                # if self.accelerator.is_main_process and key == "completion_ids":
                #     print(f"[Debug] Unified padding: {key} -> shape {combined[key].shape}, max_seq_len={max_seq_len}")
            
            # Handle special scalar fields  
            elif key == "num_items_in_batch":
                # Sum the counts - MUST keep as 0-dim tensor or int consistently
                if isinstance(val1, torch.Tensor) or isinstance(val2, torch.Tensor):
                    # Convert to tensor if either is tensor
                    v1 = val1 if isinstance(val1, torch.Tensor) else torch.tensor(val1, device=device)
                    v2 = val2 if isinstance(val2, torch.Tensor) else torch.tensor(val2, device=device)  
                    combined[key] = v1 + v2  # Result is 0-dim tensor
                else:
                    # Both are ints - keep as int
                    combined[key] = val1 + val2
            
            # Handle advantages (already computed, just concatenate)
            elif key == "advantages" and isinstance(val1, torch.Tensor):
                combined[key] = torch.cat([val1, val2], dim=0)
            
            # Handle other tensor fields
            elif isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
                # Check dimension before concatenating
                if val1.dim() == 0 or val2.dim() == 0:
                    # Scalar tensors - skip or use val1
                    combined[key] = val1
                else:
                    # Multi-dimensional tensors
                    try:
                        combined[key] = torch.cat([val1, val2], dim=0)
                    except RuntimeError as e:
                        # If concatenation fails, just use val1
                        # if self.accelerator.is_main_process:
                        #     print(f"[Warning] Failed to cat {key}: {e}")
                        combined[key] = val1
            
            # Handle lists
            elif isinstance(val1, list) and isinstance(val2, list):
                combined[key] = val1 + val2
            
            # Handle other types (scalars, etc)
            else:
                combined[key] = val1
        
        return combined
    
    def _fix_param_name_to_vllm(self, name, extra_prefixes=None):
        """Clean up parameter names for vLLM."""
        extra_prefixes = extra_prefixes or []
        prefixes = ["_checkpoint_wrapped_module."] + extra_prefixes
        for prefix in prefixes:
            name = name.replace(prefix, "")
        return name
    
    def _move_model_to_vllm(self):
        """
        Override: Follow TRL's original approach for FSDP2 + PEFT
        1. Merge adapter
        2. Sync merged parameters to vLLM  
        3. Unmerge adapter
        """
        from accelerate.utils import is_peft_model
        
        if not self.use_vllm:
            return super()._move_model_to_vllm()
        
        # For non-PEFT models, use parent implementation
        if not is_peft_model(self.model):
            return super()._move_model_to_vllm()
        
        # FSDP (both v1 and v2) + PEFT: Manually merge LoRA weights into base model
        # Note: merge_adapter() doesn't work properly with FSDP due to sharded weights
        #       - FSDP2: causes "mixed Tensor and DTensor" errors
        #       - FSDP1: causes "inconsistent tensor size" errors in get_delta_weight
        # Solution: Manually compute delta_weight = lora_B @ lora_A and add to base weight
        
        state_dict = self.model.state_dict()
        
        # Collect base weights and LoRA weights
        base_weights = {}
        lora_A_weights = {}
        lora_B_weights = {}
        
        # Debug: print parameter names to understand structure
        # if self.accelerator.is_main_process:
        #     all_keys = list(state_dict.keys())
        #     print(f"[Debug] Total parameters in state_dict: {len(all_keys)}")
        #     print(f"[Debug] First 20 parameter names:")
        #     for n in all_keys[:20]:
        #         print(f"  {n}")
        
        for name, param in state_dict.items():
            
            if param.is_cpu:
                param = param.to(torch.device("cuda"))
            # full_tensor() is only for FSDP2's DTensor, FSDP1 uses regular Tensor
            if hasattr(param, 'full_tensor'):
                param = param.full_tensor()
            
            # Clean PEFT prefixes
            clean_name = name.removeprefix("base_model.model.")
            
            if "lora_A" in name:
                # Extract module name (e.g., "model.layers.0.self_attn.q_proj")
                module_name = clean_name.replace(".lora_A.default.weight", "").replace(".lora_A.weight", "")
                lora_A_weights[module_name] = param
            elif "lora_B" in name:
                module_name = clean_name.replace(".lora_B.default.weight", "").replace(".lora_B.weight", "")
                lora_B_weights[module_name] = param
            elif "lora_" not in name.lower():
                # Base weight
                clean_name = clean_name.replace(".base_layer", "")
                if hasattr(self.model, 'prefix') and self.model.prefix in clean_name:
                    continue
                if "original_module" in clean_name:
                    continue
                base_weights[clean_name] = param
        
        # # Debug: check what we collected
        # if self.accelerator.is_main_process:
        #     print(f"[Debug] Collected {len(base_weights)} base weights")
        #     print(f"[Debug] Collected {len(lora_A_weights)} LoRA A weights")
        #     print(f"[Debug] Collected {len(lora_B_weights)} LoRA B weights")
        #     if base_weights:
        #         print(f"[Debug] Sample base_weight keys: {list(base_weights.keys())[:3]}")
        #     if lora_A_weights:
        #         print(f"[Debug] Sample lora_A keys: {list(lora_A_weights.keys())[:3]}")
        #     if lora_B_weights:
        #         print(f"[Debug] Sample lora_B keys: {list(lora_B_weights.keys())[:3]}")
        
        # Merge LoRA weights into base weights
        lora_config = self.model.peft_config['default']
        lora_alpha = lora_config.lora_alpha
        lora_r = lora_config.r
        scaling = lora_alpha / lora_r
        
        # Use Triton-accelerated batch merge if available
        if TRITON_AVAILABLE:
            # if self.accelerator.is_main_process:
            #     print(f"[LoRA Merge] Using Triton-accelerated batch merge...")
            merged_weights, merged_count = lora_merge_batch_triton(
                base_weights=base_weights,
                lora_A_weights=lora_A_weights,
                lora_B_weights=lora_B_weights,
                scaling=scaling,
                verbose=self.accelerator.is_main_process,
            )
        else:
            # Fallback to PyTorch implementation
            # if self.accelerator.is_main_process:
            #     print(f"[LoRA Merge] Using PyTorch fallback (Triton unavailable)...")
            
            merged_weights = {}
            merged_count = 0
            
            # Merge LoRA into base weights
            # Note: base_weights keys have .weight/.bias suffix, lora keys don't
            for base_name, base_weight in base_weights.items():
                # Extract module name without .weight/.bias suffix
                if base_name.endswith('.weight'):
                    module_name = base_name[:-7]  # remove '.weight'
                elif base_name.endswith('.bias'):
                    module_name = base_name[:-5]  # remove '.bias'
                else:
                    module_name = base_name
                
                if module_name in lora_A_weights and module_name in lora_B_weights and base_name.endswith('.weight'):
                    # Manually compute LoRA delta: delta = (lora_B @ lora_A) * scaling
                    lora_A = lora_A_weights[module_name]
                    lora_B = lora_B_weights[module_name]
                    
                    # Compute delta_weight = lora_B @ lora_A * scaling
                    delta_weight = (lora_B @ lora_A) * scaling
                    
                    # Add to base weight
                    merged_weight = base_weight + delta_weight
                    merged_weights[base_name] = merged_weight
                    merged_count += 1
                else:
                    # No LoRA for this weight, or it's a bias -> use base weight as-is
                    merged_weights[base_name] = base_weight
        
        if self.accelerator.is_main_process:
            backend = "Triton" if TRITON_AVAILABLE else "PyTorch"
            print(f"[LoRA Merge] {backend}: Merged {merged_count} LoRA weights (scaling={scaling:.2f})")
        
        # Sync merged weights to vLLM
        for name, param in merged_weights.items():
            # Clean additional prefixes for vLLM
            clean_name = self._fix_param_name_to_vllm(name, extra_prefixes=["modules_to_save.default."])
            
            if self.vllm_mode == "server" and self.accelerator.is_main_process:
                self.vllm_client.update_named_param(clean_name, param)
            elif self.vllm_mode == "colocate":
                llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
                llm_model.load_weights([(clean_name, param)])
        
        # Reset vLLM cache
        if self.vllm_mode == "colocate":
            self.llm.reset_prefix_cache()
        elif self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()

    def compute_liger_loss(self, model, inputs):
        """
        GSPO loss with Liger Kernel acceleration.
        
        Uses LigerFusedLinearGSPOLoss (~2-3x speedup) when importance_sampling_level='sequence_token'.
        Falls back to TRL's Liger implementation for other importance sampling levels.
        """
        if self.importance_sampling_level != "sequence_token":
            # Use TRL's Liger kernel for non-GSPO levels
            return super().compute_liger_loss(model, inputs)
        
        # ===== GSPO with Liger acceleration =====
        from liger_gspo_loss import LigerFusedLinearGSPOLoss
        
        # Get inputs
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        ref_per_token_logps = inputs.get("ref_per_token_logps")
        
        # Get model outputs (hidden states before LM head)
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Get visual inputs
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        num_images = inputs.get("num_images")
        pixel_attention_mask = inputs.get("pixel_attention_mask")
        image_sizes = inputs.get("image_sizes")
        
        # Forward pass through model to get hidden states
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        if num_images is not None:
            model_inputs["num_images"] = num_images
        if pixel_attention_mask is not None:
            model_inputs["pixel_attention_mask"] = pixel_attention_mask
        if image_sizes is not None:
            model_inputs["image_sizes"] = image_sizes
        
        outputs = model(**model_inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Last layer hidden states
        
        # Extract completion hidden states
        seq_len = completion_ids.size(1)
        completion_hidden = hidden_states[:, -seq_len:, :]
        
        # Get LM head weight
        lm_head = model.get_output_embeddings()
        lin_weight = lm_head.weight
        lin_bias = lm_head.bias if hasattr(lm_head, 'bias') else None
        
        # Create Liger GSPO loss
        gspo_loss_fn = LigerFusedLinearGSPOLoss(
            epsilon_low=self.epsilon_low,
            epsilon_high=self.epsilon_high,
            delta=getattr(self.args, 'delta', None),
            loss_type=self.loss_type,
            beta=self.beta,
            temperature=self.temperature if hasattr(self, 'temperature') else 1.0,
            use_ref_model=self.beta != 0.0,
        )
        
        # Compute loss
        loss, per_token_logps, ref_logps = gspo_loss_fn(
            _input=completion_hidden,
            lin_weight=lin_weight,
            selected_token_ids=completion_ids,
            attention_mask=completion_mask,
            advantages=advantages,
            bias=lin_bias,
            ref_per_token_logps=ref_per_token_logps,
            old_per_token_logps=old_per_token_logps,
        )
        
        return loss

    def _compute_loss(self, model, inputs):
        """
        Override TRL's _compute_loss to add GSPO (sequence_token) importance sampling support.
        
        GSPO adds a hybrid importance sampling level that combines sequence-level and token-level weights:
        - sequence_token: sg[si(θ)] * πθ(yi,t)/sg[πθ(yi,t)]
        
        Reference: gspo_trainer.py lines 75-79
        """
        # Call parent's _compute_loss for standard computation
        parent_loss = super()._compute_loss(model, inputs)
        
        # If using standard importance_sampling_level, return parent's result
        if self.importance_sampling_level != "sequence_token":
            return parent_loss
        
        # ===== GSPO: Re-compute with sequence_token importance sampling =====
        # We need to recompute the loss with GSPO-specific importance weights
        
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        
        # Compute per-token log probs and entropies
        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )
        
        # Get entropy mask if needed
        if self.top_entropy_quantile < 1.0:
            entropy_mask = self.get_high_entropy_mask(entropies, completion_mask, 1 - self.top_entropy_quantile)
        else:
            entropy_mask = None
        
        # Compute KL divergence if using reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )
        
        # Get advantages and old log probs
        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps
        
        log_ratio = per_token_logps - old_per_token_logps
        
        # ===== GSPO-specific importance sampling (sequence_token) =====
        # GSPO-token: sg[si(θ)] * πθ(yi,t)/sg[πθ(yi,t)]
        seq_level_log_weight = (log_ratio * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        seq_level_log_weight = seq_level_log_weight.detach().unsqueeze(-1)  # Stop gradient
        log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
        # Shape: (B, T) - token-level weights with sequence-level guidance
        
        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
        
        # GSPO delta clipping (if specified)
        if hasattr(self.args, 'delta') and self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)
        
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        if entropy_mask is not None:
            per_token_loss = per_token_loss * entropy_mask
        
        if self.use_vllm and self.vllm_importance_sampling_correction:
            per_token_loss = per_token_loss * inputs["importance_sampling_ratio"]
        
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        
        # Compute final loss based on loss_type
        if self.loss_type == "grpo":
            loss = ((per_token_loss * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)).mean()
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "bnpo":
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (per_token_loss.size(0) * self.max_completion_length)
            loss = loss / self.current_gradient_accumulation_steps
        elif self.loss_type == "dapo":
            normalizer = inputs["num_items_in_batch"] / self.accelerator.num_processes
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Log metrics (reuse parent's logging in compute_loss)
        return loss

    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        """
        Override to support micro-batching when return_both_responses doubles batch size.
        Works with both FSDP1 and FSDP2 (not DeepSpeed ZeRO-3).
        """
        if return_outputs:
            raise ValueError("return_outputs=True is not supported with micro-batching")
        
        # If no micro-batching needed, use parent implementation
        if self.micro_batch_size is None:
            return super().compute_loss(model, inputs, num_items_in_batch)
        
        batch_size = inputs["completion_ids"].shape[0]
        if batch_size <= self.micro_batch_size:
            return super().compute_loss(model, inputs, num_items_in_batch)
        
        # Check if we're using FSDP (micro-batching supported for both FSDP1 and FSDP2)
        if not self.is_fsdp_enabled:
            # Fall back to parent for non-FSDP
            if self.accelerator.is_main_process:
                print(f"[Warning] micro_batch_size requires FSDP, falling back to full batch")
            return super().compute_loss(model, inputs, num_items_in_batch)
        
        # FSDP: Implement micro-batching by manually splitting forward passes
        num_micro_batches = (batch_size + self.micro_batch_size - 1) // self.micro_batch_size
        
        # Sort by sequence length to reduce padding within each micro-batch
        # Use completion_mask to compute actual sequence lengths
        completion_mask = inputs.get("completion_mask")
        if completion_mask is not None:
            seq_lengths = completion_mask.sum(dim=1)
            sorted_indices = torch.argsort(seq_lengths, descending=True)
            
            # Pre-compute original visual data indices before reordering (needed for proper slicing)
            has_visual_data_presort = "image_grid_thw" in inputs and "pixel_values" in inputs
            if has_visual_data_presort:
                orig_image_grid_thw = inputs["image_grid_thw"]
                orig_num_images = inputs.get("num_images")
                
                # Build patch/image ranges for original order
                if orig_num_images is not None:
                    orig_rows_per_image = orig_image_grid_thw.prod(dim=-1)
                    orig_rows_per_sample = torch.split(orig_rows_per_image, orig_num_images)
                    orig_rows_per_sample = torch.stack([s.sum() for s in orig_rows_per_sample])
                    orig_patch_starts = torch.cat([torch.tensor([0], device=orig_rows_per_sample.device), orig_rows_per_sample.cumsum(0)])
                    orig_image_starts = torch.tensor([0] + orig_num_images, device=orig_image_grid_thw.device).cumsum(0)
                else:
                    orig_patches_per_sample = orig_image_grid_thw[:, 0] * orig_image_grid_thw[:, 1] * orig_image_grid_thw[:, 2]
                    orig_patch_starts = torch.cat([torch.tensor([0], device=orig_patches_per_sample.device), orig_patches_per_sample.cumsum(0)])
                    orig_image_starts = torch.arange(batch_size + 1, device=orig_image_grid_thw.device)
            
            # Reorder all inputs according to sorted indices
            sorted_inputs = {}
            for key, val in inputs.items():
                if key == "pixel_values" and has_visual_data_presort:
                    # Reorder pixel_values by gathering patches for each sorted sample
                    sorted_patches = []
                    for idx in sorted_indices.tolist():
                        patch_start = orig_patch_starts[idx].item()
                        patch_end = orig_patch_starts[idx + 1].item()
                        sorted_patches.append(val[patch_start:patch_end])
                    sorted_inputs[key] = torch.cat(sorted_patches, dim=0)
                elif key == "image_grid_thw" and has_visual_data_presort:
                    # Reorder image_grid_thw
                    if orig_num_images is not None:
                        sorted_grids = []
                        for idx in sorted_indices.tolist():
                            img_start = orig_image_starts[idx].item()
                            img_end = orig_image_starts[idx + 1].item()
                            sorted_grids.append(val[img_start:img_end])
                        sorted_inputs[key] = torch.cat(sorted_grids, dim=0)
                    else:
                        sorted_inputs[key] = val[sorted_indices]
                elif key == "pixel_attention_mask" and has_visual_data_presort:
                    # Reorder pixel_attention_mask
                    sorted_masks = []
                    for idx in sorted_indices.tolist():
                        patch_start = orig_patch_starts[idx].item()
                        patch_end = orig_patch_starts[idx + 1].item()
                        sorted_masks.append(val[patch_start:patch_end])
                    sorted_inputs[key] = torch.cat(sorted_masks, dim=0)
                elif key == "image_sizes" and has_visual_data_presort:
                    # Reorder image_sizes
                    if orig_num_images is not None:
                        sorted_sizes = []
                        for idx in sorted_indices.tolist():
                            img_start = orig_image_starts[idx].item()
                            img_end = orig_image_starts[idx + 1].item()
                            sorted_sizes.append(val[img_start:img_end])
                        sorted_inputs[key] = torch.cat(sorted_sizes, dim=0)
                    else:
                        sorted_inputs[key] = val[sorted_indices]
                elif isinstance(val, torch.Tensor):
                    if val.dim() == 0:
                        sorted_inputs[key] = val
                    else:
                        sorted_inputs[key] = val[sorted_indices]
                elif isinstance(val, list):
                    sorted_inputs[key] = [val[i] for i in sorted_indices.tolist()]
                else:
                    sorted_inputs[key] = val
            inputs = sorted_inputs
        
        if self.accelerator.is_main_process:
            if completion_mask is not None:
                sorted_seq_lengths = seq_lengths[sorted_indices]
                print(f"[Micro-batching] Sorted by length (descending): min={sorted_seq_lengths.min().item()}, max={sorted_seq_lengths.max().item()}, mean={sorted_seq_lengths.float().mean().item():.1f}")
            print(f"[Micro-batching] Splitting batch of {batch_size} into {num_micro_batches} micro-batches of size {self.micro_batch_size}")
        
        # Pre-compute visual data indices for slicing (after sorting)
        has_visual_data = "image_grid_thw" in inputs and "pixel_values" in inputs
        if has_visual_data:
            image_grid_thw = inputs["image_grid_thw"]
            num_images = inputs.get("num_images")
            
            if num_images is not None:
                rows_per_image = image_grid_thw.prod(dim=-1)
                rows_per_sample = torch.split(rows_per_image, num_images)
                rows_per_sample = torch.stack([s.sum() for s in rows_per_sample])
                patch_starts = torch.cat([torch.tensor([0], device=rows_per_sample.device), rows_per_sample.cumsum(0)])
                image_starts = torch.tensor([0] + num_images, device=image_grid_thw.device).cumsum(0)
            else:
                patches_per_sample = image_grid_thw[:, 0] * image_grid_thw[:, 1] * image_grid_thw[:, 2]
                patch_starts = torch.cat([torch.tensor([0], device=patches_per_sample.device), patches_per_sample.cumsum(0)])
                image_starts = torch.arange(batch_size + 1, device=image_grid_thw.device)
        
        # Accumulate loss across micro-batches with proper gradient management
        total_loss = 0.0
        use_paged_accumulation = hasattr(self, 'paged_grad_accumulator')
        
        for i in range(num_micro_batches):
            start_idx = i * self.micro_batch_size
            end_idx = min(start_idx + self.micro_batch_size, batch_size)
            
            # Slice inputs for this micro-batch
            micro_inputs = {}
            for key, val in inputs.items():
                if key == "pixel_values" and has_visual_data:
                    patch_start = patch_starts[start_idx].item()
                    patch_end = patch_starts[end_idx].item()
                    micro_inputs[key] = val[patch_start:patch_end]
                elif key == "image_grid_thw" and has_visual_data:
                    img_start = image_starts[start_idx].item()
                    img_end = image_starts[end_idx].item()
                    micro_inputs[key] = val[img_start:img_end]
                elif key == "num_images" and has_visual_data:
                    micro_inputs[key] = val[start_idx:end_idx]
                elif key == "pixel_attention_mask" and has_visual_data:
                    patch_start = patch_starts[start_idx].item()
                    patch_end = patch_starts[end_idx].item()
                    micro_inputs[key] = val[patch_start:patch_end]
                elif key == "image_sizes" and has_visual_data:
                    img_start = image_starts[start_idx].item()
                    img_end = image_starts[end_idx].item()
                    micro_inputs[key] = val[img_start:img_end]
                elif isinstance(val, torch.Tensor):
                    if val.dim() == 0:
                        micro_inputs[key] = val
                    else:
                        micro_inputs[key] = val[start_idx:end_idx]
                elif isinstance(val, list):
                    micro_inputs[key] = val[start_idx:end_idx]
                else:
                    micro_inputs[key] = val
            
            # Compute micro_num_items_in_batch
            if num_items_in_batch is not None:
                if isinstance(num_items_in_batch, torch.Tensor):
                    total_items = num_items_in_batch.item() if num_items_in_batch.dim() == 0 else num_items_in_batch
                else:
                    total_items = num_items_in_batch
                micro_num_items = int(total_items * (end_idx - start_idx) / batch_size)
            else:
                micro_num_items = None
            
            # Forward pass for this micro-batch (calls parent's compute_loss)
            micro_loss = super().compute_loss(model, micro_inputs, micro_num_items)
            
            # Scale loss by micro-batch size proportion
            weight = (end_idx - start_idx) / batch_size
            weighted_loss = micro_loss * weight
            
            # Backward pass
            if use_paged_accumulation:
                # With paged accumulation: always backward immediately and accumulate
                weighted_loss.backward()
                self.paged_grad_accumulator.accumulate(scale_factor=1.0)
                
                if i < num_micro_batches - 1:
                    # Not last: zero gradients to save memory (already in 8-bit)
                    model.zero_grad(set_to_none=True)
                    total_loss += weighted_loss.detach()
                else:
                    # Last: apply all accumulated gradients
                    self.paged_grad_accumulator.apply_accumulated_gradients(normalize=False)
                    total_loss = total_loss + weighted_loss.detach()
            else:
                # Without paged accumulation: original behavior
                if i < num_micro_batches - 1:
                    weighted_loss.backward()
                    total_loss += weighted_loss.detach()
                else:
                    # Last micro-batch: accumulate without backward (trainer will handle it)
                    total_loss = total_loss + weighted_loss
        
        return total_loss
