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
                 micro_batch_size: int = None, **kwargs):
        """
        Initialize the GRPO trainer with optional self-refine capability.
        
        Args:
            enable_self_refine: If True, will attempt to self-refine incorrect responses
            use_critique: If True, will first generate a critique before refining (two-stage refine)
            refine_log_file: Path to save self-refine logs (default: "self_refine_log.jsonl")
            return_both_responses: If True, return both original and refined responses (doubles sample size)
            micro_batch_size: If set and using FSDP2, split large batches to avoid OOM
        """
        super().__init__(*args, **kwargs)
        self.enable_self_refine = enable_self_refine
        self.use_critique = use_critique
        self.refine_log_file = refine_log_file
        self.return_both_responses = return_both_responses
        self.micro_batch_size = micro_batch_size
        self.batch_counter = 0
        
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
        
        # If some samples need refinement, regenerate the ENTIRE batch with modified prompts
        if refine_indices:
            # Store batch size (current batch only, not cumulative!)
            current_batch_size = len(completions_text)
            self.batch_counter += 1
            
            if self.accelerator.is_main_process:
                # Calculate batch-level accuracy at t1
                batch_correct_t1 = sum(correctness_at_t1)
                batch_acc_t1 = 100 * batch_correct_t1 / current_batch_size
                
                print(f"\n{'='*70}")
                print(f"[Self-Refine] Batch {self.batch_counter} Statistics")
                print(f"{'='*70}")
                print(f"Batch size: {current_batch_size}")
                print(f"Batch Accuracy@t1: {batch_correct_t1}/{current_batch_size} = {batch_acc_t1:.2f}%")
                print(f"Samples needing refine: {len(refine_indices)}")
                print(f"Log file: {self.refine_log_file}")
                print(f"{'-'*70}")
            
            critiques_text = None
            
            # Stage 1: Generate critiques if enabled
            if self.use_critique:
                if self.accelerator.is_main_process:
                    print(f"[Self-Refine] Stage 1: Generating critiques...")
                
                # Create critique prompts for incorrect samples
                critique_inputs = []
                for idx, inp in enumerate(inputs):
                    if needs_refine_mask[idx]:
                        # Generate critique for this sample
                        original_prompt = prompts[idx]
                        original_completion = completions_text[idx]
                        # Remove <think> tags - only show final answer for critique
                        completion_for_critique = self._remove_think_tags(original_completion)
                        
                        # Create critique prompt
                        if is_conversational(inp):
                            if isinstance(original_prompt, list):
                                critique_prompt = []
                                for i, msg in enumerate(original_prompt):
                                    critique_msg = msg.copy()
                                    
                                    # Replace system prompt for critique agent
                                    if msg["role"] == "system":
                                        critique_msg["content"] = (
                                            "You are an expert evaluator. Your task is to:\n"
                                            "1. Analyze the given solution carefully\n"
                                            "2. Identify any errors in reasoning or calculation\n"
                                            "3. Point out what's correct and what needs improvement\n"
                                            "4. Be specific and constructive in your feedback"
                                        )
                                    
                                    is_last_user_msg = (
                                        msg["role"] == "user" and
                                        all(m["role"] != "user" for m in original_prompt[i+1:])
                                    )
                                    
                                    if is_last_user_msg:
                                        original_content = msg["content"]
                                        
                                        # Critique instruction - ask evaluator to review the solution
                                        # completion_for_critique now contains "Self-review:\n...\n\nSolution:\n..."
                                        critique_instruction = (
                                            f"\n\nHere is a student's attempt:\n\n"
                                            f"{completion_for_critique}\n\n"
                                            f"Please evaluate this solution:\n"
                                            f"- Is the self-review accurate?\n"
                                            f"- Are the reasoning steps correct?\n"
                                            f"- Are calculations accurate?\n"
                                            f"- Is the final answer correct?\n\n"
                                            f"Provide your analysis and point out any errors or issues you find."
                                        )
                                        
                                        if isinstance(original_content, str):
                                            critique_msg["content"] = f"{original_content}{critique_instruction}"
                                        elif isinstance(original_content, list):
                                            critique_msg["content"] = original_content.copy()
                                            critique_msg["content"].append({
                                                "type": "text",
                                                "text": critique_instruction
                                            })
                                    
                                    critique_prompt.append(critique_msg)
                            else:
                                critique_prompt = original_prompt
                        else:
                            critique_prompt = (
                                f"{original_prompt}\n\n"
                                f"{completion_for_critique}\n\n"
                                f"Evaluate the above. Check the self-review accuracy, verify calculations, "
                                f"and confirm the answer in \\boxed{{}}. Point out errors or issues."
                            )
                        
                        critique_input = inp.copy()
                        critique_input["prompt"] = critique_prompt
                        critique_inputs.append(critique_input)
                    else:
                        # Correct samples don't need critique
                        critique_inputs.append(inp)
                
                # Generate critiques
                critique_result = super()._generate_and_score_completions(critique_inputs)
                critiques_text_raw = self.processing_class.batch_decode(
                    critique_result["completion_ids"], 
                    skip_special_tokens=True
                )
                
                # Normalize critique output - convert <thinking> to <think> tags
                critiques_text = [self._clean_critique_output(c) for c in critiques_text_raw]
                
                if self.accelerator.is_main_process:
                    print(f"[Self-Refine] Stage 1 complete. Example:")
                    example_idx = refine_indices[0]
                    print(f"  Answer: {self._remove_think_tags(completions_text[example_idx])[:150]}...")
                    print(f"  Critique: {critiques_text[example_idx][:150]}...")
            
            # Stage 2: Refine based on critique (or directly if no critique)
            if self.accelerator.is_main_process:
                stage_name = "Stage 2: Refining based on critique..." if self.use_critique else "Refining directly..."
                print(f"[Self-Refine] {stage_name}")
            
            # Create refine inputs
            refine_inputs = []
            for idx, inp in enumerate(inputs):
                if needs_refine_mask[idx]:
                    # This sample needs refine - modify its prompt
                    original_prompt = prompts[idx]
                    original_completion = completions_text[idx]
                    # Remove <think> tags - only show final answer for refine
                    completion_for_refine = self._remove_think_tags(original_completion)
                    critique = critiques_text[idx] if critiques_text else None
                    
                    # Create refine prompt
                    if is_conversational(inp):
                        if isinstance(original_prompt, list):
                            refine_prompt = []
                            for i, msg in enumerate(original_prompt):
                                refine_msg = msg.copy()
                                
                                is_last_user_msg = (
                                    msg["role"] == "user" and
                                    all(m["role"] != "user" for m in original_prompt[i+1:])
                                )
                                
                                if is_last_user_msg:
                                    original_content = msg["content"]
                                    
                                    if self.use_critique and critique:
                                        # Natural format: provide original answer as reference + critique as guidance
                                        # Frame it as collaborative refinement
                                        refine_instruction = (
                                            f"\n\nA reference solution:\n{completion_for_refine}\n\n"
                                            f"Key observations: {critique}\n\n"
                                            f"Now provide your solution to this problem."
                                        )
                                    else:
                                        # Without critique, show attempt and ask to refine
                                        refine_instruction = (
                                            f"\n\nAn initial attempt:\n{completion_for_refine}\n\n"
                                            f"Verify and refine this solution."
                                        )
                                    
                                    if isinstance(original_content, str):
                                        refine_msg["content"] = f"{original_content}{refine_instruction}"
                                    elif isinstance(original_content, list):
                                        refine_msg["content"] = original_content.copy()
                                        refine_msg["content"].append({
                                            "type": "text",
                                            "text": refine_instruction
                                        })
                                
                                refine_prompt.append(refine_msg)
                        else:
                            refine_prompt = original_prompt
                    else:
                        if self.use_critique and critique:
                            refine_prompt = (
                                f"{original_prompt}\n\n"
                                f"A reference solution:\n{completion_for_refine}\n\n"
                                f"Key observations: {critique}\n\n"
                                f"Provide your solution."
                            )
                        else:
                            refine_prompt = (
                                f"{original_prompt}\n\n"
                                f"An initial attempt:\n{completion_for_refine}\n\n"
                                f"Verify and refine."
                            )
                    
                    refine_input = inp.copy()
                    refine_input["prompt"] = refine_prompt
                    refine_inputs.append(refine_input)
                else:
                    # This sample is already correct, keep original prompt
                    refine_inputs.append(inp)
            
            # Regenerate for the ENTIRE batch with the modified inputs
            refined_result = super()._generate_and_score_completions(refine_inputs)
            
            # Check correctness at t2 (after refine) and update statistics
            if self.accelerator.is_main_process:
                refined_completions = self.processing_class.batch_decode(
                    refined_result["completion_ids"], 
                    skip_special_tokens=True
                )
                
                # Calculate batch-level statistics
                batch_correctness_t2 = []
                batch_transitions = {"i→c": 0, "i→i": 0, "c→c": 0, "c→i": 0}
                
                # Track transitions for all samples in this batch
                for idx in range(len(completions_text)):
                    was_correct_at_t1 = correctness_at_t1[idx]
                    is_correct_at_t2 = self._check_if_correct(refined_completions[idx], solutions[idx])
                    batch_correctness_t2.append(is_correct_at_t2)
                    
                    # Update batch transition counters
                    if not was_correct_at_t1 and is_correct_at_t2:
                        batch_transitions["i→c"] += 1
                    elif not was_correct_at_t1 and not is_correct_at_t2:
                        batch_transitions["i→i"] += 1
                    elif was_correct_at_t1 and is_correct_at_t2:
                        batch_transitions["c→c"] += 1
                    elif was_correct_at_t1 and not is_correct_at_t2:
                        batch_transitions["c→i"] += 1
                
                # Calculate batch accuracy at t2 (current batch only, not cumulative!)
                batch_correct_t2 = sum(batch_correctness_t2)
                batch_acc_t2 = 100 * batch_correct_t2 / current_batch_size
                batch_delta = batch_acc_t2 - batch_acc_t1
                
                # Print batch-level results (current batch only)
                print(f"[Self-Refine] {'Stage 2 complete!' if self.use_critique else 'Refine complete!'}")
                print(f"\n📊 Current Batch Results (n={current_batch_size}):")
                print(f"  Accuracy@t1: {batch_correct_t1}/{current_batch_size} = {batch_acc_t1:.2f}%")
                print(f"  Accuracy@t2: {batch_correct_t2}/{current_batch_size} = {batch_acc_t2:.2f}%")
                print(f"  Δ(batch):    {batch_delta:+.2f}% {'📈' if batch_delta > 0 else '📉' if batch_delta < 0 else '➡️'}")
                print(f"\nBatch Transitions:")
                print(f"  i→c: {batch_transitions['i→c']:2d}  |  i→i: {batch_transitions['i→i']:2d}")
                print(f"  c→c: {batch_transitions['c→c']:2d}  |  c→i: {batch_transitions['c→i']:2d}")
                print(f"{'='*70}")
                
                # Show example trace (without <think> tags for clarity)
                if refine_indices:
                    print(f"\n[Self-Refine] Example trace:")
                    example_idx = refine_indices[0]
                    
                    # Show cleaned versions (without <think> tags)
                    original_clean = self._remove_think_tags(completions_text[example_idx])
                    refined_clean = self._remove_think_tags(refined_completions[example_idx])
                    
                    print(f"  Original answer: {original_clean[:150]}...")
                    if self.use_critique and critiques_text:
                        print(f"  Critique: {critiques_text[example_idx][:150]}...")
                    print(f"  Refined answer: {refined_clean[:150]}...")
                    is_correct_now = self._check_if_correct(refined_completions[example_idx], solutions[example_idx])
                    print(f"  Result: {'✓ Correct' if is_correct_now else '✗ Still incorrect'}")
                
                # Log all samples to file
                import time
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                
                for idx in range(len(completions_text)):
                    log_entry = {
                        "batch": self.batch_counter,
                        "sample_idx": idx,
                        "timestamp": timestamp,
                        "was_correct_at_t1": correctness_at_t1[idx],
                        "is_correct_at_t2": batch_correctness_t2[idx],
                        "transition": None,
                        "original_answer_full": completions_text[idx],
                        "original_answer_clean": self._remove_think_tags(completions_text[idx]),
                        "solution": solutions[idx],
                    }
                    
                    # Determine transition
                    if not correctness_at_t1[idx] and batch_correctness_t2[idx]:
                        log_entry["transition"] = "i→c"
                    elif not correctness_at_t1[idx] and not batch_correctness_t2[idx]:
                        log_entry["transition"] = "i→i"
                    elif correctness_at_t1[idx] and batch_correctness_t2[idx]:
                        log_entry["transition"] = "c→c"
                    else:
                        log_entry["transition"] = "c→i"
                    
                    # Add critique if available
                    if self.use_critique and critiques_text:
                        log_entry["critique"] = critiques_text[idx]
                    
                    # Add refined answer
                    log_entry["refined_answer_full"] = refined_completions[idx]
                    log_entry["refined_answer_clean"] = self._remove_think_tags(refined_completions[idx])
                    
                    # Save to log
                    self._log_refine_sample(log_entry)
                
                if self.accelerator.is_main_process:
                    print(f"✓ Logged {len(completions_text)} samples to {self.refine_log_file}")
            
            # Return both original and refined results if enabled
            if self.return_both_responses:
                if self.accelerator.is_main_process:
                    print(f"\n[Self-Refine] Returning BOTH original and refined responses")
                    print(f"  Original batch size: {current_batch_size}")
                    
                    # # DEBUG: Print all keys in both results
                    # print(f"\n[DEBUG] Result keys:")
                    # print(f"  Original keys: {list(result.keys())}")
                    # print(f"  Refined keys: {list(refined_result.keys())}")
                    
                    # # Check shape of each field
                    # print(f"\n[DEBUG] Field shapes:")
                    # for key in result.keys():
                    #     if isinstance(result[key], torch.Tensor):
                    #         print(f"    {key}: orig={result[key].shape}, refined={refined_result.get(key, 'N/A')}")
                
                # Concatenate original and refined results
                combined_result = self._combine_results(result, refined_result)
                
                if self.accelerator.is_main_process:
                    print(f"  Total samples returned: {combined_result['completion_ids'].shape[0]} (should be {current_batch_size * 2})")
                
                return combined_result
            else:
                # Only return the refined result
                return refined_result
        
        return result
    
    def _combine_results(self, result1: dict, result2: dict) -> dict:
        """
        Combine two batches by simple concatenation.
        Both batches must have same structure.
        """
        combined = {}
        batch_size = result1["completion_ids"].shape[0]
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
                
                # Unpad
                list1 = [val1[i][val1[i] != padding_value] if (val1[i] != padding_value).any() else val1[i] 
                         for i in range(batch_size)]
                list2 = [val2[i][val2[i] != padding_value] if (val2[i] != padding_value).any() else val2[i]
                         for i in range(batch_size)]
                
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
        
        if self.accelerator.is_main_process:
            print(f"[Micro-batching] Splitting batch of {batch_size} into {num_micro_batches} micro-batches of size {self.micro_batch_size}")
        
        # Pre-compute visual data indices for slicing
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
            
            # For all but the last micro-batch: backward immediately and detach
            # This releases the computation graph to save memory
            if i < num_micro_batches - 1:
                weighted_loss.backward()
                total_loss += weighted_loss.detach()
            else:
                # Last micro-batch: accumulate without backward (trainer will handle it)
                total_loss = total_loss + weighted_loss
        
        return total_loss
