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

"""
Self-Refine Agent: Multi-round refinement with Answer/Critique/Refine sub-agents
"""

import torch
from typing import List, Tuple, Optional, Dict, Any


class SelfRefineAgent:
    """
    Multi-round self-refinement with three sub-agents:
    1. Answer Agent: Generate initial/refined answers
    2. Critique Agent: Evaluate and critique current answers
    3. Refine Agent: Improve answers based on critique
    
    Supports configurable number of refinement rounds.
    
    Usage:
        agent = SelfRefineAgent(trainer, max_refine_rounds=2)
        
        # For single-round refine (current implementation)
        refined_prompts, refined_batch = agent.refine_incorrect_samples(
            original_prompts, answers, correctness_mask
        )
        
        # For multi-round refine (future implementation)
        history = agent.run_multi_round_refine(
            original_prompts, answers, correctness_mask
        )
    """
    
    def __init__(self, trainer, max_refine_rounds: int = 1):
        """
        Args:
            trainer: The GRPOTrainer instance
            max_refine_rounds: Number of refinement rounds (default: 1)
        """
        self.trainer = trainer
        self.max_refine_rounds = max_refine_rounds
        
        # System prompts for each agent
        self.critique_system_prompt = (
            "You are a critical reviewer. Evaluate the given solution and provide constructive feedback.\n\n"
            "Focus on:\n"
            "- Correctness of reasoning and calculations\n"
            "- Completeness of the solution\n"
            "- Clarity and organization\n"
            "- Common mistakes or edge cases\n\n"
            "Format your critique in <think></think> tags for internal analysis, "
            "followed by clear, actionable feedback."
        )
        
        self.refine_system_prompt = (
            "You are a problem solver. Given a question and feedback on a previous attempt, "
            "provide an improved solution.\n\n"
            "Use the standard format:\n"
            "<think>Your unstructured internal reasoning</think>\n"
            "<answer>Clear, well-formatted solution with final answer in \\boxed{}</answer>"
        )
    
    def critique_agent(self, prompts: List, answers: List[str], round_idx: int = 0) -> List[str]:
        """
        Generate critiques for given answers.
        
        Args:
            prompts: List of conversation prompts (list of message dicts)
            answers: List of answer strings to critique
            round_idx: Current refinement round (for logging)
        
        Returns:
            List of critique strings
        """
        import copy
        
        critique_prompts = []
        for prompt, answer in zip(prompts, answers):
            # Extract answer content for critique
            answer_content = self.trainer._extract_answer_for_critique(answer)
            
            # Build critique prompt
            critique_prompt = copy.deepcopy(prompt)
            
            # Override system prompt for critique agent
            critique_prompt[0] = {
                "role": "system",
                "content": self.critique_system_prompt
            }
            
            # Add the answer to critique
            critique_prompt.append({
                "role": "assistant",
                "content": f"Here is my solution:\n\n{answer_content}"
            })
            
            # Request critique
            critique_prompt.append({
                "role": "user",
                "content": (
                    "Please review this solution carefully. "
                    "What aspects are correct? What could be improved? "
                    "Are there any errors in reasoning or calculation?"
                )
            })
            
            critique_prompts.append(critique_prompt)
        
        # Generate critiques using sampler
        critique_result = self.trainer.sampler.generate_from_prompts(critique_prompts, images=None)
        
        # Extract critique texts
        critique_texts = [
            self.trainer.processing_class.decode(ids, skip_special_tokens=True)
            for ids in critique_result['completion_ids']
        ]
        
        # Clean critique output (normalize thinking tags)
        critique_texts = [
            self.trainer._clean_critique_output(text)
            for text in critique_texts
        ]
        
        return critique_texts
    
    def refine_agent(self, prompts: List, answers: List[str], critiques: List[str], 
                     round_idx: int = 0) -> Tuple[List, Dict[str, Any]]:
        """
        Generate refined answers based on critiques.
        
        Args:
            prompts: List of conversation prompts (list of message dicts)
            answers: List of original answer strings
            critiques: List of critique strings
            round_idx: Current refinement round (for logging)
        
        Returns:
            Tuple of (refined_prompts, generation_batch) for scoring
        """
        import copy
        
        refined_prompts = []
        for prompt, answer, critique in zip(prompts, answers, critiques):
            # Extract answer content
            answer_content = self.trainer._extract_answer_for_critique(answer)
            
            # Build refine prompt
            refine_prompt = copy.deepcopy(prompt)
            
            # Override system prompt for refine agent
            refine_prompt[0] = {
                "role": "system",
                "content": self.refine_system_prompt
            }
            
            # Add original attempt
            refine_prompt.append({
                "role": "assistant",
                "content": f"My previous solution:\n\n{answer_content}"
            })
            
            # Add critique
            refine_prompt.append({
                "role": "user",
                "content": f"Feedback on your solution:\n\n{critique}\n\nPlease provide an improved solution."
            })
            
            refined_prompts.append(refine_prompt)
        
        # Return only prompts (sampler will handle generation)
        return refined_prompts
    
    def refine_batch(
        self,
        inputs: List[Dict],
        original_result: Dict,
        original_prompts: List,
        answers: List[str],
        solutions: List[str],
        correctness_mask: torch.Tensor,
        batch_counter: int = 0
    ) -> Optional[Dict]:
        """
        ⭐ ULTIMATE UNIFIED INTERFACE ⭐
        
        Single-call interface: Agent handles EVERYTHING internally.
        Trainer just passes inputs and waits for refined results.
        
        Args:
            inputs: Original batch inputs (for regeneration)
            original_result: Original generation result
            original_prompts: List of original conversation prompts
            answers: List of answer strings
            solutions: List of ground truth solutions
            correctness_mask: Boolean tensor indicating which samples are correct
            batch_counter: Batch number for logging
        
        Returns:
            Refined result dict (or None if no incorrect samples)
            Agent internally handles:
            - Critique generation
            - Refine prompt building
            - Regeneration
            - Evaluation
            - Logging
        """
        # Find incorrect samples
        incorrect_mask = ~correctness_mask
        incorrect_indices = torch.where(incorrect_mask)[0].tolist()
        
        if not incorrect_indices:
            return None
        
        # Internal logging
        if self.trainer.accelerator.is_main_process:
            agent_type = "Critique + Refine" if self.trainer.use_critique else "Direct Refine"
            print(f"\n[Self-Refine Agent] Processing {len(incorrect_indices)} incorrect samples using {agent_type}")
        
        # Extract incorrect samples
        incorrect_prompts = [original_prompts[i] for i in incorrect_indices]
        incorrect_answers = [answers[i] for i in incorrect_indices]
        
        # Generate refined prompts (internal)
        critiques = None
        if self.trainer.use_critique:
            # Two-stage: Critique → Refine
            critiques = self.critique_agent(incorrect_prompts, incorrect_answers, round_idx=0)
            refined_prompts = self.refine_agent(
                incorrect_prompts, incorrect_answers, critiques, round_idx=0
            )
        else:
            # Direct refine without critique
            import copy
            refined_prompts = []
            for prompt, answer in zip(incorrect_prompts, incorrect_answers):
                answer_content = self.trainer._extract_answer_for_critique(answer)
                
                refine_prompt = copy.deepcopy(prompt)
                refine_prompt.append({
                    "role": "assistant",
                    "content": answer_content
                })
                refine_prompt.append({
                    "role": "user",
                    "content": "Your previous answer may be incorrect. Please reconsider and provide an improved solution."
                })
                refined_prompts.append(refine_prompt)
        
        # Expand refined prompts to full batch (internal)
        needs_refine_mask = [i in incorrect_indices for i in range(len(answers))]
        refine_inputs = []
        refined_prompt_idx = 0
        for idx, inp in enumerate(inputs):
            if needs_refine_mask[idx]:
                # Use refined prompt from agent
                refine_input = inp.copy()
                refine_input["prompt"] = refined_prompts[refined_prompt_idx]
                refined_prompt_idx += 1
                refine_inputs.append(refine_input)
            else:
                # Keep original input for correct samples
                refine_inputs.append(inp)
        
        # Regenerate refined answers (internal - use sampler)
        refined_result = self.trainer.sampler.generate_and_score(refine_inputs)
        
        # Decode and evaluate (internal)
        refined_completions = self.trainer.processing_class.batch_decode(
            refined_result["completion_ids"],
            skip_special_tokens=True
        )
        
        # Evaluate and log (internal)
        correctness_at_t1_list = correctness_mask.cpu().tolist()
        self._evaluate_and_log_internal(
            original_answers=answers,
            refined_answers=refined_completions,
            solutions=solutions,
            correctness_at_t1=correctness_at_t1_list,
            incorrect_indices=incorrect_indices,
            critiques=critiques,
            batch_counter=batch_counter
        )
        
        return refined_result
    
    def _evaluate_and_log_internal(
        self,
        original_answers: List[str],
        refined_answers: List[str],
        solutions: List[str],
        correctness_at_t1: List[bool],
        incorrect_indices: List[int],
        critiques: Optional[List[str]],
        batch_counter: int
    ):
        """Internal evaluation and logging (called by refine_batch)."""
        if not self.trainer.accelerator.is_main_process:
            return
        
        current_batch_size = len(original_answers)
        
        # Calculate batch-level statistics
        correctness_at_t2 = []
        batch_transitions = {"i→c": 0, "i→i": 0, "c→c": 0, "c→i": 0}
        
        for idx in range(len(original_answers)):
            was_correct_at_t1 = correctness_at_t1[idx]
            is_correct_at_t2 = self.trainer._check_if_correct(refined_answers[idx], solutions[idx])
            correctness_at_t2.append(is_correct_at_t2)
            
            # Update transition counters
            if not was_correct_at_t1 and is_correct_at_t2:
                batch_transitions["i→c"] += 1
            elif not was_correct_at_t1 and not is_correct_at_t2:
                batch_transitions["i→i"] += 1
            elif was_correct_at_t1 and is_correct_at_t2:
                batch_transitions["c→c"] += 1
            elif was_correct_at_t1 and not is_correct_at_t2:
                batch_transitions["c→i"] += 1
        
        # Calculate accuracies
        batch_correct_t1 = sum(correctness_at_t1)
        batch_correct_t2 = sum(correctness_at_t2)
        batch_acc_t1 = 100 * batch_correct_t1 / current_batch_size
        batch_acc_t2 = 100 * batch_correct_t2 / current_batch_size
        batch_delta = batch_acc_t2 - batch_acc_t1
        
        # Print results
        print(f"\n📊 Self-Refine Agent Evaluation (Batch {batch_counter}):")
        print(f"  Accuracy@t1: {batch_correct_t1}/{current_batch_size} = {batch_acc_t1:.2f}%")
        print(f"  Accuracy@t2: {batch_correct_t2}/{current_batch_size} = {batch_acc_t2:.2f}%")
        print(f"  Δ(batch):    {batch_delta:+.2f}% {'📈' if batch_delta > 0 else '📉' if batch_delta < 0 else '➡️'}")
        print(f"\n  Transitions:")
        print(f"    i→c: {batch_transitions['i→c']:2d}  |  i→i: {batch_transitions['i→i']:2d}")
        print(f"    c→c: {batch_transitions['c→c']:2d}  |  c→i: {batch_transitions['c→i']:2d}")
        
        # Show example
        if incorrect_indices:
            print(f"\n  Example trace:")
            example_idx = incorrect_indices[0]
            original_clean = self.trainer._extract_answer_for_critique(original_answers[example_idx])
            refined_clean = self.trainer._extract_answer_for_critique(refined_answers[example_idx])
            print(f"    Original: {original_clean[:100]}...")
            if self.trainer.use_critique and critiques:
                print(f"    Critique: {critiques[0][:100]}...")
            print(f"    Refined:  {refined_clean[:100]}...")
            is_correct_now = self.trainer._check_if_correct(refined_answers[example_idx], solutions[example_idx])
            print(f"    Result: {'✓ Correct' if is_correct_now else '✗ Still incorrect'}")
        
        # Log all samples to file
        import time
        import json
        from pathlib import Path
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_path = Path(self.trainer.refine_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Expand critiques to full batch
        critiques_full = []
        if critiques:
            critique_idx = 0
            for idx in range(len(original_answers)):
                if idx in incorrect_indices:
                    critiques_full.append(critiques[critique_idx])
                    critique_idx += 1
                else:
                    critiques_full.append(None)
        else:
            critiques_full = [None] * len(original_answers)
        
        for idx in range(len(original_answers)):
            log_entry = {
                "batch": batch_counter,
                "sample_idx": idx,
                "timestamp": timestamp,
                "was_correct_at_t1": correctness_at_t1[idx],
                "is_correct_at_t2": correctness_at_t2[idx],
                "transition": None,
                "original_answer_full": original_answers[idx],
                "original_answer_clean": self.trainer._extract_answer_for_critique(original_answers[idx]),
                "solution": solutions[idx],
            }
            
            # Determine transition
            if not correctness_at_t1[idx] and correctness_at_t2[idx]:
                log_entry["transition"] = "i→c"
            elif not correctness_at_t1[idx] and not correctness_at_t2[idx]:
                log_entry["transition"] = "i→i"
            elif correctness_at_t1[idx] and correctness_at_t2[idx]:
                log_entry["transition"] = "c→c"
            else:
                log_entry["transition"] = "c→i"
            
            # Add critique if available
            if critiques_full[idx]:
                log_entry["critique"] = critiques_full[idx]
            
            # Add refined answer
            log_entry["refined_answer_full"] = refined_answers[idx]
            log_entry["refined_answer_clean"] = self.trainer._extract_answer_for_critique(refined_answers[idx])
            
            # Save to log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"[Warning] Failed to write to log file: {e}")
        
        print(f"  ✓ Logged {len(original_answers)} samples to {self.trainer.refine_log_file}\n")
    
    def evaluate_and_log(
        self,
        original_answers: List[str],
        refined_answers: List[str],
        solutions: List[str],
        correctness_at_t1: List[bool],
        stats: Dict[str, Any],
        batch_counter: int
    ):
        """
        Agent-internal evaluation and logging.
        Called after refined answers are generated.
        
        Args:
            original_answers: Original answer strings
            refined_answers: Refined answer strings
            solutions: Ground truth solutions
            correctness_at_t1: Correctness before refinement
            stats: Stats dict from refine_incorrect_samples
            batch_counter: Batch number
        """
        if not self.trainer.accelerator.is_main_process:
            return
        
        current_batch_size = len(original_answers)
        incorrect_indices = stats['incorrect_indices']
        critiques = stats.get('critiques')
        
        # Calculate batch-level statistics
        correctness_at_t2 = []
        batch_transitions = {"i→c": 0, "i→i": 0, "c→c": 0, "c→i": 0}
        
        for idx in range(len(original_answers)):
            was_correct_at_t1 = correctness_at_t1[idx]
            is_correct_at_t2 = self.trainer._check_if_correct(refined_answers[idx], solutions[idx])
            correctness_at_t2.append(is_correct_at_t2)
            
            # Update transition counters
            if not was_correct_at_t1 and is_correct_at_t2:
                batch_transitions["i→c"] += 1
            elif not was_correct_at_t1 and not is_correct_at_t2:
                batch_transitions["i→i"] += 1
            elif was_correct_at_t1 and is_correct_at_t2:
                batch_transitions["c→c"] += 1
            elif was_correct_at_t1 and not is_correct_at_t2:
                batch_transitions["c→i"] += 1
        
        # Calculate accuracies
        batch_correct_t1 = sum(correctness_at_t1)
        batch_correct_t2 = sum(correctness_at_t2)
        batch_acc_t1 = 100 * batch_correct_t1 / current_batch_size
        batch_acc_t2 = 100 * batch_correct_t2 / current_batch_size
        batch_delta = batch_acc_t2 - batch_acc_t1
        
        # Print results
        print(f"\n📊 Self-Refine Agent Evaluation (Batch {batch_counter}):")
        print(f"  Accuracy@t1: {batch_correct_t1}/{current_batch_size} = {batch_acc_t1:.2f}%")
        print(f"  Accuracy@t2: {batch_correct_t2}/{current_batch_size} = {batch_acc_t2:.2f}%")
        print(f"  Δ(batch):    {batch_delta:+.2f}% {'📈' if batch_delta > 0 else '📉' if batch_delta < 0 else '➡️'}")
        print(f"\n  Transitions:")
        print(f"    i→c: {batch_transitions['i→c']:2d}  |  i→i: {batch_transitions['i→i']:2d}")
        print(f"    c→c: {batch_transitions['c→c']:2d}  |  c→i: {batch_transitions['c→i']:2d}")
        
        # Show example
        if incorrect_indices:
            print(f"\n  Example trace:")
            example_idx = incorrect_indices[0]
            original_clean = self.trainer._extract_answer_for_critique(original_answers[example_idx])
            refined_clean = self.trainer._extract_answer_for_critique(refined_answers[example_idx])
            print(f"    Original: {original_clean[:100]}...")
            if self.trainer.use_critique and critiques:
                print(f"    Critique: {critiques[0][:100]}...")
            print(f"    Refined:  {refined_clean[:100]}...")
            is_correct_now = self.trainer._check_if_correct(refined_answers[example_idx], solutions[example_idx])
            print(f"    Result: {'✓ Correct' if is_correct_now else '✗ Still incorrect'}")
        
        # Log all samples to file
        import time
        import json
        from pathlib import Path
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_path = Path(self.trainer.refine_log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Expand critiques to full batch
        critiques_full = []
        if critiques:
            critique_idx = 0
            for idx in range(len(original_answers)):
                if idx in incorrect_indices:
                    critiques_full.append(critiques[critique_idx])
                    critique_idx += 1
                else:
                    critiques_full.append(None)
        else:
            critiques_full = [None] * len(original_answers)
        
        for idx in range(len(original_answers)):
            log_entry = {
                "batch": batch_counter,
                "sample_idx": idx,
                "timestamp": timestamp,
                "was_correct_at_t1": correctness_at_t1[idx],
                "is_correct_at_t2": correctness_at_t2[idx],
                "transition": None,
                "original_answer_full": original_answers[idx],
                "original_answer_clean": self.trainer._extract_answer_for_critique(original_answers[idx]),
                "solution": solutions[idx],
            }
            
            # Determine transition
            if not correctness_at_t1[idx] and correctness_at_t2[idx]:
                log_entry["transition"] = "i→c"
            elif not correctness_at_t1[idx] and not correctness_at_t2[idx]:
                log_entry["transition"] = "i→i"
            elif correctness_at_t1[idx] and correctness_at_t2[idx]:
                log_entry["transition"] = "c→c"
            else:
                log_entry["transition"] = "c→i"
            
            # Add critique if available
            if critiques_full[idx]:
                log_entry["critique"] = critiques_full[idx]
            
            # Add refined answer
            log_entry["refined_answer_full"] = refined_answers[idx]
            log_entry["refined_answer_clean"] = self.trainer._extract_answer_for_critique(refined_answers[idx])
            
            # Save to log
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"[Warning] Failed to write to log file: {e}")
        
        print(f"  ✓ Logged {len(original_answers)} samples to {self.trainer.refine_log_file}\n")
    
    def run_multi_round_refine(
        self,
        original_prompts: List,
        answers: List[str],
        correctness_mask: torch.Tensor
    ) -> List[Dict[str, Any]]:
        """
        Multi-round self-refinement on incorrect samples.
        
        This is the future implementation for iterative refinement.
        Currently returns empty history as placeholder.
        
        Args:
            original_prompts: List of original conversation prompts
            answers: List of answer strings
            correctness_mask: Boolean tensor indicating which samples are correct
        
        Returns:
            List of dicts tracking each round's results:
            [
                {
                    'round': 0,
                    'prompts': [...],
                    'answers': [...],
                    'critiques': [...],
                    'correctness': [...]
                },
                ...
            ]
        """
        # Extract incorrect samples
        incorrect_mask = ~correctness_mask
        incorrect_indices = torch.where(incorrect_mask)[0].tolist()
        
        if not incorrect_indices:
            return []
        
        # TODO: Implement multi-round refinement loop
        # For now, return empty history as this is handled by refine_incorrect_samples
        
        refine_history = []
        current_prompts = [original_prompts[i] for i in incorrect_indices]
        current_answers = [answers[i] for i in incorrect_indices]
        
        for round_idx in range(self.max_refine_rounds):
            # Generate critiques
            if self.trainer.use_critique:
                critiques = self.critique_agent(current_prompts, current_answers, round_idx)
            else:
                critiques = [None] * len(current_prompts)
            
            # Generate refined prompts
            refined_prompts = self.refine_agent(
                current_prompts, current_answers, critiques, round_idx
            )
            
            # Generate and score using sampler
            refined_result = self.trainer.sampler.generate_from_prompts(refined_prompts, images=None)
            
            # Extract new answers
            new_answers = [
                self.trainer.processor.decode(ids, skip_special_tokens=True)
                for ids in refined_result['completion_ids']
            ]
            
            # Log this round
            refine_history.append({
                'round': round_idx + 1,
                'prompts': refined_prompts,
                'answers': new_answers,
                'critiques': critiques,
                # Note: Would need to score again to get correctness
            })
            
            # Update for next round
            current_prompts = refined_prompts
            current_answers = new_answers
        
        return refine_history
