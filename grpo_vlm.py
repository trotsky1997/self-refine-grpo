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

# /// script
# dependencies = [
#     "trl",
#     "Pillow",
#     "peft",
#     "math-verify",
#     "latex2sympy2_extended",
#     "torchvision",
#     "trackio",
#     "kernels",
# ]
# ///

"""
pip install math_verify

# For Qwen/Qwen2.5-VL-3B-Instruct
accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --output_dir grpo-Qwen2.5-VL-3B-Instruct \
    --learning_rate 1e-5 \
    --gradient_checkpointing \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_vllm \
    --vllm_mode colocate \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions

# For HuggingFaceTB/SmolVLM2-2.2B-Instruct
pip install num2words==0.5.14

accelerate launch \
    --config_file examples/accelerate_configs/deepspeed_zero3.yaml \
    examples/scripts/grpo_vlm.py \
    --model_name_or_path HuggingFaceTB/SmolVLM2-2.2B-Instruct \
    --output_dir grpo-SmolVLM2-2.2B-Instruct \
    --learning_rate 1e-5 \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 1024 \
    --use_peft \
    --lora_target_modules "q_proj", "v_proj" \
    --log_completions \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --num_generations 2

"""

import os
import random
import torch
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from trl.rewards import think_format_reward
from trainer import GRPOTrainer
from unsloth_gc import apply_unsloth_offloaded_gradient_checkpoint


# Enable logging in a Hugging Face Space
os.environ.setdefault("TRACKIO_SPACE_ID", "trl-trackio")


def is_triplet_atk(string,k,n,probe_num):
    probe_positions = random.sample(range(1,k),probe_num)
    bools_list = [[string[-j] == string[-j-k*i] for i in range(1,n)] for j in probe_positions]
    bools = [all(bools_list[j]) for j in range(probe_num)]
    bool_val = all(bools)

    if bool_val:
        return True
    return False

def is_triplet_repeat(string,k,n,probe_num):
    L = len(string)
    max_K = L // n
    for i in range(k,max_K):
        if is_triplet_atk(string,i,n,probe_num):
            return True
    return False


if __name__ == "__main__":
    # Apply Unsloth's offloaded gradient checkpointing BEFORE parsing args
    # This monkey-patches all PreTrainedModel.gradient_checkpointing_enable() calls
    # 
    # Available modes (ordered by memory savings):
    #   "offload"        - Standard CPU offload (~50% savings, most compatible)
    #   "quantized"      - CPU offload + 8-bit (~87.5% savings, ~0.1% precision loss)
    #   "uvm"            - CUDA Unified Memory (auto-paging, BnB style)
    #   "uvm_alt"        - Alternative UVM (ctypes + cudaMemAdvise, BnB style)
    #   "uvm_quantized"  - UVM + 8-bit (ULTIMATE ~93.75% savings, combines both!)
    #
    # Recommendation:
    #   1. Start with "offload" or "uvm_alt"
    #   2. If still OOM → try "uvm_quantized" (ultimate memory saver!)
    #   3. Fallback to "quantized" if UVM not available
    # 
    # NOTE: Fixed backward pass tuple unpacking issue - now using uvm_quantized!
    apply_unsloth_offloaded_gradient_checkpoint(mode="uvm_quantized")
    
    # Prepare Liger Kernel for GSPO support
    # 
    # Performance tiers:
    #   use_triton=False → LigerFusedLinearGSPOLoss (~2-3x speedup) ⭐ STABLE
    #   use_triton=True  → TritonGSPOLoss (Triton forward + PyTorch backward) 🚀 EXPERIMENTAL
    # 
    from patch_liger_gspo import patch_liger_kernel_for_gspo
    patch_liger_kernel_for_gspo(use_triton=True)  # ✅ Triton acceleration enabled!
    
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    
    # Ensure gradient checkpointing is enabled
    training_args.gradient_checkpointing = True
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    print(f"✅ Gradient checkpointing config: {training_args.gradient_checkpointing}")
    
    ################
    # Model & Processor
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config

    ################
    # Dataset
    ################
    dataset = load_dataset("lmms-lab/multimodal-open-r1-8k-verified", split="train")
    dataset = dataset.train_test_split(test_size=100, seed=42)

    SYSTEM_PROMPT = (
        "A conversation between user and assistant. The user asks a question, and the assistant solves it step by step.\n\n"
        "Response Format:\n\n"
        "1. <think>\n"
        "Your internal thought process - completely unstructured, like talking to yourself:\n"
        "⚠️ ZERO READABILITY REQUIRED - This is your private scratchpad:\n"
        "- Jot down raw thoughts, calculations, and observations as they come\n"
        "- Make mistakes, backtrack, correct yourself freely\n"
        "- Use abbreviations, fragments, messy formatting - anything goes\n"
        "- Prioritize speed and thoroughness over clarity\n"
        "- No one will read this except you during problem-solving\n"
        "</think>\n\n"
        
        "2. <answer>\n"
        "Present the solution for HUMAN READERS:\n"
        "✅ STRICT READABILITY REQUIRED - This will be graded on clarity:\n"
        "- Use clear step-by-step structure (Step 1:, Step 2:, etc.)\n"
        "- Explain each step thoroughly with proper formatting\n"
        "- Use paragraphs and spacing for maximum readability\n"
        "- Show all calculations and intermediate results\n"
        "- Conclude with the final answer in \\boxed{}\n"
        "- Make it polished, professional, and easy to follow\n"
        "</answer>\n\n"
    )

    def make_conversation(example):
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["problem"]},
        ]
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)

    # Filter have big images
    def filter_big_images(example):
        image = example["image"]
        return image.size[0] < 512 and image.size[1] < 512

    dataset = dataset.filter(filter_big_images)

    def convert_to_rgb(example):
        image = example["image"]
        if image.mode != "RGB":
            image = image.convert("RGB")
        example["image"] = image
        return example

    dataset = dataset.map(convert_to_rgb)

    train_dataset = dataset["train"]
    eval_dataset = dataset["test"] if training_args.eval_strategy != "no" else None

    ################
    # Reward Function for Training
    ################
    def accuracy_reward(completions, solution: list[str], **kwargs):
        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.
        """
        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        for content, sol in zip(contents, solution):
            try:
                gold_parsed = parse(sol, extraction_mode="first_match")
            except Exception:
                gold_parsed = []

            if len(gold_parsed) != 0:
                # Try parsing predicted answer too
                try:
                    answer_parsed = parse(
                        content,
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
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception as e:
                    print(f"verify failed: {e}, answer: {content}, gold: {sol}")
                    reward = None
            else:
                # fallback to text match
                reward = float(content.strip().lower() == sol.strip().lower())
            # print(f"accuracy_reward: {reward}")
            rewards.append(reward)

        return rewards

    def tandem_repeat_reward(completions, solution: list[str], **kwargs):
        """Reward function that checks if the completion matches the ground truth.
        - If both gold and prediction are parseable → use math verification.
        - If not parseable → compare as normalized text.
        """
        rewards = []
        contents = [completion[0]["content"] for completion in completions]
        for content, sol in zip(contents, solution):

            reward = int(not is_triplet_repeat(content,8,3,5))
            # print(f"tandem_repeat_reward: {reward}")
            rewards.append(reward)

        return rewards

    ################
    # Training
    ################

    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        reward_funcs=[think_format_reward, accuracy_reward],
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        enable_self_refine=True,
        use_critique=True,
        refine_log_file="self_refine_log.jsonl",
        return_both_responses=True,
        micro_batch_size=8,  # Reduced to 1 for extremely long sequences (4096 tokens)
        peft_config=get_peft_config(model_args),
    )
    
    # Explicitly enable gradient checkpointing on the model
    # This will use Unsloth's offloaded checkpointing due to monkey-patch
    if hasattr(trainer.model, "gradient_checkpointing_enable"):
        trainer.model.gradient_checkpointing_enable()
        print(f"✅ Unsloth offloaded gradient checkpointing applied to model")
    
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)
