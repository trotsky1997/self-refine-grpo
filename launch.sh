cd /home/diz/cvpr/

# Reduce memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

accelerate launch \
    --config_file fsdp1.yaml \
    grpo_vlm.py \
    --model_name_or_path /home/diz/cvpr/Qwen2.5-VL-3B-Instruct \
    --output_dir grpo-Qwen2.5-VL-3B-Instruct \
    --learning_rate 1e-5 \
    --logging_steps 1 \
    --gradient_checkpointing \
    --attn_implementation flash_attention_2 \
    --use_liger_loss \
    --use_liger_kernel \
    --dtype bfloat16 \
    --max_prompt_length 2048 \
    --max_completion_length 8192 \
    --use_vllm \
    --vllm_mode colocate \
    --lora_target_modules "all-linear" \
    --use_peft \
    --optim paged_adamw_8bit \
    --loss_type grpo \
    --importance_sampling_level sequence_token \
    --delta 10.0  # Optional: GSPO delta clipping (uncomment to enable)


    # --log_completions \
#
