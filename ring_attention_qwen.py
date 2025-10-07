# Copyright 2025 The HuggingFace Team. All rights reserved.
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
Qwen2VL-specific RingAttention implementation.

This module provides a specialized RingAttention implementation for Qwen2VL models,
replacing the standard attention mechanism with sequence-parallel RingAttention.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple
from ring_attention import RingAttention


def patch_qwen2vl_attention(
    model: nn.Module,
    sequence_parallel_size: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    enable_chunking: bool = False,
    chunk_size: int = 512,
    quantize_kv: bool = True,
    use_uvm: bool = True,
) -> nn.Module:
    """
    Patch Qwen2VL model to use RingAttention for sequence parallelism.
    
    This function replaces Qwen2Attention layers with RingAttention-enabled versions.
    Can optionally enable ChunkedAttention for local computation.
    
    Args:
        model: Qwen2VL model to patch
        sequence_parallel_size: Number of GPUs for sequence parallelism (default: all)
        process_group: Process group for sequence parallelism (optional)
        enable_chunking: Whether to enable ChunkedAttention for local computation
        chunk_size: Chunk size for ChunkedAttention
        quantize_kv: Whether to quantize K/V in ChunkedAttention
        use_uvm: Whether to use UVM for offloading in ChunkedAttention
        
    Returns:
        Patched model with RingAttention (optionally with ChunkedAttention)
    """
    if not dist.is_initialized():
        print("[RingAttention] Warning: Distributed not initialized, skipping RingAttention")
        return model
    
    world_size = dist.get_world_size()
    if world_size == 1:
        print("[RingAttention] Warning: World size is 1, RingAttention not needed")
        return model
    
    # Determine sequence parallel size
    if sequence_parallel_size is None:
        sp_size = world_size
    else:
        sp_size = sequence_parallel_size
        if process_group is not None:
            sp_size = dist.get_world_size(process_group)
    
    config = model.config if hasattr(model, 'config') else None
    if config is None:
        print("[RingAttention] Error: Could not find model config")
        return model
    
    # Extract model parameters
    hidden_size = getattr(config, 'hidden_size', None)
    num_attention_heads = getattr(config, 'num_attention_heads', None)
    
    if hidden_size is None or num_attention_heads is None:
        print("[RingAttention] Error: Could not extract attention parameters")
        return model
    
    print(f"[RingAttention] Patching Qwen2VL model")
    print(f"[RingAttention] - Hidden size: {hidden_size}")
    print(f"[RingAttention] - Attention heads: {num_attention_heads}")
    print(f"[RingAttention] - World size: {world_size}")
    print(f"[RingAttention] - Sequence parallel size: {sp_size}")
    if enable_chunking:
        print(f"[RingAttention] - ChunkedAttention: ENABLED (chunk_size={chunk_size})")
        print(f"[RingAttention]   └─ K/V quantization: {quantize_kv} (8-bit)")
        print(f"[RingAttention]   └─ UVM offloading: {use_uvm}")
        print(f"[RingAttention]   └─ Mode: HYBRID (Ring + Chunked)")
    else:
        print(f"[RingAttention] - ChunkedAttention: disabled (Ring only)")
    
    # Count replaced layers
    replaced_count = 0
    
    # Walk through model and replace attention layers
    for name, module in model.named_modules():
        # Check if this is a Qwen2Attention or Qwen2VLAttention layer
        if 'Attention' in type(module).__name__ and any(x in type(module).__name__ for x in ['Qwen2', 'Qwen2VL']):
            # Create RingAttention wrapper
            ring_attn = Qwen2RingAttentionWrapper(
                original_attention=module,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                process_group=process_group,
                enable_chunking=enable_chunking,
                chunk_size=chunk_size,
                quantize_kv=quantize_kv,
                use_uvm=use_uvm,
            )
            
            # Replace the module
            # This requires navigating the parent module
            parent_name = '.'.join(name.split('.')[:-1])
            child_name = name.split('.')[-1]
            
            if parent_name:
                parent = dict(model.named_modules())[parent_name]
                setattr(parent, child_name, ring_attn)
                replaced_count += 1
    
    print(f"[RingAttention] Replaced {replaced_count} attention layers")
    
    return model


class Qwen2RingAttentionWrapper(nn.Module):
    """
    Wrapper that adds RingAttention to Qwen2/Qwen2VL attention layers.
    
    This wrapper intercepts the forward pass and applies RingAttention
    for sequence-parallel computation.
    """
    
    def __init__(
        self,
        original_attention: nn.Module,
        hidden_size: int,
        num_attention_heads: int,
        process_group: Optional[dist.ProcessGroup] = None,
        enable_chunking: bool = False,
        chunk_size: int = 512,
        quantize_kv: bool = True,
        use_uvm: bool = True,
    ):
        super().__init__()
        self.original_attention = original_attention
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.process_group = process_group
        
        # Create RingAttention module (with optional ChunkedAttention)
        self.ring_attention = RingAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=getattr(original_attention, 'dropout', 0.0),
            use_flash_attn=True,
            causal=True,  # Qwen2 uses causal attention
            process_group=process_group,
            enable_chunking=enable_chunking,
            chunk_size=chunk_size,
            quantize_kv=quantize_kv,
            use_uvm=use_uvm,
        )
        
        # Copy over projection layers from original attention
        if hasattr(original_attention, 'q_proj'):
            self.q_proj = original_attention.q_proj
        if hasattr(original_attention, 'k_proj'):
            self.k_proj = original_attention.k_proj
        if hasattr(original_attention, 'v_proj'):
            self.v_proj = original_attention.v_proj
        if hasattr(original_attention, 'o_proj'):
            self.o_proj = original_attention.o_proj
        
        if process_group is not None:
            self.world_size = dist.get_world_size(process_group)
            self.rank = dist.get_rank(process_group)
        else:
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
            self.rank = dist.get_rank() if dist.is_initialized() else 0
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        Forward pass with RingAttention.
        
        Args:
            hidden_states: Input hidden states (batch, seq_len, hidden_size)
            attention_mask: Attention mask
            position_ids: Position IDs for rotary embeddings
            past_key_value: Cached key/value for generation
            output_attentions: Whether to output attention weights
            use_cache: Whether to cache key/value
            
        Returns:
            attn_output: Attention output (batch, seq_len, hidden_size)
            attn_weights: Attention weights (if output_attentions=True)
            past_key_value: Updated cache (if use_cache=True)
        """
        # If not in training mode or world_size=1, use original attention
        if not self.training or self.world_size == 1 or use_cache or past_key_value is not None:
            return self.original_attention(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                **kwargs,
            )
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        
        # Apply RingAttention
        attn_output = self.ring_attention(
            query=query_states,
            key=key_states,
            value=value_states,
            attention_mask=attention_mask,
        )
        
        # Project output
        attn_output = self.o_proj(attn_output)
        
        # Return in format expected by Qwen2
        if output_attentions:
            # RingAttention doesn't return attention weights
            return attn_output, None, None
        else:
            return attn_output, None, None


def enable_ring_attention_for_qwen2vl(trainer) -> None:
    """
    Enable RingAttention for Qwen2VL model in the trainer.
    
    Args:
        trainer: GRPOTrainer instance with Qwen2VL model
    """
    if hasattr(trainer, 'model'):
        trainer.model = patch_qwen2vl_attention(trainer.model)
        
        if trainer.accelerator.is_main_process:
            print("[RingAttention] Qwen2VL model patched successfully")
    else:
        print("[RingAttention] Error: Trainer does not have a model attribute")

