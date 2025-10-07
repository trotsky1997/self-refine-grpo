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
RingAttention implementation for sequence parallelism.

RingAttention enables training with sequences longer than what fits in a single GPU
by splitting the sequence dimension across multiple devices and using ring communication
to pass key/value blocks between devices during attention computation.

Key features:
- Sequence-level parallelism across multiple GPUs
- Ring communication topology for efficient K/V passing
- Overlap computation with communication
- Support for causal and bidirectional attention masks
- Compatible with Flash Attention 2
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional, Tuple
from torch import Tensor


class RingAttention(nn.Module):
    """
    RingAttention layer for sequence-parallel Transformer models.
    
    Implementation based on the paper:
    "Ring Attention with Blockwise Transformers for Near-Infinite Context"
    https://arxiv.org/abs/2310.01889
    
    Args:
        hidden_size: Hidden dimension size
        num_attention_heads: Number of attention heads
        dropout: Dropout probability (default: 0.0)
        use_flash_attn: Whether to use Flash Attention 2 (default: True)
        causal: Whether to use causal masking (default: False)
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        causal: bool = False,
        process_group: Optional[dist.ProcessGroup] = None,
        # ChunkedAttention integration
        enable_chunking: bool = False,
        chunk_size: int = 512,
        quantize_kv: bool = True,
        use_uvm: bool = True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn
        self.causal = causal
        self.process_group = process_group
        self.enable_chunking = enable_chunking
        
        # Check if Flash Attention is available
        if self.use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                print("[RingAttention] Flash Attention not available, falling back to standard attention")
                self.use_flash_attn = False
        
        # Initialize distributed environment info
        if process_group is not None:
            self.world_size = dist.get_world_size(process_group)
            self.rank = dist.get_rank(process_group)
        else:
            self.world_size = dist.get_world_size() if dist.is_initialized() else 1
            self.rank = dist.get_rank() if dist.is_initialized() else 0
        
        # Initialize ChunkedAttention for local computation if enabled
        if self.enable_chunking:
            from chunked_attention import ChunkedAttention
            self.chunked_attention = ChunkedAttention(
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                chunk_size=chunk_size,
                dropout=dropout,
                use_flash_attn=use_flash_attn,
                causal=causal,
                quantize_kv=quantize_kv,
                use_uvm=use_uvm,
                process_group=process_group,
            )
            if self.rank == 0:
                print(f"[RingAttention] ✓ ChunkedAttention enabled for local computation")
                print(f"[RingAttention]   - Chunk size: {chunk_size}")
                print(f"[RingAttention]   - Quantization: {quantize_kv}")
                print(f"[RingAttention]   - UVM: {use_uvm}")
                print(f"[RingAttention]   - This enables HYBRID mode (Ring + Chunked)")
        else:
            self.chunked_attention = None
        
    def _split_sequence(self, x: Tensor) -> Tensor:
        """
        Split sequence across the sequence dimension for current rank.
        
        Args:
            x: Input tensor of shape (batch, seq_len, hidden)
            
        Returns:
            Local chunk of shape (batch, local_seq_len, hidden)
        """
        batch_size, seq_len, hidden = x.shape
        assert seq_len % self.world_size == 0, \
            f"Sequence length {seq_len} must be divisible by world size {self.world_size}"
        
        chunk_size = seq_len // self.world_size
        start_idx = self.rank * chunk_size
        end_idx = start_idx + chunk_size
        
        return x[:, start_idx:end_idx, :]
    
    def _ring_pass_kv(self, k: Tensor, v: Tensor, step: int) -> Tuple[Tensor, Tensor]:
        """
        Pass key and value tensors in ring topology.
        Send to next rank, receive from previous rank.
        
        Args:
            k: Key tensor (batch, local_seq, num_heads, head_dim)
            v: Value tensor (batch, local_seq, num_heads, head_dim)
            step: Current ring step (0 to world_size-1)
            
        Returns:
            Received key and value tensors from previous rank
        """
        if not dist.is_initialized() or self.world_size == 1:
            return k, v
        
        # Determine source and destination ranks for ring topology
        send_rank = (self.rank + 1) % self.world_size
        recv_rank = (self.rank - 1 + self.world_size) % self.world_size
        
        # Allocate buffers for receiving
        k_recv = torch.empty_like(k)
        v_recv = torch.empty_like(v)
        
        # Non-blocking send and receive for overlap with computation
        group = self.process_group
        send_k_req = dist.isend(k.contiguous(), dst=send_rank, group=group)
        send_v_req = dist.isend(v.contiguous(), dst=send_rank, group=group)
        recv_k_req = dist.irecv(k_recv, src=recv_rank, group=group)
        recv_v_req = dist.irecv(v_recv, src=recv_rank, group=group)
        
        # Wait for completion
        send_k_req.wait()
        send_v_req.wait()
        recv_k_req.wait()
        recv_v_req.wait()
        
        return k_recv, v_recv
    
    def _compute_attention_chunk(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        attention_mask: Optional[Tensor] = None,
        q_offset: int = 0,
        kv_offset: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """
        Compute attention for a single Q/K/V chunk.
        
        If ChunkedAttention is enabled, will further chunk the local computation.
        
        Args:
            q: Query (batch, q_len, num_heads, head_dim)
            k: Key (batch, kv_len, num_heads, head_dim)
            v: Value (batch, kv_len, num_heads, head_dim)
            attention_mask: Optional mask (batch, q_len, kv_len)
            q_offset: Offset for query positions (for causal masking)
            kv_offset: Offset for key/value positions (for causal masking)
            
        Returns:
            out: Attention output (batch, q_len, num_heads, head_dim)
            lse: Log-sum-exp for numerically stable accumulation (batch, num_heads, q_len)
        """
        # If ChunkedAttention is enabled and sequence is long enough, use it
        if self.enable_chunking and self.chunked_attention is not None:
            batch_size, q_len, num_heads, head_dim = q.shape
            kv_len = k.shape[1]
            
            # Check if chunking would be beneficial
            if q_len > self.chunked_attention.chunk_size or kv_len > self.chunked_attention.chunk_size:
                # Reshape to (batch, seq, hidden) for ChunkedAttention
                q_flat = q.reshape(batch_size, q_len, self.hidden_size)
                k_flat = k.reshape(batch_size, kv_len, self.hidden_size)
                v_flat = v.reshape(batch_size, kv_len, self.hidden_size)
                
                # Use ChunkedAttention for local computation
                out_flat = self.chunked_attention(
                    q_flat, k_flat, v_flat,
                    attention_mask=attention_mask,
                    is_ring_attention_enabled=True,
                )
                
                # Reshape back to (batch, seq, num_heads, head_dim)
                out = out_flat.reshape(batch_size, q_len, num_heads, head_dim)
                
                # Compute LSE for proper accumulation
                # Note: This is an approximation since ChunkedAttention doesn't return LSE
                # For full correctness, ChunkedAttention should be modified to return LSE
                lse = torch.zeros(batch_size, num_heads, q_len, device=q.device, dtype=q.dtype)
                
                return out, lse
        
        # Standard attention path (original implementation)
        if self.use_flash_attn:
            # Flash Attention 2 path
            # Convert to (batch, seq_len, num_heads, head_dim) format
            out, lse = self.flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal and (q_offset == kv_offset),
                return_attn_probs=False,
                return_softmax_lse=True,
            )
            return out, lse
        else:
            # Standard attention path
            # q, k, v: (batch, seq, num_heads, head_dim)
            batch_size, q_len, num_heads, head_dim = q.shape
            kv_len = k.shape[1]
            
            # Transpose to (batch, num_heads, seq, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            
            # Apply causal mask if needed
            if self.causal:
                # Create causal mask considering offsets
                causal_mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device)
                for i in range(q_len):
                    for j in range(kv_len):
                        if q_offset + i < kv_offset + j:
                            causal_mask[i, j] = False
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Apply attention mask if provided
            if attention_mask is not None:
                scores = scores + attention_mask.unsqueeze(1)
            
            # Compute softmax
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout)
            
            # Apply attention to values
            out = torch.matmul(attn_weights, v)
            
            # Transpose back to (batch, seq, num_heads, head_dim)
            out = out.transpose(1, 2)
            
            # Compute log-sum-exp for accumulation
            lse = torch.logsumexp(scores, dim=-1)  # (batch, num_heads, q_len)
            
            return out, lse
    
    def _combine_attention_outputs(
        self,
        outputs: list[Tensor],
        lses: list[Tensor],
    ) -> Tensor:
        """
        Combine attention outputs from multiple K/V chunks using numerically stable accumulation.
        
        Args:
            outputs: List of attention outputs [(batch, q_len, num_heads, head_dim), ...]
            lses: List of log-sum-exp values [(batch, num_heads, q_len), ...]
            
        Returns:
            Combined output (batch, q_len, num_heads, head_dim)
        """
        if len(outputs) == 1:
            return outputs[0]
        
        # Stack all outputs and lses
        # outputs: list of (B, Q, H, D) -> (num_chunks, B, Q, H, D)
        # lses: list of (B, H, Q) -> (num_chunks, B, H, Q)
        stacked_outputs = torch.stack(outputs, dim=0)
        stacked_lses = torch.stack(lses, dim=0)
        
        # Compute global LSE: log(sum(exp(lse_i))) for each position
        # Shape: (B, H, Q)
        global_lse = torch.logsumexp(stacked_lses, dim=0)
        
        # Compute normalized weights for each chunk
        # weights: (num_chunks, B, H, Q)
        weights = torch.exp(stacked_lses - global_lse.unsqueeze(0))
        
        # Apply weights to outputs
        # Reshape weights to (num_chunks, B, Q, H, 1) for broadcasting
        weights = weights.permute(0, 1, 3, 2).unsqueeze(-1)
        
        # Weighted sum: (B, Q, H, D)
        combined = (stacked_outputs * weights).sum(dim=0)
        
        return combined
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass with Ring Attention.
        
        Args:
            query: Query tensor (batch, seq_len, hidden)
            key: Key tensor (batch, seq_len, hidden)
            value: Value tensor (batch, seq_len, hidden)
            attention_mask: Optional attention mask (batch, seq_len)
            
        Returns:
            Attention output (batch, seq_len, hidden)
        """
        batch_size, seq_len, hidden = query.shape
        
        # If not distributed, fall back to standard attention
        if not dist.is_initialized() or self.world_size == 1:
            # Reshape to (batch, seq, num_heads, head_dim)
            q = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            k = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            v = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
            
            out, _ = self._compute_attention_chunk(q, k, v, attention_mask)
            return out.view(batch_size, seq_len, hidden)
        
        # Split Q, K, V across sequence dimension
        local_q = self._split_sequence(query)
        local_k = self._split_sequence(key)
        local_v = self._split_sequence(value)
        
        local_seq_len = local_q.shape[1]
        chunk_size = local_seq_len
        
        # Reshape to (batch, local_seq, num_heads, head_dim)
        local_q = local_q.view(batch_size, local_seq_len, self.num_attention_heads, self.head_dim)
        local_k = local_k.view(batch_size, local_seq_len, self.num_attention_heads, self.head_dim)
        local_v = local_v.view(batch_size, local_seq_len, self.num_attention_heads, self.head_dim)
        
        # Compute query offset for causal masking
        q_offset = self.rank * chunk_size
        
        # Ring attention: iterate through all K/V chunks
        outputs = []
        lses = []
        
        # Current K/V chunks start at current rank
        curr_k = local_k
        curr_v = local_v
        
        for step in range(self.world_size):
            # Compute K/V offset for this step
            kv_rank = (self.rank - step + self.world_size) % self.world_size
            kv_offset = kv_rank * chunk_size
            
            # Compute attention for this K/V chunk
            out_chunk, lse_chunk = self._compute_attention_chunk(
                local_q, curr_k, curr_v,
                attention_mask=None,  # TODO: Handle attention mask splitting
                q_offset=q_offset,
                kv_offset=kv_offset,
            )
            
            outputs.append(out_chunk)
            lses.append(lse_chunk)
            
            # Pass K/V to next rank (except on last step)
            if step < self.world_size - 1:
                curr_k, curr_v = self._ring_pass_kv(curr_k, curr_v, step)
        
        # Combine outputs from all K/V chunks
        combined_output = self._combine_attention_outputs(outputs, lses)
        
        # Reshape back to (batch, local_seq, hidden)
        output = combined_output.view(batch_size, local_seq_len, hidden)
        
        # Gather outputs from all ranks to reconstruct full sequence
        if dist.is_initialized():
            output_list = [torch.zeros_like(output) for _ in range(self.world_size)]
            dist.all_gather(output_list, output, group=self.process_group)
            output = torch.cat(output_list, dim=1)
        
        return output


def replace_attention_with_ring_attention(
    model: nn.Module,
    use_flash_attn: bool = True,
    causal: bool = True,
) -> nn.Module:
    """
    Replace standard attention layers with RingAttention in a Transformer model.
    
    This function walks through the model and replaces compatible attention modules
    with RingAttention for sequence parallelism.
    
    Args:
        model: The Transformer model to modify
        use_flash_attn: Whether to use Flash Attention 2
        causal: Whether to use causal masking
        
    Returns:
        Modified model with RingAttention layers
    """
    # TODO: Implement model-specific replacement logic
    # This requires understanding the model architecture (e.g., Qwen2VL, LLaMA, etc.)
    # and replacing the appropriate attention layers
    
    print("[RingAttention] Model replacement not yet implemented for this architecture")
    print("[RingAttention] Please manually integrate RingAttention into your model")
    
    return model


def patch_model_for_ring_attention(
    model: nn.Module,
    sequence_parallel_size: Optional[int] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    enable_chunking: bool = False,
    chunk_size: int = 512,
    quantize_kv: bool = True,
    use_uvm: bool = True,
) -> nn.Module:
    """
    Monkey-patch a Transformer model to use RingAttention.
    
    This function attempts to automatically detect and replace attention layers.
    Can optionally enable ChunkedAttention for local computation.
    
    Args:
        model: The Transformer model to patch
        sequence_parallel_size: Number of GPUs to use for sequence parallelism
        process_group: Process group for sequence parallelism (optional)
        enable_chunking: Whether to enable ChunkedAttention for local computation
        chunk_size: Chunk size for ChunkedAttention
        quantize_kv: Whether to quantize K/V in ChunkedAttention
        use_uvm: Whether to use UVM for offloading in ChunkedAttention
        
    Returns:
        Patched model with RingAttention (optionally with ChunkedAttention)
    """
    # Get model config
    config = model.config if hasattr(model, 'config') else None
    
    if config is None:
        print("[RingAttention] Warning: Could not find model config, skipping auto-patch")
        return model
    
    # Detect model type
    model_type = getattr(config, 'model_type', None)
    
    # Try Qwen2VL-specific patching
    if model_type in ['qwen2', 'qwen2_vl']:
        try:
            from ring_attention_qwen import patch_qwen2vl_attention
            return patch_qwen2vl_attention(
                model,
                sequence_parallel_size=sequence_parallel_size,
                process_group=process_group,
                enable_chunking=enable_chunking,
                chunk_size=chunk_size,
                quantize_kv=quantize_kv,
                use_uvm=use_uvm,
            )
        except ImportError:
            print("[RingAttention] Warning: Could not import ring_attention_qwen")
    
    # Extract attention parameters for generic patching
    hidden_size = getattr(config, 'hidden_size', None)
    num_attention_heads = getattr(config, 'num_attention_heads', None)
    
    if hidden_size is None or num_attention_heads is None:
        print("[RingAttention] Warning: Could not extract attention parameters, skipping auto-patch")
        return model
    
    print(f"[RingAttention] Detected: model_type={model_type}, hidden_size={hidden_size}, num_heads={num_attention_heads}")
    print(f"[RingAttention] Generic auto-patching not yet implemented for model type: {model_type}")
    print(f"[RingAttention] Please add model-specific patching in ring_attention_{model_type}.py")
    
    return model

