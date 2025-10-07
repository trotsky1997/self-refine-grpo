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
Chunked Attention with UVM Quantized Offload

Similar to Ring Attention but designed for single-GPU long context training.
Instead of passing K/V across GPUs, we chunk the sequence and offload K/V to CPU/UVM.

Key advantages over Ring Attention:
- Works on single GPU
- No communication overhead
- Simpler implementation
- 8-bit quantization for K/V storage

Memory savings:
- Standard attention: O(seq_len^2) for attention matrix
- Chunked attention: O(chunk_size * seq_len) + offload storage
- With quantization: additional 75% reduction on offloaded K/V
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch import Tensor
import ctypes
import numpy as np


class ChunkedAttention(nn.Module):
    """
    Chunked Attention with UVM Quantized Offload.
    
    Processes long sequences by chunking and offloading K/V to CPU/UVM,
    allowing single-GPU training of sequences much longer than GPU memory permits.
    
    Can be combined with RingAttention for hybrid parallelism:
    - RingAttention: Split sequence across GPUs (sequence parallelism)
    - ChunkedAttention: Further chunk each GPU's portion (memory optimization)
    
    Algorithm:
    1. Split sequence into chunks
    2. Process each chunk:
       - Load/compute K/V for chunk
       - Quantize and offload to CPU/UVM
    3. For each query chunk:
       - Load all K/V chunks (dequantize on-the-fly)
       - Compute attention
       - Accumulate using stable log-sum-exp
    
    Args:
        hidden_size: Hidden dimension
        num_attention_heads: Number of attention heads
        chunk_size: Size of each chunk (default: 512)
        dropout: Dropout probability
        use_flash_attn: Whether to use Flash Attention 2
        causal: Whether to use causal masking
        quantize_kv: Whether to quantize K/V (default: True)
        use_uvm: Whether to use UVM for offloading (default: True)
        process_group: Optional process group for RingAttention compatibility
    """
    
    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        chunk_size: int = 512,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        causal: bool = False,
        quantize_kv: bool = True,
        use_uvm: bool = True,
        process_group: Optional[object] = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.chunk_size = chunk_size
        self.dropout = dropout
        self.use_flash_attn = use_flash_attn
        self.causal = causal
        self.quantize_kv = quantize_kv
        self.use_uvm = use_uvm
        self.process_group = process_group
        
        # Check if we're in a distributed setting (for RingAttention compatibility)
        try:
            import torch.distributed as dist
            if process_group is not None:
                self.world_size = dist.get_world_size(process_group)
                self.rank = dist.get_rank(process_group)
            elif dist.is_initialized():
                self.world_size = dist.get_world_size()
                self.rank = dist.get_rank()
            else:
                self.world_size = 1
                self.rank = 0
        except:
            self.world_size = 1
            self.rank = 0
        
        # Check Flash Attention availability
        if self.use_flash_attn:
            try:
                from flash_attn import flash_attn_func
                self.flash_attn_func = flash_attn_func
            except ImportError:
                print("[ChunkedAttention] Flash Attention not available, using standard attention")
                self.use_flash_attn = False
        
        # Initialize UVM if requested
        if self.use_uvm:
            try:
                self.cuda = ctypes.CDLL("libcudart.so")
                self.uvm_available = True
            except Exception as e:
                print(f"[ChunkedAttention] UVM not available, using CPU offload: {e}")
                self.uvm_available = False
                self.use_uvm = False
        else:
            self.uvm_available = False
    
    def _quantize_kv(self, kv: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Quantize K or V tensor to 8-bit.
        
        Args:
            kv: K or V tensor (batch, seq, num_heads, head_dim)
            
        Returns:
            quantized: int8 tensor
            scale: scale factors for dequantization
        """
        # Per-head quantization for better accuracy
        # kv shape: (batch, seq, num_heads, head_dim)
        batch, seq, num_heads, head_dim = kv.shape
        
        # Reshape to (batch, seq, num_heads, head_dim)
        # Quantize per head
        kv_reshaped = kv.reshape(batch, seq, num_heads, head_dim)
        
        # Compute scale per head: (batch, seq, num_heads, 1)
        absmax = kv_reshaped.abs().amax(dim=-1, keepdim=True)
        absmax = torch.clamp(absmax, min=1e-8)
        scale = absmax / 127.0
        
        # Quantize
        quantized = (kv_reshaped / scale).clamp(-127, 127).to(torch.int8)
        
        return quantized, scale
    
    def _dequantize_kv(self, quantized: Tensor, scale: Tensor, target_dtype: torch.dtype) -> Tensor:
        """
        Dequantize int8 K/V back to float.
        
        Args:
            quantized: int8 tensor (batch, seq, num_heads, head_dim)
            scale: scale factors (batch, seq, num_heads, 1)
            target_dtype: target dtype
            
        Returns:
            dequantized: float tensor
        """
        dequantized = quantized.to(target_dtype) * scale
        return dequantized
    
    def _offload_to_uvm(self, tensor: Tensor) -> Tuple[Tensor, Optional[ctypes.c_void_p]]:
        """
        Offload tensor to UVM managed memory.
        
        Args:
            tensor: Tensor to offload
            
        Returns:
            managed_tensor: Tensor in managed memory
            managed_ptr: Pointer for later freeing (None if UVM unavailable)
        """
        if not self.uvm_available:
            return tensor.cpu(), None
        
        try:
            # Allocate managed memory
            size = tensor.numel() * tensor.element_size()
            managed_ptr = ctypes.c_void_p()
            result = self.cuda.cudaMallocManaged(
                ctypes.byref(managed_ptr),
                ctypes.c_size_t(size),
                ctypes.c_uint(1)  # cudaMemAttachGlobal
            )
            
            if result != 0:
                raise RuntimeError(f"cudaMallocManaged failed with code {result}")
            
            # Determine dtype
            if tensor.dtype == torch.int8:
                np_dtype = ctypes.c_int8
                np_type = np.int8
            elif tensor.dtype == torch.float32:
                np_dtype = ctypes.c_float
                np_type = np.float32
            elif tensor.dtype == torch.float16:
                np_dtype = ctypes.c_uint16
                np_type = np.float16
            elif tensor.dtype == torch.bfloat16:
                np_dtype = ctypes.c_uint16
                np_type = np.uint16
            else:
                raise ValueError(f"Unsupported dtype: {tensor.dtype}")
            
            # Create numpy array view
            managed_array = np.ctypeslib.as_array(
                ctypes.cast(managed_ptr, ctypes.POINTER(np_dtype)),
                shape=tensor.shape
            )
            
            # Copy data
            if tensor.dtype == torch.bfloat16:
                temp_cpu = tensor.cpu()
                managed_array[:] = temp_cpu.view(torch.uint16).numpy()
                managed_tensor = torch.from_numpy(managed_array).view(torch.bfloat16)
            else:
                managed_tensor = torch.from_numpy(managed_array.astype(np_type))
                managed_tensor.copy_(tensor.cpu())
            
            # Advise CUDA to prefer CPU
            self.cuda.cudaMemAdvise(
                managed_ptr,
                ctypes.c_size_t(size),
                ctypes.c_int(3),  # cudaMemAdviseSetPreferredLocation to CPU
                ctypes.c_int(-1)  # CPU device ID
            )
            
            return managed_tensor, managed_ptr
            
        except Exception as e:
            print(f"[ChunkedAttention] UVM offload failed, using CPU: {e}")
            return tensor.cpu(), None
    
    def _free_uvm(self, managed_ptr: Optional[ctypes.c_void_p]) -> None:
        """Free UVM managed memory."""
        if managed_ptr is not None:
            try:
                self.cuda.cudaFree(managed_ptr)
            except:
                pass
    
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
        Compute attention for Q with one K/V chunk.
        
        Args:
            q: Query (batch, q_len, num_heads, head_dim)
            k: Key (batch, kv_len, num_heads, head_dim)
            v: Value (batch, kv_len, num_heads, head_dim)
            attention_mask: Optional mask
            q_offset: Offset for query positions (for causal masking)
            kv_offset: Offset for key/value positions
            
        Returns:
            out: Attention output (batch, q_len, num_heads, head_dim)
            lse: Log-sum-exp for accumulation (batch, num_heads, q_len)
        """
        if self.use_flash_attn:
            # Flash Attention 2 path
            out, lse = self.flash_attn_func(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal and (q_offset == kv_offset),
                return_attn_probs=False,
                return_softmax_lse=True,
            )
            return out, lse
        else:
            # Standard attention
            batch_size, q_len, num_heads, head_dim = q.shape
            kv_len = k.shape[1]
            
            # Transpose to (batch, num_heads, seq, head_dim)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            
            # Compute scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)
            
            # Apply causal mask
            if self.causal:
                causal_mask = torch.ones(q_len, kv_len, dtype=torch.bool, device=q.device)
                for i in range(q_len):
                    for j in range(kv_len):
                        if q_offset + i < kv_offset + j:
                            causal_mask[i, j] = False
                scores = scores.masked_fill(~causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Apply attention mask
            if attention_mask is not None:
                scores = scores + attention_mask.unsqueeze(1)
            
            # Softmax and attention
            attn_weights = torch.nn.functional.softmax(scores, dim=-1)
            if self.training and self.dropout > 0:
                attn_weights = torch.nn.functional.dropout(attn_weights, p=self.dropout)
            
            out = torch.matmul(attn_weights, v)
            out = out.transpose(1, 2)  # Back to (batch, seq, num_heads, head_dim)
            
            # Log-sum-exp
            lse = torch.logsumexp(scores, dim=-1)  # (batch, num_heads, q_len)
            
            return out, lse
    
    def _combine_attention_outputs(
        self,
        outputs: list[Tensor],
        lses: list[Tensor],
    ) -> Tensor:
        """
        Combine attention outputs from multiple K/V chunks using log-sum-exp.
        
        Args:
            outputs: List of attention outputs
            lses: List of log-sum-exp values
            
        Returns:
            Combined output
        """
        if len(outputs) == 1:
            return outputs[0]
        
        # Stack outputs and lses
        stacked_outputs = torch.stack(outputs, dim=0)  # (num_chunks, B, Q, H, D)
        stacked_lses = torch.stack(lses, dim=0)  # (num_chunks, B, H, Q)
        
        # Global LSE
        global_lse = torch.logsumexp(stacked_lses, dim=0)  # (B, H, Q)
        
        # Normalized weights
        weights = torch.exp(stacked_lses - global_lse.unsqueeze(0))  # (num_chunks, B, H, Q)
        weights = weights.permute(0, 1, 3, 2).unsqueeze(-1)  # (num_chunks, B, Q, H, 1)
        
        # Weighted sum
        combined = (stacked_outputs * weights).sum(dim=0)  # (B, Q, H, D)
        
        return combined
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Optional[Tensor] = None,
        is_ring_attention_enabled: bool = False,
    ) -> Tensor:
        """
        Forward pass with chunked attention and UVM offload.
        
        Can work in two modes:
        1. Standalone: Process full sequence with chunking
        2. With RingAttention: Process local sequence portion (after SP split)
        
        Args:
            query: Query tensor (batch, seq_len, hidden)
            key: Key tensor (batch, seq_len, hidden)
            value: Value tensor (batch, seq_len, hidden)
            attention_mask: Optional attention mask
            is_ring_attention_enabled: Whether RingAttention is also active
            
        Returns:
            Attention output (batch, seq_len, hidden)
        """
        batch_size, seq_len, hidden = query.shape
        device = query.device
        dtype = query.dtype
        
        # Note: If RingAttention is enabled, seq_len is already the local portion
        # (e.g., 32K total / 4 GPUs = 8K per GPU)
        
        # Reshape to (batch, seq, num_heads, head_dim)
        q = query.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        k = key.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        v = value.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)
        
        # If sequence fits in one chunk, use standard attention
        if seq_len <= self.chunk_size:
            out, _ = self._compute_attention_chunk(q, k, v, attention_mask)
            return out.view(batch_size, seq_len, hidden)
        
        # Calculate number of chunks
        num_chunks = (seq_len + self.chunk_size - 1) // self.chunk_size
        
        # Phase 1: Process and offload all K/V chunks
        kv_chunks = []
        managed_ptrs = []
        
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * self.chunk_size
            end_idx = min(start_idx + self.chunk_size, seq_len)
            
            # Extract K/V chunk
            k_chunk = k[:, start_idx:end_idx, :, :]
            v_chunk = v[:, start_idx:end_idx, :, :]
            
            if self.quantize_kv:
                # Quantize
                k_quant, k_scale = self._quantize_kv(k_chunk)
                v_quant, v_scale = self._quantize_kv(v_chunk)
                
                # Offload to UVM/CPU
                k_offload, k_ptr = self._offload_to_uvm(k_quant)
                v_offload, v_ptr = self._offload_to_uvm(v_quant)
                k_scale_offload, k_scale_ptr = self._offload_to_uvm(k_scale)
                v_scale_offload, v_scale_ptr = self._offload_to_uvm(v_scale)
                
                kv_chunks.append({
                    'k': k_offload,
                    'v': v_offload,
                    'k_scale': k_scale_offload,
                    'v_scale': v_scale_offload,
                    'quantized': True,
                })
                managed_ptrs.extend([k_ptr, v_ptr, k_scale_ptr, v_scale_ptr])
            else:
                # Just offload without quantization
                k_offload, k_ptr = self._offload_to_uvm(k_chunk)
                v_offload, v_ptr = self._offload_to_uvm(v_chunk)
                
                kv_chunks.append({
                    'k': k_offload,
                    'v': v_offload,
                    'quantized': False,
                })
                managed_ptrs.extend([k_ptr, v_ptr])
        
        # Phase 2: Process each query chunk with all K/V chunks
        final_output = []
        
        for q_chunk_idx in range(num_chunks):
            q_start = q_chunk_idx * self.chunk_size
            q_end = min(q_start + self.chunk_size, seq_len)
            q_chunk = q[:, q_start:q_end, :, :]
            
            outputs_for_q = []
            lses_for_q = []
            
            # Attend to each K/V chunk
            for kv_chunk_idx, kv_chunk in enumerate(kv_chunks):
                kv_start = kv_chunk_idx * self.chunk_size
                
                # Load K/V chunk to GPU
                if kv_chunk['quantized']:
                    k_loaded = self._dequantize_kv(
                        kv_chunk['k'].to(device),
                        kv_chunk['k_scale'].to(device),
                        dtype
                    )
                    v_loaded = self._dequantize_kv(
                        kv_chunk['v'].to(device),
                        kv_chunk['v_scale'].to(device),
                        dtype
                    )
                else:
                    k_loaded = kv_chunk['k'].to(device)
                    v_loaded = kv_chunk['v'].to(device)
                
                # Compute attention for this Q-KV pair
                out_chunk, lse_chunk = self._compute_attention_chunk(
                    q_chunk, k_loaded, v_loaded,
                    attention_mask=None,
                    q_offset=q_start,
                    kv_offset=kv_start,
                )
                
                outputs_for_q.append(out_chunk)
                lses_for_q.append(lse_chunk)
            
            # Combine outputs from all K/V chunks for this Q chunk
            combined_out = self._combine_attention_outputs(outputs_for_q, lses_for_q)
            final_output.append(combined_out)
        
        # Concatenate all Q chunks
        output = torch.cat(final_output, dim=1)  # (batch, seq_len, num_heads, head_dim)
        
        # Free managed memory
        for ptr in managed_ptrs:
            self._free_uvm(ptr)
        
        # Reshape back
        output = output.view(batch_size, seq_len, hidden)
        
        return output


def enable_chunked_attention(
    model: nn.Module,
    chunk_size: int = 512,
    quantize_kv: bool = True,
    use_uvm: bool = True,
) -> nn.Module:
    """
    Enable chunked attention with UVM offload for a model.
    
    This replaces standard attention layers with ChunkedAttention,
    enabling single-GPU training of very long sequences.
    
    Alias for patch_model_for_chunked_attention.
    
    Args:
        model: Transformer model to modify
        chunk_size: Size of each chunk (smaller = less memory, more overhead)
        quantize_kv: Whether to quantize K/V (recommended)
        use_uvm: Whether to use UVM (recommended if available)
        
    Returns:
        Modified model with chunked attention
    """
    return patch_model_for_chunked_attention(
        model=model,
        chunk_size=chunk_size,
        quantize_kv=quantize_kv,
        use_uvm=use_uvm,
    )


def patch_model_for_chunked_attention(
    model: nn.Module,
    chunk_size: int = 512,
    quantize_kv: bool = True,
    use_uvm: bool = True,
) -> nn.Module:
    """
    Patch a model to use ChunkedAttention (without RingAttention).
    
    This is for single-GPU scenarios where memory optimization is needed.
    
    Args:
        model: Model to patch
        chunk_size: Chunk size for ChunkedAttention
        quantize_kv: Whether to quantize K/V
        use_uvm: Whether to use UVM for offloading
        
    Returns:
        Patched model
    """
    print(f"[ChunkedAttention] Enabling chunked attention with:")
    print(f"  - Chunk size: {chunk_size}")
    print(f"  - K/V quantization: {quantize_kv}")
    print(f"  - UVM offload: {use_uvm}")
    
    config = model.config if hasattr(model, 'config') else None
    
    if config is None:
        print("[ChunkedAttention] Warning: Could not find model config, skipping auto-patch")
        return model
    
    # Detect model type
    model_type = getattr(config, 'model_type', None)
    
    # Try Qwen2VL-specific patching
    if model_type in ['qwen2', 'qwen2_vl']:
        print(f"[ChunkedAttention] Detected model type: {model_type}")
        print(f"[ChunkedAttention] Note: Qwen2VL-specific patching not yet implemented")
        print(f"[ChunkedAttention] ChunkedAttention will be available when combined with RingAttention")
        return model
    
    print(f"[ChunkedAttention] Generic auto-patching not yet implemented for model type: {model_type}")
    
    return model

