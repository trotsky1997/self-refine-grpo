"""
Triton kernels for fast LoRA weight merging.

Fuses the LoRA merge operations:
1. delta_weight = (lora_B @ lora_A) * scaling
2. merged_weight = base_weight + delta_weight

This is significantly faster than PyTorch for large matrices.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def lora_merge_kernel(
    # Pointers
    base_ptr,
    lora_A_ptr,
    lora_B_ptr,
    output_ptr,
    # Shapes
    M, N, K,  # base: MxN, lora_A: KxN, lora_B: MxK
    # LoRA scaling
    scaling,
    # Strides
    stride_base_m, stride_base_n,
    stride_A_k, stride_A_n,
    stride_B_m, stride_B_k,
    stride_out_m, stride_out_n,
    # Meta-parameters
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused LoRA merge kernel: output = base + (lora_B @ lora_A) * scaling
    
    Grid: (M // BLOCK_M, N // BLOCK_N)
    
    Improved precision:
    - Use float32 accumulation
    - Convert base to float32 before addition
    - Explicit type casting
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block ranges
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Masks for boundary conditions
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    # Load base weight block and convert to float32 for higher precision
    base_block_ptrs = base_ptr + offs_m[:, None] * stride_base_m + offs_n[None, :] * stride_base_n
    base_block = tl.load(base_block_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    base_block = base_block.to(tl.float32)  # Ensure float32 for precision
    
    # Compute delta_weight = lora_B @ lora_A
    # Accumulator for the matrix multiplication (always float32)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over K dimension in chunks
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        mask_k = k_offs < K
        
        # Load lora_B block: (BLOCK_M, BLOCK_K)
        lora_B_ptrs = lora_B_ptr + offs_m[:, None] * stride_B_m + k_offs[None, :] * stride_B_k
        lora_B_block = tl.load(lora_B_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        lora_B_block = lora_B_block.to(tl.float32)  # Ensure float32
        
        # Load lora_A block: (BLOCK_K, BLOCK_N)
        lora_A_ptrs = lora_A_ptr + k_offs[:, None] * stride_A_k + offs_n[None, :] * stride_A_n
        lora_A_block = tl.load(lora_A_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)
        lora_A_block = lora_A_block.to(tl.float32)  # Ensure float32
        
        # Accumulate: delta += B @ A (float32 precision)
        acc += tl.dot(lora_B_block, lora_A_block, allow_tf32=False)
    
    # Apply scaling and add to base (all in float32)
    # output = base + delta * scaling
    delta_scaled = acc * scaling
    output_block = base_block + delta_scaled
    
    # Store result
    output_ptrs = output_ptr + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(output_ptrs, output_block, mask=mask_m[:, None] & mask_n[None, :])


def lora_merge_triton(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """
    Fast LoRA merge using Triton.
    
    Args:
        base_weight: Base model weight (M x N)
        lora_A: LoRA A matrix (K x N), where K is the rank
        lora_B: LoRA B matrix (M x K)
        scaling: LoRA scaling factor (alpha / rank)
    
    Returns:
        Merged weight: base_weight + (lora_B @ lora_A) * scaling
    
    Shape analysis:
        base_weight: (M, N)  - e.g., (4096, 4096) for attention
        lora_A:      (K, N)  - e.g., (8, 4096) for rank-8 LoRA
        lora_B:      (M, K)  - e.g., (4096, 8)
        output:      (M, N)  - e.g., (4096, 4096)
    """
    # Validate shapes
    assert base_weight.dim() == 2, "base_weight must be 2D"
    assert lora_A.dim() == 2, "lora_A must be 2D"
    assert lora_B.dim() == 2, "lora_B must be 2D"
    
    M, N = base_weight.shape
    K_A, N_A = lora_A.shape
    M_B, K_B = lora_B.shape
    
    assert N == N_A, f"Dimension mismatch: base_weight.shape[1]={N} != lora_A.shape[1]={N_A}"
    assert M == M_B, f"Dimension mismatch: base_weight.shape[0]={M} != lora_B.shape[0]={M_B}"
    assert K_A == K_B, f"Dimension mismatch: lora_A.shape[0]={K_A} != lora_B.shape[1]={K_B}"
    
    K = K_A
    
    # Ensure all tensors are on the same device and contiguous
    device = base_weight.device
    dtype = base_weight.dtype
    
    lora_A = lora_A.to(device=device, dtype=dtype).contiguous()
    lora_B = lora_B.to(device=device, dtype=dtype).contiguous()
    base_weight = base_weight.contiguous()
    
    # Allocate output
    output = torch.empty_like(base_weight)
    
    # Optimal block sizes (tuned for A100/H100)
    # Note: Triton requires M, N, K >= 16 for tl.dot
    # For smaller ranks, we still use BLOCK_K=16 but only compute valid elements
    if K <= 16:
        BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 16
    elif K <= 32:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
    else:
        BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 64
    
    # Launch grid
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    
    # Launch kernel
    lora_merge_kernel[grid](
        base_weight, lora_A, lora_B, output,
        M, N, K,
        scaling,
        base_weight.stride(0), base_weight.stride(1),
        lora_A.stride(0), lora_A.stride(1),
        lora_B.stride(0), lora_B.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    
    return output


def lora_merge_batch_triton(
    base_weights: dict[str, torch.Tensor],
    lora_A_weights: dict[str, torch.Tensor],
    lora_B_weights: dict[str, torch.Tensor],
    scaling: float,
    verbose: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Batch LoRA merge for multiple weight matrices using Triton.
    
    Args:
        base_weights: Dict of base model weights {name: tensor}
        lora_A_weights: Dict of LoRA A matrices {module_name: tensor}
        lora_B_weights: Dict of LoRA B matrices {module_name: tensor}
        scaling: LoRA scaling factor
        verbose: Print progress
    
    Returns:
        Dict of merged weights {name: tensor}
    """
    merged_weights = {}
    merged_count = 0
    
    for base_name, base_weight in base_weights.items():
        # Extract module name without .weight/.bias suffix
        if base_name.endswith('.weight'):
            module_name = base_name[:-7]
        elif base_name.endswith('.bias'):
            module_name = base_name[:-5]
        else:
            module_name = base_name
        
        # Check if this weight has LoRA
        if module_name in lora_A_weights and module_name in lora_B_weights and base_name.endswith('.weight'):
            lora_A = lora_A_weights[module_name]
            lora_B = lora_B_weights[module_name]
            
            # Use Triton kernel for merging
            merged_weight = lora_merge_triton(base_weight, lora_A, lora_B, scaling)
            merged_weights[base_name] = merged_weight
            merged_count += 1
            
            # if verbose and merged_count % 10 == 0:
            #     print(f"[Triton LoRA Merge] Processed {merged_count} layers...")
        else:
            # No LoRA for this weight -> use base as-is
            merged_weights[base_name] = base_weight
    
    # if verbose:
    #     print(f"[Triton LoRA Merge] Completed: {merged_count} layers merged with Triton (scaling={scaling:.2f})")
    
    return merged_weights, merged_count


# Fallback to PyTorch if Triton not available
def lora_merge_pytorch(
    base_weight: torch.Tensor,
    lora_A: torch.Tensor,
    lora_B: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """PyTorch fallback for LoRA merge."""
    delta_weight = (lora_B @ lora_A) * scaling
    return base_weight + delta_weight


# Auto-select best implementation
try:
    # Test if Triton is available and working
    _ = triton.__version__
    TRITON_AVAILABLE = True
    lora_merge = lora_merge_triton
    print("✅ Triton LoRA merge enabled")
except (ImportError, AttributeError):
    TRITON_AVAILABLE = False
    lora_merge = lora_merge_pytorch
    print("⚠️  Triton not available, using PyTorch fallback for LoRA merge")

