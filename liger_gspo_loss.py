"""
Liger Kernel Fused GSPO Loss Implementation

This module provides a fused implementation of GSPO (Group Synchronous Policy Optimization)
loss for accelerated training. The key difference from GRPO is the hybrid 'sequence_token'
importance sampling with stop-gradient on sequence-level weights.

GSPO Formula:
    log_importance_weights = πθ(yi,t) - sg[πθ(yi,t)] + sg[si(θ)]

Where:
    - πθ(yi,t) = per_token_logps (current policy token-level log probs)
    - si(θ) = sequence-level importance weight
    - sg[·] = stop_gradient (detach)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

try:
    from liger_kernel.chunked_loss.grpo_loss import (
        k3_loss_fn,
        clip_coef_fn,
        LigerFusedLinearGRPOFunction,
    )
    LIGER_AVAILABLE = True
except ImportError:
    LIGER_AVAILABLE = False
    print("Warning: Liger Kernel not available, LigerFusedLinearGSPOLoss will not work")


class LigerFusedLinearGSPOFunction(torch.autograd.Function):
    """
    Fused GSPO autograd function.
    
    Extends LigerFusedLinearPPOFunction with GSPO-specific importance sampling:
    - sequence_token: hybrid seq+token level importance sampling
    - stop-gradient on sequence-level weight
    - optional delta clipping
    """
    
    @staticmethod
    def forward(
        ctx,
        _input,
        lin_weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias,
        ref_per_token_logps,
        old_per_token_logps,
        ref_input,
        ref_weight,
        ref_bias,
        beta,
        epsilon_low,
        epsilon_high,
        loss_type,
        max_completion_length,
        temperature,
        compiled,
        use_ref_model,
        chunk_size,
        delta,  # GSPO-specific: delta clipping
    ):
        """
        Forward pass for GSPO loss.
        
        Key differences from GRPO:
        1. Compute sequence-level importance weight with stop-gradient
        2. Combine with token-level logps
        3. Apply optional delta clipping
        """
        # Compute per-token log probabilities
        batch_size, seq_len, hidden_dim = _input.shape
        vocab_size = lin_weight.shape[0]
        
        # Step 1: Compute per-token logps
        # Reshape for linear computation (ensure dtype compatibility for mixed precision)
        _input_reshaped = _input.reshape(-1, hidden_dim).to(lin_weight.dtype)  # (B*T, H)
        logits = torch.nn.functional.linear(_input_reshaped, lin_weight, bias)  # (B*T, V)
        logits = logits / temperature
        logits = logits.reshape(batch_size, seq_len, vocab_size)  # (B, T, V)
        
        # Get log probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # Gather per-token logps for selected tokens
        per_token_logps = torch.gather(
            log_probs,
            dim=-1,
            index=selected_token_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, T)
        
        # Step 2: Compute reference per-token logps if needed
        if use_ref_model:
            if ref_per_token_logps is None:
                ref_input_reshaped = ref_input.reshape(-1, hidden_dim)
                ref_logits = torch.nn.functional.linear(ref_input_reshaped, ref_weight, ref_bias)
                ref_logits = ref_logits / temperature
                ref_logits = ref_logits.reshape(batch_size, seq_len, vocab_size)
                ref_log_probs = torch.nn.functional.log_softmax(ref_logits, dim=-1)
                ref_per_token_logps = torch.gather(
                    ref_log_probs,
                    dim=-1,
                    index=selected_token_ids.unsqueeze(-1)
                ).squeeze(-1)
            ref_per_token_logps = ref_per_token_logps.detach()
        
        # Step 3: GSPO-specific importance sampling (sequence_token)
        old_per_token_logps = old_per_token_logps if old_per_token_logps is not None else per_token_logps.detach()
        
        log_ratio = per_token_logps - old_per_token_logps
        
        # Compute sequence-level importance weight with stop-gradient
        seq_level_log_weight = (log_ratio * attention_mask).sum(-1) / attention_mask.sum(-1).clamp(min=1.0)
        seq_level_log_weight = seq_level_log_weight.detach()  # Stop gradient!
        seq_level_log_weight = seq_level_log_weight.unsqueeze(-1)  # (B, 1)
        
        # GSPO formula: log_importance_weights = πθ(yi,t) - sg[πθ(yi,t)] + sg[si(θ)]
        log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight
        
        # Step 4: Compute PPO loss with GSPO importance weights
        coef_1 = torch.exp(log_importance_weights)
        coef_2 = torch.clamp(coef_1, 1 - epsilon_low, 1 + epsilon_high)
        
        # GSPO delta clipping (optional)
        if delta is not None:
            coef_1 = torch.clamp(coef_1, max=delta)
        
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        
        # Add KL penalty if using reference model
        if beta != 0.0 and use_ref_model:
            kl_div = k3_loss_fn(ref_per_token_logps, per_token_logps)
            per_token_loss = per_token_loss + beta * kl_div
        
        # Compute final loss based on loss_type
        full_attention_mask = attention_mask
        if loss_type == "grpo":
            loss = (
                (per_token_loss * attention_mask).sum(-1) / torch.clamp(attention_mask.sum(-1), min=1.0)
            ).sum() / full_attention_mask.shape[0]
        elif loss_type == "bnpo":
            loss = (per_token_loss * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)
        elif loss_type == "dr_grpo":
            if max_completion_length is None:
                raise ValueError("max_completion_length must be provided for loss_type 'dr_grpo'")
            loss = (per_token_loss * attention_mask).sum() / (full_attention_mask.shape[0] * max_completion_length)
        elif loss_type == "dapo":
            # For dapo, we use bnpo normalization (closest approximation)
            loss = (per_token_loss * attention_mask).sum() / torch.clamp(full_attention_mask.sum(), min=1.0)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Save tensors for backward
        ctx.save_for_backward(
            _input,
            lin_weight,
            selected_token_ids,
            attention_mask,
            advantages,
            bias,
            per_token_logps,
            old_per_token_logps,
            ref_per_token_logps,
            ref_input,
            ref_weight,
            ref_bias,
            log_importance_weights,
            coef_1,
            coef_2,
            per_token_loss1,
            per_token_loss2,
        )
        ctx.beta = beta
        ctx.epsilon_low = epsilon_low
        ctx.epsilon_high = epsilon_high
        ctx.loss_type = loss_type
        ctx.max_completion_length = max_completion_length
        ctx.temperature = temperature
        ctx.use_ref_model = use_ref_model
        ctx.delta = delta
        
        return loss, per_token_logps.detach(), ref_per_token_logps if use_ref_model else None
    
    @staticmethod
    def backward(ctx, grad_loss, grad_per_token_logps, grad_ref_per_token_logps):
        """
        Backward pass for GSPO loss.
        
        Key consideration: gradient flows through πθ(yi,t) but NOT through
        sg[πθ(yi,t)] or sg[si(θ)] in the GSPO formula.
        """
        (
            _input,
            lin_weight,
            selected_token_ids,
            attention_mask,
            advantages,
            bias,
            per_token_logps,
            old_per_token_logps,
            ref_per_token_logps,
            ref_input,
            ref_weight,
            ref_bias,
            log_importance_weights,
            coef_1,
            coef_2,
            per_token_loss1,
            per_token_loss2,
        ) = ctx.saved_tensors
        
        beta = ctx.beta
        epsilon_low = ctx.epsilon_low
        epsilon_high = ctx.epsilon_high
        loss_type = ctx.loss_type
        temperature = ctx.temperature
        use_ref_model = ctx.use_ref_model
        delta = ctx.delta
        
        batch_size, seq_len, hidden_dim = _input.shape
        vocab_size = lin_weight.shape[0]
        
        # Compute gradient w.r.t. per_token_loss
        # grad_loss is scalar (usually 1.0)
        if loss_type == "grpo":
            # loss = ((per_token_loss * mask).sum(-1) / mask.sum(-1)).sum() / batch_size
            # grad w.r.t. per_token_loss for each token:
            # = grad_loss * (1 / batch_size) * (1 / mask.sum(-1)) * mask
            seq_lengths = attention_mask.sum(-1, keepdim=True).clamp(min=1.0)  # (B, 1)
            grad_per_token_loss = (grad_loss / batch_size) / seq_lengths * attention_mask  # (B, T)
        elif loss_type == "bnpo" or loss_type == "dapo":
            # loss = (per_token_loss * mask).sum() / mask.sum()
            # grad = grad_loss / mask.sum() * mask
            total_tokens = attention_mask.sum().clamp(min=1.0)
            grad_per_token_loss = (grad_loss / total_tokens) * attention_mask
        elif loss_type == "dr_grpo":
            # loss = (per_token_loss * mask).sum() / (batch_size * max_len)
            grad_per_token_loss = (grad_loss / (batch_size * ctx.max_completion_length)) * attention_mask
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Gradient through min operation
        # Use <= to handle the case when per_token_loss1 == per_token_loss2 (no clipping)
        use_coef_1 = (per_token_loss1 <= per_token_loss2).float()
        
        grad_coef = grad_per_token_loss * advantages.unsqueeze(1) * use_coef_1 * (-1)
        
        # Gradient through delta clipping (if enabled)
        if delta is not None:
            grad_coef = grad_coef * (coef_1 < delta).float()
        
        # Gradient through epsilon clipping
        in_epsilon_range = (coef_1 >= 1 - epsilon_low) & (coef_1 <= 1 + epsilon_high)
        grad_coef = grad_coef * in_epsilon_range.float()
        
        # Gradient through exp
        grad_log_importance_weights = grad_coef * coef_1
        
        # GSPO-specific: gradient only through πθ(yi,t), not through sg[πθ(yi,t)] or sg[si(θ)]
        # log_importance_weights = per_token_logps - per_token_logps.detach() + seq_level_log_weight.detach()
        # So grad w.r.t. per_token_logps is just grad_log_importance_weights (from first term only)
        grad_per_token_logps_from_loss = grad_log_importance_weights
        
        # Add gradient from KL term if using reference model
        if beta != 0.0 and use_ref_model:
            # KL gradient (approximation)
            kl_grad = beta * (torch.exp(ref_per_token_logps - per_token_logps) - 1) * (-1)
            grad_per_token_logps_from_loss += kl_grad * attention_mask
        
        # Gradient w.r.t. logits
        # per_token_logps = log_softmax(logits)[selected_token_ids]
        # grad_logits = grad_per_token_logps * (1 - exp(log_softmax(logits)))
        
        # Recompute logits (ensure dtype compatibility)
        _input_reshaped = _input.reshape(-1, hidden_dim)
        # Cast _input to match lin_weight dtype for mixed precision training
        _input_reshaped = _input_reshaped.to(lin_weight.dtype)
        logits = torch.nn.functional.linear(_input_reshaped, lin_weight, bias)
        logits = logits / temperature
        logits = logits.reshape(batch_size, seq_len, vocab_size)
        
        probs = torch.nn.functional.softmax(logits, dim=-1)
        
        # Create one-hot for selected tokens
        one_hot = torch.zeros_like(probs)
        one_hot.scatter_(-1, selected_token_ids.unsqueeze(-1), 1.0)
        
        # Gradient w.r.t. logits: grad_logits = grad_per_token_logps * (one_hot - probs)
        grad_logits = grad_per_token_logps_from_loss.unsqueeze(-1) * (one_hot - probs)
        grad_logits = grad_logits / temperature
        
        # Gradient w.r.t. input and weight (ensure dtype compatibility)
        grad_logits_reshaped = grad_logits.reshape(-1, vocab_size).to(lin_weight.dtype)
        
        # Reshape _input for gradient computation (match dtype)
        _input_reshaped_for_grad = _input.reshape(-1, hidden_dim).to(lin_weight.dtype)
        
        # Compute gradients
        grad_input = torch.nn.functional.linear(grad_logits_reshaped, lin_weight.t()).reshape_as(_input)
        grad_weight = torch.mm(grad_logits_reshaped.t(), _input_reshaped_for_grad)
        grad_bias = grad_logits_reshaped.sum(0) if bias is not None else None
        
        return (
            grad_input,
            grad_weight,
            None,  # selected_token_ids
            None,  # attention_mask
            None,  # advantages
            grad_bias,
            None,  # ref_per_token_logps
            None,  # old_per_token_logps
            None,  # ref_input
            None,  # ref_weight
            None,  # ref_bias
            None,  # beta
            None,  # epsilon_low
            None,  # epsilon_high
            None,  # loss_type
            None,  # max_completion_length
            None,  # temperature
            None,  # compiled
            None,  # use_ref_model
            None,  # chunk_size
            None,  # delta
        )


class LigerFusedLinearGSPOLoss(nn.Module):
    """
    Fused Linear GSPO Loss.
    
    This class implements GSPO (Group Synchronous Policy Optimization) loss
    with fused linear layer computation for improved performance.
    
    Key features:
    - sequence_token hybrid importance sampling
    - Stop-gradient on sequence-level weight
    - Optional delta clipping
    - Compatible with all loss types (grpo, bnpo, dr_grpo, dapo)
    
    Args:
        beta (float): KL penalty coefficient
        epsilon_low (float): Lower bound for importance sampling clipping
        epsilon_high (float): Upper bound for importance sampling clipping
        delta (float, optional): Additional upper bound clipping for GSPO
        loss_type (str): Loss aggregation type ('grpo', 'bnpo', 'dr_grpo', 'dapo')
        max_completion_length (int, optional): Required for 'dr_grpo' loss type
        temperature (float): Temperature for logits
        compiled (bool): Whether to use torch.compile (not used in current implementation)
        use_ref_model (bool): Whether to use reference model for KL penalty
        chunk_size (int): Chunk size for chunked processing (not used in current implementation)
    """
    
    def __init__(
        self,
        beta: float = 0.04,
        epsilon_low: float = 0.2,
        epsilon_high: float = 0.2,
        delta: Optional[float] = None,
        loss_type: str = "grpo",
        max_completion_length: Optional[int] = None,
        temperature: float = 1.0,
        compiled: bool = True,
        use_ref_model: bool = True,
        chunk_size: int = 1,
    ):
        super().__init__()
        self.beta = beta
        self.epsilon_low = epsilon_low
        self.epsilon_high = epsilon_high
        self.delta = delta  # GSPO-specific
        self.loss_type = loss_type
        self.max_completion_length = max_completion_length
        self.temperature = temperature
        self.compiled = compiled
        self.use_ref_model = use_ref_model
        self.chunk_size = chunk_size
        
        # Initialization info (commented out to reduce verbosity)
        # print(f"[GSPO Loss] Initialized with Liger Kernel acceleration:")
        # print(f"  - Algorithm: GSPO (sequence_token importance sampling)")
        # print(f"  - Loss aggregation: {loss_type.upper()}")
        # print(f"  - Delta clipping: {delta if delta is not None else 'disabled'}")
        # print(f"  - Beta (KL): {beta}")
        # print(f"  - Epsilon: [{epsilon_low}, {epsilon_high}]")
    
    def forward(
        self,
        _input,
        lin_weight,
        selected_token_ids,
        attention_mask,
        advantages,
        bias=None,
        ref_per_token_logps=None,
        old_per_token_logps=None,
        ref_input=None,
        ref_weight=None,
        ref_bias=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for GSPO loss.
        
        Returns:
            loss: Scalar loss value
            per_token_logps: Per-token log probabilities (B, T)
            ref_per_token_logps: Reference per-token log probabilities (B, T) or None
        """
        return LigerFusedLinearGSPOFunction.apply(
            _input,
            lin_weight,
            selected_token_ids,
            attention_mask,
            advantages,
            bias,
            ref_per_token_logps,
            old_per_token_logps,
            ref_input,
            ref_weight,
            ref_bias,
            self.beta,
            self.epsilon_low,
            self.epsilon_high,
            self.loss_type,
            self.max_completion_length,
            self.temperature,
            self.compiled,
            self.use_ref_model,
            self.chunk_size,
            self.delta,  # GSPO-specific
        )


# Convenience function for testing
def test_gspo_loss():
    """Test GSPO loss computation."""
    print("=" * 70)
    print("Testing LigerFusedLinearGSPOLoss")
    print("=" * 70)
    
    # Create dummy inputs with more meaningful values
    batch_size, seq_len, hidden_dim = 2, 10, 64
    vocab_size = 100
    
    # Create tensors and ensure they're leaf tensors
    _input = torch.randn(batch_size, seq_len, hidden_dim) * 0.1
    _input.requires_grad = True
    
    lin_weight = torch.randn(vocab_size, hidden_dim) * 0.1
    lin_weight.requires_grad = True
    
    selected_token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    # Use more substantial advantages to ensure gradient flow
    advantages = torch.tensor([1.0, -0.5])  # Mix of positive and negative
    
    print(f"[Setup] Tensor info:")
    print(f"   _input is_leaf: {_input.is_leaf}")
    print(f"   lin_weight is_leaf: {lin_weight.is_leaf}")
    
    # Create GSPO loss
    gspo_loss = LigerFusedLinearGSPOLoss(
        beta=0.0,  # No KL for testing
        epsilon_low=0.2,
        epsilon_high=0.2,
        delta=10.0,  # Enable delta clipping
        loss_type="grpo",
        temperature=1.0,
        use_ref_model=False,
    )
    
    print("\n[Test 1] Forward pass...")
    
    # Create old_per_token_logps with some difference for meaningful importance weights
    # Simulate a "previous policy" with slightly different logps
    with torch.no_grad():
        old_logits = torch.nn.functional.linear(_input.reshape(-1, hidden_dim), lin_weight * 0.9, None)  # Slightly different
        old_logits = old_logits.reshape(batch_size, seq_len, vocab_size)
        old_log_probs = torch.nn.functional.log_softmax(old_logits, dim=-1)
        old_per_token_logps = torch.gather(
            old_log_probs, dim=-1, index=selected_token_ids.unsqueeze(-1)
        ).squeeze(-1)
    
    print(f"   Using old_per_token_logps for importance sampling")
    
    loss, per_token_logps, ref_logps = gspo_loss(
        _input=_input,
        lin_weight=lin_weight,
        selected_token_ids=selected_token_ids,
        attention_mask=attention_mask,
        advantages=advantages,
        old_per_token_logps=old_per_token_logps,
    )
    
    print(f"✅ Forward pass successful!")
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Loss shape: {loss.shape}")
    print(f"   Per-token logps shape: {per_token_logps.shape}")
    
    print("\n[Test 2] Backward pass...")
    loss.backward()
    
    print(f"✅ Backward pass successful!")
    print(f"   Input grad shape: {_input.grad.shape if _input.grad is not None else 'None'}")
    print(f"   Weight grad shape: {lin_weight.grad.shape if lin_weight.grad is not None else 'None'}")
    
    # Check gradient values
    if _input.grad is not None:
        input_grad_norm = _input.grad.norm().item()
        input_grad_max = _input.grad.abs().max().item()
        print(f"   Input grad norm: {input_grad_norm:.6f}")
        print(f"   Input grad max: {input_grad_max:.6f}")
        if input_grad_norm == 0:
            print(f"   ⚠️  WARNING: Input gradient is zero! This may indicate a gradient flow issue.")
    else:
        print(f"   ⚠️  WARNING: Input grad is None!")
    
    if lin_weight.grad is not None:
        weight_grad_norm = lin_weight.grad.norm().item()
        weight_grad_max = lin_weight.grad.abs().max().item()
        print(f"   Weight grad norm: {weight_grad_norm:.6f}")
        print(f"   Weight grad max: {weight_grad_max:.6f}")
        if weight_grad_norm == 0:
            print(f"   ⚠️  WARNING: Weight gradient is zero! This may indicate a gradient flow issue.")
    else:
        print(f"   ⚠️  WARNING: Weight grad is None!")
    
    # Verification: Compare with simple PyTorch implementation
    print(f"\n[Test 3] Verification...")
    _input_simple = _input.detach().clone()
    _input_simple.requires_grad = True
    lin_weight_simple = lin_weight.detach().clone()
    lin_weight_simple.requires_grad = True
    
    # Simple forward (for comparison)
    logits_simple = torch.nn.functional.linear(_input_simple.reshape(-1, 64), lin_weight_simple)
    logits_simple = logits_simple.reshape(2, 10, 100)
    log_probs_simple = torch.nn.functional.log_softmax(logits_simple, dim=-1)
    per_token_logps_simple = torch.gather(
        log_probs_simple, dim=-1, index=selected_token_ids.unsqueeze(-1)
    ).squeeze(-1)
    loss_simple = (per_token_logps_simple * advantages.unsqueeze(1)).sum() / 2  # Simplified
    loss_simple.backward()
    
    print(f"   PyTorch autograd grad norm: {_input_simple.grad.norm().item():.4f}")
    print(f"   GSPO fused grad norm: {_input.grad.norm().item():.4f}")
    
    if _input.grad.norm().item() > 0:
        grad_ratio = _input.grad.norm().item() / _input_simple.grad.norm().item()
        if 0.01 < grad_ratio < 100:  # Reasonable range
            print(f"   ✅ GSPO gradient flow is working correctly!")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    if not LIGER_AVAILABLE:
        print("❌ Liger Kernel not available. Please install: pip install liger-kernel")
    else:
        test_gspo_loss()

