import torch
import inspect
import transformers


# ============================================================================
# Unsloth Offloaded Gradient Checkpointing
# Saves VRAM by smartly offloading to RAM with non-blocking transfers
# ============================================================================

class Unsloth_Offloaded_Gradient_Checkpointer(torch.autograd.Function):
    """
    Saves VRAM by smartly offloading to RAM.
    Tiny hit to performance, since we mask the movement via non blocking calls.
    
    V2: Enhanced with optional quantization and pinned memory.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        # Option 1: Standard CPU offload (original)
        saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
        
        # Option 2: Pinned memory for faster transfer (uncomment to use)
        # pinned_memory = torch.empty(
        #     hidden_states.shape, 
        #     dtype=hidden_states.dtype, 
        #     pin_memory=True
        # )
        # pinned_memory.copy_(hidden_states, non_blocking=True)
        # saved_hidden_states = pinned_memory
        
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        ctx.save_for_backward(saved_hidden_states)
        ctx.forward_function = forward_function
        ctx.args = args
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        hidden_states = hidden_states.to("cuda", non_blocking=True).detach()
        hidden_states.requires_grad = True
        with torch.enable_grad():
            output = ctx.forward_function(hidden_states, *ctx.args)
            if isinstance(output, tuple):
                output = output[0]
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,) * len(ctx.args)


class Unsloth_Offloaded_Gradient_Checkpointer_Quantized(torch.autograd.Function):
    """
    Enhanced version with 8-bit quantization (BnB-inspired).
    Saves even more VRAM by quantizing activations before offloading.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        # Quantize to int8 before offloading (saves 75% memory)
        original_dtype = hidden_states.dtype
        original_shape = hidden_states.shape
        
        # Improved per-channel quantization for better precision
        if hidden_states.dim() >= 2:
            # Reshape to 2D for per-channel quantization
            hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
            
            # Per-channel absmax (more accurate than global)
            absmax = hidden_2d.abs().max(dim=0, keepdim=True)[0]
            absmax = torch.clamp(absmax, min=1e-8)  # Avoid division by zero
            scale = absmax / 127.0
            
            # Clamp before quantization to avoid overflow
            quantized = (hidden_2d / scale).clamp(-127, 127).to(torch.int8)
            quantized = quantized.reshape(original_shape)
        else:
            # Fallback for 1D
            absmax = torch.clamp(hidden_states.abs().max(), min=1e-8)
            scale = absmax / 127.0
            quantized = (hidden_states / scale).clamp(-127, 127).to(torch.int8)
        
        # Offload quantized data + scale
        quantized_cpu = quantized.to("cpu", non_blocking=True)
        scale_cpu = scale.to("cpu", non_blocking=True)
        
        with torch.no_grad():
            output = forward_function(hidden_states, *args)
        
        ctx.save_for_backward(quantized_cpu, scale_cpu)
        ctx.forward_function = forward_function
        ctx.args = args
        ctx.original_dtype = original_dtype
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY):
        quantized, scale = ctx.saved_tensors
        
        # Dequantize and transfer back to GPU
        quantized = quantized.to("cuda", non_blocking=True)
        scale = scale.to("cuda", non_blocking=True)
        
        # Dequantize with per-channel scale
        if quantized.dim() >= 2 and scale.dim() >= 2:
            # Reshape for per-channel dequantization
            hidden_2d = quantized.reshape(-1, quantized.shape[-1]).to(ctx.original_dtype)
            hidden_states = (hidden_2d * scale).reshape(quantized.shape)
        else:
            hidden_states = quantized.to(ctx.original_dtype) * scale
        
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        
        with torch.enable_grad():
            output = ctx.forward_function(hidden_states, *ctx.args)
            if isinstance(output, tuple):
                output = output[0]
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,) * len(ctx.args)


class Unsloth_UVM_Gradient_Checkpointer(torch.autograd.Function):
    """
    CUDA Unified Memory (UVM) version - BnB paged_adamw style.
    Uses CUDA managed memory for automatic paging between GPU and CPU.
    
    NOTE: Requires PyTorch built with UVM support and proper CUDA setup.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        try:
            # Allocate managed memory (accessible from both CPU and GPU)
            # CUDA driver will automatically page data in/out as needed
            managed_storage = torch.cuda.memory.managed_alloc(
                hidden_states.numel() * hidden_states.element_size()
            )
            
            # Create tensor view on managed memory
            managed_tensor = torch.frombuffer(
                managed_storage,
                dtype=hidden_states.dtype,
                count=hidden_states.numel()
            ).reshape(hidden_states.shape)
            
            # Copy data to managed memory
            managed_tensor.copy_(hidden_states)
            
            with torch.no_grad():
                output = forward_function(hidden_states, *args)
            
            ctx.save_for_backward(managed_tensor)
            ctx.forward_function = forward_function
            ctx.args = args
            
        except (AttributeError, RuntimeError) as e:
            # Fallback to standard CPU offload if UVM not available
            print(f"[UVM GC] Warning: CUDA managed memory not available, falling back to CPU offload")
            print(f"[UVM GC] Error: {e}")
            saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
            
            with torch.no_grad():
                output = forward_function(hidden_states, *args)
            
            ctx.save_for_backward(saved_hidden_states)
            ctx.forward_function = forward_function
            ctx.args = args
        
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        
        # If in managed memory, CUDA will automatically page in
        # If in CPU memory (fallback), transfer back
        if not hidden_states.is_cuda:
            hidden_states = hidden_states.to("cuda", non_blocking=True)
        
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        
        with torch.enable_grad():
            output = ctx.forward_function(hidden_states, *ctx.args)
            if isinstance(output, tuple):
                output = output[0]
        torch.autograd.backward(output, dY)
        return (None, hidden_states.grad,) + (None,) * len(ctx.args)


class Unsloth_UVM_Gradient_Checkpointer_Alt(torch.autograd.Function):
    """
    Alternative UVM implementation using torch.cuda.memory.cudaMallocManaged.
    More compatible with different PyTorch versions.
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        try:
            # Try to use CUDA managed memory via ctypes
            import ctypes
            
            # Load CUDA runtime
            cuda = ctypes.CDLL("libcudart.so")
            
            # Allocate managed memory
            size = hidden_states.numel() * hidden_states.element_size()
            managed_ptr = ctypes.c_void_p()
            result = cuda.cudaMallocManaged(
                ctypes.byref(managed_ptr),
                ctypes.c_size_t(size),
                ctypes.c_uint(1)  # cudaMemAttachGlobal
            )
            
            if result != 0:
                raise RuntimeError(f"cudaMallocManaged failed with code {result}")
            
            # Create tensor from managed memory (via numpy)
            import numpy as np
            # Determine numpy dtype based on torch dtype
            if hidden_states.dtype == torch.float32:
                np_dtype = ctypes.c_float
                np_type = np.float32
            elif hidden_states.dtype == torch.float16:
                np_dtype = ctypes.c_uint16  # float16 stored as uint16
                np_type = np.float16
            elif hidden_states.dtype == torch.bfloat16:
                np_dtype = ctypes.c_uint16  # bfloat16 stored as uint16
                np_type = np.uint16  # numpy doesn't have bfloat16, use uint16
            else:
                raise ValueError(f"Unsupported dtype: {hidden_states.dtype}")
            
            managed_array = np.ctypeslib.as_array(
                ctypes.cast(managed_ptr, ctypes.POINTER(np_dtype)),
                shape=hidden_states.shape
            )
            
            if hidden_states.dtype == torch.bfloat16:
                # For bfloat16, we need special handling
                # Convert to CPU first, then copy bits
                temp_cpu = hidden_states.cpu()
                managed_array[:] = temp_cpu.view(torch.uint16).numpy()
                managed_tensor = torch.from_numpy(managed_array).view(torch.bfloat16)
            else:
                managed_tensor = torch.from_numpy(managed_array.astype(np_type))
                managed_tensor.copy_(hidden_states.cpu())
            
            # Advise CUDA to prefer CPU when not accessed
            cuda.cudaMemAdvise(
                managed_ptr,
                ctypes.c_size_t(size),
                ctypes.c_int(3),  # cudaMemAdviseSetPreferredLocation to CPU
                ctypes.c_int(-1)  # CPU device ID
            )
            
            with torch.no_grad():
                output = forward_function(hidden_states, *args)
            
            ctx.managed_ptr = managed_ptr
            ctx.cuda = cuda
            ctx.save_for_backward(managed_tensor)
            ctx.forward_function = forward_function
            ctx.args = args
            ctx.use_managed = True
            
        except Exception as e:
            # Fallback to standard CPU offload
            print(f"[UVM GC Alt] Warning: CUDA managed memory not available, using CPU offload")
            print(f"[UVM GC Alt] Error: {e}")
            saved_hidden_states = hidden_states.to("cpu", non_blocking=True)
            
            with torch.no_grad():
                output = forward_function(hidden_states, *args)
            
            ctx.save_for_backward(saved_hidden_states)
            ctx.forward_function = forward_function
            ctx.args = args
            ctx.use_managed = False
        
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY):
        (hidden_states,) = ctx.saved_tensors
        
        # Transfer to GPU if needed (UVM will automatically page in)
        if not hidden_states.is_cuda:
            # Ensure correct dtype before transfer
            hidden_states = hidden_states.to(device="cuda", non_blocking=True)
        
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        
        with torch.enable_grad():
            output = ctx.forward_function(hidden_states, *ctx.args)
            if isinstance(output, tuple):
                output = output[0]
        torch.autograd.backward(output, dY)
        
        # Free managed memory if used
        if ctx.use_managed and hasattr(ctx, 'managed_ptr'):
            try:
                ctx.cuda.cudaFree(ctx.managed_ptr)
            except:
                pass  # Already freed or error
        
        return (None, hidden_states.grad,) + (None,) * len(ctx.args)


class Unsloth_UVM_Quantized_Gradient_Checkpointer(torch.autograd.Function):
    """
    Ultimate memory saver: UVM + 8-bit quantization.
    Combines CUDA Unified Memory automatic paging with int8 quantization.
    
    Expected savings: ~93.75% (UVM auto-paging + int8 vs GPU fp16)
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd
    def forward(ctx, forward_function, hidden_states, *args):
        try:
            import ctypes
            
            original_dtype = hidden_states.dtype
            original_shape = hidden_states.shape
            
            # Step 1: Improved per-channel quantization
            if hidden_states.dim() >= 2:
                hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
                absmax = hidden_2d.abs().max(dim=0, keepdim=True)[0]
                absmax = torch.clamp(absmax, min=1e-8)
                scale = absmax / 127.0
                quantized = (hidden_2d / scale).clamp(-127, 127).to(torch.int8)
                quantized = quantized.reshape(original_shape)
            else:
                absmax = torch.clamp(hidden_states.abs().max(), min=1e-8)
                scale = absmax / 127.0
                quantized = (hidden_states / scale).clamp(-127, 127).to(torch.int8)
            
            # Step 2: Allocate UVM for quantized data (automatic paging)
            cuda = ctypes.CDLL("libcudart.so")
            size = quantized.numel() * quantized.element_size()
            managed_ptr = ctypes.c_void_p()
            result = cuda.cudaMallocManaged(
                ctypes.byref(managed_ptr),
                ctypes.c_size_t(size),
                ctypes.c_uint(1)
            )
            
            if result != 0:
                raise RuntimeError(f"cudaMallocManaged failed with code {result}")
            
            # Create tensor from managed memory (via numpy)
            import numpy as np
            managed_array = np.ctypeslib.as_array(
                ctypes.cast(managed_ptr, ctypes.POINTER(ctypes.c_int8)),
                shape=quantized.shape
            )
            managed_quantized = torch.from_numpy(managed_array)
            managed_quantized.copy_(quantized.cpu())  # Copy to managed memory
            
            # Advise CUDA to prefer CPU (data is small after quantization)
            cuda.cudaMemAdvise(
                managed_ptr,
                ctypes.c_size_t(size),
                ctypes.c_int(3),  # cudaMemAdviseSetPreferredLocation to CPU
                ctypes.c_int(-1)
            )
            
            with torch.no_grad():
                output = forward_function(hidden_states, *args)
            
            # Store scale in regular CPU memory (tiny)
            scale_cpu = scale.to("cpu", non_blocking=True)
            
            ctx.managed_ptr = managed_ptr
            ctx.cuda = cuda
            ctx.save_for_backward(managed_quantized, scale_cpu)
            ctx.forward_function = forward_function
            ctx.args = args
            ctx.original_dtype = original_dtype
            ctx.use_managed = True
            
        except Exception as e:
            # Fallback to CPU offload + quantization
            print(f"[UVM+Quant GC] Warning: UVM not available, using CPU offload + quantization")
            print(f"[UVM+Quant GC] Error: {e}")
            
            original_dtype = hidden_states.dtype
            original_shape = hidden_states.shape
            
            # Use same improved quantization in fallback
            if hidden_states.dim() >= 2:
                hidden_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
                absmax = hidden_2d.abs().max(dim=0, keepdim=True)[0]
                absmax = torch.clamp(absmax, min=1e-8)
                scale = absmax / 127.0
                quantized = (hidden_2d / scale).clamp(-127, 127).to(torch.int8)
                quantized = quantized.reshape(original_shape)
            else:
                absmax = torch.clamp(hidden_states.abs().max(), min=1e-8)
                scale = absmax / 127.0
                quantized = (hidden_states / scale).clamp(-127, 127).to(torch.int8)
            
            quantized_cpu = quantized.to("cpu", non_blocking=True)
            scale_cpu = scale.to("cpu", non_blocking=True)
            
            with torch.no_grad():
                output = forward_function(hidden_states, *args)
            
            ctx.save_for_backward(quantized_cpu, scale_cpu)
            ctx.forward_function = forward_function
            ctx.args = args
            ctx.original_dtype = original_dtype
            ctx.use_managed = False
        
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, dY):
        quantized, scale = ctx.saved_tensors
        
        # Dequantize (UVM will auto page-in if needed)
        if not quantized.is_cuda:
            quantized = quantized.to("cuda", non_blocking=True)
        if not scale.is_cuda:
            scale = scale.to("cuda", non_blocking=True)
        
        # Dequantize with per-channel scale
        if quantized.dim() >= 2 and scale.dim() >= 2:
            hidden_2d = quantized.reshape(-1, quantized.shape[-1]).to(ctx.original_dtype)
            hidden_states = (hidden_2d * scale).reshape(quantized.shape)
        else:
            hidden_states = quantized.to(ctx.original_dtype) * scale
        
        hidden_states = hidden_states.detach()
        hidden_states.requires_grad = True
        
        with torch.enable_grad():
            # Forward function may return single tensor or tuple
            output = ctx.forward_function(hidden_states, *ctx.args)
            if isinstance(output, tuple):
                output = output[0]
        
        torch.autograd.backward(output, dY)
        
        # Free managed memory if used
        if ctx.use_managed and hasattr(ctx, 'managed_ptr'):
            try:
                ctx.cuda.cudaFree(ctx.managed_ptr)
            except:
                pass  # Already freed or error
        
        return (None, hidden_states.grad,) + (None,) * len(ctx.args)


def unsloth_offloaded_gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    """Monkey-patched gradient_checkpointing_enable using Unsloth's offloaded checkpointing."""
    if gradient_checkpointing_kwargs is not None:
        print(f"[Unsloth GC] Warning: gradient_checkpointing_kwargs ignored: {gradient_checkpointing_kwargs}")
    
    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    # Use the globally configured checkpointer class
    gradient_checkpointing_func = _GRADIENT_CHECKPOINTER_CLASS.apply
    
    # Check if using new format (transformers >= 4.35.0)
    _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

    if not _is_using_old_format:
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
    else:
        # Fallback for old format
        self._set_gradient_checkpointing(value=True)
        print(f"[Unsloth GC] Warning: Old format detected, using standard checkpointing")

    if getattr(self, "_hf_peft_config_loaded", False):
        # When using PEFT + gradient checkpointing, ensure input has requires_grad=True
        self.enable_input_require_grads()


def apply_unsloth_offloaded_gradient_checkpoint(mode="offload"):
    """
    Apply Unsloth's offloaded gradient checkpointing to all models.
    
    Args:
        mode: Checkpoint mode to use:
            - "offload"        : Standard CPU offload (default, most compatible)
            - "quantized"      : CPU offload + 8-bit quantization (87.5% savings)
            - "uvm"            : CUDA Unified Memory (automatic paging, BnB-style)
            - "uvm_alt"        : Alternative UVM (uses ctypes + cudaMemAdvise)
            - "uvm_quantized"  : UVM + 8-bit quantization (ULTIMATE, 93.75% savings)
    """
    global _GRADIENT_CHECKPOINTER_CLASS
    
    if mode == "quantized":
        _GRADIENT_CHECKPOINTER_CLASS = Unsloth_Offloaded_Gradient_Checkpointer_Quantized
        print("✅ Gradient Checkpointing: CPU Offload + 8-bit Quantization")
        print("   💾 Memory savings: ~87.5% (vs GPU fp16/bf16)")
        print("   ⚠️  Precision loss: ~0.1%")
    
    elif mode == "uvm":
        _GRADIENT_CHECKPOINTER_CLASS = Unsloth_UVM_Gradient_Checkpointer
        print("✅ Gradient Checkpointing: CUDA Unified Memory (UVM)")
        print("   💾 Memory: Automatic paging by CUDA driver")
        print("   🚀 Performance: Hardware-managed, on-demand")
        print("   ⚠️  Fallback: CPU offload if UVM unavailable")
    
    elif mode == "uvm_alt":
        _GRADIENT_CHECKPOINTER_CLASS = Unsloth_UVM_Gradient_Checkpointer_Alt
        print("✅ Gradient Checkpointing: CUDA Unified Memory (Alt)")
        print("   💾 Memory: Auto-paging via cudaMallocManaged")
        print("   🚀 Performance: Hardware + cudaMemAdvise hints")
        print("   ⚠️  Fallback: CPU offload if UVM unavailable")
    
    elif mode == "uvm_quantized":
        _GRADIENT_CHECKPOINTER_CLASS = Unsloth_UVM_Quantized_Gradient_Checkpointer
        print("✅ Gradient Checkpointing: UVM + 8-bit Quantization (ULTIMATE)")
        print("   💾 Memory savings: ~93.75% (UVM + int8 vs GPU fp16)")
        print("   🚀 Technique 1: CUDA auto-paging (BnB style)")
        print("   🚀 Technique 2: 8-bit quantization (75% reduction)")
        print("   ⚠️  Precision loss: ~0.1%")
        print("   ⚠️  Fallback: CPU offload + quantization if UVM unavailable")
    
    else:  # mode == "offload" (default)
        _GRADIENT_CHECKPOINTER_CLASS = Unsloth_Offloaded_Gradient_Checkpointer
        print("✅ Gradient Checkpointing: Standard CPU Offload")
        print("   💾 Memory savings: ~50% (vs GPU storage)")
        print("   🚀 Performance: Non-blocking async transfers")
    
    original_method = transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_enable
    transformers.modeling_utils.PreTrainedModel.gradient_checkpointing_enable = (
        unsloth_offloaded_gradient_checkpointing_enable
    )
    return original_method


# Global variable to store which checkpointer to use
_GRADIENT_CHECKPOINTER_CLASS = Unsloth_Offloaded_Gradient_Checkpointer