"""
Memory optimization utilities for APPFL components.

This module provides reusable functions for common memory optimization patterns
used throughout the APPFL codebase to reduce memory usage and improve garbage collection.
"""

import gc
import torch
import io
from typing import Dict, Any, Optional, Union, OrderedDict
from contextlib import contextmanager


def clone_state_dict_optimized(
    state_dict: Union[Dict[str, torch.Tensor], OrderedDict[str, torch.Tensor]],
    include_buffers: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient cloning of model state dict using tensor.clone().detach()
    instead of copy.deepcopy().

    Args:
        state_dict: The state dictionary to clone
        include_buffers: Whether to include buffer tensors (default: True)

    Returns:
        Cloned state dictionary
    """
    result = {}
    with torch.no_grad():
        for name, tensor in state_dict.items():
            if tensor is not None:
                result[name] = tensor.clone().detach()
    gc.collect()
    return result


def extract_model_state_optimized(
    model: torch.nn.Module, include_buffers: bool = True, cpu_transfer: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient extraction of model state using tensor cloning.

    Args:
        model: PyTorch model
        include_buffers: Whether to include named buffers (default: True)
        cpu_transfer: Whether to transfer tensors to CPU (default: False)

    Returns:
        Model state dictionary
    """
    state = {}
    with torch.no_grad():
        # Extract parameters
        for name, param in model.named_parameters():
            tensor = param.clone().detach()
            if cpu_transfer and tensor.device.type != "cpu":
                tensor = tensor.cpu()
            state[name] = tensor

        # Extract buffers if requested
        if include_buffers:
            for name, buffer in model.named_buffers():
                if buffer is not None:
                    tensor = buffer.clone().detach()
                    if cpu_transfer and tensor.device.type != "cpu":
                        tensor = tensor.cpu()
                    state[name] = tensor

    gc.collect()
    return state


def safe_inplace_operation(
    tensor: torch.Tensor,
    operation: str,
    operand: Union[torch.Tensor, float, int],
    alpha: Optional[float] = None,
) -> torch.Tensor:
    """
    Safely perform in-place operations with dtype checking to avoid errors
    with integer tensors.

    Args:
        tensor: Target tensor for operation
        operation: Operation type ('add', 'sub', 'mul', 'div')
        operand: Operand for the operation
        alpha: Optional scaling factor for add/sub operations

    Returns:
        Result tensor (may be the same tensor if in-place was used)
    """
    # Check if tensor supports in-place operations (avoid integer tensors)
    if tensor.dtype in [torch.int32, torch.int64, torch.long]:
        # For integer tensors, use regular operations
        if operation == "add":
            if alpha is not None:
                return tensor + alpha * operand
            else:
                return tensor + operand
        elif operation == "sub":
            if alpha is not None:
                return tensor - alpha * operand
            else:
                return tensor - operand
        elif operation == "mul":
            return tensor * operand
        elif operation == "div":
            return torch.div(tensor, operand).type(tensor.dtype)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
    else:
        # For float tensors, use in-place operations
        if operation == "add":
            if alpha is not None:
                tensor.add_(operand, alpha=alpha)
            else:
                tensor.add_(operand)
        elif operation == "sub":
            if alpha is not None:
                tensor.sub_(operand, alpha=alpha)
            else:
                tensor.sub_(operand)
        elif operation == "mul":
            tensor.mul_(operand)
        elif operation == "div":
            tensor.div_(operand)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        return tensor


def efficient_tensor_aggregation(
    tensors: Dict[str, torch.Tensor],
    weights: Optional[Dict[str, float]] = None,
    cleanup_intermediate: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Memory-efficient tensor aggregation with dtype-aware operations.

    Args:
        tensors: Dictionary of tensors to aggregate
        weights: Optional weights for weighted averaging
        cleanup_intermediate: Whether to clean up intermediate results

    Returns:
        Aggregated tensors
    """
    if not tensors:
        return {}

    # Initialize result with zeros
    result = {}
    tensor_names = set()
    for tensor_dict in tensors.values():
        tensor_names.update(tensor_dict.keys())

    with torch.no_grad():
        for name in tensor_names:
            # Find a tensor to use as template for this parameter
            template_tensor = None
            for tensor_dict in tensors.values():
                if name in tensor_dict:
                    template_tensor = tensor_dict[name]
                    break

            if template_tensor is not None:
                result[name] = torch.zeros_like(template_tensor)

        for client_id, tensor_dict in tensors.items():
            weight = (
                1.0 / len(tensors) if weights is None else weights.get(client_id, 0.0)
            )

            for name, tensor in tensor_dict.items():
                if name in result:
                    # Use safe in-place operations
                    weighted_tensor = tensor * weight
                    result[name] = safe_inplace_operation(
                        result[name], "add", weighted_tensor
                    )

                    if cleanup_intermediate:
                        del weighted_tensor

        if cleanup_intermediate:
            gc.collect()

    return result


@contextmanager
def memory_efficient_model_io(optimize_memory: bool = True, force_cpu: bool = True):
    """
    Context manager for memory-efficient model serialization/deserialization.

    Args:
        optimize_memory: Whether to use memory optimizations
        force_cpu: Whether to force CPU loading

    Yields:
        BytesIO buffer for model operations
    """
    if optimize_memory:
        buffer = io.BytesIO()
        try:
            yield buffer
        finally:
            # Explicit cleanup
            if hasattr(buffer, "close"):
                buffer.close()
            del buffer
            gc.collect()
    else:
        # Standard BytesIO without explicit cleanup
        buffer = io.BytesIO()
        yield buffer


def efficient_bytearray_concatenation(
    data_chunks: list, optimize_memory: bool = True
) -> bytes:
    """
    Memory-efficient concatenation using bytearray instead of bytes.

    Args:
        data_chunks: List of byte chunks to concatenate
        optimize_memory: Whether to use memory optimizations

    Returns:
        Concatenated bytes
    """
    if optimize_memory:
        # Use bytearray for more efficient concatenation
        result = bytearray()
        for chunk in data_chunks:
            result.extend(chunk)
        return bytes(result)
    else:
        # Standard bytes concatenation
        return b"".join(data_chunks)


def optimize_memory_cleanup(
    *objects, force_gc: bool = True, clear_cuda_cache: bool = False
) -> None:
    """
    Comprehensive memory cleanup utility.

    Args:
        *objects: Objects to delete
        force_gc: Whether to force garbage collection
        clear_cuda_cache: Whether to clear CUDA cache
    """
    # Delete objects
    for obj in objects:
        if obj is not None:
            del obj

    # Force garbage collection
    if force_gc:
        gc.collect()

    # Clear CUDA cache if requested and available
    if clear_cuda_cache and torch.cuda.is_available():
        torch.cuda.empty_cache()


def get_tensor_memory_info(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Get memory usage information for a tensor.

    Args:
        tensor: PyTorch tensor

    Returns:
        Dictionary with memory information
    """
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": tuple(tensor.shape),
        "numel": tensor.numel(),
        "element_size": tensor.element_size(),
        "memory_bytes": tensor.numel() * tensor.element_size(),
        "memory_mb": (tensor.numel() * tensor.element_size()) / (1024 * 1024),
    }


def log_optimization_status(
    component_name: str, optimize_memory: bool, logger=None
) -> None:
    """
    Log memory optimization status for a component.

    Args:
        component_name: Name of the component
        optimize_memory: Whether optimization is enabled
        logger: Optional logger object
    """
    status = "ENABLED" if optimize_memory else "DISABLED (original behavior)"
    message = f"{component_name}: Memory optimization {status}"

    if logger and hasattr(logger, "info"):
        logger.info(message)
    else:
        print(message)


# Backwards compatibility aliases
memory_efficient_clone = clone_state_dict_optimized
memory_efficient_extract = extract_model_state_optimized
safe_tensor_operation = safe_inplace_operation
