"""CUDA device introspection.

This is the single source of truth for hardware identity used by both the
benchmark harness and the measurement cache's hardware fingerprint.
"""

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


class HardwareProfilerError(RuntimeError):
    """Raised when GPU introspection is requested without a CUDA device."""


def get_hardware_identity(device_index: int = 0) -> dict[str, Any]:
    """Record the current GPU and CUDA state for measurement provenance."""
    if not torch.cuda.is_available():
        raise HardwareProfilerError("CUDA is required for hardware introspection")
    props = torch.cuda.get_device_properties(device_index)
    return {
        "gpu_name": torch.cuda.get_device_name(device_index),
        "compute_capability": torch.cuda.get_device_capability(device_index),
        "total_memory_bytes": props.total_memory,
        "multi_processor_count": props.multi_processor_count,
        "cuda_version": torch.version.cuda,
        "device_index": device_index,
    }


def current_memory_snapshot(device_index: int = 0) -> dict[str, float]:
    """Report current/peak allocated and reserved memory in MB for the device."""
    if not torch.cuda.is_available():
        raise HardwareProfilerError("CUDA is required for memory introspection")
    device = f"cuda:{device_index}"
    return {
        "allocated_mb": torch.cuda.memory_allocated(device=device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device=device) / 1024**2,
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device=device) / 1024**2,
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device=device) / 1024**2,
    }
