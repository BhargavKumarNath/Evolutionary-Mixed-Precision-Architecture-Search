"""Hardware benchmark harness for GPU measurements and analysis."""

import logging
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.cuda
from torch import nn

from ..hardware.profiler import get_hardware_identity as _get_hardware_identity

logger = logging.getLogger(__name__)


class BenchmarkError(RuntimeError):
    """Raised when benchmark conditions are not met."""


@dataclass(frozen=True)
class MemoryStats:
    baseline_mb: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    incremental_mb: float


@dataclass(frozen=True)
class LatencyStats:
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    p50_ms: float | None = None
    p95_ms: float | None = None
    p99_ms: float | None = None


@dataclass(frozen=True)
class ThroughputStats:
    tokens_per_sec: float
    batches_per_sec: float


@dataclass(frozen=True)
class QualityStats:
    mean_loss: float
    perplexity: float
    loss_samples: int


@dataclass(frozen=True)
class BenchmarkResult:
    genome: list[int]
    backend: str
    model_name: str
    batch_size: int
    sequence_length: int
    warmup_iterations: int
    benchmark_iterations: int
    load_memory: MemoryStats
    inference_memory: MemoryStats
    inference_latency: LatencyStats
    inference_throughput: ThroughputStats
    validation_quality: QualityStats | None = None
    hardware_identity: dict[str, Any] | None = None
    raw_latencies_ms: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["load_memory"] = asdict(self.load_memory)
        result["inference_memory"] = asdict(self.inference_memory)
        result["inference_latency"] = asdict(self.inference_latency)
        result["inference_throughput"] = asdict(self.inference_throughput)
        if self.validation_quality:
            result["validation_quality"] = asdict(self.validation_quality)
        return result


def get_hardware_identity() -> dict[str, Any]:
    """Record the current GPU and CUDA state for measurement provenance.

    Delegates to ``src.hardware.profiler`` (the single source of truth for
    device introspection) and re-raises its error as ``BenchmarkError`` so
    existing callers of this module don't need to catch a second exception
    type.
    """
    from ..hardware.profiler import HardwareProfilerError

    try:
        return _get_hardware_identity()
    except HardwareProfilerError as exc:
        raise BenchmarkError(str(exc)) from exc


def measure_model_load(model_loader, device: str = "cuda") -> tuple[nn.Module, MemoryStats]:
    """Load a model and measure memory consumption."""
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise BenchmarkError("CUDA required but not available")

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device=device) / 1024**2

    model = model_loader()
    torch.cuda.synchronize()

    peak_allocated = torch.cuda.max_memory_allocated(device=device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device=device) / 1024**2

    return model, MemoryStats(
        baseline_mb=baseline_allocated,
        peak_allocated_mb=peak_allocated,
        peak_reserved_mb=peak_reserved,
        incremental_mb=peak_allocated - baseline_allocated,
    )


def measure_inference_batch(
    model: nn.Module,
    batch: torch.Tensor,
    device: str = "cuda",
) -> tuple[float, float]:
    """Run a single inference batch and return loss and token count."""
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise BenchmarkError("CUDA required but not available")

    model.eval()
    batch = batch.to(device)
    with torch.no_grad():
        output = model(batch, labels=batch)
    loss = output.loss.item()
    token_count = batch.numel()
    return loss, float(token_count)


def benchmark_inference(
    model: nn.Module,
    batches: list[torch.Tensor],
    device: str = "cuda",
    warmup_iterations: int = 2,
    benchmark_iterations: int = 5,
) -> tuple[LatencyStats, ThroughputStats, MemoryStats]:
    """Benchmark inference latency, throughput, and memory."""
    if not torch.cuda.is_available() and device.startswith("cuda"):
        raise BenchmarkError("CUDA required but not available")

    model.eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    baseline_allocated = torch.cuda.memory_allocated(device=device) / 1024**2

    for _ in range(warmup_iterations):
        batch = batches[0].to(device)
        with torch.no_grad():
            _ = model(batch)
        torch.cuda.synchronize()

    latencies_ms = []
    total_tokens = 0

    for _ in range(benchmark_iterations):
        for batch in batches:
            batch = batch.to(device)
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            with torch.no_grad():
                _ = model(batch)
            end_event.record()
            torch.cuda.synchronize()
            latencies_ms.append(start_event.elapsed_time(end_event))
            total_tokens += batch.numel()

    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(device=device) / 1024**2
    peak_reserved = torch.cuda.max_memory_reserved(device=device) / 1024**2

    latencies_ms.sort()
    mean_latency = sum(latencies_ms) / len(latencies_ms)
    if len(latencies_ms) > 1:
        variance = sum((x - mean_latency) ** 2 for x in latencies_ms) / len(latencies_ms)
        std_latency = variance ** 0.5
    else:
        std_latency = 0.0

    latency_stats = LatencyStats(
        mean_ms=mean_latency,
        std_ms=std_latency,
        min_ms=min(latencies_ms),
        max_ms=max(latencies_ms),
        p50_ms=latencies_ms[len(latencies_ms) // 2] if latencies_ms else None,
        p95_ms=latencies_ms[int(len(latencies_ms) * 0.95)] if len(latencies_ms) >= 20 else None,
        p99_ms=latencies_ms[int(len(latencies_ms) * 0.99)] if len(latencies_ms) >= 100 else None,
    )

    total_batch_time_sec = sum(latencies_ms) / 1000
    throughput_stats = ThroughputStats(
        tokens_per_sec=total_tokens / total_batch_time_sec if total_batch_time_sec > 0 else 0,
        batches_per_sec=len(latencies_ms) / total_batch_time_sec if total_batch_time_sec > 0 else 0,
    )

    memory_stats = MemoryStats(
        baseline_mb=baseline_allocated,
        peak_allocated_mb=peak_allocated,
        peak_reserved_mb=peak_reserved,
        incremental_mb=peak_allocated - baseline_allocated,
    )

    return latency_stats, throughput_stats, memory_stats


def percentile(data: list[float], p: float) -> float | None:
    """Compute p-th percentile; returns None if insufficient data."""
    if not data or p < 0 or p > 100:
        return None
    sorted_data = sorted(data)
    index = int((p / 100) * len(sorted_data))
    return sorted_data[min(index, len(sorted_data) - 1)]
