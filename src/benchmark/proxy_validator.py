"""Proxy validation and measurement integration for search."""

import logging
from pathlib import Path
from typing import Any

import torch

from src.core.evaluator import FitnessMetrics

from .cache import MeasurementCache
from .proxy_calibration import ProxyCalibration

logger = logging.getLogger(__name__)


def compute_additive_quality_loss(genome: list[int], base_model_name: str) -> float:
    """Placeholder for additive sensitivity-based quality loss prediction.

    In a real implementation, this would compute the sum of per-layer sensitivity
    impacts for the given precision configuration. For now, it returns a proxy
    based on average bitwidth deviation from FP16.
    """
    num_layers = len(genome)
    avg_bitwidth = sum(genome) / num_layers if num_layers > 0 else 16
    # Assume FP16 (16-bit) is baseline
    deviation = (16 - avg_bitwidth) / 16
    # Rough proxy: lower bitwidth → more quality loss
    return max(0.0, deviation * 0.5)


class ProxyValidator:
    """Validate and calibrate proxy metrics against measured hardware behavior."""

    def __init__(
        self,
        cache_dir: str | Path,
        calibration_report_path: str | Path | None = None,
    ):
        self.cache = MeasurementCache(cache_dir)
        self.calibration = ProxyCalibration()
        self.calibration_report_path = calibration_report_path

        if calibration_report_path:
            self.calibration.load_report(calibration_report_path)

    def predict_quality_loss(
        self,
        genome: list[int],
        base_model_name: str,
        reference_loss: float,
    ) -> float:
        """Predict quality loss using additive sensitivity."""
        predicted_loss = compute_additive_quality_loss(genome, base_model_name)
        return predicted_loss

    def measure_candidate(
        self,
        genome: list[int],
        measured_fitness: FitnessMetrics,
        predicted_quality_loss: float,
        hardware_identity: dict[str, Any],
    ) -> None:
        """Record a candidate's proxy and measured metrics."""
        self.calibration.add_point(
            genome=genome,
            predicted_quality_loss=predicted_quality_loss,
            measured_loss=measured_fitness.mean_loss,
            measured_perplexity=0.0,  # Would compute from loss
            measured_memory_mb=measured_fitness.peak_memory_mb,
            measured_latency_ms=measured_fitness.latency_ms,
            measured_throughput=0.0,  # Would compute from throughput
            hardware_identity=hardware_identity,
        )

    def validate_proxy(self) -> dict[str, Any]:
        """Compute correlation report between proxy and measured metrics."""
        return self.calibration.compute_correlation()

    def save_calibration(self) -> None:
        """Save calibration report to disk."""
        if self.calibration_report_path:
            self.calibration.save_report(self.calibration_report_path)
            logger.info("Saved proxy calibration to %s", self.calibration_report_path)


def measure_initial_design_of_experiments(
    validator_func,
    num_layers: int,
    hardware_identity: dict[str, Any],
) -> dict[str, Any]:
    """Measure a diverse set of genomes for proxy calibration."""
    designs = {
        "uniform_fp16": [16] * num_layers,
        "uniform_int8": [8] * num_layers,
        "uniform_int4": [4] * num_layers,
        "alternating_4_8": [4 if i % 2 == 0 else 8 for i in range(num_layers)],
        "alternating_8_16": [8 if i % 2 == 0 else 16 for i in range(num_layers)],
    }

    results = {}
    for design_name, genome in designs.items():
        try:
            logger.info("Measuring %s: %s", design_name, genome)
            fitness = validator_func(genome)
            results[design_name] = {
                "genome": genome,
                "fitness": {
                    "mean_loss": fitness.mean_loss,
                    "peak_memory_mb": fitness.peak_memory_mb,
                    "latency_ms": fitness.latency_ms,
                },
            }
        except (RuntimeError, ValueError, torch.cuda.OutOfMemoryError) as e:
            logger.warning("Failed to measure %s: %s", design_name, e)
            results[design_name] = {"error": str(e)}

    return results
