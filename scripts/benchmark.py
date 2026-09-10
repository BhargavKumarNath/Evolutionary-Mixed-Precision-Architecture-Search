"""Hydra-based benchmark CLI.

Runs the Phase 6 design-of-experiments proxy calibration for real: measures
a diverse set of genomes on real GPU hardware via the bitsandbytes backend,
records proxy (predicted) vs. measured metrics, computes Pearson
correlations, and persists both a calibration report and per-design
measurement-cache entries.
"""

import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.benchmark import MeasurementCache, ProxyCalibration, get_hardware_identity
from src.core.configuration import validate_config
from src.core.proxy_evaluator import ProxyEvaluator
from src.core.search_space import Genome
from src.core.validator import Validator

logger = logging.getLogger(__name__)

# The same diverse initial design used by
# ``src.benchmark.proxy_validator.measure_initial_design_of_experiments``,
# inlined here so each design can be measured against real hardware with a
# fresh model load per genome.
DESIGNS = {
    "uniform_fp16": lambda n: [16] * n,
    "uniform_int8": lambda n: [8] * n,
    "uniform_int4": lambda n: [4] * n,
    "alternating_4_8": lambda n: [4 if i % 2 == 0 else 8 for i in range(n)],
    "alternating_8_16": lambda n: [8 if i % 2 == 0 else 16 for i in range(n)],
}


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def benchmark_cli(config: DictConfig) -> None:
    """Main benchmark CLI entry point."""
    resolved_cfg = validate_config(config)
    logger.info("Starting hardware benchmark + proxy calibration")
    logger.info("Model: %s", resolved_cfg["model_name"])
    logger.info("Backend: %s", resolved_cfg["quantization"]["backend"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking")

    hardware_identity = get_hardware_identity()
    logger.info("Hardware: %s", hardware_identity)

    cache_dir = Path(resolved_cfg.get("benchmark", {}).get("cache_dir", "./benchmark_cache"))
    cache = MeasurementCache(cache_dir)

    num_layers = resolved_cfg["search_space"]["num_layers"]
    evaluator = ProxyEvaluator(resolved_cfg["evaluator"]["profile_path"])
    validation_protocol = resolved_cfg["data"]["protocol"]["validation"]

    calibration = ProxyCalibration(model_version="v1")

    for design_name, genome_fn in DESIGNS.items():
        genome_genes = genome_fn(num_layers)
        genome = Genome(genes=genome_genes)
        logger.info("=== Measuring design: %s -> %s ===", design_name, genome_genes)

        # Proxy: cheap additive-sensitivity prediction. Reported as a
        # quality-loss *delta* over the FP16 baseline (matching the
        # semantics ProxyCalibration expects), not the absolute loss.
        predicted = evaluator.evaluate(genome)
        predicted_quality_loss = max(0.0, predicted.mean_loss - evaluator.baseline_loss)

        # Measured: real GPU execution via the configured backend.
        validator = Validator(
            resolved_cfg["model_name"],
            device=resolved_cfg["target_device"],
            seq_len=validation_protocol["seq_len"],
            nsamples=validation_protocol["nsamples"],
            seed=validation_protocol["seed"],
            dataset_split=validation_protocol["split"],
            allow_synthetic_data=resolved_cfg["data"]["allow_synthetic_data"],
            quantization_backend=resolved_cfg["quantization"]["backend"],
        )
        validator.apply_genome(genome)

        torch.cuda.reset_peak_memory_stats()
        result = validator.validate()
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024**2

        logger.info(
            "%s -> predicted_delta=%.4f measured_loss=%.4f perplexity=%.4f "
            "latency=%.2fms peak_mem=%.0fMB",
            design_name,
            predicted_quality_loss,
            result.mean_loss,
            result.perplexity,
            result.latency_ms,
            peak_memory_mb,
        )

        calibration.add_point(
            genome=genome_genes,
            predicted_quality_loss=predicted_quality_loss,
            measured_loss=result.mean_loss,
            measured_perplexity=result.perplexity,
            measured_memory_mb=peak_memory_mb,
            measured_latency_ms=result.latency_ms,
            measured_throughput=result.throughput_tokens_per_sec,
            hardware_identity=hardware_identity,
        )

        cache_key = cache.get_key(
            genome=genome_genes,
            backend=resolved_cfg["quantization"]["backend"],
            hardware_identity=hardware_identity,
            batch_size=1,
            sequence_length=validation_protocol["seq_len"],
        )
        cache.store(
            cache_key,
            {
                "design_name": design_name,
                "genome": genome_genes,
                "predicted_quality_loss": predicted_quality_loss,
                "measured_mean_loss": result.mean_loss,
                "measured_perplexity": result.perplexity,
                "measured_latency_ms": result.latency_ms,
                "measured_peak_memory_mb": peak_memory_mb,
                "measured_throughput_tokens_per_sec": result.throughput_tokens_per_sec,
                "hardware_identity": hardware_identity,
            },
        )

        # Free the model before the next design loads a fresh one.
        del validator
        torch.cuda.empty_cache()

    correlation = calibration.compute_correlation()
    logger.info("=== Proxy calibration correlation report ===")
    for key, value in correlation.items():
        logger.info("  %s: %s", key, value)

    report_path = Path(resolved_cfg["paths"]["calibration_report_path"])
    calibration.save_report(report_path)
    logger.info("Saved calibration report to %s", report_path)
    logger.info("Benchmark + calibration complete")


if __name__ == "__main__":
    benchmark_cli()
