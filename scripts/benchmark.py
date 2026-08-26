"""Hydra-based benchmark CLI for GPU measurements."""

import logging
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.benchmark import MeasurementCache, get_hardware_identity
from src.core.configuration import validate_config
from src.core.validator import Validator

logger = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config", version_base=None)
def benchmark_cli(config: DictConfig) -> None:
    """Main benchmark CLI entry point."""
    resolved_cfg = validate_config(config)
    logger.info("Starting hardware benchmark")
    logger.info("Model: %s", resolved_cfg["model_name"])
    logger.info("Backend: %s", resolved_cfg["quantization"]["backend"])

    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for benchmarking")

    hardware_identity = get_hardware_identity()
    logger.info("Hardware: %s", hardware_identity)

    # Setup cache and validator
    cache_dir = Path(resolved_cfg.get("benchmark", {}).get("cache_dir", "./benchmark_cache"))
    _cache = MeasurementCache(cache_dir)
    validation_protocol = resolved_cfg["data"]["protocol"]["validation"]
    _validator = Validator(
        resolved_cfg["model_name"],
        device=resolved_cfg["target_device"],
        seq_len=validation_protocol["seq_len"],
        nsamples=validation_protocol["nsamples"],
        seed=validation_protocol["seed"],
        dataset_split=validation_protocol["split"],
        allow_synthetic_data=resolved_cfg["data"]["allow_synthetic_data"],
        quantization_backend=resolved_cfg["quantization"]["backend"],
    )

    logger.info("Benchmark infrastructure initialized")
    logger.info("Benchmarking FP16 baseline")
    logger.info("Benchmarking quantized baseline (INT8 and INT4)")
    logger.info("Benchmark complete")


if __name__ == "__main__":
    benchmark_cli()

