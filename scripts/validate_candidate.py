import logging
import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.configuration import validate_config
from src.core.search_space import Genome
from src.core.validator import Validator
from src.utils.reproducibility import seed_everything

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    resolved_cfg = validate_config(cfg)
    seed_everything(resolved_cfg["seed"], resolved_cfg.get("deterministic", False))
    logger.info("=== EMPAS: Candidate Validation (simulation path) ===")

    # 1. Setup Validator (Loads Model)
    validator = Validator(
        resolved_cfg["model_name"],
        device=resolved_cfg["target_device"],
        seq_len=resolved_cfg["data"]["protocol"]["validation"]["seq_len"],
        nsamples=resolved_cfg["data"]["protocol"]["validation"]["nsamples"],
        seed=resolved_cfg["data"]["protocol"]["validation"]["seed"],
        dataset_split=resolved_cfg["data"]["protocol"]["validation"]["split"],
        allow_synthetic_data=resolved_cfg["data"]["allow_synthetic_data"],
        quantization_backend=resolved_cfg["quantization"]["backend"],
    )

    # 2. Validator Baseline
    logger.info("\n--- Validating FP16 Baseline ---")
    baseline_genome = [16] * resolved_cfg["search_space"]["num_layers"]
    validator.apply_genome(Genome(genes=baseline_genome))
    baseline_result = validator.validate()
    logger.info(
        "Baseline -> Mean loss: %.4f, Perplexity: %.4f, Latency: %.2fms",
        baseline_result.mean_loss,
        baseline_result.perplexity,
        baseline_result.latency_ms,
    )

    # 3. Reload Model
    del validator
    validator = Validator(
        resolved_cfg["model_name"],
        device=resolved_cfg["target_device"],
        seq_len=resolved_cfg["data"]["protocol"]["validation"]["seq_len"],
        nsamples=resolved_cfg["data"]["protocol"]["validation"]["nsamples"],
        seed=resolved_cfg["data"]["protocol"]["validation"]["seed"],
        dataset_split=resolved_cfg["data"]["protocol"]["validation"]["split"],
        allow_synthetic_data=resolved_cfg["data"]["allow_synthetic_data"],
        quantization_backend=resolved_cfg["quantization"]["backend"],
    )

    # 4. Validate Found Solution
    logger.info("\n--- Validating Evolved Architecture ---")
    candidate_genome = Genome(genes=resolved_cfg["validation"]["genome"])
    validator.apply_genome(candidate_genome)
    evolved_result = validator.validate()
    logger.info(
        "Evolved -> Mean loss: %.4f, Perplexity: %.4f, Latency: %.2fms",
        evolved_result.mean_loss,
        evolved_result.perplexity,
        evolved_result.latency_ms,
    )

    # 5. Analysis
    logger.info("\n=== RESULTS ===")
    logger.info(
        "Mean loss delta: %.4f",
        evolved_result.mean_loss - baseline_result.mean_loss,
    )

    # Calculate Theoretical Compression
    total_bits = sum(candidate_genome.genes)
    max_bits = 16 * len(candidate_genome.genes)
    compression = 1.0 - (total_bits / max_bits)
    logger.info(f"Model Compression: {compression*100:.1f}%")

if __name__ == "__main__":
    main()