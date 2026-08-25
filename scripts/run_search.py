import logging
import os
import sys

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.configuration import validate_config
from src.core.proxy_evaluator import ProxyEvaluator
from src.core.search_space import MixedPrecisionSearchSpace
from src.engine.ga import GAEngine
from src.utils.manifest import build_manifest, write_manifest
from src.utils.reproducibility import seed_everything

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("=== EMPAS: Hardware-Aware Evolutionary Search ===")
    resolved_cfg = validate_config(cfg)
    seed_everything(resolved_cfg["seed"], resolved_cfg.get("deterministic", False))

    # 1. Setup Search Space
    logger.info(f"Initializing Search Space with {cfg.search_space.num_layers} layers.")
    space = MixedPrecisionSearchSpace(
        num_layers=cfg.search_space.num_layers,
        choices=cfg.search_space.choices,
        min_avg_bitwidth=cfg.search_space.get("min_avg_bitwidth"),
        max_avg_bitwidth=cfg.search_space.get("max_avg_bitwidth"),
    )

    # 2. Setup Evaluator
    if cfg.evaluator.type == "proxy":
        logger.info(f"Loading Proxy Profile from: {resolved_cfg['evaluator']['profile_path']}")
        evaluator = ProxyEvaluator(profile_path=resolved_cfg["evaluator"]["profile_path"])
    else:
        raise ValueError("Please set evaluator.type='proxy' in config")
        
    # Setup Configs
    algo_config = OmegaConf.to_container(cfg.algorithm, resolve=True)
    tracking_config = OmegaConf.to_container(cfg.tracking, resolve=True)
    
    algo_config["profile_path"] = resolved_cfg["evaluator"]["profile_path"]

    engine = GAEngine(
        search_space=space,
        evaluator=evaluator,
        config=algo_config,
        tracking_config=tracking_config, # Pass tracking config
        output_dir=resolved_cfg["paths"]["search_output_dir"]
    )

    manifest = build_manifest(resolved_cfg, {"profile_path": resolved_cfg["evaluator"]["profile_path"]})
    write_manifest(resolved_cfg["paths"]["manifest_path"], manifest)
    engine.run(resume_from=resolved_cfg["paths"].get("resume_from"))
    
    # 5. Show Best Results
    logger.info("--- Pareto Frontier ---")
    best_loss = min(
        engine.population, key=lambda g: engine.evaluate_population([g])[0].validation_loss
    )
    best_vram = min(
        engine.population, key=lambda g: engine.evaluate_population([g])[0].vram_peak_mb
    )
    
    fit_loss = engine.evaluate_population([best_loss])[0]
    fit_vram = engine.evaluate_population([best_vram])[0]
    
    logger.info(f"Best Accuracy Model: {fit_loss}")
    logger.info(f"Genome: {best_loss.genes}")
    
    logger.info(f"Lowest VRAM Model: {fit_vram}")
    logger.info(f"Genome: {best_vram.genes}")

if __name__ == "__main__":
    main()