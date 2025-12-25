import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.search_space import MixedPrecisionSearchSpace
from src.core.proxy_evaluator import ProxyEvaluator
from src.engine.ga import GAEngine

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("=== EMPAS: Hardware-Aware Evolutionary Search ===")

    # 1. Setup Search Space
    logger.info(f"Initializing Search Space with {cfg.search_space.num_layers} layers.")
    space = MixedPrecisionSearchSpace(
        num_layers=cfg.search_space.num_layers,
        choices=cfg.search_space.choices
    )

    # 2. Setup Evaluator
    if cfg.evaluator.type == "proxy":
        logger.info(f"Loading Proxy Profile from: {cfg.evaluator.profile_path}")
        evaluator = ProxyEvaluator(profile_path=cfg.evaluator.profile_path)
    else:
        raise ValueError("Please set evaluator.type='proxy' in config")
        
    # 3. Setup Engine
    algo_config = OmegaConf.to_container(cfg.algorithm, resolve=True)
    
    # increase generations for the real run
    algo_config['n_generations'] = 20
    algo_config['pop_size'] = 50
    
    engine = GAEngine(
        search_space=space,
        evaluator=evaluator,
        config=algo_config,
        output_dir="./data/logs/search_tinyllama"
    )
    
    # 4. Run Search
    engine.run()
    
    # 5. Show Best Results
    logger.info("--- Pareto Frontier ---")
    best_loss = sorted(engine.population, key=lambda g: engine.evaluate_population([g])[0].validation_loss)[0]
    best_vram = sorted(engine.population, key=lambda g: engine.evaluate_population([g])[0].vram_peak_mb)[0]
    
    fit_loss = engine.evaluate_population([best_loss])[0]
    fit_vram = engine.evaluate_population([best_vram])[0]
    
    logger.info(f"Best Accuracy Model: {fit_loss}")
    logger.info(f"Genome: {best_loss.genes}")
    
    logger.info(f"Lowest VRAM Model: {fit_vram}")
    logger.info(f"Genome: {best_vram.genes}")

if __name__ == "__main__":
    main()