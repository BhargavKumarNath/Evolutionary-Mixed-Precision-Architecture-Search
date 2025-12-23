import hydra
from omegaconf import DictConfig, OmegaConf
import sys
import os
import logging
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.search_space import MixedPrecisionSearchSpace
from src.core.evaluator import DummyEvaluator
from src.engine.ga import GAEngine

# Setup basic logging to stdout
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

@hydra.main(version_base=None, config_path="../conf", config_name="config")

def main(cfg: DictConfig):
    print("=== EMPAS: Evolutionary Search (Dummy Simulation) ===")
    
    # 1. initialise Search Space
    space = MixedPrecisionSearchSpace(
        num_layers=cfg.search_space.num_layers,
        choices=cfg.search_space.choices
    )

    # 2. Initialise Evaluator
    if cfg.evaluator.type == "dummy":
        evaluator = DummyEvaluator()
    else:
        raise NotImplementedError("Only dummy supported for this script")
    
    # 3. Initialise Engine
    algo_config = OmegaConf.to_container(cfg.algorithm, resolve=True)

    engine = GAEngine(
        search_space=space,
        evaluator=evaluator,
        config=algo_config,
        output_dir="./data/logs/run_001"
    )

    # 4. Run 
    engine.run()

if __name__ == "__main__":
    main()
    
