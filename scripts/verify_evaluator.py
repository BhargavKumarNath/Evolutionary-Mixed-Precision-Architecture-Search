import hydra
from omegaconf import DictConfig
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.search_space import MixedPrecisionSearchSpace
from src.core.evaluator import DummyEvaluator

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"=== EMPAS: Testing Evaluator ({cfg.evaluator.type}) ===")

    # 1. Setup Search Space
    space = MixedPrecisionSearchSpace(
        num_layers=cfg.search_space.num_layers,
        choices=cfg.search_space.choices
    )

    # 2. Setup Evaluator
    if cfg.evaluator.type == "dummy":
        evaluator = DummyEvaluator()
    else:
        raise ValueError(f"Unknown evaluator type: {cfg.evaluator.type}")
    
    # 3. Sample and Evaluate
    print("\n--- Simulating 5 Evaluations ---")
    for i in range(5):
        genome = space.sample()
        fitness = evaluator.evaluate(genome)

        avg_bits = sum(genome.genes) / len(genome.genes)
        print(f"Arch {i+1} [Avg Bits: {avg_bits:.2f}]: {fitness}")
    
    print("\nSUCCESS: Evaluator interface is ready")

if __name__ == "__main__":
    main()
    