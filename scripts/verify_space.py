import hydra
from omegaconf import DictConfig
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.search_space import MixedPrecisionSearchSpace
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print(f"=== EMPAS: Initialising Search Space: {cfg.search_space.name} ===")

    space = MixedPrecisionSearchSpace(
        num_layers=cfg.search_space.num_layers,
        choices=cfg.search_space.choices
    )

    print(f"Configuration loaded")
    print(f"Layers: {space.num_layers}")
    print(f"Bit-width choices: {space.choices}")

    # Test Sampling
    print("\n--- Test 1: Random Sampling ---")
    genome = space.sample()
    print(f"Sampled Genome: {genome}")
    print(f"Genes (first 5): {genome.genes[:5]}...")
    
    # Test Decoding
    print("\n --- Test 2: Decoding to Phenotype ---")
    config_map = space.decode(genome)
    print(f"Layer 0 Config: {config_map['layer_0']}-bit")
    print(f"Layer 10 Config: {config_map['layer_10']}-bit")

    # Test Validation
    print("\n --- Test 3: Validation ---")
    is_valid = space.validate(genome)
    print(f"Is genome valid? {is_valid}")

    print("\nSUCCESS: Search Space infrastructure is ready")


if __name__ == "__main__":
    main()