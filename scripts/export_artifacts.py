import json
import logging
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.evaluator import FitnessMetrics
from src.core.proxy_evaluator import ProxyEvaluator
from src.core.search_space import Genome
from src.engine.pareto import get_pareto_front

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

CHECKPOINT_PATH = "./data/logs/search_tinyllama_wandb/checkpoint_gen_50.json"
EXPORT_DIR = "./deployment/artifacts"

def load_checkpoint(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)

def find_knee_point(fitnesses: list[FitnessMetrics], pareto_indices: list[int]) -> int:
    """
    Finds the 'Balanced' solution.
    Method: Normalize Loss and VRAM to 0-1 scale, find point closest to origin (0,0).
    """
    pareto_fits = [fitnesses[i] for i in pareto_indices]
    
    losses = np.array([f.validation_loss for f in pareto_fits])
    vrams = np.array([f.vram_peak_mb for f in pareto_fits])
    
    # Min-Max Normalization
    norm_loss = (losses - losses.min()) / (losses.max() - losses.min() + 1e-6)
    norm_vram = (vrams - vrams.min()) / (vrams.max() - vrams.min() + 1e-6)
    
    # Distance to origin
    distances = np.sqrt(norm_loss**2 + norm_vram**2)
    best_idx_in_pareto = np.argmin(distances)
    
    return pareto_indices[best_idx_in_pareto]

def main():
    logger.info("=== EMPAS: Exporting Artifacts ===")
    
    # 1. Load Data
    data = load_checkpoint(CHECKPOINT_PATH)
    logger.info(f"Loaded Generation {data['generation']}")    
    
    # Load Proxy
    profile_path = "./data/profiles/tinyllama_sensitivity.json"
    evaluator = ProxyEvaluator(profile_path)
    
    population = [Genome(genes=g) for g in data['population']]
    fitnesses = [evaluator.evaluate(g) for g in population]
    
    # 2. Identify Pareto Front
    pareto_indices = get_pareto_front(fitnesses)
    logger.info(f"Found {len(pareto_indices)} solutions on the Pareto Frontier.")
    
    # 3. Select Archetypes
    # A. Max Accuracy (Lowest Loss)
    # Filter only pareto solutions
    pareto_loss_idx = min(pareto_indices, key=lambda i: fitnesses[i].validation_loss)
    
    # B. Max Compression (Lowest VRAM)
    pareto_vram_idx = min(pareto_indices, key=lambda i: fitnesses[i].vram_peak_mb)
    
    # C. Balanced (Knee Point)
    pareto_balanced_idx = find_knee_point(fitnesses, pareto_indices)
    
    selection = {
        "max_accuracy": pareto_loss_idx,
        "max_compression": pareto_vram_idx,
        "balanced": pareto_balanced_idx
    }
    
    # 4. Export
    os.makedirs(EXPORT_DIR, exist_ok=True)
    
    for name, idx in selection.items():
        genome = population[idx]
        fit = fitnesses[idx]
        
        artifact = {
            "archetype": name,
            "metrics": {
                "predicted_loss": fit.validation_loss,
                "predicted_vram_mb": fit.vram_peak_mb,
                "predicted_latency_score": fit.latency_ms
            },
            "config": {
                "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "quantization_map": {
                    f"layer_{i}": bit for i, bit in enumerate(genome.genes)
                }
            }
        }
        
        filename = os.path.join(EXPORT_DIR, f"{name}.json")
        with open(filename, 'w') as f:
            json.dump(artifact, f, indent=4)
            
        logger.info(f"Exported [{name}]: Loss={fit.validation_loss:.4f}, VRAM={fit.vram_peak_mb:.0f}MB -> {filename}")

if __name__ == "__main__":
    main()
