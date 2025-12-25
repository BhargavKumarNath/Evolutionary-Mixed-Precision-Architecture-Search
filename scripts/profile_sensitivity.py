import sys
import os
import torch
import json
import hydra
import logging
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.data import get_calibration_dataset
from src.models.wrapper import ModelWrapper
from src.models.sensitivity import SensitivityProfiler

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

# Using TinyLlama for fast profiling validation
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("=== EMPAS: Sensitivity Profiling (Enriched) ===")
    
    dataset = get_calibration_dataset(MODEL_ID, seq_len=128, nsamples=4)
    wrapper = ModelWrapper(MODEL_ID, device="cuda")
    
    profiler = SensitivityProfiler(wrapper, dataset)
    # Unpack new return values
    sensitivity_table, layer_counts, baseline = profiler.profile()
    
    # Save Results
    output = {
        "model_name": MODEL_ID,
        "baseline_loss": baseline,
        # Convert integer keys to strings for JSON
        "layer_params": {str(k): v for k, v in layer_counts.items()},
        "sensitivity": {f"{k[0]}_{k[1]}": v for k, v in sensitivity_table.items()}
    }
    
    os.makedirs("./data/profiles", exist_ok=True)
    save_path = "./data/profiles/tinyllama_sensitivity.json"
    
    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)
        
    logger.info(f"Enriched profile saved to {save_path}")
    
    # Sanity Check
    l0_params = output["layer_params"]["0"]
    logger.info(f"Layer 0 Parameters: {l0_params:,}")
    logger.info("SUCCESS: Parameter counts captured.")

if __name__ == "__main__":
    main()