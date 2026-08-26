import os
import sys

import hydra
import torch
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.models.wrapper import ModelWrapper
from src.utils.data import get_calibration_dataset

# Override config to use TinyLlama for development
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    print("=== EMPAS: Hardware & Data Verification ===")
    print(f"Target GPU: {torch.cuda.get_device_name(0)}")

    # 1. Load Data
    print("\n--- Step 1: Loading Calibration Data ---")
    dataset = get_calibration_dataset(MODEL_ID, seq_len=128, nsamples=4)
    print(f"Loaded {len(dataset)} samples. Shape: {dataset[0].shape}")

    # 2. Load Model
    print("\n--- Step 2: Loading Model to GPU ---")
    wrapper = ModelWrapper(MODEL_ID, device="cuda")

    mem_usage = wrapper.get_memory_footprint()
    print(f"Model loaded. VRAM Footprint: {mem_usage:.2f} MB")

    num_layers = wrapper.get_layer_count()
    print(f"Detected {num_layers} layers.")

    # 3. Test Inference
    print("\n--- Step 3: Running Inference Check---")
    loss = wrapper.forward_pass_check(dataset[0])
    print(f"Forward pass successful. Loss {loss:.4f}")

    print("\nSUCCESS: Data and Model Infrastructure Ready!")

if __name__ == "__main__":
    main()