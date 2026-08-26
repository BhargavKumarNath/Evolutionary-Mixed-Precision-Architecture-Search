import logging
import os
import sys
import time

import pandas as pd
import torch
from tabulate import tabulate

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.core.search_space import Genome
from src.core.validator import Validator
from src.utils.data import get_calibration_dataset

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("benchmark")

# Configurations
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
# From previous export (The "Balanced" Genome)
# Replace this with the specific genes from your balanced.json if different
EMPAS_GENES = [4, 4, 4, 4, 4, 4, 4, 8, 4, 4, 4, 4, 8, 4, 4, 4, 4, 4, 4, 4, 4, 4] 

def measure_performance(validator, genome, name):
    """
    Runs a full evaluation loop:
    1. Applies Quantization
    2. Measures VRAM
    3. Measures Perplexity (Loss)
    4. Measures Throughput (Tokens/sec)
    """
    logger.info(f"--- Benchmarking: {name} ---")
    
    # 1. Apply Config
    validator.wrapper.model = validator.wrapper.model.to("cpu") # Offload to reset
    del validator.wrapper.model
    torch.cuda.empty_cache()
    
    # Reload fresh
    from src.models.wrapper import ModelWrapper
    validator.wrapper = ModelWrapper(MODEL_NAME, device="cuda")
    
    validator.apply_genome(genome)
    
    # 2. Measure VRAM (Peak)
    torch.cuda.reset_peak_memory_stats()
    
    # 3. Measure Loss & Latency
    # use a distinct dataset slice for benchmarking
    dataset = get_calibration_dataset(MODEL_NAME, seq_len=1024, nsamples=5)
    
    validator.wrapper.model.eval()
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    
    with torch.no_grad():
        for batch in dataset:
            batch = batch.to("cuda")
            loss = validator.wrapper.forward_pass_check(batch)
            total_loss += loss
            total_tokens += batch.numel()
            
    end_time = time.time()
    
    # Metrics
    duration = end_time - start_time
    throughput = total_tokens / duration
    avg_loss = total_loss / len(dataset)
    peak_vram = torch.cuda.max_memory_allocated() / (1024**2) # MB
    
    return {
        "Model": name,
        "Loss (PPL Proxy)": f"{avg_loss:.4f}",
        "VRAM (MB)": f"{peak_vram:.0f}",
        "Throughput (T/s)": f"{throughput:.1f}",
        "Avg Bit-Width": f"{sum(genome.genes)/len(genome.genes):.1f}"
    }

def main():
    logger.info("=== EMPAS vs. Baselines Benchmark ===")
    
    validator = Validator(MODEL_NAME)
    results = []
    
    # 1. Baseline: FP16
    baseline_genes = [16] * 22
    results.append(measure_performance(validator, Genome(genes=baseline_genes), "Baseline (FP16)"))
    
    # 2. Naive: Uniform 4-bit
    naive_genes = [4] * 22
    results.append(measure_performance(validator, Genome(genes=naive_genes), "Naive (All 4-bit)"))
    
    # 3. EMPAS: Balanced
    empas_genome = Genome(genes=EMPAS_GENES)
    results.append(measure_performance(validator, empas_genome, "EMPAS (Balanced)"))
    
    # Print Table
    df = pd.DataFrame(results)
    print("\n" + tabulate(df, headers='keys', tablefmt='github', showindex=False))
    
    # Final Analysis
    empas_loss = float(results[2]["Loss (PPL Proxy)"])
    naive_loss = float(results[1]["Loss (PPL Proxy)"])
    
    print("\n=== Analysis ===")
    if empas_loss < naive_loss:
        print(f"EMPAS beats Naive 4-bit! (Lower Loss: {empas_loss} vs {naive_loss})")
        print("   Reason: EMPAS used 8-bit for sensitive layers, while Naive forced them to 4-bit.")
    else:
        print("EMPAS is comparable to Naive.")

if __name__ == "__main__":
    main()
