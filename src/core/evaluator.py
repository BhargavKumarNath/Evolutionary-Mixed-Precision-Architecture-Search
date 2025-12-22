import abc
import time
import random
from dataclasses import dataclass
from typing import Dict, Tuple
from .search_space import Genome

@dataclass(order=True)
class FitnessMetrics:
    """The objectuves we are optimising."""
    validation_loss: float      # Proxy for Perplecity (lower is better)
    vram_peak_mb: float     # Memory footprints (lower is better)
    latency_ms: float       # Inference speed (lower is better)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.validation_loss, self.vram_peak_mb, self.latency_ms)
    
    def __repr__(self):
        return (f"Fitness(Loss={self.validation_loss:.4f}, "
                f"VRAM={self.vram_peak_mb:.1f}MB, "
                f"Lat={self.latency_ms:.2f}ms)")

class Evaluator(abc.ABC):
    """Base class for mapping a Genome (Architecture) to FitnessMetrics"""
    
    @abc.abstractmethod
    def evaluate(self, genome: Genome) -> FitnessMetrics:
        """Run the evaluation pipeline (inference/profiling) on the genome"""
        pass

class DummyEvaluator(Evaluator):
    """Used for testing the Search Engine without loading real models.
    Simulates the trade-off: Lower bits -> Higher Loss, Lower VRAM"""
    def evaluate(self, genome: Genome) -> FitnessMetrics:
        # Simulate computational cost
        avg_bits = sum(genome.genes)/len(genome.genes)

        # Artificial correlation logic for simulation:
        # 1. Low bits = High Error (bad)
        # 2. Low bits = Low VRAM (good)
        # 3. Low bits = Low Latency (good)

        # Base error + penalty for lower precision
        simulated_loss = 2.5 + (16.0 / (avg_bits + 0.1)) * 0.1

        # Base VRAM + cost per bit
        # Llama-3-8B approx: 8B params. 
        # 4-bit ~= 4GB + overhead. 
        simulated_vram = 1024 + (8000 * (avg_bits / 8.0))

        # Base Latency (Memory bandwidth bound)
        simulated_latency = 50 + (avg_bits * 2)

        return FitnessMetrics(
            validation_loss=simulated_loss + random.uniform(-0.05, 0.05),
            vram_peak_mb=simulated_vram + random.uniform(-100, 100),
            latency_ms=simulated_latency + random.uniform(-2, 2)
        )