import abc
import random
from dataclasses import dataclass

from .search_space import Genome


@dataclass(order=True)
class FitnessMetrics:
    """Named minimization metrics used by the search evaluator."""
    mean_loss: float
    peak_memory_mb: float
    latency_ms: float
    source: str = "predicted"

    @property
    def validation_loss(self) -> float:
        """Compatibility alias for the legacy search field."""
        return self.mean_loss

    @property
    def vram_peak_mb(self) -> float:
        """Compatibility alias for the legacy search field."""
        return self.peak_memory_mb

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.mean_loss, self.peak_memory_mb, self.latency_ms)
    
    def __repr__(self):
        return (f"Fitness(Loss={self.mean_loss:.4f}, "
            f"Memory={self.peak_memory_mb:.1f}MB, "
                f"Lat={self.latency_ms:.2f}ms)")

class Evaluator(abc.ABC):
    """Base class for mapping a Genome (Architecture) to FitnessMetrics"""
    
    @abc.abstractmethod
    def evaluate(self, genome: Genome) -> FitnessMetrics:
        """Run the evaluation pipeline (inference/profiling) on the genome"""

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
            mean_loss=simulated_loss + random.uniform(-0.05, 0.05),
            peak_memory_mb=simulated_vram + random.uniform(-100, 100),
            latency_ms=simulated_latency + random.uniform(-2, 2)
        )