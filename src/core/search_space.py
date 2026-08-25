import abc
import random
from typing import Any

import numpy as np
from pydantic import BaseModel


class Genome(BaseModel):
    """Represents a single candidate in the population.
    genes: List of integers representing bit-widths per layer"""
    genes: list[int]

    def __repr__(self):
        return f"Genome(len={len(self.genes)}, avg_bit={np.mean(self.genes):.2f})"

class SearchSpace(abc.ABC):
    """Abstract base class for any architecture search space"""

    @abc.abstractmethod
    def sample(self) -> Genome:
        """Randomly sample a genome from the search space"""

    @abc.abstractmethod
    def validate(self, genome: Genome) -> bool:
        """CHeck if a genome is valid within constraints"""

    @abc.abstractmethod
    def decode(self, genome: Genome) -> dict[str, Any]:
        """Convert Genotype (genes) to Phenotype (Model Config)
        Returns a dict mapping layer indices to configurations"""

class MixedPrecisionSearchSpace(SearchSpace):
    """Search space where each layer can have a different bit-width"""
    def __init__(
        self,
        num_layers: int,
        choices: list[int],
        min_avg_bitwidth: float | None = None,
        max_avg_bitwidth: float | None = None,
    ):
        if num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if not choices or any(choice <= 0 for choice in choices):
            raise ValueError("choices must contain positive bit widths")
        if len(set(choices)) != len(choices):
            raise ValueError("choices must be unique")
        if (
            min_avg_bitwidth is not None
            and max_avg_bitwidth is not None
            and min_avg_bitwidth > max_avg_bitwidth
        ):
            raise ValueError("min_avg_bitwidth cannot exceed max_avg_bitwidth")
        self.num_layers = num_layers
        self.choices = list(choices)
        self.min_avg_bitwidth = min_avg_bitwidth
        self.max_avg_bitwidth = max_avg_bitwidth
    
    def sample(self) -> Genome:
        """Uniformly sample a valid bit-width vector."""
        for _ in range(10000):
            genes = [random.choice(self.choices) for _ in range(self.num_layers)]
            genome = Genome(genes=genes)
            if self.validate(genome):
                return genome
        raise RuntimeError("Could not sample a genome satisfying search-space constraints")

    def validate(self, genome: Genome) -> bool:
        if len(genome.genes) != self.num_layers:
            return False
        if not all(g in self.choices for g in genome.genes):
            return False
        average = sum(genome.genes) / len(genome.genes)
        if self.min_avg_bitwidth is not None and average < self.min_avg_bitwidth:
            return False
        return not (
            self.max_avg_bitwidth is not None and average > self.max_avg_bitwidth
        )
    
    def decode(self, genome: Genome) -> dict[str, int]:
        """Returns: { "layer_0": 4, "layer_1": 2, ... }"""
        return {f"layer_{i}": bitwidth for i, bitwidth in enumerate(genome.genes)}
