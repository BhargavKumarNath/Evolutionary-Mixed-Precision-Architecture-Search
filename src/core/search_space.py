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
        """Sample a valid bit-width vector (uniform, then repaired if the average cap is tight)."""
        return self.sample_with(random)

    def sample_with(self, rng) -> Genome:
        """As ``sample`` using ``rng`` (a ``random.Random`` or the ``random`` module).

        Tries rejection sampling first (exactly uniform over feasible genomes). When the
        feasible region is tiny (a tight average-bit cap) it falls back to uniform sampling
        followed by ``repair``.
        """
        for _ in range(200):
            genome = Genome(genes=[rng.choice(self.choices) for _ in range(self.num_layers)])
            if self.validate(genome):
                return genome
        genome = Genome(genes=[rng.choice(self.choices) for _ in range(self.num_layers)])
        return self.repair(genome, rng)

    def repair(self, genome: Genome, rng=random) -> Genome:
        """Move a genome into the feasible average-bit range by changing random layers one step."""
        ordered = sorted(self.choices)
        genes = list(genome.genes)
        guard = 0
        while guard < 100_000:
            guard += 1
            avg = sum(genes) / len(genes)
            if self.max_avg_bitwidth is not None and avg > self.max_avg_bitwidth:
                movable = [i for i, g in enumerate(genes) if ordered.index(g) > 0]
                step = -1
            elif self.min_avg_bitwidth is not None and avg < self.min_avg_bitwidth:
                movable = [i for i, g in enumerate(genes) if ordered.index(g) < len(ordered) - 1]
                step = 1
            else:
                return Genome(genes=genes)
            if not movable:
                break
            i = rng.choice(movable)
            genes[i] = ordered[ordered.index(genes[i]) + step]
        raise RuntimeError("Could not repair genome into the search-space constraints")

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
