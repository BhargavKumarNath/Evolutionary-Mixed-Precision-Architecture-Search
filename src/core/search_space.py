import abc
import random
from typing import List, Dict, Any, Union
import numpy as np
from pydantic import BaseModel, Field

class Genome(BaseModel):
    """Represents a single candidate in the population.
    genes: List of integers representing bit-widths per layer"""
    genes: List[int]

    def __repr__(self):
        return f"Genome(len={len(self.genes)}, avg_bit={np.mean(self.genes):.2f})"

class SearchSpace(abc.ABC):
    """Abstract base class for any architecture search space"""

    @abc.abstractmethod
    def sample(self) -> Genome:
        """Randomly sample a genome from the search space"""
        pass

    @abc.abstractmethod
    def validate(self, genome: Genome) -> bool:
        """CHeck if a genome is valid within constraints"""
        pass

    @abc.abstractmethod
    def decode(self, genome: Genome) -> Dict[str, Any]:
        """Convert Genotype (genes) to Phenotype (Model Config)
        Returns a dict mapping layer indices to configurations"""
        pass

class MixedPrecisionSearchSpace(SearchSpace):
    """Search space where each layer can have a different bit-width"""
    def __init__(self, num_layers: int, choices: List[int]):
        self.num_layers = num_layers
        self.choices = choices
    
    def sample(self) -> Genome:
        """Uniformly sample bit-widths for each layer"""
        genes = [random.choice(self.choices) for _ in range(self.num_layers)]
        return Genome(genes=genes)

    def validate(self, genome: Genome) -> bool:
        if len(genome.genes) != self.num_layers:
            return False
        return all(g in self.choices for g in genome.genes)
    
    def decode(self, genome: Genome) -> Dict[str, int]:
        """Returns: { "layer_0": 4, "layer_1": 2, ... }"""
        return {f"layer_{i}": bitwidth for i, bitwidth in enumerate(genome.genes)}
