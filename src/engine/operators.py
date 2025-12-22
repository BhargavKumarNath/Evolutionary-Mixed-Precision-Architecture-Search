import random
from typing import List, Tuple
from ..core.search_space import Genome, MixedPrecisionSearchSpace

def crossover_uniform(parent1: Genome, parent2: Genome) -> Tuple[Genome, Genome]:
    """
    Uniform Crossover: Each gene (layer) is randomly selected from either parent.
    Returns two children.
    """
    genes1 = []
    genes2 = []
    
    assert len(parent1.genes) == len(parent2.genes), "Parents must have same length"
    
    for g1, g2 in zip(parent1.genes, parent2.genes):
        # 50% chance to swap
        if random.random() < 0.5:
            genes1.append(g1)
            genes2.append(g2)
        else:
            genes1.append(g2)
            genes2.append(g1)
            
    return Genome(genes=genes1), Genome(genes=genes2)

def mutate_random_gene(genome: Genome, search_space: MixedPrecisionSearchSpace, mutation_rate: float = 0.1) -> Genome:
    """
    With probability `mutation_rate`, change a gene to a random valid choice.
    Returns a NEW genome (does not modify in place).
    """
    new_genes = list(genome.genes) # Copy
    
    for i in range(len(new_genes)):
        if random.random() < mutation_rate:
            # Pick a new choice from the search space
            # Ensure we don't pick the same one (optional, but efficient)
            current = new_genes[i]
            possible_choices = [c for c in search_space.choices if c != current]
            if possible_choices:
                new_genes[i] = random.choice(possible_choices)
    
    return Genome(genes=new_genes)