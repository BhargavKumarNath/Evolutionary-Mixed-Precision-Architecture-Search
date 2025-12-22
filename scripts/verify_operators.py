import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.search_space import Genome, MixedPrecisionSearchSpace
from src.core.evaluator import FitnessMetrics
from src.engine.pareto import dominates, get_pareto_front
from src.engine.operators import crossover_uniform, mutate_random_gene

def test_pareto():
    print("--- Testing Pareto Logic ---")
    # A: Good Loss, High VRAM (2.0, 1000)
    # B: Bad Loss, Low VRAM   (3.0, 500)
    # C: Bad Loss, High VRAM  (3.0, 1000) -> Dominated by A (in Loss) and B (in VRAM)? 
    # Wait, A dominates C? 2.0 <= 3.0 (T), 1000 <= 1000 (T). A is strictly better in Loss. YES.
    
    fA = FitnessMetrics(2.0, 1000, 50)
    fB = FitnessMetrics(3.0, 500, 50)
    fC = FitnessMetrics(3.0, 1000, 55) # Worse than A and B
    
    print(f"A dominates C? {dominates(fA, fC)} (Expected: True)")
    print(f"B dominates C? {dominates(fB, fC)} (Expected: True)")
    print(f"A dominates B? {dominates(fA, fB)} (Expected: False)")
    print(f"B dominates A? {dominates(fB, fA)} (Expected: False)")
    
    pop = [fA, fB, fC]
    indices = get_pareto_front(pop)
    print(f"Pareto Indices: {indices} (Expected: [0, 1] i.e., A and B)")

def test_operators():
    print("\n--- Testing Genetic Operators ---")
    space = MixedPrecisionSearchSpace(num_layers=4, choices=[2, 4, 8])
    
    p1 = Genome(genes=[2, 2, 2, 2])
    p2 = Genome(genes=[8, 8, 8, 8])
    
    print(f"Parent 1: {p1.genes}")
    print(f"Parent 2: {p2.genes}")
    
    # Crossover
    c1, c2 = crossover_uniform(p1, p2)
    print(f"Child 1:  {c1.genes}")
    print(f"Child 2:  {c2.genes}")
    
    # Mutation
    m1 = mutate_random_gene(p1, space, mutation_rate=0.5)
    print(f"Mutant 1: {m1.genes} (Base was [2,2,2,2])")

if __name__ == "__main__":
    test_pareto()
    test_operators()