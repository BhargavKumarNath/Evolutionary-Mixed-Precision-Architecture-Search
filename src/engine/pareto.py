from typing import List
from ..core.evaluator import FitnessMetrics

def dominates(f1: FitnessMetrics, f2: FitnessMetrics) -> bool:
    """
    Returns True if fitness f1 dominates f2.
    We are MINIMIZING all objectives (Loss, VRAM, Latency).
    
    A dominates B if:
        all(A_i <= B_i) AND any(A_i < B_i)
    """
    # Convert to tuples for easy comparison
    val1 = f1.to_tuple()
    val2 = f2.to_tuple()
    
    # Check if f1 is no worse than f2 in all objectives
    all_better_or_equal = all(x <= y for x, y in zip(val1, val2))
    
    # Check if f1 is strictly better in at least one objective
    any_strictly_better = any(x < y for x, y in zip(val1, val2))
    
    return all_better_or_equal and any_strictly_better

def get_pareto_front(population_fitness: List[FitnessMetrics]) -> List[int]:
    """
    Identifies the indices of the non-dominated solutions in the population.
    Naive approach: O(N^2). Sufficient for N < 500.
    """
    pareto_indices = []
    num_pop = len(population_fitness)
    
    for i in range(num_pop):
        is_dominated = False
        for j in range(num_pop):
            if i == j: 
                continue
            
            if dominates(population_fitness[j], population_fitness[i]):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_indices.append(i)
            
    return pareto_indices