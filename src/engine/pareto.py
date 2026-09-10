import math

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

def get_pareto_front(population_fitness: list[FitnessMetrics]) -> list[int]:
    """
    Identifies the indices of the non-dominated solutions in the population.
    Naive approach: O(N^2). Sufficient for N < 500.

    Note: this returns only the single first front (rank 0). For a full
    NSGA-II environmental selection across all ranks, use
    ``fast_nondominated_sort`` instead.
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


def fast_nondominated_sort(population_fitness: list[FitnessMetrics]) -> list[list[int]]:
    """
    Deb et al.'s O(N^2) fast non-dominated sort (NSGA-II, 2002).

    Partitions the population into successive Pareto fronts: front 0 is the
    set of non-dominated solutions, front 1 is what remains non-dominated
    after front 0 is removed, and so on. Returns a list of fronts, each a
    list of population indices.
    """
    num_pop = len(population_fitness)
    dominated_by: list[list[int]] = [[] for _ in range(num_pop)]
    domination_count = [0] * num_pop

    fronts: list[list[int]] = [[]]

    for p in range(num_pop):
        for q in range(num_pop):
            if p == q:
                continue
            if dominates(population_fitness[p], population_fitness[q]):
                dominated_by[p].append(q)
            elif dominates(population_fitness[q], population_fitness[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_by[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    # Drop the trailing empty front produced by the loop's termination check.
    if not fronts[-1]:
        fronts.pop()
    return fronts


def crowding_distance(front: list[int], population_fitness: list[FitnessMetrics]) -> dict[int, float]:
    """
    Deb et al.'s crowding-distance assignment for a single front.

    For each objective, sorts the front by that objective and accumulates
    the normalized distance between an individual's neighbors. Boundary
    solutions (min/max per objective) get infinite distance so they are
    always preferred, preserving spread at the extremes of the front.
    Returns a mapping from population index to crowding distance.
    """
    distance = {idx: 0.0 for idx in front}
    if len(front) <= 2:
        return {idx: math.inf for idx in front}

    num_objectives = len(population_fitness[front[0]].to_tuple())

    for obj_idx in range(num_objectives):
        front_sorted = sorted(front, key=lambda idx: population_fitness[idx].to_tuple()[obj_idx])
        obj_min = population_fitness[front_sorted[0]].to_tuple()[obj_idx]
        obj_max = population_fitness[front_sorted[-1]].to_tuple()[obj_idx]
        obj_range = obj_max - obj_min

        distance[front_sorted[0]] = math.inf
        distance[front_sorted[-1]] = math.inf

        if obj_range == 0:
            continue

        for k in range(1, len(front_sorted) - 1):
            prev_val = population_fitness[front_sorted[k - 1]].to_tuple()[obj_idx]
            next_val = population_fitness[front_sorted[k + 1]].to_tuple()[obj_idx]
            distance[front_sorted[k]] += (next_val - prev_val) / obj_range

    return distance


def crowded_comparison_key(
    idx: int,
    rank: dict[int, int],
    distance: dict[int, float],
) -> tuple[int, float]:
    """
    NSGA-II's crowded-comparison operator as a sort key: prefer lower
    (better) front rank, and within the same rank prefer larger crowding
    distance (less crowded, i.e. more diverse).
    """
    return (rank[idx], -distance[idx])
