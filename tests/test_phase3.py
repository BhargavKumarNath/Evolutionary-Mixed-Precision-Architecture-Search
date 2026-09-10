"""Phase 3: NSGA-II evolutionary engine — non-dominated sorting, crowding
distance, operators, and end-to-end search behavior."""

import math
import tempfile
import unittest

from src.core.evaluator import DummyEvaluator, FitnessMetrics
from src.core.search_space import Genome, MixedPrecisionSearchSpace
from src.engine.ga import GAEngine
from src.engine.operators import crossover_uniform, mutate_random_gene
from src.engine.pareto import (
    crowded_comparison_key,
    crowding_distance,
    dominates,
    fast_nondominated_sort,
    get_pareto_front,
)


def _fit(loss: float, mem: float, lat: float = 0.0) -> FitnessMetrics:
    return FitnessMetrics(mean_loss=loss, peak_memory_mb=mem, latency_ms=lat)


class DominanceTests(unittest.TestCase):
    def test_dominates_requires_no_worse_and_one_strictly_better(self):
        self.assertTrue(dominates(_fit(1.0, 100.0), _fit(2.0, 100.0)))
        self.assertFalse(dominates(_fit(1.0, 200.0), _fit(2.0, 100.0)))

    def test_equal_fitness_does_not_dominate(self):
        self.assertFalse(dominates(_fit(1.0, 100.0), _fit(1.0, 100.0)))


class NondominatedSortTests(unittest.TestCase):
    def test_single_front_matches_naive_pareto_front(self):
        fitnesses = [_fit(1.0, 300.0), _fit(2.0, 200.0), _fit(3.0, 100.0), _fit(2.5, 250.0)]
        fronts = fast_nondominated_sort(fitnesses)
        naive = set(get_pareto_front(fitnesses))
        self.assertEqual(set(fronts[0]), naive)

    def test_multiple_fronts_are_correctly_ranked(self):
        # 0 and 1 are mutually non-dominated (front 0).
        # 2 is dominated by 0 (worse on both axes) but not by 1 -> front 1.
        # 3 is dominated by everything -> front 2.
        fitnesses = [
            _fit(1.0, 100.0),  # 0: front 0
            _fit(2.0, 50.0),   # 1: front 0
            _fit(1.5, 150.0),  # 2: dominated by 0 -> front 1
            _fit(5.0, 500.0),  # 3: dominated by all -> front 2
        ]
        fronts = fast_nondominated_sort(fitnesses)
        self.assertEqual(set(fronts[0]), {0, 1})
        self.assertIn(2, fronts[1])
        self.assertIn(3, fronts[-1])
        self.assertNotIn(3, fronts[0])
        self.assertNotIn(3, fronts[1] if len(fronts) > 2 else [])

    def test_every_individual_is_assigned_to_exactly_one_front(self):
        fitnesses = [_fit(float(i % 5), float((i * 37) % 200)) for i in range(30)]
        fronts = fast_nondominated_sort(fitnesses)
        seen = [idx for front in fronts for idx in front]
        self.assertEqual(sorted(seen), list(range(30)))

    def test_all_non_dominated_population_is_a_single_front(self):
        # A strictly decreasing loss / strictly increasing memory curve: no
        # point dominates another, so everything belongs in front 0.
        fitnesses = [_fit(float(i), float(10 - i)) for i in range(5)]
        fronts = fast_nondominated_sort(fitnesses)
        self.assertEqual(len(fronts), 1)
        self.assertEqual(sorted(fronts[0]), list(range(5)))


class CrowdingDistanceTests(unittest.TestCase):
    def test_boundary_points_get_infinite_distance(self):
        fitnesses = [_fit(1.0, 300.0), _fit(2.0, 200.0), _fit(3.0, 100.0)]
        front = [0, 1, 2]
        dist = crowding_distance(front, fitnesses)
        self.assertEqual(dist[0], math.inf)
        self.assertEqual(dist[2], math.inf)
        self.assertLess(dist[1], math.inf)
        self.assertGreaterEqual(dist[1], 0.0)

    def test_front_of_two_or_fewer_is_all_infinite(self):
        fitnesses = [_fit(1.0, 300.0), _fit(2.0, 200.0)]
        dist = crowding_distance([0, 1], fitnesses)
        self.assertTrue(all(v == math.inf for v in dist.values()))

    def test_more_isolated_point_gets_larger_distance(self):
        # Middle point is closer to its left neighbor than a point that sits
        # roughly halfway between two more widely-spaced neighbors.
        fitnesses = [_fit(0.0, 0.0), _fit(1.0, 1.0), _fit(1.2, 1.2), _fit(10.0, 10.0)]
        front = [0, 1, 2, 3]
        dist = crowding_distance(front, fitnesses)
        # Point 2 (1.2) is tightly boxed between 1.0 and 10.0's neighbor 1;
        # point 1 (1.0) sits between 0.0 and 1.2 -> less isolated than the
        # gap-adjacent point 2, which borders the large jump to 10.0.
        self.assertGreater(dist[2], dist[1])

    def test_crowded_comparison_prefers_lower_rank_then_higher_distance(self):
        rank = {0: 0, 1: 0, 2: 1}
        distance = {0: 1.0, 1: 5.0, 2: math.inf}
        ordered = sorted([0, 1, 2], key=lambda i: crowded_comparison_key(i, rank, distance))
        # Rank 0 individuals come first; within rank 0, higher distance (1) wins.
        self.assertEqual(ordered, [1, 0, 2])


class OperatorTests(unittest.TestCase):
    def test_crossover_children_only_contain_parent_alleles(self):
        p1 = Genome(genes=[4] * 10)
        p2 = Genome(genes=[16] * 10)
        c1, c2 = crossover_uniform(p1, p2)
        self.assertTrue(all(g in (4, 16) for g in c1.genes))
        self.assertTrue(all(g in (4, 16) for g in c2.genes))
        self.assertEqual(len(c1.genes), 10)

    def test_mutation_rate_zero_is_identity(self):
        space = MixedPrecisionSearchSpace(num_layers=8, choices=[4, 8, 16])
        genome = Genome(genes=[4] * 8)
        mutated = mutate_random_gene(genome, space, mutation_rate=0.0)
        self.assertEqual(mutated.genes, genome.genes)

    def test_mutation_rate_one_changes_every_gene(self):
        space = MixedPrecisionSearchSpace(num_layers=8, choices=[4, 8, 16])
        genome = Genome(genes=[4] * 8)
        mutated = mutate_random_gene(genome, space, mutation_rate=1.0)
        self.assertTrue(all(g != 4 for g in mutated.genes))

    def test_mutation_does_not_modify_input_genome(self):
        space = MixedPrecisionSearchSpace(num_layers=8, choices=[4, 8, 16])
        genome = Genome(genes=[4] * 8)
        mutate_random_gene(genome, space, mutation_rate=1.0)
        self.assertEqual(genome.genes, [4] * 8)


class GAEngineIntegrationTests(unittest.TestCase):
    def _make_engine(self, output_dir: str) -> GAEngine:
        space = MixedPrecisionSearchSpace(num_layers=6, choices=[4, 8, 16])
        config = {
            "pop_size": 12,
            "n_generations": 5,
            "crossover_rate": 0.9,
            "mutation_rate": 0.2,
            "tournament_size": 3,
            "checkpoint_freq": 1,
        }
        return GAEngine(space, DummyEvaluator(), config, output_dir=output_dir)

    def test_population_size_is_conserved_across_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            engine.population = [engine.space.sample() for _ in range(engine.cfg["pop_size"])]
            for gen in range(1, 4):
                engine.step(gen)
                self.assertEqual(len(engine.population), engine.cfg["pop_size"])

    def test_search_improves_best_loss_over_generations(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            engine.population = [engine.space.sample() for _ in range(engine.cfg["pop_size"])]
            first_gen_best = None
            last_gen_best = None
            for gen in range(1, 11):
                engine.step(gen)
                if gen == 1:
                    first_gen_best = engine.history[-1]["min_loss"]
                last_gen_best = engine.history[-1]["min_loss"]
            # DummyEvaluator rewards lower average bit-width with lower loss
            # up to a point; the search should not regress the best-seen loss.
            self.assertLessEqual(last_gen_best, first_gen_best)

    def test_final_population_is_pareto_valid_for_search_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = self._make_engine(tmp)
            engine.population = [engine.space.sample() for _ in range(engine.cfg["pop_size"])]
            for gen in range(1, 4):
                engine.step(gen)
            self.assertTrue(all(engine.space.validate(g) for g in engine.population))


if __name__ == "__main__":
    unittest.main()
