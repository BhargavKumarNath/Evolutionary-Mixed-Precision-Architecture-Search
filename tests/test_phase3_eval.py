"""Tests for the Phase 3 evaluation tooling (CPU only)."""

import itertools
import json
import random

import numpy as np
import pytest

from src.baselines.methods import (
    AdditiveProfile,
    avg_bits,
    dp_optimal,
    exact_pareto_front,
    greedy_downgrade,
    hypervolume_2d,
    random_search,
)
from src.core.search_space import Genome, MixedPrecisionSearchSpace
from src.stats.bootstrap import _rank, corr_with_ci, paired_diff_ci


def make_profile(n: int = 6, seed: int = 0) -> AdditiveProfile:
    rng = np.random.default_rng(seed)
    delta = tuple(
        {4: float(rng.uniform(0.0, 0.05)), 8: float(rng.uniform(0.0, 0.005)), 16: 0.0} for _ in range(n)
    )
    return AdditiveProfile(delta=delta, bits=(4, 8, 16), baseline_loss=2.0)


def brute_force(profile: AdditiveProfile, budget: float) -> float:
    best = float("inf")
    for genes in itertools.product(profile.bits, repeat=profile.n_layers):
        if avg_bits(list(genes)) <= budget + 1e-9:
            best = min(best, profile.loss(list(genes)))
    return best


@pytest.mark.parametrize("budget", [4.0, 5.0, 6.7, 8.0, 12.0])
def test_dp_matches_brute_force(budget: float) -> None:
    profile = make_profile()
    genes = dp_optimal(profile, budget)
    assert avg_bits(genes) <= budget + 1e-9
    assert profile.loss(genes) == pytest.approx(brute_force(profile, budget))


def test_greedy_is_feasible_and_never_beats_dp() -> None:
    profile = make_profile(8, seed=3)
    for budget in (5.0, 6.0, 8.0, 10.0):
        g = greedy_downgrade(profile, budget)
        assert avg_bits(g) <= budget + 1e-9
        assert profile.loss(g) >= profile.loss(dp_optimal(profile, budget)) - 1e-12


def test_infeasible_budget_raises() -> None:
    with pytest.raises(ValueError):
        dp_optimal(make_profile(), 3.0)
    with pytest.raises(ValueError):
        greedy_downgrade(make_profile(), 3.0)


def test_exact_front_is_monotone_and_matches_dp() -> None:
    profile = make_profile()
    front = exact_pareto_front(profile)
    xs = [p[0] for p in front]
    ys = [p[1] for p in front]
    assert xs == sorted(xs) and all(a > b for a, b in zip(ys, ys[1:]))
    for x, y in front:
        assert y == pytest.approx(profile.loss(dp_optimal(profile, x)))


def test_hypervolume_simple() -> None:
    assert hypervolume_2d([(1.0, 1.0)], (2.0, 2.0)) == pytest.approx(1.0)
    assert hypervolume_2d([(1.0, 1.5), (1.5, 1.0)], (2.0, 2.0)) == pytest.approx(0.5 * 1.0 + 0.5 * 0.5)


def test_repair_and_tight_budget_sampling() -> None:
    space = MixedPrecisionSearchSpace(22, [4, 8, 16], max_avg_bitwidth=5.0)
    rng = random.Random(0)
    for _ in range(20):
        g = space.sample_with(rng)
        assert space.validate(g)
    fixed = space.repair(Genome(genes=[16] * 22), rng)
    assert space.validate(fixed)


def test_random_search_feasible_and_improves_with_budget() -> None:
    profile = make_profile(10, seed=5)
    space = MixedPrecisionSearchSpace(10, [4, 8, 16], max_avg_bitwidth=6.0)
    few = profile.loss(random_search(space, profile, 5, random.Random(1)))
    many = profile.loss(random_search(space, profile, 2000, random.Random(1)))
    assert many <= few


def test_rank_ties_average() -> None:
    assert list(_rank(np.array([1.0, 1.0, 3.0]))) == [0.5, 0.5, 2.0]


def test_corr_ci_contains_truth_for_perfect_relation() -> None:
    x = np.arange(30, dtype=float)
    out = corr_with_ci(x, 2 * x + 1, n_boot=200)
    assert out["value"] == pytest.approx(1.0)
    sp = corr_with_ci(x, x**3, "spearman", n_boot=200)
    assert sp["value"] == pytest.approx(1.0)


def test_paired_diff_ci() -> None:
    a = np.full(50, 1.0)
    b = np.full(50, 0.5)
    out = paired_diff_ci(a, b, n_boot=200)
    assert out["mean_diff"] == pytest.approx(0.5)
    assert out["ci95"] == [pytest.approx(0.5), pytest.approx(0.5)]


def test_profile_from_json(tmp_path) -> None:
    data = {
        "baseline_loss": 2.0,
        "layer_params": {"0": 10, "1": 10},
        "sensitivity": {"0_4": 0.1, "0_8": 0.01, "1_4": 0.2, "1_8": 0.02},
    }
    path = tmp_path / "p.json"
    path.write_text(json.dumps(data))
    prof = AdditiveProfile.from_json(path)
    assert prof.loss([4, 8]) == pytest.approx(0.12)
    assert prof.loss([16, 16]) == 0.0


def test_latency_mode_none_zeroes_the_objective(tmp_path) -> None:
    from src.core.proxy_evaluator import ProxyEvaluator

    path = tmp_path / "p.json"
    path.write_text(
        json.dumps(
            {
                "baseline_loss": 2.0,
                "layer_params": {"0": 1000, "1": 1000},
                "sensitivity": {"0_4": 0.1, "1_4": 0.1},
            }
        )
    )
    genome = Genome(genes=[4, 16])
    assert ProxyEvaluator(str(path)).evaluate(genome).latency_ms > 0
    assert ProxyEvaluator(str(path), latency_mode="none").evaluate(genome).latency_ms == 0.0
    with pytest.raises(ValueError):
        ProxyEvaluator(str(path), latency_mode="bogus")


def test_memory_model_changes_vram(tmp_path) -> None:
    from src.core.proxy_evaluator import ProxyEvaluator

    prof = tmp_path / "p.json"
    prof.write_text(
        json.dumps({"baseline_loss": 2.0, "layer_params": {"0": 2**20 * 8}, "sensitivity": {"0_4": 0.1}})
    )
    mem = tmp_path / "m.json"
    mem.write_text(json.dumps({"intercept_mb": 300.0, "effective_bits": {"4": 4.5}}))
    default = ProxyEvaluator(str(prof)).evaluate(Genome(genes=[4])).peak_memory_mb
    fitted = ProxyEvaluator(str(prof), str(mem)).evaluate(Genome(genes=[4])).peak_memory_mb
    assert default == pytest.approx(1024 + 4.0)
    assert fitted == pytest.approx(300 + 4.5)


def test_crowding_ignores_degenerate_objective() -> None:
    from src.core.evaluator import FitnessMetrics
    from src.engine.pareto import crowding_distance

    # three-point front, third objective constant: only the loss/memory extremes are infinite
    fits = [FitnessMetrics(1.0, 3.0, 0.0), FitnessMetrics(2.0, 2.0, 0.0), FitnessMetrics(3.0, 1.0, 0.0),
            FitnessMetrics(1.5, 2.5, 0.0)]
    d = crowding_distance([0, 1, 2, 3], fits)
    assert d[0] == float("inf") and d[2] == float("inf")
    assert d[1] < float("inf") and d[3] < float("inf")
