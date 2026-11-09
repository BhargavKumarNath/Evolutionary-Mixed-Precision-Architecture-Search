"""Run the repo's NSGA-II against the additive proxy (no wandb, no disk checkpoints)."""

import random
import tempfile

import numpy as np
import torch
from common import ROOT  # noqa: F401  (adds repo root to sys.path)

from src.baselines.methods import AdditiveProfile
from src.core.proxy_evaluator import ProxyEvaluator
from src.core.search_space import MixedPrecisionSearchSpace
from src.engine.ga import GAEngine


def run_nsga2(
    profile_path: str,
    memory_model_path: str | None,
    seed: int,
    pop_size: int,
    n_generations: int,
    crossover_rate: float,
    mutation_rate: float,
    tournament_size: int,
    max_avg_bitwidth: float | None = None,
    n_layers: int = 22,
) -> tuple[GAEngine, int]:
    """Returns (engine after the run, number of unique proxy evaluations used)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    space = MixedPrecisionSearchSpace(n_layers, [4, 8, 16], max_avg_bitwidth=max_avg_bitwidth)
    evaluator = ProxyEvaluator(profile_path, memory_model_path, latency_mode="none")
    cfg = {
        "pop_size": pop_size,
        "n_generations": n_generations,
        "crossover_rate": crossover_rate,
        "mutation_rate": mutation_rate,
        "tournament_size": tournament_size,
        "checkpoint_freq": 10**9,  # never checkpoint
    }
    with tempfile.TemporaryDirectory() as tmp:
        engine = GAEngine(space, evaluator, cfg, {"enabled": False}, output_dir=tmp)
        engine.run()
    return engine, len(engine.fitness_cache)


def additive(profile_path: str) -> AdditiveProfile:
    return AdditiveProfile.from_json(profile_path)
