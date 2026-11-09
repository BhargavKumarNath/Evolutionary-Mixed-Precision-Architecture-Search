"""Reference methods that choose a per-layer bit-width vector under a bit budget.

All methods consume the same additive proxy (``AdditiveProfile``), so differences between
them are differences in *search*, not in what they are shown. Budget ``B`` is a cap on the
average bit-width across layers (layers have equal parameter counts, asserted).
"""

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..core.search_space import Genome, MixedPrecisionSearchSpace


@dataclass(frozen=True)
class AdditiveProfile:
    delta: tuple[dict[int, float], ...]  # delta[layer][bits] -> signed loss increase vs FP16
    bits: tuple[int, ...]
    baseline_loss: float

    @property
    def n_layers(self) -> int:
        return len(self.delta)

    @classmethod
    def from_json(cls, path: str | Path, bits: tuple[int, ...] = (4, 8, 16)) -> "AdditiveProfile":
        data = json.loads(Path(path).read_text())
        counts = set(data["layer_params"].values())
        assert len(counts) == 1, "methods assume equal parameter counts per layer"
        n = len(data["layer_params"])
        delta = tuple({b: float(data["sensitivity"].get(f"{i}_{b}", 0.0)) for b in bits} for i in range(n))
        return cls(delta=delta, bits=tuple(sorted(bits)), baseline_loss=float(data["baseline_loss"]))

    def loss(self, genes: list[int]) -> float:
        """Predicted loss increase over FP16 (sum of single-layer deltas)."""
        assert len(genes) == self.n_layers
        return float(sum(self.delta[i][b] for i, b in enumerate(genes)))


def avg_bits(genes: list[int]) -> float:
    return sum(genes) / len(genes)


def uniform(n_layers: int, bits: int) -> list[int]:
    return [bits] * n_layers


def greedy_downgrade(profile: AdditiveProfile, budget: float) -> list[int]:
    """Start at FP16; repeatedly apply the downgrade with the least added loss per bit saved."""
    genes = [max(profile.bits)] * profile.n_layers
    cap = budget * profile.n_layers
    ladder = sorted(profile.bits, reverse=True)
    while sum(genes) > cap + 1e-9:
        best: tuple[float, int, int] | None = None
        for i, b in enumerate(genes):
            k = ladder.index(b)
            if k + 1 >= len(ladder):
                continue
            nb = ladder[k + 1]
            cost = (profile.delta[i][nb] - profile.delta[i][b]) / (b - nb)
            if best is None or cost < best[0]:
                best = (cost, i, nb)
        if best is None:
            raise ValueError("budget below the smallest feasible average bit-width")
        genes[best[1]] = best[2]
    return genes


def _dp_table(profile: AdditiveProfile) -> list[list[float]]:
    """best[i][w] = min predicted loss using layers 0..i-1 with total bits exactly w."""
    total = max(profile.bits) * profile.n_layers
    inf = float("inf")
    best = [[inf] * (total + 1) for _ in range(profile.n_layers + 1)]
    best[0][0] = 0.0
    for i in range(profile.n_layers):
        for w in range(total + 1):
            if best[i][w] == inf:
                continue
            for b in profile.bits:
                if w + b <= total:
                    v = best[i][w] + profile.delta[i][b]
                    best[i + 1][w + b] = min(best[i + 1][w + b], v)
    return best


def _reconstruct(profile: AdditiveProfile, best: list[list[float]], w: int) -> list[int]:
    genes: list[int] = []
    for i in range(profile.n_layers, 0, -1):
        for b in profile.bits:
            if w - b >= 0 and abs(best[i - 1][w - b] + profile.delta[i - 1][b] - best[i][w]) < 1e-12:
                genes.append(b)
                w -= b
                break
        else:  # pragma: no cover
            raise RuntimeError("DP reconstruction failed")
    return genes[::-1]


def dp_optimal(profile: AdditiveProfile, budget: float) -> list[int]:
    """Exact minimiser of the additive proxy subject to sum(bits) <= budget * n_layers."""
    best = _dp_table(profile)
    cap = int(budget * profile.n_layers + 1e-9)
    w_best = min(range(cap + 1), key=lambda w: best[profile.n_layers][w])
    if best[profile.n_layers][w_best] == float("inf"):
        raise ValueError("budget below the smallest feasible average bit-width")
    return _reconstruct(profile, best, w_best)


def exact_pareto_front(profile: AdditiveProfile) -> list[tuple[float, float]]:
    """Exact (avg_bits, proxy loss) Pareto front of the additive proxy, via the DP table."""
    best = _dp_table(profile)
    last = best[profile.n_layers]
    front: list[tuple[float, float]] = []
    running = float("inf")
    for w, v in enumerate(last):
        if v < running:
            running = v
            front.append((w / profile.n_layers, v))
    return front


def random_search(
    space: MixedPrecisionSearchSpace, profile: AdditiveProfile, n_evals: int, rng: random.Random
) -> list[int]:
    """Best of ``n_evals`` independent feasible random genomes under the proxy."""
    best_genes: list[int] | None = None
    best_loss = float("inf")
    for _ in range(n_evals):
        genes = space.sample_with(rng).genes
        loss = profile.loss(genes)
        if loss < best_loss:
            best_genes, best_loss = genes, loss
    assert best_genes is not None
    return best_genes


def hypervolume_2d(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    """Hypervolume of a 2-objective minimisation set (x = avg bits, y = loss) w.r.t. ``ref``."""
    pts = sorted(p for p in points if p[0] < ref[0] and p[1] < ref[1])
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y < prev_y:
            hv += (ref[0] - x) * (prev_y - y)
            prev_y = y
    return hv


def genome_of(genes: list[int]) -> Genome:
    return Genome(genes=list(genes))


__all__ = [n for n in dir() if not n.startswith("_")] + ["np"]
