"""Small, dependency-light bootstrap helpers (numpy only)."""

from collections.abc import Callable

import numpy as np


def _rank(v: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank), so Spearman = Pearson of ranks."""
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(len(v), dtype=np.float64)
    ranks[order] = np.arange(len(v), dtype=np.float64)
    for value in np.unique(v):
        mask = v == value
        if mask.sum() > 1:
            ranks[mask] = ranks[mask].mean()
    return ranks


def percentile_ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    lo = (1 - level) / 2 * 100
    return float(np.percentile(samples, lo)), float(np.percentile(samples, 100 - lo))


def bootstrap_stat(
    n: int,
    stat: Callable[[np.ndarray], float],
    n_boot: int = 10_000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, tuple[float, float]]:
    """Resample indices 0..n-1 with replacement; ``stat`` maps an index array to a scalar.

    Returns (point estimate on the full sample, percentile CI). Resamples where ``stat``
    is undefined (NaN) are dropped.
    """
    rng = np.random.default_rng(seed)
    point = float(stat(np.arange(n)))
    draws = np.array([stat(rng.integers(0, n, n)) for _ in range(n_boot)], dtype=np.float64)
    draws = draws[np.isfinite(draws)]
    return point, percentile_ci(draws, level)


def corr_with_ci(
    x: np.ndarray, y: np.ndarray, kind: str = "pearson", n_boot: int = 10_000, seed: int = 0
) -> dict[str, float | list[float]]:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    assert x.shape == y.shape and x.ndim == 1

    def stat(idx: np.ndarray) -> float:
        xs, ys = x[idx], y[idx]
        if xs.std() == 0 or ys.std() == 0:
            return float("nan")
        if kind == "pearson":
            return float(np.corrcoef(xs, ys)[0, 1])
        return float(np.corrcoef(_rank(xs), _rank(ys))[0, 1])

    point, (lo, hi) = bootstrap_stat(len(x), stat, n_boot, seed)
    return {"value": point, "ci95": [lo, hi], "n": len(x)}


def paired_diff_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0
) -> dict[str, float | list[float]]:
    """Mean(a - b) over matched units (e.g. test chunks), with a bootstrap CI."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    point, (lo, hi) = bootstrap_stat(len(d), lambda i: float(d[i].mean()), n_boot, seed)
    return {"mean_diff": point, "ci95": [lo, hi], "n": len(d)}
