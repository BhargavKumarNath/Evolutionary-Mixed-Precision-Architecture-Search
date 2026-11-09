"""Fit and cross-validate the VRAM model from measured memory."""

from dataclasses import dataclass

import numpy as np

MIB = 2**20


@dataclass(frozen=True)
class MemoryFit:
    form: str  # "A" (nominal bits, intercept only) or "B" (intercept + effective bits at 4 and 8)
    intercept_mb: float
    effective_bits: dict[int, float]
    loo_rmse_mb: float
    train_rmse_mb: float


def features(genomes: list[list[int]], layer_params: list[int]) -> np.ndarray:
    """(n_genomes, 3): MiB of weights *as nominal bytes/8* grouped by width [4, 8, 16]."""
    x = np.zeros((len(genomes), 3))
    for r, genes in enumerate(genomes):
        assert len(genes) == len(layer_params)
        for p, b in zip(layer_params, genes):
            x[r, {4: 0, 8: 1, 16: 2}[b]] += p / 8 / MIB
    return x


def _solve(form: str, x: np.ndarray, y: np.ndarray) -> tuple[float, dict[int, float]]:
    if form == "A":
        c = float(np.mean(y - x @ np.array([4.0, 8.0, 16.0])))
        return c, {4: 4.0, 8: 8.0, 16: 16.0}
    resid = y - 16.0 * x[:, 2]
    design = np.column_stack([np.ones(len(y)), x[:, 0], x[:, 1]])
    coef, *_ = np.linalg.lstsq(design, resid, rcond=None)
    return float(coef[0]), {4: float(coef[1]), 8: float(coef[2]), 16: 16.0}


def predict(form_fit: tuple[float, dict[int, float]], x: np.ndarray) -> np.ndarray:
    c, e = form_fit
    return c + x @ np.array([e[4], e[8], e[16]])


def fit_form(form: str, x: np.ndarray, y: np.ndarray) -> MemoryFit:
    sol = _solve(form, x, y)
    train = float(np.sqrt(np.mean((predict(sol, x) - y) ** 2)))
    errs = []
    for i in range(len(y)):
        keep = np.arange(len(y)) != i
        errs.append(predict(_solve(form, x[keep], y[keep]), x[i : i + 1])[0] - y[i])
    return MemoryFit(form, sol[0], sol[1], float(np.sqrt(np.mean(np.square(errs)))), train)


def bootstrap_params(form: str, x: np.ndarray, y: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    cs, e4, e8 = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        try:
            c, e = _solve(form, x[idx], y[idx])
        except np.linalg.LinAlgError:
            continue
        cs.append(c)
        e4.append(e[4])
        e8.append(e[8])
    q = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {"intercept_mb": q(cs), "e4": q(e4), "e8": q(e8)}
