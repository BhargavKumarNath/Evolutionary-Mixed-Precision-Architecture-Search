"""Reproducibility helpers for search and evaluation experiments."""

import base64
import os
import pickle
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, deterministic: bool = False) -> dict[str, Any]:
    """Seed supported random number generators and return applied settings."""
    if seed < 0:
        raise ValueError("seed must be non-negative")

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return {
        "seed": seed,
        "deterministic_algorithms": deterministic,
        "cuda_available": cuda_available,
        "determinism_note": (
            "CUDA operations may remain nondeterministic when no deterministic "
            "kernel exists for an operation."
        ),
    }


def capture_rng_state() -> dict[str, str]:
    """Capture RNG state in JSON-safe base64 encoded form."""
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return {key: _encode(value) for key, value in state.items()}


def restore_rng_state(state: dict[str, str]) -> None:
    """Restore a state returned by :func:`capture_rng_state`."""
    random.setstate(_decode(state["python"]))
    np.random.set_state(_decode(state["numpy"]))
    torch.set_rng_state(_decode(state["torch"]))
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(_decode(state["cuda"]))


def _encode(value: Any) -> str:
    return base64.b64encode(
        pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    ).decode("ascii")


def _decode(value: str) -> Any:
    return pickle.loads(base64.b64decode(value.encode("ascii")))