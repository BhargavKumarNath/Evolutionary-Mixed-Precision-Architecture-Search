"""Validation and stable identity helpers for resolved experiment configuration."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from omegaconf import ListConfig


class ConfigurationError(ValueError):
    """Raised when an experiment configuration is incomplete or contradictory."""


def as_plain_dict(config: Mapping[str, Any]) -> dict[str, Any]:
    """Convert nested OmegaConf-like mappings into ordinary dictionaries."""
    return {str(key): _as_plain_value(value) for key, value in config.items()}


def _as_plain_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return as_plain_dict(value)
    if isinstance(value, (list, tuple, ListConfig)):
        return [_as_plain_value(item) for item in value]
    return value


def validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the Phase 1 configuration contract and return a plain mapping."""
    resolved = as_plain_dict(config)
    required = ("seed", "target_device", "search_space", "algorithm", "evaluator")
    missing = [key for key in required if key not in resolved]
    if missing:
        raise ConfigurationError(f"Missing required configuration keys: {missing}")

    space = resolved["search_space"]
    if space.get("num_layers", 0) <= 0:
        raise ConfigurationError("search_space.num_layers must be positive")
    choices = space.get("choices", [])
    if not choices or len(set(choices)) != len(choices):
        raise ConfigurationError("search_space.choices must be a non-empty unique list")
    minimum = space.get("min_avg_bitwidth")
    maximum = space.get("max_avg_bitwidth")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise ConfigurationError("min_avg_bitwidth cannot exceed max_avg_bitwidth")
    choice_min = min(choices)
    choice_max = max(choices)
    if minimum is not None and minimum > choice_max:
        raise ConfigurationError("min_avg_bitwidth is infeasible for the precision choices")
    if maximum is not None and maximum < choice_min:
        raise ConfigurationError("max_avg_bitwidth is infeasible for the precision choices")

    algorithm = resolved["algorithm"]
    for key in ("pop_size", "n_generations", "tournament_size", "checkpoint_freq"):
        if int(algorithm.get(key, 0)) <= 0:
            raise ConfigurationError(f"algorithm.{key} must be positive")
    for key in ("crossover_rate", "mutation_rate"):
        value = float(algorithm.get(key, -1))
        if not 0 <= value <= 1:
            raise ConfigurationError(f"algorithm.{key} must be between 0 and 1")

    if not isinstance(resolved["seed"], int) or resolved["seed"] < 0:
        raise ConfigurationError("seed must be a non-negative integer")
    if not resolved["target_device"]:
        raise ConfigurationError("target_device must not be empty")
    if resolved["evaluator"].get("type") not in {"proxy"}:
        raise ConfigurationError("evaluator.type must be 'proxy' during Phase 1")
    return resolved


def configuration_hash(config: Mapping[str, Any]) -> str:
    payload = json.dumps(as_plain_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()