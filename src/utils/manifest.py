"""Machine-readable experiment manifest creation."""

import hashlib
import json
import subprocess
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from ..core.configuration import as_plain_dict, configuration_hash


def build_manifest(
    config: Mapping[str, Any],
    input_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable manifest without requiring GPU initialization."""
    resolved = as_plain_dict(config)
    git_commit, git_dirty = _git_state()
    hardware: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        hardware.update(
            {
                "gpu_model": torch.cuda.get_device_name(0),
                "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
            }
        )
    config_digest = configuration_hash(resolved)
    return {
        "manifest_version": 1,
        "experiment_id": config_digest[:16],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty_state": git_dirty,
        "command": sys.argv,
        "config": resolved,
        "config_hash": config_digest,
        "model_name": resolved.get("model_name"),
        "model_revision": resolved.get("model_revision"),
        "dataset": resolved.get("data", {}).get("dataset"),
        "random_seed": resolved.get("seed"),
        "search_algorithm": resolved.get("algorithm", {}).get("name"),
        "precision_space": resolved.get("search_space"),
        "software": {
            "python": sys.version,
            "pytorch": torch.__version__,
        },
        "hardware": hardware,
        "inputs": dict(input_metadata or {}),
    }


def write_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state() -> tuple[str | None, bool | None]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return commit, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None