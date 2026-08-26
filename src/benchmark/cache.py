"""Measurement cache and provenance tracking."""

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MeasurementCacheKey:
    """Unique identifier for a benchmark measurement."""

    genome_tuple: tuple[int, ...]
    backend_name: str
    hardware_hash: str
    batch_size: int
    sequence_length: int
    benchmark_protocol: str

    def as_filename(self) -> str:
        """Generate a unique cache filename."""
        key_parts = (
            tuple(self.genome_tuple),
            self.backend_name,
            self.hardware_hash,
            self.batch_size,
            self.sequence_length,
            self.benchmark_protocol,
        )
        key_str = "_".join(str(p) for p in key_parts)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()[:16]
        return f"measurement_{key_hash}.json"


def hardware_fingerprint(hardware_identity: dict[str, Any]) -> str:
    """Hash hardware identity for cache key consistency."""
    core_fields = (
        hardware_identity.get("gpu_name"),
        hardware_identity.get("compute_capability"),
        hardware_identity.get("total_memory_bytes"),
        hardware_identity.get("cuda_version"),
    )
    serialized = json.dumps(core_fields, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


class MeasurementCache:
    """Track measured benchmark results by unique genome and hardware."""

    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_key(
        self,
        genome: list[int],
        backend: str,
        hardware_identity: dict[str, Any],
        batch_size: int,
        sequence_length: int,
    ) -> MeasurementCacheKey:
        """Construct a cache key."""
        return MeasurementCacheKey(
            genome_tuple=tuple(genome),
            backend_name=backend,
            hardware_hash=hardware_fingerprint(hardware_identity),
            batch_size=batch_size,
            sequence_length=sequence_length,
            benchmark_protocol="v1",
        )

    def store(self, key: MeasurementCacheKey, result: dict[str, Any]) -> None:
        """Save a measurement result."""
        path = self.cache_dir / key.as_filename()
        path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        logger.info("Cached measurement: %s", key.as_filename())

    def retrieve(self, key: MeasurementCacheKey) -> dict[str, Any] | None:
        """Retrieve a cached measurement if it exists."""
        path = self.cache_dir / key.as_filename()
        if path.exists():
            logger.info("Retrieved cached measurement: %s", key.as_filename())
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def list_measurements(self) -> list[dict[str, Any]]:
        """List all cached measurements."""
        results = []
        for path in sorted(self.cache_dir.glob("measurement_*.json")):
            try:
                results.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Could not load cache file %s: %s", path, e)
        return results
