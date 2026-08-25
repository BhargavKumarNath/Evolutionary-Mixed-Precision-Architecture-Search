import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, cast

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class DatasetLoadError(RuntimeError):
    """Raised when the calibration corpus cannot be loaded or prepared."""


@dataclass(frozen=True)
class DatasetSplit:
    """An immutable, identified collection of tokenized evaluation samples."""

    name: str
    samples: tuple[torch.Tensor, ...]
    sample_ids: tuple[str, ...]
    token_count: int
    fingerprint: str

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


class DatasetProtocol:
    """Build disjoint, reproducible calibration, validation, and test splits."""

    def __init__(
        self,
        tokenizer_name: str,
        dataset: str = "wikitext",
        split_config: Mapping[str, Mapping[str, object]] | None = None,
        allow_synthetic_data: bool = False,
    ):
        self.tokenizer_name = tokenizer_name
        self.dataset = dataset
        self.split_config = dict(split_config or {})
        self.allow_synthetic_data = allow_synthetic_data

    def load(self) -> dict[str, DatasetSplit]:
        if not self.split_config:
            raise ValueError("split_config must define at least one named protocol split")
        result: dict[str, DatasetSplit] = {}
        seen_ids: set[str] = set()
        for name, config in self.split_config.items():
            split = str(config.get("split", name))
            samples, offsets = _load_samples(
                tokenizer_name=self.tokenizer_name,
                dataset=self.dataset,
                seq_len=_config_int(config, "seq_len", 2048),
                nsamples=_config_int(config, "nsamples", 128),
                seed=_config_int(config, "seed", 42),
                split=split,
                allow_synthetic_data=self.allow_synthetic_data,
            )
            sample_ids = tuple(f"{split}:{offset}" for offset in offsets)
            overlap = seen_ids.intersection(sample_ids)
            if overlap:
                raise DatasetLoadError(
                    f"Protocol split {name!r} overlaps sample IDs: {sorted(overlap)[:3]}"
                )
            seen_ids.update(sample_ids)
            result[name] = DatasetSplit(
                name=name,
                samples=tuple(samples),
                sample_ids=sample_ids,
                token_count=sum(sample.numel() for sample in samples),
                fingerprint=fingerprint_samples(samples, sample_ids),
            )
        return result


def get_calibration_dataset(
    tokenizer_name: str,
    dataset: str = "wikitext",
    seq_len: int = 2048,
    nsamples: int = 128,
    seed: int = 42,
    split: str = "test",
    allow_synthetic_data: bool = False,
) -> list[torch.Tensor]:
    """Loads Wikitext2 and prepares a list of input_idx for calibration.
    Returns: List [torch.Tensor]"""
    samples, _ = _load_samples(
        tokenizer_name=tokenizer_name,
        dataset=dataset,
        seq_len=seq_len,
        nsamples=nsamples,
        seed=seed,
        split=split,
        allow_synthetic_data=allow_synthetic_data,
    )
    return samples


def _load_samples(
    tokenizer_name: str,
    dataset: str,
    seq_len: int,
    nsamples: int,
    seed: int,
    split: str,
    allow_synthetic_data: bool,
) -> tuple[list[torch.Tensor], list[int]]:
    if dataset != "wikitext":
        raise ValueError(f"Unsupported dataset {dataset!r}; Phase 1 supports 'wikitext'")
    if seq_len <= 0 or nsamples <= 0:
        raise ValueError("seq_len and nsamples must be positive")

    try:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    except Exception as exc:
        if allow_synthetic_data:
            logger.warning("Using synthetic data because dataset loading failed: %s", exc)
            generator = torch.Generator().manual_seed(seed)
            synthetic_samples = [
                torch.randint(0, 1000, (1, seq_len), generator=generator)
                for _ in range(nsamples)
            ]
            return synthetic_samples, list(range(nsamples))
        raise DatasetLoadError(
            f"Could not load Wikitext2 split {split!r}; pass allow_synthetic_data "
            "only for explicit tests"
        ) from exc
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    except Exception as exc:
        raise DatasetLoadError(f"Could not load tokenizer {tokenizer_name!r}") from exc
    tokenizer.model_max_length = int(1e30)

    # tokenize the entire dataset into one long stream
    try:
        encodings = tokenizer(
            "\n\n".join(data["text"]),
            return_tensors="pt",
            add_special_tokens=False,
        )
    except Exception as exc:
        raise DatasetLoadError("Could not tokenize the calibration corpus") from exc

    # Chunk into nsamples of seq_len
    num_tokens = encodings.input_ids.size(1)
    if num_tokens < seq_len:
        raise DatasetLoadError(
            f"Dataset split {split!r} contains {num_tokens} tokens, "
            f"but seq_len is {seq_len}"
        )

    samples: list[torch.Tensor] = []
    offsets: list[int] = []

    # Use a datetministic generator for reproducibility
    rng = torch.Generator().manual_seed(seed)

    for _ in range(nsamples):
        i = int(torch.randint(0, num_tokens - seq_len + 1, (1,), generator=rng).item())
        chunk = encodings.input_ids[:, i : i + seq_len] 
        samples.append(chunk)
        offsets.append(i)
    
    return samples, offsets


def fingerprint_samples(samples: list[torch.Tensor] | tuple[torch.Tensor, ...], sample_ids: tuple[str, ...]) -> str:
    """Hash sample identities and token IDs for experiment provenance."""
    digest = hashlib.sha256()
    for sample_id, sample in zip(sample_ids, samples):
        digest.update(sample_id.encode("utf-8"))
        digest.update(sample.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _config_int(config: Mapping[str, Any], key: str, default: int) -> int:
    return int(cast(Any, config.get(key, default)))
