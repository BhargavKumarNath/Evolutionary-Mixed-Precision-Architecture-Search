import logging

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


class DatasetLoadError(RuntimeError):
    """Raised when the calibration corpus cannot be loaded or prepared."""


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
            return [
                torch.randint(0, 1000, (1, seq_len), generator=generator)
                for _ in range(nsamples)
            ]
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

    # Use a datetministic generator for reproducibility
    rng = torch.Generator().manual_seed(seed)

    for _ in range(nsamples):
        i = torch.randint(0, num_tokens - seq_len + 1, (1,), generator=rng).item()
        chunk = encodings.input_ids[:, i : i + seq_len] 
        samples.append(chunk)
    
    return samples
