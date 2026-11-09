"""Non-overlapping, contiguous token chunks of a WikiText-2 split.

The legacy loader (``utils/data.py``) draws chunk start offsets at random, which can
overlap and covers only a tiny part of a split. For trustworthy perplexity we instead
tile a split into disjoint ``seq_len`` chunks and either use all of them or a seeded
random subset of the chunk indices.
"""

import hashlib

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def load_chunks(
    tokenizer_name: str,
    split: str,
    seq_len: int,
    max_chunks: int | None = None,
    seed: int = 0,
) -> tuple[torch.Tensor, str]:
    """Return ``(chunks, fingerprint)``; chunks has shape ``(n_chunks, seq_len)`` (int64)."""
    if seq_len <= 1:
        raise ValueError("seq_len must be > 1")
    data = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.model_max_length = int(1e30)
    ids = tokenizer("\n\n".join(data["text"]), return_tensors="pt", add_special_tokens=False).input_ids[0]
    n_chunks = ids.numel() // seq_len
    if n_chunks == 0:
        raise ValueError(f"split {split!r} has fewer than seq_len={seq_len} tokens")
    chunks = ids[: n_chunks * seq_len].reshape(n_chunks, seq_len)
    if max_chunks is not None and max_chunks < n_chunks:
        gen = torch.Generator().manual_seed(seed)
        keep = torch.randperm(n_chunks, generator=gen)[:max_chunks].sort().values
        chunks = chunks[keep]
    assert chunks.ndim == 2 and chunks.shape[1] == seq_len
    digest = hashlib.sha256(chunks.numpy().tobytes()).hexdigest()
    return chunks, digest
