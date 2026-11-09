"""Per-chunk negative log-likelihood, computed in fp32 and token-exact."""

import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def chunk_nll(model: nn.Module, chunks: torch.Tensor, batch_size: int = 8, device: str = "cuda") -> torch.Tensor:
    """Mean next-token NLL (nats) of every chunk.

    Args:
        chunks: (n_chunks, seq_len) token ids.
    Returns:
        (n_chunks,) float64 CPU tensor. Each entry averages over seq_len - 1 predicted tokens,
        so the mean of the vector equals the corpus-level token-weighted NLL.
    """
    assert chunks.ndim == 2
    model.eval()
    out: list[torch.Tensor] = []
    for start in range(0, chunks.shape[0], batch_size):
        batch = chunks[start : start + batch_size].to(device)
        logits = model(batch).logits  # (b, seq_len, vocab)
        assert logits.shape[:2] == batch.shape
        loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            batch[:, 1:].reshape(-1),
            reduction="none",
        ).reshape(batch.shape[0], batch.shape[1] - 1)
        out.append(loss.mean(dim=1).double().cpu())
    return torch.cat(out)
