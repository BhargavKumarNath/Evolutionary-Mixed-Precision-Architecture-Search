"""Per-layer quantization sensitivity, measured with signed deltas on per-chunk NLL.

Two quantizers can be profiled:
  * ``fake``: absmax fake quantization (the original EMPAS method)
  * ``bnb``:  the real bitsandbytes backend (NF4 / LLM.int8), the thing actually deployed
Deltas are **not** clipped at zero (clipping biases small effects upward).
"""

import logging
from collections.abc import Iterator
from contextlib import contextmanager

import numpy as np
import torch
from torch import nn

from ..eval.perplexity import chunk_nll
from .quantization_backend import BitsAndBytesBackend, _parent_module
from .quantizer import fake_quantize_tensor

logger = logging.getLogger(__name__)


def _linears(layer: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(n, m) for n, m in layer.named_modules() if isinstance(m, nn.Linear)]


@contextmanager
def quantized_layer(layer: nn.Module, bits: int, mode: str, backend: BitsAndBytesBackend | None) -> Iterator[None]:
    """Temporarily quantize every Linear in ``layer``; always restores the original."""
    linears = _linears(layer)
    if mode == "fake":
        originals = {n: m.weight.data.clone() for n, m in linears}
        try:
            for _, m in linears:
                m.weight.data = fake_quantize_tensor(m.weight.data, bits)
            yield
        finally:
            for n, m in linears:
                m.weight.data = originals[n]
    elif mode == "bnb":
        assert backend is not None
        swapped: list[tuple[nn.Module, str, nn.Module]] = []
        try:
            for name, module in linears:
                parent, child = _parent_module(layer, name)
                swapped.append((parent, child, module))
                setattr(parent, child, backend._replace_linear(module, bits))
            yield
        finally:
            for parent, child, module in swapped:
                setattr(parent, child, module)
            torch.cuda.empty_cache()
    else:
        raise ValueError(mode)


def profile_sensitivity(
    model: nn.Module,
    chunks: torch.Tensor,
    mode: str,
    bit_choices: tuple[int, ...] = (4, 8),
    batch_size: int = 8,
    device: str = "cuda",
) -> dict:
    """Return a profile dict compatible with ``ProxyEvaluator`` plus standard errors."""
    layers = model.model.layers
    backend = BitsAndBytesBackend(device) if mode == "bnb" else None
    base = chunk_nll(model, chunks, batch_size, device).numpy()
    sensitivity: dict[str, float] = {}
    se: dict[str, float] = {}
    layer_params: dict[str, int] = {}
    for i, layer in enumerate(layers):
        layer_params[str(i)] = int(sum(m.weight.numel() for _, m in _linears(layer)))
        sensitivity[f"{i}_16"] = 0.0
        se[f"{i}_16"] = 0.0
        for bits in bit_choices:
            with quantized_layer(layer, bits, mode, backend):
                nll = chunk_nll(model, chunks, batch_size, device).numpy()
            delta = nll - base  # (n_chunks,) paired by chunk
            sensitivity[f"{i}_{bits}"] = float(delta.mean())
            se[f"{i}_{bits}"] = float(delta.std(ddof=1) / np.sqrt(len(delta)))
        logger.info("layer %d/%d profiled (%s)", i + 1, len(layers), mode)
    total = int(sum(p.numel() for p in model.parameters()))
    return {
        "quantizer": mode,
        "baseline_loss": float(base.mean()),
        "layer_params": layer_params,
        "non_layer_params": total - sum(layer_params.values()),
        "sensitivity": sensitivity,
        "sensitivity_se": se,
        "n_chunks": int(chunks.shape[0]),
        "seq_len": int(chunks.shape[1]),
        "n_tokens": int(chunks.numel()),
    }
