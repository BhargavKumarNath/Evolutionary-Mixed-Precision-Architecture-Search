"""Warm-up + repeated CUDA-event latency for prefill and single-token decode."""

import subprocess
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from transformers import DynamicCache


@dataclass
class Prepared:
    model: nn.Module
    prompt: torch.Tensor  # (1, ctx)
    past: DynamicCache
    next_token: torch.Tensor  # (1, 1)
    ctx: int


def prepare(model: nn.Module, prompt: torch.Tensor) -> Prepared:
    """Run prefill once to build the KV cache used by the decode timing."""
    assert prompt.ndim == 2 and prompt.shape[0] == 1
    prompt = prompt.cuda()
    with torch.no_grad():
        out = model(prompt, use_cache=True)
    nxt = out.logits[:, -1:].argmax(-1)
    return Prepared(model, prompt, out.past_key_values, nxt, prompt.shape[1])


def _time(fn, iters: int) -> np.ndarray:
    ms = np.empty(iters)
    for i in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        ms[i] = a.elapsed_time(b)
    return ms


def time_prefill(p: Prepared, warmup: int, iters: int) -> np.ndarray:
    def fn() -> None:
        with torch.no_grad():
            p.model(p.prompt, use_cache=False)

    for _ in range(warmup):
        fn()
    return _time(fn, iters)


def time_decode(p: Prepared, warmup: int, iters: int) -> np.ndarray:
    def fn() -> None:
        with torch.no_grad():
            p.model(p.next_token, past_key_values=p.past, use_cache=True)
        p.past.crop(p.ctx)  # keep the context length constant across iterations

    for _ in range(warmup):
        fn()
    return _time(fn, iters)


def gpu_state() -> dict[str, float]:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.mem,power.draw", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=False,
    ).stdout.strip().split(",")
    keys = ("temp_c", "sm_mhz", "mem_mhz", "power_w")
    return {k: float(v) for k, v in zip(keys, out)} if len(out) == 4 else {}


def summarize(samples: np.ndarray, n_boot: int = 4000, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    meds = np.array([np.median(rng.choice(samples, len(samples))) for _ in range(n_boot)])
    q1, med, q3 = np.percentile(samples, [25, 50, 75])
    return {
        "median_ms": float(med), "iqr_ms": [float(q1), float(q3)],
        "median_ci95_ms": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
        "min_ms": float(samples.min()), "n": len(samples),
    }
