"""Shared plumbing for the Phase 3 experiment scripts."""

import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.utils.chunks import load_chunks


def cfg() -> dict:
    return yaml.safe_load((ROOT / "conf" / "phase3.yaml").read_text())


def results_dir() -> Path:
    d = ROOT / cfg()["results_dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def split(name: str) -> tuple[torch.Tensor, str]:
    c = cfg()
    s = c["splits"][name]
    return load_chunks(c["model_name"], s["split"], c["seq_len"], s.get("max_chunks"), s.get("seed", 0))


def load_fp16(device: str = "cuda"):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        cfg()["model_name"], dtype=torch.float16, device_map=device, low_cpu_mem_usage=True
    )
    m.eval()
    return m


def quantized_model(genome: list[int], device: str = "cuda"):
    from src.models.quantization_backend import BitsAndBytesBackend

    model = load_fp16(device)
    BitsAndBytesBackend(device).quantize(model, genome)
    return model


def dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")
