import logging
import math
import time
from collections.abc import Iterator
from dataclasses import dataclass

import torch
from torch import nn

from ..models.quantizer import fake_quantize_tensor
from ..models.wrapper import ModelWrapper
from ..utils.data import get_calibration_dataset
from .search_space import Genome

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ValidationResult:
    mean_loss: float
    perplexity: float
    latency_ms: float
    throughput_tokens_per_sec: float

    @property
    def validation_loss(self) -> float:
        """Compatibility alias for callers using the legacy field name."""
        return self.mean_loss

    def __iter__(self) -> Iterator[float]:
        """Preserve the legacy ``loss, latency = validate()`` interface."""
        yield self.mean_loss
        yield self.latency_ms


class Validator:
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        seq_len: int = 2048,
        nsamples: int = 8,
        seed: int = 42,
        dataset_split: str = "test",
        allow_synthetic_data: bool = False,
    ):
        self.wrapper = ModelWrapper(model_name, device)
        self.dataset = get_calibration_dataset(
            model_name,
            seq_len=seq_len,
            nsamples=nsamples,
            seed=seed,
            split=dataset_split,
            allow_synthetic_data=allow_synthetic_data,
        )
    
    def apply_genome(self, genome: Genome):
        """Aplies the mixed precision configuration to the loaded model"""
        layers = self.wrapper.model.model.layers
        if len(genome.genes) != len(layers):
            raise ValueError(f"Genome length {len(genome.genes)} != Model layers {len(layers)}")
        
        logger.info("Applying Mixed-Precision Quantization...")

        for i, (layer, bits) in enumerate(zip(layers, genome.genes)):
            linear_modules = [m for m in layer.modules() if isinstance(m, nn.Linear)]

            for m in linear_modules:
                with torch.no_grad():
                    w = m.weight.data
                    m.weight.data = fake_quantize_tensor(w, bits)
        logger.info(f"Model quantized. Configuration: {genome.genes}")
    
    def validate(self) -> ValidationResult:
        """Evaluate loss and latency without conflating the two metrics."""
        logger.info("Starting Validation Loop...")
        self.wrapper.model.eval()
        total_loss_nats = 0.0
        total_tokens = 0
        total_time = 0.0

        with torch.no_grad():
            for batch in self.dataset:
                start = time.time()
                loss = self.wrapper.forward_pass_check(batch)
                end = time.time()

                tokens = batch.numel()
                total_loss_nats += loss * tokens
                total_tokens += tokens
                total_time += (end - start)
        if total_tokens == 0 or not self.dataset:
            raise ValueError("Validation dataset must contain at least one token")

        avg_loss = total_loss_nats / total_tokens
        avg_latency = (total_time / len(self.dataset)) * 1000
        throughput = total_tokens / total_time if total_time > 0 else math.inf

        return ValidationResult(
            mean_loss=avg_loss,
            perplexity=math.exp(avg_loss),
            latency_ms=avg_latency,
            throughput_tokens_per_sec=throughput,
        )
    