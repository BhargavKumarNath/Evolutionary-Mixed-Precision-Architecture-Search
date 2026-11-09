import json
import logging
import os

from .evaluator import Evaluator, FitnessMetrics
from .search_space import Genome

logger = logging.getLogger(__name__)

class ProxyEvaluator(Evaluator):
    def __init__(self, profile_path: str, memory_model_path: str | None = None,
                 latency_mode: str = 'bits_proxy'):
        if not os.path.exists(profile_path):
            raise FileNotFoundError(f"Profile not found at {profile_path}. Run profile_sensitivity.py first")
        
        with open(profile_path, 'r') as f:
            data = json.load(f)
        
        self.baseline_loss = data['baseline_loss']
        self.sensitivity = data['sensitivity']
        self.layer_params = data['layer_params']

        if latency_mode not in ('bits_proxy', 'none'):
            raise ValueError(f"latency_mode must be 'bits_proxy' or 'none', got {latency_mode!r}")
        self.latency_mode = latency_mode
        self.context_overhead_mb = 1024.0
        # effective bits stored per weight, by nominal bit-width. Defaults to the nominal width.
        self.effective_bits: dict[int, float] = {}
        if memory_model_path is not None:
            with open(memory_model_path, 'r') as f:
                mm = json.load(f)
            self.context_overhead_mb = float(mm['intercept_mb'])
            self.effective_bits = {int(k): float(v) for k, v in mm.get('effective_bits', {}).items()}

    def evaluate(self, genome: Genome) -> FitnessMetrics:
        """Predict metrics based on the genome configuration."""
        if not genome.genes:
            raise ValueError("Genome must contain at least one gene")
        predicted_loss = self.baseline_loss
        total_weight_size_bytes = 0.0
        weighted_bits = 0.0

        for i, bits in enumerate(genome.genes):
            # 1. Loss Prediction: Add sensitivity penalty
            # if bits == 16, penalty is 0
            key = f"{i}_{bits}"
            penalty = self.sensitivity.get(key, 0.0)
            predicted_loss += penalty

            # 2. VRAM Prediction
            # Params * (bits / 8) = bytes
            params = self.layer_params.get(str(i), 0)

            total_weight_size_bytes += params * (self.effective_bits.get(bits, float(bits)) / 8.0)

            # 3. Latency Proxy
            weighted_bits += bits
        
        # Convert to MB
        weight_mb = total_weight_size_bytes / (1024 ** 2)
        total_vram = weight_mb + self.context_overhead_mb

        max_bits = 16 * len(genome.genes)
        latency_proxy = (weighted_bits / max_bits) * 100.0

        return FitnessMetrics(
            mean_loss=predicted_loss,
            peak_memory_mb=total_vram,
            latency_ms=latency_proxy if self.latency_mode == 'bits_proxy' else 0.0
        )