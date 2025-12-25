import torch
import torch.nn as nn
import logging
from typing import Dict, List, Tuple

from .wrapper import ModelWrapper
from .quantizer import fake_quantize_tensor

logger = logging.getLogger(__name__)

class SensitivityProfiler:
    def __init__(self, wrapper: ModelWrapper, dataset: List[torch.Tensor]):
        self.wrapper = wrapper
        self.dataset = dataset
        self.choices = [2, 4, 8]
        
    def get_baseline_loss(self) -> float:
        total_loss = 0.0
        with torch.no_grad():
            for batch in self.dataset:
                total_loss += self.wrapper.forward_pass_check(batch)
        return total_loss / len(self.dataset)

    def profile(self) -> Tuple[Dict[Tuple[int, int], float], Dict[int, int], float]:
        """
        Returns:
            1. Sensitivity Map: {(layer_idx, bits): impact}
            2. Layer Sizes: {layer_idx: num_params}
            3. Baseline Loss: float
        """
        sensitivity = {}
        layer_param_counts = {}
        
        logger.info("Computing baseline FP16 loss...")
        baseline_loss = self.get_baseline_loss()
        logger.info(f"Baseline Loss: {baseline_loss:.4f}")
        
        layers = self.wrapper.model.model.layers
        
        for i, layer in enumerate(layers):
            logger.info(f"Profiling Layer {i}/{len(layers)-1}...")
            
            linear_modules = [m for m in layer.modules() if isinstance(m, nn.Linear)]
            
            # Count parameters that we are quantizing (weights only)
            param_count = sum(m.weight.numel() for m in linear_modules)
            layer_param_counts[i] = param_count
            
            original_weights = {m: m.weight.detach().cpu().clone() for m in linear_modules}
            
            for bits in self.choices:
                # Quantize
                for m in linear_modules:
                    w = m.weight.data
                    m.weight.data = fake_quantize_tensor(w, bits)
                
                # Measure
                total_loss = 0.0
                with torch.no_grad():
                    for batch in self.dataset:
                        total_loss += self.wrapper.forward_pass_check(batch)
                avg_loss = total_loss / len(self.dataset)
                
                # Record
                delta = max(0.0, avg_loss - baseline_loss)
                sensitivity[(i, bits)] = delta
                
                # Restore
                for m in linear_modules:
                    m.weight.data = original_weights[m].to(self.wrapper.device)
            
            # Final restore just in case
            for m in linear_modules:
                m.weight.data = original_weights[m].to(self.wrapper.device)
                
        return sensitivity, layer_param_counts, baseline_loss