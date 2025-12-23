import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

class ModelWrapper:
    def __init__(self, model_name: str, device: str = "cuda"):
        self.device = device
        self.model_name = model_name

        print(f"Loading model: {model_name}...")
        self.config = AutoConfig.from_pretrained(model_name)

        # Load model in FP16 to save memory immediately
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.float16, device_map=device, low_cpu_mem_usage=True
        )
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def get_layer_count(self) -> int:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return len(self.model.model.layers)
        else:
            raise ValueError(f"Unknown model structure for {self.model_name}")
    
    def get_memory_footprint(self) -> float:
        """Returns VRAM usage in MB"""
        return self.model.get_memory_footprint() / 1024**2
    
    def forward_pass_check(self, input_ids: torch.Tensor):
        """Run a quick forward pass to verify functionality"""
        with torch.no_grad():
            input_ids = input_ids.to(self.device)
            output = self.model(input_ids, labels=input_ids)
            return output.loss.item()
        