import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from src.models.quantization_backend import (
    BackendCapabilityError,
    BitsAndBytesBackend,
)


class TinyBody(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Sequential(nn.Linear(128, 64, bias=False), nn.Linear(64, 128))]
        )


class TinyCausalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = TinyBody()

    def forward(self, inputs):
        for layer in self.model.layers:
            inputs = layer(inputs)
        return inputs


@unittest.skipUnless(torch.cuda.is_available(), "Phase 4 GPU tests require CUDA")
class BitsAndBytesBackendTests(unittest.TestCase):
    def test_capabilities_are_explicit(self):
        backend = BitsAndBytesBackend("cuda")
        capabilities = backend.capabilities(TinyCausalModel().cuda().half())
        self.assertEqual(capabilities.supported_precisions, (4, 8, 16))
        self.assertTrue(capabilities.packed_weights)
        self.assertIn("bitsandbytes", capabilities.kernel)

    def test_four_bit_map_executes_with_packed_weights(self):
        model = TinyCausalModel().cuda().half()
        backend = BitsAndBytesBackend("cuda")
        backend.quantize(model, [4])
        output = model(torch.randn(2, 128, device="cuda", dtype=torch.float16))
        quantized_weights = [
            module.weight
            for module in model.modules()
            if module.__class__.__name__ == "Linear4bit"
        ]
        self.assertTrue(quantized_weights)
        self.assertTrue(all(weight.dtype == torch.uint8 for weight in quantized_weights))
        self.assertTrue(torch.isfinite(output).all())

    def test_eight_bit_map_executes(self):
        model = TinyCausalModel().cuda().half()
        backend = BitsAndBytesBackend("cuda")
        backend.quantize(model, [8])
        output = model(torch.randn(2, 128, device="cuda", dtype=torch.float16))
        self.assertTrue(any(module.__class__.__name__ == "Linear8bitLt" for module in model.modules()))
        self.assertTrue(torch.isfinite(output).all())

    def test_unsupported_two_bit_map_is_rejected(self):
        backend = BitsAndBytesBackend("cuda")
        with self.assertRaises(BackendCapabilityError):
            backend.quantize(TinyCausalModel().cuda().half(), [2])

    def test_backend_artifact_metadata_round_trip(self):
        model = TinyCausalModel().cuda().half()
        backend = BitsAndBytesBackend("cuda")
        backend.quantize(model, [4])
        with tempfile.TemporaryDirectory() as directory:
            backend.save(
                model,
                directory,
                {"model_name": "fixture", "precision_map": [4]},
            )
            metadata = Path(directory, "backend.json").read_text()
            self.assertIn('"backend": "bitsandbytes"', metadata)
            self.assertTrue(Path(directory, "model.safetensors.pt").exists())


if __name__ == "__main__":
    unittest.main()
