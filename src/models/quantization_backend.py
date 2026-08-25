"""Real quantization backend interfaces and the bitsandbytes adapter."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

logger = logging.getLogger(__name__)


class BackendCapabilityError(ValueError):
    """Raised when a model or precision map is unsupported by a backend."""


@dataclass(frozen=True)
class BackendCapabilities:
    name: str
    supported_precisions: tuple[int, ...]
    packed_weights: bool
    group_size: int | None
    scale_dtype: str | None
    zero_point: str | None
    accumulation_dtype: str
    kernel: str
    quantizable_module: str


class QuantizationBackend(Protocol):
    name: str

    def capabilities(self, model: nn.Module) -> BackendCapabilities:
        """Return the backend's model and precision capabilities."""

    def quantize(
        self,
        model: nn.Module,
        precision_map: list[int],
        calibration_data: Any | None = None,
    ) -> nn.Module:
        """Apply a supported precision map and return the executable model."""

    def save(self, model: nn.Module, output_path: str | Path, metadata: dict[str, Any]) -> None:
        """Save backend metadata and model state."""

    def load(self, artifact_path: str | Path, device: str) -> nn.Module:
        """Load an executable model artifact."""


class BitsAndBytesBackend:
    """Layer-wise 4-bit and 8-bit Linear backend using bitsandbytes kernels."""

    name = "bitsandbytes"
    supported_precisions = (4, 8, 16)

    def __init__(self, device: str = "cuda", compute_dtype: torch.dtype = torch.float16):
        if not torch.cuda.is_available() and device.startswith("cuda"):
            raise BackendCapabilityError("bitsandbytes requires an available CUDA device")
        try:
            import bitsandbytes as bnb
        except ImportError as exc:
            raise BackendCapabilityError(
                "bitsandbytes is required for the bitsandbytes backend"
            ) from exc
        self.bnb = bnb
        self.device = device
        self.compute_dtype = compute_dtype

    def capabilities(self, model: nn.Module) -> BackendCapabilities:
        layers = _get_transformer_layers(model)
        if not layers:
            raise BackendCapabilityError("Model does not expose transformer layers")
        return BackendCapabilities(
            name=self.name,
            supported_precisions=self.supported_precisions,
            packed_weights=True,
            group_size=64,
            scale_dtype="float32",
            zero_point="nf4 implicit symmetric" if 4 in self.supported_precisions else None,
            accumulation_dtype=str(self.compute_dtype).replace("torch.", ""),
            kernel="bitsandbytes Linear4bit/Linear8bitLt CUDA kernels",
            quantizable_module="torch.nn.Linear within transformer layers",
        )

    def quantize(
        self,
        model: nn.Module,
        precision_map: list[int],
        calibration_data: Any | None = None,
    ) -> nn.Module:
        del calibration_data
        layers = _get_transformer_layers(model)
        if len(precision_map) != len(layers):
            raise BackendCapabilityError(
                f"Precision map length {len(precision_map)} does not match model layers {len(layers)}"
            )
        unsupported = sorted(set(precision_map).difference(self.supported_precisions))
        if unsupported:
            raise BackendCapabilityError(
                f"bitsandbytes backend does not support precisions: {unsupported}; "
                f"supported precisions are {self.supported_precisions}"
            )

        for layer_index, (layer, bits) in enumerate(zip(layers, precision_map)):
            if bits == 16:
                continue
            for module_name, module in list(layer.named_modules()):
                if not isinstance(module, nn.Linear):
                    continue
                parent, child_name = _parent_module(layer, module_name)
                replacement = self._replace_linear(module, bits)
                setattr(parent, child_name, replacement)
                logger.debug(
                    "Quantized layer %d module %s to %d-bit", layer_index, module_name, bits
                )
        model.to(self.device)
        model.eval()
        return model

    def save(self, model: nn.Module, output_path: str | Path, metadata: dict[str, Any]) -> None:
        target = Path(output_path)
        target.mkdir(parents=True, exist_ok=True)
        if not metadata.get("model_name"):
            raise ValueError("metadata.model_name is required to reload a backend artifact")
        (target / "backend.json").write_text(
            json.dumps(
                {
                    "backend": self.name,
                    "metadata": metadata,
                    "state_policy": "packed_state_is_retained_for_inspection; load rebuilds from base revision",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        torch.save(model.state_dict(), target / "model.safetensors.pt")

    def load(self, artifact_path: str | Path, device: str) -> nn.Module:
        target = Path(artifact_path)
        metadata_file = target / "backend.json"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Backend metadata not found at {metadata_file}")
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata.get("backend") != self.name:
            raise BackendCapabilityError("Artifact backend does not match bitsandbytes")
        model_metadata = metadata["metadata"]
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_metadata["model_name"],
            revision=model_metadata.get("model_revision"),
            dtype=torch.float16,
            device_map=device,
            low_cpu_mem_usage=True,
        )
        self.quantize(model, model_metadata["precision_map"])
        # bitsandbytes quantization auxiliaries are runtime state in this version.
        # Rebuilding from the immutable base revision preserves valid scales and packing.
        return model

    def _replace_linear(self, original: nn.Linear, bits: int) -> nn.Module:
        if bits == 4:
            replacement: nn.Module = self.bnb.nn.Linear4bit(
                original.in_features,
                original.out_features,
                bias=original.bias is not None,
                compute_dtype=self.compute_dtype,
                compress_statistics=True,
                quant_type="nf4",
            )
        elif bits == 8:
            replacement = self.bnb.nn.Linear8bitLt(
                original.in_features,
                original.out_features,
                bias=original.bias is not None,
                has_fp16_weights=False,
            )
        else:
            raise BackendCapabilityError(f"Unsupported replacement precision: {bits}")

        state = {"weight": original.weight.detach().to(device="cpu", dtype=torch.float16)}
        if original.bias is not None:
            state["bias"] = original.bias.detach().to(device="cpu", dtype=torch.float16)
        replacement.load_state_dict(state)
        return replacement.to(self.device)


def _get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    model_body = getattr(model, "model", None)
    layers = getattr(model_body, "layers", None)
    if not isinstance(layers, nn.ModuleList):
        raise BackendCapabilityError(
            "Expected a model.model.layers ModuleList for layer-wise quantization"
        )
    return layers


def _parent_module(root: nn.Module, module_name: str) -> tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]
