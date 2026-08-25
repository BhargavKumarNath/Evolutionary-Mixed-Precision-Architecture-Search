import json
import logging
import os
import sys

import hydra
from omegaconf import DictConfig

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.configuration import validate_config
from src.models.sensitivity import SensitivityProfiler
from src.models.wrapper import ModelWrapper
from src.utils.data import DatasetProtocol
from src.utils.manifest import build_manifest, write_manifest
from src.utils.reproducibility import seed_everything

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    logger.info("=== EMPAS: Sensitivity Profiling (Enriched) ===")
    resolved_cfg = validate_config(cfg)
    seed_everything(resolved_cfg["seed"], resolved_cfg.get("deterministic", False))
    model_id = resolved_cfg["model_name"]
    data_cfg = resolved_cfg["data"]
    protocol = DatasetProtocol(
        tokenizer_name=model_id,
        dataset=data_cfg["dataset"],
        split_config=data_cfg["protocol"],
        allow_synthetic_data=data_cfg["allow_synthetic_data"],
    ).load()
    dataset = list(protocol["calibration"])
    wrapper = ModelWrapper(model_id, device=resolved_cfg["target_device"])
    
    profiler = SensitivityProfiler(wrapper, dataset)
    # Unpack new return values
    sensitivity_table, layer_counts, baseline = profiler.profile()
    
    # Save Results
    output = {
        "model_name": model_id,
        "baseline_loss": baseline,
        # Convert integer keys to strings for JSON
        "layer_params": {str(k): v for k, v in layer_counts.items()},
        "sensitivity": {f"{k[0]}_{k[1]}": v for k, v in sensitivity_table.items()}
    }
    
    os.makedirs("./data/profiles", exist_ok=True)
    save_path = resolved_cfg["paths"]["profile_path"]
    
    with open(save_path, "w") as f:
        json.dump(output, f, indent=4)
        
    logger.info(f"Enriched profile saved to {save_path}")
    
    # Sanity Check
    l0_params = output["layer_params"]["0"]
    logger.info(f"Layer 0 Parameters: {l0_params:,}")
    logger.info("SUCCESS: Parameter counts captured.")
    write_manifest(
        resolved_cfg["paths"]["manifest_path"].replace("search_manifest", "profile_manifest"),
        build_manifest(
            resolved_cfg,
            {
                "profile_path": save_path,
                "dataset_protocol": {
                    name: {
                        "sample_ids": split.sample_ids,
                        "token_count": split.token_count,
                        "fingerprint": split.fingerprint,
                    }
                    for name, split in protocol.items()
                },
            },
        ),
    )

if __name__ == "__main__":
    main()