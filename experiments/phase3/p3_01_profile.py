"""Step 1: sensitivity profiles (fake and real bitsandbytes), signed deltas, 128 x 512 tokens."""

import logging
import time

import torch
from common import cfg, dump, load_fp16, results_dir, split

from src.models.layer_profile import profile_sensitivity

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")


def main() -> None:
    c = cfg()
    chunks, fp = split("calibration")
    print(f"calibration chunks {tuple(chunks.shape)} fingerprint {fp[:12]}")
    model = load_fp16()
    for mode in ("bnb", "fake"):
        t0 = time.time()
        prof = profile_sensitivity(model, chunks, mode, (4, 8), c["eval_batch_size"])
        prof["split"] = c["splits"]["calibration"]
        prof["fingerprint"] = fp
        prof["wall_seconds"] = time.time() - t0
        dump(prof, results_dir() / f"sensitivity_{mode}.json")
        print(mode, "baseline", prof["baseline_loss"], f"{prof['wall_seconds']:.0f}s")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
