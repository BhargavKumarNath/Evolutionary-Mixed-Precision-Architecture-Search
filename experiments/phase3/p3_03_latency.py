"""Step 3: latency with warm-up, repeats, interleaving and a sweep over the 30 calibration genomes."""

import gc
import json

import numpy as np
import torch
from common import cfg, dump, quantized_model, results_dir, split

from src.baselines.methods import AdditiveProfile, avg_bits, dp_optimal
from src.benchmark.latency import gpu_state, prepare, summarize, time_decode, time_prefill
from src.stats.bootstrap import corr_with_ci


def main() -> None:
    c = cfg()
    out = results_dir()
    lat = c["latency"]
    chunks, _ = split("validation")
    prompt = chunks[:1]  # one fixed 512-token prompt
    prof = AdditiveProfile.from_json(out / "sensitivity_bnb.json")
    n = prof.n_layers
    configs = {
        "fp16": [16] * n,
        "int8": [8] * n,
        "int4": [4] * n,
        "dp_budget6": dp_optimal(prof, 6.0),
        "dp_budget8": dp_optimal(prof, 8.0),
    }
    print({k: round(avg_bits(v), 2) for k, v in configs.items()})
    models = {k: prepare(quantized_model(g), prompt) for k, g in configs.items()}
    print("resident MB", torch.cuda.memory_allocated() / 2**20)

    rng = np.random.default_rng(lat["seed"])
    pre = {k: [] for k in configs}
    dec = {k: [] for k in configs}
    round_log = []
    for r in range(lat["rounds"]):
        order = list(rng.permutation(list(configs)))
        for k in order:
            pre[k].append(time_prefill(models[k], lat["warmup"], lat["iters"]))
            dec[k].append(time_decode(models[k], lat["warmup"], lat["iters"]))
        round_log.append({"round": r, "order": order, "gpu": gpu_state()})
        print("round", r, order, round_log[-1]["gpu"], flush=True)

    result = {"protocol": lat, "ctx": int(prompt.shape[1]), "rounds": round_log, "configs": {}}
    for k, g in configs.items():
        p, d = np.concatenate(pre[k]), np.concatenate(dec[k])
        result["configs"][k] = {
            "genome": g, "avg_bits": avg_bits(g),
            "prefill": summarize(p), "decode": summarize(d),
            "prefill_round_medians_ms": [float(np.median(x)) for x in pre[k]],
            "decode_round_medians_ms": [float(np.median(x)) for x in dec[k]],
        }
    del models
    gc.collect()
    torch.cuda.empty_cache()

    # sweep: every calibration genome, sequential, FP16 drift anchors at start and end
    genomes = json.loads((out / "calibration_genomes.json").read_text())
    sweep = []

    def one(name: str, g: list[int]) -> dict:
        m = prepare(quantized_model(g), prompt)
        row = {
            "name": name, "avg_bits": avg_bits(g),
            "prefill_median_ms": float(np.median(time_prefill(m, lat["warmup"], 50))),
            "decode_median_ms": float(np.median(time_decode(m, lat["warmup"], 50))),
        }
        del m
        gc.collect()
        torch.cuda.empty_cache()
        return row

    anchor_start = one("fp16_anchor_start", [16] * n)
    for g in genomes:
        sweep.append({**one(g["name"], g["genes"]), "kind": g["kind"]})
        print(sweep[-1], flush=True)
    anchor_end = one("fp16_anchor_end", [16] * n)
    result["sweep"] = {"anchors": [anchor_start, anchor_end], "rows": sweep}
    for key in ("decode_median_ms", "prefill_median_ms"):
        x = np.array([r["avg_bits"] for r in sweep])  # proxy latency score is exactly avg_bits/16*100
        y = np.array([r[key] for r in sweep])
        result["sweep"][f"proxy_vs_{key}"] = {
            "pearson": corr_with_ci(x, y, "pearson", c["bootstrap"]["n_boot"], 0),
            "spearman": corr_with_ci(x, y, "spearman", c["bootstrap"]["n_boot"], 0),
        }
    dump(result, out / "latency.json")


if __name__ == "__main__":
    main()
