"""Step 2: measure 30 genomes on the full validation split with the real backend.

The genome list is written to disk BEFORE any measurement (pre-registration of what is tested).
"""

import gc
import random
import time

import numpy as np
import torch
from common import cfg, dump, quantized_model, results_dir, split
from search_utils import run_nsga2

from src.baselines.methods import avg_bits
from src.core.search_space import MixedPrecisionSearchSpace
from src.engine.pareto import fast_nondominated_sort
from src.eval.perplexity import chunk_nll

DOE = {
    "uniform_fp16": lambda n: [16] * n,
    "uniform_int8": lambda n: [8] * n,
    "uniform_int4": lambda n: [4] * n,
    "alternating_4_8": lambda n: [4 if i % 2 == 0 else 8 for i in range(n)],
    "alternating_8_16": lambda n: [8 if i % 2 == 0 else 16 for i in range(n)],
}


def build_genomes() -> list[dict]:
    c = cfg()
    out_dir = results_dir()
    prof = str(out_dir / "sensitivity_bnb.json")
    n = 22
    genomes: list[dict] = [{"name": k, "kind": "doe", "genes": f(n)} for k, f in DOE.items()]
    seen = {tuple(g["genes"]) for g in genomes}

    gc_cfg = c["calibration_genomes"]
    rng = random.Random(gc_cfg["seed"])
    targets = np.linspace(4.5, 15.5, gc_cfg["n_random"])
    for k, t in enumerate(targets):
        space = MixedPrecisionSearchSpace(n, c["bits"], min_avg_bitwidth=t - 0.3, max_avg_bitwidth=t + 0.3)
        while True:
            g = space.sample_with(rng).genes
            if tuple(g) not in seen:
                break
        seen.add(tuple(g))
        genomes.append({"name": f"random_{k:02d}", "kind": "random", "genes": g})

    # preliminary proxy front (bnb profile, default memory model), spread evenly along it
    cmp_cfg = c["comparison"]
    engine, _ = run_nsga2(
        prof, None, seed=99, pop_size=cmp_cfg["pop_size"], n_generations=30,
        crossover_rate=cmp_cfg["crossover_rate"], mutation_rate=cmp_cfg["mutation_rate"],
        tournament_size=cmp_cfg["tournament_size"],
    )
    pop = engine.population
    fits = engine.evaluate_population(pop)
    front0 = fast_nondominated_sort(fits)[0]
    cands = sorted({tuple(pop[i].genes) for i in front0}, key=lambda g: avg_bits(list(g)))
    cands = [g for g in cands if g not in seen]
    idx = np.unique(np.linspace(0, len(cands) - 1, gc_cfg["n_front"]).round().astype(int))
    for j, i in enumerate(idx):
        genomes.append({"name": f"front_{j:02d}", "kind": "front", "genes": list(cands[i])})
    return genomes


def main() -> None:
    out_dir = results_dir()
    gfile = out_dir / "calibration_genomes.json"
    if gfile.exists():
        import json

        genomes = json.loads(gfile.read_text())
    else:
        genomes = build_genomes()
        dump(genomes, gfile)
    print(f"{len(genomes)} genomes: " + ", ".join(f"{k}={sum(g['kind']==k for g in genomes)}" for k in ("doe", "random", "front")))

    chunks, fp = split("validation")
    print(f"validation chunks {tuple(chunks.shape)} fingerprint {fp[:12]}")
    cache = out_dir / "calib"
    cache.mkdir(exist_ok=True)
    bs = cfg()["eval_batch_size"]
    for k, g in enumerate(genomes):
        path = cache / f"{k:02d}.npz"
        if path.exists():
            continue
        t0 = time.time()
        torch.cuda.empty_cache()
        model = quantized_model(g["genes"])
        torch.cuda.synchronize()
        static_mb = torch.cuda.memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            model(chunks[:1].cuda())  # batch 1 x seq_len forward
        torch.cuda.synchronize()
        peak_mb = torch.cuda.max_memory_allocated() / 2**20
        nll = chunk_nll(model, chunks, bs).numpy()
        np.savez(path, nll=nll, static_mb=static_mb, peak_mb=peak_mb, fingerprint=fp)
        print(f"[{k+1}/{len(genomes)}] {g['name']:<18} avg_bits={avg_bits(g['genes']):5.2f} nll={nll.mean():.4f} static={static_mb:.0f}MB peak={peak_mb:.0f}MB {time.time()-t0:.0f}s", flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
