"""Step 4: GA vs uniform / random / greedy / exact-DP at matched bit budgets, scored once on the test split.

Stage A (proxy only): every method picks its genome from the validation-calibrated bnb proxy.
          Picks are written to disk before any test data is touched.
Stage B: each unique picked genome is measured once on the full test split (real backend).
Stage C: statistics.

The script refuses to run again once comparison_test.json exists (held-out data is evaluated once).
"""

import argparse
import hashlib
import json
import random

import numpy as np
import torch
from common import cfg, dump, quantized_model, results_dir, split
from search_utils import additive, run_nsga2

from src.baselines.methods import (
    avg_bits,
    dp_optimal,
    exact_pareto_front,
    greedy_downgrade,
    hypervolume_2d,
    random_search,
    uniform,
)
from src.core.search_space import MixedPrecisionSearchSpace
from src.eval.perplexity import chunk_nll
from src.stats.bootstrap import bootstrap_stat, paired_diff_ci


def key(genes: list[int]) -> str:
    return hashlib.sha1(json.dumps(genes).encode()).hexdigest()[:16]


def stage_a(out) -> dict:
    c = cfg()["comparison"]
    prof_path = str(out / "sensitivity_bnb.json")
    mem_path = str(out / "memory_model.json")
    prof = additive(prof_path)
    n = prof.n_layers
    picks: dict = {"budgets": {}, "fronts": {}}
    for B in c["budgets"]:
        space = MixedPrecisionSearchSpace(n, [4, 8, 16], max_avg_bitwidth=float(B))
        entry: dict = {
            "dp": dp_optimal(prof, B),
            "greedy": greedy_downgrade(prof, B),
            "uniform4": uniform(n, 4),
            "ga": {}, "random": {}, "n_evals": {},
        }
        if B >= 8:
            entry["uniform8"] = uniform(n, 8)
        for s in c["seeds"]:
            engine, n_unique = run_nsga2(
                prof_path, mem_path, s, c["pop_size"], c["n_generations"],
                c["crossover_rate"], c["mutation_rate"], c["tournament_size"], float(B), n,
            )
            best = min(engine.population, key=lambda g: prof.loss(g.genes))
            entry["ga"][str(s)] = best.genes
            entry["n_evals"][str(s)] = n_unique
            entry["random"][str(s)] = random_search(space, prof, n_unique, random.Random(1000 + s))
        picks["budgets"][str(B)] = entry
        print("budget", B, "dp", round(prof.loss(entry["dp"]), 5), "greedy", round(prof.loss(entry["greedy"]), 5),
              "ga mean", round(float(np.mean([prof.loss(g) for g in entry["ga"].values()])), 5),
              "rand mean", round(float(np.mean([prof.loss(g) for g in entry["random"].values()])), 5), flush=True)

    # proxy-space Pareto fronts (avg bits vs predicted loss), unconstrained
    exact = exact_pareto_front(prof)
    ref = (16.0 + 1e-6, max(prof.loss([4] * n), max(y for _, y in exact)) * 1.05)
    picks["fronts"]["exact"] = {"points": exact, "hv": hypervolume_2d(exact, ref)}
    picks["fronts"]["ref"] = ref
    picks["fronts"]["ga"], picks["fronts"]["random"] = {}, {}
    space = MixedPrecisionSearchSpace(n, [4, 8, 16])
    for s in c["seeds"]:
        engine, n_unique = run_nsga2(
            prof_path, mem_path, s, c["pop_size"], c["n_generations"],
            c["crossover_rate"], c["mutation_rate"], c["tournament_size"], None, n,
        )
        # a front over every genome the GA ever evaluated (what a user could pick from)
        seen = [list(json.loads(k)) if k.startswith("[") else None for k in engine.fitness_cache]
        seen = [g for g in seen if g]
        pts = [(avg_bits(g), prof.loss(g)) for g in seen]
        nd = sorted(set(pts))
        front, run = [], float("inf")
        for x, y in nd:
            if y < run:
                front.append((x, y))
                run = y
        picks["fronts"]["ga"][str(s)] = {"hv": hypervolume_2d(front, ref), "n_evals": n_unique}
        rng = random.Random(2000 + s)
        rpts = [(avg_bits(g), prof.loss(g)) for g in (space.sample_with(rng).genes for _ in range(n_unique))]
        rfront, run = [], float("inf")
        for x, y in sorted(set(rpts)):
            if y < run:
                rfront.append((x, y))
                run = y
        picks["fronts"]["random"][str(s)] = {"hv": hypervolume_2d(rfront, ref)}
    return picks


def unique_genomes(picks: dict) -> dict[str, list[int]]:
    u: dict[str, list[int]] = {}
    for entry in picks["budgets"].values():
        for name, v in entry.items():
            if name == "n_evals":
                continue
            if isinstance(v, dict):
                for g in v.values():
                    u[key(g)] = g
            else:
                u[key(v)] = v
    u[key([16] * 22)] = [16] * 22
    return u


def stage_b(out, picks: dict) -> dict[str, np.ndarray]:
    chunks, fp = split("test")
    print(f"test chunks {tuple(chunks.shape)} fingerprint {fp[:12]}")
    cache = out / "test_cache"
    cache.mkdir(exist_ok=True)
    u = unique_genomes(picks)
    print(len(u), "unique genomes to evaluate on test")
    bs = cfg()["eval_batch_size"]
    for i, (k, g) in enumerate(u.items()):
        f = cache / f"{k}.npy"
        if f.exists():
            continue
        model = quantized_model(g)
        np.save(f, chunk_nll(model, chunks, bs).numpy())
        del model
        torch.cuda.empty_cache()
        print(f"[{i+1}/{len(u)}] {k} avg_bits={avg_bits(g):.2f} nll={np.load(f).mean():.4f}", flush=True)
    return {k: np.load(cache / f"{k}.npy") for k in u}, fp


def stage_c(picks: dict, nll: dict[str, np.ndarray], fp: str) -> dict:
    c = cfg()
    nb = c["bootstrap"]["n_boot"]
    prof = additive(str(results_dir() / "sensitivity_bnb.json"))
    fp16 = nll[key([16] * 22)]
    res: dict = {"test_fingerprint": fp, "n_test_chunks": len(fp16), "fp16_nll": float(fp16.mean()), "budgets": {}}
    for B, entry in picks["budgets"].items():
        row: dict = {}
        for name in ("dp", "greedy", "uniform4", "uniform8"):
            if name in entry:
                g = entry[name]
                v = nll[key(g)]
                row[name] = {"genome": g, "avg_bits": avg_bits(g), "proxy": prof.loss(g),
                             "test_nll": float(v.mean()), "delta_vs_fp16": float((v - fp16).mean())}
        for name in ("ga", "random"):
            gs = [entry[name][s] for s in sorted(entry[name], key=int)]
            per_seed = np.array([nll[key(g)].mean() for g in gs])
            chunk_mean = np.mean([nll[key(g)] for g in gs], axis=0)  # mean over seeds, per chunk
            _, ci = bootstrap_stat(len(per_seed), lambda i, p=per_seed: float(p[i].mean()), nb, 0)
            row[name] = {
                "per_seed_test_nll": per_seed.tolist(), "mean_test_nll": float(per_seed.mean()),
                "seed_ci95": list(ci), "sd_across_seeds": float(per_seed.std(ddof=1)),
                "mean_avg_bits": float(np.mean([avg_bits(g) for g in gs])),
                "mean_proxy": float(np.mean([prof.loss(g) for g in gs])),
                "n_distinct_genomes": len({key(g) for g in gs}),
                "mean_unique_evals": float(np.mean(list(entry["n_evals"].values()))),
                "delta_vs_fp16": float((chunk_mean - fp16).mean()),
            }
            row[f"_chunks_{name}"] = chunk_mean
        pair = lambda a, b: paired_diff_ci(a, b, nb, 0)
        ga_c, rs_c = row.pop("_chunks_ga"), row.pop("_chunks_random")
        gr_c, dp_c = nll[key(entry["greedy"])], nll[key(entry["dp"])]
        row["paired_test_nll_differences"] = {
            "ga_minus_greedy": pair(ga_c, gr_c), "ga_minus_dp": pair(ga_c, dp_c),
            "ga_minus_random": pair(ga_c, rs_c), "greedy_minus_dp": pair(gr_c, dp_c),
            "random_minus_dp": pair(rs_c, dp_c),
        }
        res["budgets"][B] = row
    ga_hv = [v["hv"] for v in picks["fronts"]["ga"].values()]
    rs_hv = [v["hv"] for v in picks["fronts"]["random"].values()]
    ex = picks["fronts"]["exact"]["hv"]
    res["proxy_front"] = {
        "exact_hv": ex, "ga_hv_mean_ratio": float(np.mean(ga_hv) / ex), "ga_hv_min_ratio": float(np.min(ga_hv) / ex),
        "random_hv_mean_ratio": float(np.mean(rs_hv) / ex), "random_hv_min_ratio": float(np.min(rs_hv) / ex),
    }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out = results_dir()
    final = out / "comparison_test.json"
    if final.exists() and not args.force:
        raise SystemExit("comparison_test.json exists: the test split is evaluated once. Use --force (it will be logged).")
    pfile = out / "comparison_picks.json"
    if pfile.exists():
        picks = json.loads(pfile.read_text())
    else:
        picks = stage_a(out)
        dump(picks, pfile)
    nll, fp = stage_b(out, picks)
    res = stage_c(picks, nll, fp)
    res["forced_rerun"] = bool(args.force)
    dump(res, final)
    np.savez_compressed(out / "test_nll_per_chunk.npz", **nll)
    print(json.dumps({B: {m: r[m]["mean_test_nll"] if "mean_test_nll" in r[m] else r[m]["test_nll"] for m in r if m in ("dp", "greedy", "ga", "random", "uniform4", "uniform8")} for B, r in res["budgets"].items()}, indent=1))


if __name__ == "__main__":
    main()
