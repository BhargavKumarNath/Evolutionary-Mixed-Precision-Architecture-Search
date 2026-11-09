"""Step 6: one reference search (seed 0, final settings), export archetypes, measure them, draw figures.

Archetype rules are unchanged from scripts/export_artifacts.py: max_accuracy = lowest predicted loss
on the front, max_compression = lowest predicted memory, balanced = front point closest to the
origin after min-max normalising loss and memory. Archetypes are measured on the VALIDATION split
(the test split was used once, in p3_04).
"""

import gc
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from common import ROOT, cfg, dump, quantized_model, results_dir, split
from p3_05_figures import ACC, BG, FG, GOLD, INK, MUTE, ROSE  # noqa: F401  (also applies the dark style)
from search_utils import run_nsga2
from src.baselines.methods import AdditiveProfile, avg_bits
from src.engine.pareto import fast_nondominated_sort
from src.eval.perplexity import chunk_nll
from src.stats.bootstrap import bootstrap_stat

N_LAYERS = 22


def measure(genes: list[int], chunks) -> dict:
    model = quantized_model(genes)
    torch.cuda.synchronize()
    static = torch.cuda.memory_allocated() / 2**20
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(chunks[:1].cuda())
    peak = torch.cuda.max_memory_allocated() / 2**20
    nll = chunk_nll(model, chunks, cfg()["eval_batch_size"]).numpy()
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return {"nll": nll, "static_mb": static, "peak_mb": peak}


def main() -> None:
    out = results_dir()
    c = cfg()["comparison"]
    prof_path, mem_path = str(out / "sensitivity_bnb.json"), str(out / "memory_model.json")
    prof = AdditiveProfile.from_json(prof_path)

    engine, n_unique = run_nsga2(
        prof_path, mem_path, seed=0, pop_size=c["pop_size"], n_generations=c["n_generations"],
        crossover_rate=c["crossover_rate"], mutation_rate=c["mutation_rate"],
        tournament_size=c["tournament_size"], max_avg_bitwidth=None, n_layers=N_LAYERS,
    )
    pop = engine.population
    fits = engine.evaluate_population(pop)
    front = fast_nondominated_sort(fits)[0]
    loss = np.array([prof.loss(g.genes) for g in pop])
    mem = np.array([f.peak_memory_mb for f in fits])
    f_loss, f_mem = loss[front], mem[front]
    nl = (f_loss - f_loss.min()) / (f_loss.max() - f_loss.min() + 1e-9)
    nm = (f_mem - f_mem.min()) / (f_mem.max() - f_mem.min() + 1e-9)
    pick = {
        "max_accuracy": front[int(np.argmin(f_loss))],
        "balanced": front[int(np.argmin(np.hypot(nl, nm)))],
        "max_compression": front[int(np.argmin(f_mem))],
    }

    chunks, _ = split("validation")
    ref = measure([16] * N_LAYERS, chunks)
    arche: dict = {"fp16": {"avg_bits": 16.0, "predicted_delta": 0.0, "predicted_peak_mb": None,
                            "measured_peak_mb": ref["peak_mb"], "measured_delta": 0.0, "delta_ci95": [0.0, 0.0]}}
    export_dir = ROOT / "deployment" / "artifacts"
    for name, idx in pick.items():
        g = pop[idx].genes
        m = measure(g, chunks)
        d = m["nll"] - ref["nll"]
        _, ci = bootstrap_stat(len(d), lambda i, d=d: float(d[i].mean()), 10_000, 0)
        arche[name] = {
            "genome": g, "avg_bits": avg_bits(g), "predicted_delta": float(loss[idx]),
            "predicted_peak_mb": float(mem[idx]), "measured_peak_mb": m["peak_mb"],
            "measured_delta": float(d.mean()), "delta_ci95": list(ci),
        }
        artifact = {
            "archetype": name,
            "metrics": {
                "predicted_loss": prof.baseline_loss + float(loss[idx]),
                "predicted_delta_loss": float(loss[idx]),
                "predicted_vram_mb": float(mem[idx]),
                "measured_validation_delta_loss": float(d.mean()),
                "measured_peak_vram_mb": m["peak_mb"],
            },
            "provenance": "results/phase3 reference search, seed 0, pop 50 x 40 generations, bnb proxy + fitted memory model",
            "config": {
                "base_model": cfg()["model_name"],
                "quantization_map": {f"layer_{i}": b for i, b in enumerate(g)},
            },
        }
        dump(artifact, export_dir / f"{name}.json")

    history = engine.history
    dump({"history": history, "front_size": len(front), "unique_evals": n_unique,
          "archetypes": arche, "fp16_val_nll": float(ref["nll"].mean())}, out / "archetypes.json")

    fig_dir = out / "figures"
    # --- 01 convergence ---
    gens = [h["gen"] for h in history]
    fig, ax = plt.subplots(1, 2, figsize=(9.2, 3.6))
    ax[0].plot(gens, [h["min_loss"] - prof.baseline_loss for h in history], c=GOLD, label="best in population")
    ax[0].plot(gens, [h["avg_loss"] - prof.baseline_loss for h in history], c=MUTE, label="mean of parents+offspring")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("generation")
    ax[0].set_ylabel("proxy Δloss (nats)")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(gens, [h["num_elites"] for h in history], c=ACC)
    ax[1].axhline(c["pop_size"], c="#6a6a6a", ls="--", lw=1)
    ax[1].set_xlabel("generation")
    ax[1].set_ylabel("rank-0 front size")
    fig.suptitle("NSGA-II reference run, seed 0 (pop 50, 40 generations)", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "p3_05_convergence.png")
    plt.close(fig)
    # --- 02 final population ---
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(mem, loss, c=MUTE, s=16, alpha=0.7, label="final population")
    order = np.argsort(f_mem)
    ax.plot(f_mem[order], f_loss[order], c=ACC, lw=1, marker="o", ms=4, label=f"rank-0 front ({len(front)})")
    for (name, idx), col in zip(pick.items(), (GOLD, INK, ROSE)):
        ax.scatter([mem[idx]], [loss[idx]], c=col, s=70, marker="D", zorder=5, label=name)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_xlabel("predicted peak memory (MiB)")
    ax.set_ylabel("proxy Δloss (nats)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Final population and archetypes", fontsize=10)
    fig.tight_layout()
    fig.savefig(fig_dir / "p3_06_pareto.png")
    plt.close(fig)
    # --- 04 archetype comparison (measured) ---
    names = list(arche)
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    cols = [MUTE, GOLD, INK, ROSE]
    ax[0].bar(names, [arche[n]["avg_bits"] for n in names], color=cols)
    ax[0].set_ylabel("average bit-width")
    lo = [arche[n]["measured_delta"] - arche[n]["delta_ci95"][0] for n in names]
    hi = [arche[n]["delta_ci95"][1] - arche[n]["measured_delta"] for n in names]
    ax[1].bar(names, [arche[n]["measured_delta"] for n in names], color=cols, yerr=[lo, hi], error_kw={"ecolor": FG, "lw": 0.8})
    ax[1].set_ylabel("measured Δloss, validation (nats)")
    ax[2].bar(names, [arche[n]["measured_peak_mb"] for n in names], color=cols)
    ax[2].set_ylabel("measured peak memory (MiB)")
    for a in ax:
        a.tick_params(axis="x", rotation=30)
        a.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "p3_08_archetypes.png")
    plt.close(fig)
    # --- 03 sensitivity (real backend, with 2 SE; fake quantizer overlaid for 8-bit) ---
    real = json.loads((out / "sensitivity_bnb.json").read_text())
    fake = json.loads((out / "sensitivity_fake.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), sharex=True)
    layers = np.arange(N_LAYERS)
    for a, bits in zip(ax, (4, 8)):
        v = np.array([real["sensitivity"][f"{i}_{bits}"] for i in layers])
        se = np.array([real["sensitivity_se"][f"{i}_{bits}"] for i in layers])
        a.bar(layers, v, yerr=2 * se, color=ACC, error_kw={"ecolor": FG, "lw": 0.8}, label="real backend (NF4/int8)")
        a.scatter(layers, [fake["sensitivity"][f"{i}_{bits}"] for i in layers], c=GOLD, s=14, zorder=4, label="fake quantizer")
        a.set_title(f"{bits}-bit Δloss per layer (±2 SE)", fontsize=10)
        a.set_xlabel("layer")
        a.set_ylabel("Δloss vs FP16 (nats)")
        a.grid(axis="x", visible=False)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "p3_07_sensitivity.png")
    plt.close(fig)
    print(json.dumps({k: {x: v[x] for x in ("avg_bits", "predicted_delta", "measured_delta", "delta_ci95", "measured_peak_mb")} for k, v in arche.items()}, indent=1))
    print("front size", len(front), "unique evals", n_unique)


if __name__ == "__main__":
    main()
