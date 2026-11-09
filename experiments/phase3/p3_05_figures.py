"""Step 5: figures + a flat summary JSON for the website, built only from saved result files."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from common import results_dir

# Dark, site-matched style (same background and mono type as the existing EMPAS plots).
BG, FG = "#0a0a0a", "#d8d8d0"
INK, ACC, GOLD, ROSE, MUTE = "#e8e6d4", "#7fd4ff", "#ffc83d", "#ff7b7b", "#8a8a8a"
plt.rcParams.update({
    "font.size": 10, "font.family": "DejaVu Sans Mono", "figure.dpi": 140,
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "axes.edgecolor": "#3a3a3a", "axes.labelcolor": FG, "text.color": FG,
    "xtick.color": "#9a9a9a", "ytick.color": "#9a9a9a", "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True, "grid.color": "#161616", "legend.labelcolor": FG,
})


def main() -> None:
    out = results_dir()
    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    cal = json.loads((out / "calibration_report.json").read_text())
    cmp_ = json.loads((out / "comparison_test.json").read_text())
    lat = json.loads((out / "latency.json").read_text())

    # 1. proxy vs measured, 30 genomes, bnb proxy
    pts = cal["points_bnb"]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for kind, col, mk in (("doe", INK, "s"), ("random", ACC, "o"), ("front", GOLD, "D")):
        p = [q for q in pts if q["kind"] == kind]
        ax.scatter([q["predicted"] for q in p], [q["measured"] for q in p], c=col, marker=mk, s=34, label=kind, zorder=3)
    lim = max(max(q["predicted"], q["measured"]) for q in pts) * 1.05
    ax.plot([0, lim], [0, lim], c="#6a6a6a", lw=1, ls="--", zorder=1, label="y = x")
    r = cal["quality"]["bnb"]["all_30"]["pearson"]
    ax.set_title(f"Proxy vs measured Δloss, n=30\nr = {r['value']:.3f}  (95% CI {r['ci95'][0]:.3f}–{r['ci95'][1]:.3f})", fontsize=10)
    ax.set_xlabel("proxy predicted Δloss (nats)")
    ax.set_ylabel("measured Δloss on validation (nats)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "p3_01_proxy_vs_measured.png")
    plt.close(fig)

    # 2. matched-budget comparison on test
    budgets = sorted(cmp_["budgets"], key=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    x = np.arange(len(budgets))
    w = 0.2
    series = [("greedy", "greedy", MUTE), ("dp", "exact DP (proxy optimum)", INK), ("random", "random search", ROSE), ("ga", "NSGA-II", GOLD)]
    for j, (k, label, col) in enumerate(series):
        ys, errs = [], []
        for B in budgets:
            row = cmp_["budgets"][B][k]
            if "mean_test_nll" in row:
                ys.append(row["mean_test_nll"] - cmp_["fp16_nll"])
                errs.append([ys[-1] - (row["seed_ci95"][0] - cmp_["fp16_nll"]), (row["seed_ci95"][1] - cmp_["fp16_nll"]) - ys[-1]])
            else:
                ys.append(row["delta_vs_fp16"])
                errs.append([0, 0])
        ax.bar(x + (j - 1.5) * w, ys, w, color=col, label=label, yerr=np.array(errs).T, error_kw={"lw": 0.8, "ecolor": FG})
    ax.set_xticks(x, [f"≤{b} bits" for b in budgets])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("test Δloss vs FP16 (nats/token)")
    ax.set_title("Same bit budget, held-out test split", fontsize=10)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "p3_02_matched_budget.png")
    plt.close(fig)

    # 3. VRAM
    vr = cal["vram"]
    p = vr["points"]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    bits = [q["avg_bits"] for q in p]
    ax.scatter(bits, [q["peak_mb"] for q in p], c=ACC, s=30, label="measured peak (batch 1×512)", zorder=3)
    ax.scatter(bits, [q["static_mb"] for q in p], c=GOLD, s=22, marker="D", label="measured weights", zorder=3)
    pf = vr["peak_fit"]
    ax.set_xlabel("average bit-width")
    ax.set_ylabel("GPU memory (MiB)")
    ax.set_title(f"VRAM model: intercept {pf['intercept_mb']:.0f} MiB (was 1024), LOO RMSE {pf['loo_rmse_mb']:.1f} MiB", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "p3_03_vram.png")
    plt.close(fig)

    # 4. latency
    names = list(lat["configs"])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8))
    for ax, phase in zip(axes, ("prefill", "decode")):
        meds = [lat["configs"][n][phase]["median_ms"] for n in names]
        lo = [lat["configs"][n][phase]["median_ci95_ms"][0] for n in names]
        hi = [lat["configs"][n][phase]["median_ci95_ms"][1] for n in names]
        ax.bar(names, meds, color=[INK, ACC, GOLD, MUTE, ROSE][: len(names)], yerr=[np.array(meds) - lo, np.array(hi) - meds], error_kw={"lw": 0.8, "ecolor": FG})
        ax.set_title(f"{phase} (median, 95% CI)", fontsize=10)
        ax.set_ylabel("ms")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(fig_dir / "p3_04_latency.png")
    plt.close(fig)
    print("figures written to", fig_dir)


if __name__ == "__main__":
    main()
