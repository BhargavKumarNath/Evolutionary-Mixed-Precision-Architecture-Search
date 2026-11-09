"""Step 2b: proxy-vs-measured statistics and VRAM fit from the saved calibration measurements."""

import json

import numpy as np
from common import cfg, dump, results_dir

from src.baselines.methods import AdditiveProfile, avg_bits
from src.stats.bootstrap import corr_with_ci
from src.stats.memory_model import bootstrap_params, features, fit_form


def main() -> None:
    c = cfg()
    out = results_dir()
    nb, seed = c["bootstrap"]["n_boot"], c["bootstrap"]["seed"]
    genomes = json.loads((out / "calibration_genomes.json").read_text())
    files = [np.load(out / "calib" / f"{k:02d}.npz") for k in range(len(genomes))]
    nll = np.stack([f["nll"] for f in files])  # (n_genomes, n_chunks)
    static = np.array([float(f["static_mb"]) for f in files])
    peak = np.array([float(f["peak_mb"]) for f in files])
    kinds = np.array([g["kind"] for g in genomes])
    names = [g["name"] for g in genomes]
    genes = [g["genes"] for g in genomes]
    fp16_idx = names.index("uniform_fp16")
    measured = nll.mean(axis=1) - nll[fp16_idx].mean()  # delta vs FP16, same chunks
    n_chunks = nll.shape[1]

    report: dict = {"n_genomes": len(genomes), "n_val_chunks": int(n_chunks), "n_val_tokens": int(n_chunks * c["seq_len"])}
    report["measured_fp16_nll"] = float(nll[fp16_idx].mean())
    report["measured_fp16_ppl"] = float(np.exp(nll[fp16_idx].mean()))

    # --- proxy vs measured (quality) ---
    subsets = {
        "all_30": np.ones(len(genomes), bool),
        "excluding_fp16": np.arange(len(genomes)) != fp16_idx,
        "non_front": kinds != "front",
        "front_only": kinds == "front",
    }
    report["quality"] = {}
    for variant in ("bnb", "fake"):
        prof = AdditiveProfile.from_json(out / f"sensitivity_{variant}.json")
        pred = np.array([prof.loss(g) for g in genes])
        per = {}
        for sname, mask in subsets.items():
            x, y = pred[mask], measured[mask]
            slope = float(np.polyfit(x, y, 1)[0]) if x.std() > 0 else float("nan")
            per[sname] = {
                "pearson": corr_with_ci(x, y, "pearson", nb, seed),
                "spearman": corr_with_ci(x, y, "spearman", nb, seed),
                "mae_nats": float(np.mean(np.abs(x - y))),
                "slope_measured_on_predicted": slope,
            }
        report["quality"][variant] = per
        report[f"points_{variant}"] = [
            {"name": n, "kind": k, "avg_bits": avg_bits(g), "predicted": float(p), "measured": float(m)}
            for n, k, g, p, m in zip(names, kinds, genes, pred, measured)
        ]

    # --- VRAM ---
    layer_params = json.loads((out / "sensitivity_bnb.json").read_text())["layer_params"]
    lp = [layer_params[str(i)] for i in range(len(layer_params))]
    x = features(genes, lp)
    fits = {f: fit_form(f, x, static) for f in ("A", "B")}
    chosen = min(fits, key=lambda f: fits[f].loo_rmse_mb)  # decided by LOO-CV RMSE (pre-registered)
    peak_fit = fit_form(chosen, x, peak)
    old_pred_static = 1024.0 + x @ np.array([4.0, 8.0, 16.0])
    report["vram"] = {
        "old_model_1024mb_constant": {
            "rmse_vs_static_mb": float(np.sqrt(np.mean((old_pred_static - static) ** 2))),
            "rmse_vs_peak_mb": float(np.sqrt(np.mean((old_pred_static - peak) ** 2))),
            "mean_overestimate_vs_peak_pct": float(np.mean((old_pred_static - peak) / peak) * 100),
        },
        "static": {f: vars(fits[f]) for f in fits},
        "chosen_form": chosen,
        "peak_fit": vars(peak_fit),
        "bootstrap_ci_static": bootstrap_params(chosen, x, static),
        "bootstrap_ci_peak": bootstrap_params(chosen, x, peak),
        "points": [
            {"name": n, "avg_bits": avg_bits(g), "static_mb": float(s), "peak_mb": float(p)}
            for n, g, s, p in zip(names, genes, static, peak)
        ],
    }
    dump(
        {"intercept_mb": peak_fit.intercept_mb, "effective_bits": {str(k): v for k, v in peak_fit.effective_bits.items()},
         "form": chosen, "target": "peak allocated MiB, batch 1 x 512 forward", "loo_rmse_mb": peak_fit.loo_rmse_mb},
        out / "memory_model.json",
    )
    dump(report, out / "calibration_report.json")
    q = report["quality"]["bnb"]["all_30"]
    print("bnb proxy  all_30  pearson", q["pearson"], "\n           spearman", q["spearman"])
    print("front_only pearson", report["quality"]["bnb"]["front_only"]["pearson"]["value"], "spearman", report["quality"]["bnb"]["front_only"]["spearman"]["value"])
    print("fake proxy all_30  pearson", report["quality"]["fake"]["all_30"]["pearson"]["value"])
    print("VRAM", {f: (fits[f].intercept_mb, fits[f].effective_bits, fits[f].loo_rmse_mb) for f in fits}, "chosen", chosen)


if __name__ == "__main__":
    main()
