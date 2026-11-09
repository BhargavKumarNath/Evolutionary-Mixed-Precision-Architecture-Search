# EMPAS Phase 3 results (TinyLlama-1.1B, RTX 4070 Laptop 8 GB, bitsandbytes 0.50.1)

Protocol: `docs/phase3_protocol.md`, written before any result existed. Raw outputs sit next to this file.
All losses are mean next-token NLL in nats on WikiText-2, 512-token non-overlapping chunks.

| split | role | chunks | tokens |
|---|---|---|---|
| train (seeded subset) | sensitivity profile | 128 | 65,536 |
| validation (whole split) | proxy calibration, VRAM fit | 585 | 299,520 |
| test (whole split) | matched-budget comparison, evaluated once | 666 | 340,992 |

FP16 reference: validation NLL 2.3614 (ppl 10.61), test NLL 2.3417 (ppl 10.40).
Earlier numbers (loss 2.5714, ppl 13.08) came from 4 x 128 tokens and are not comparable.

## 1. Does the proxy track hardware? (30 genomes, validation split)

| proxy | subset | Pearson r (95% CI) | Spearman rho (95% CI) |
|---|---|---|---|
| real backend (NF4 / int8) | all 30 | 0.988 (0.975-0.997) | 0.988 (0.945-0.998) |
| real backend | 10 front genomes | 0.995 (0.991-1.000) | 1.000 |
| real backend | 20 non-front | 0.984 (0.961-0.997) | 0.977 |
| fake absmax quantizer (original method) | all 30 | 0.933 (0.862-0.972) | 0.907 (0.740-0.979) |

Slope of measured on predicted is 0.90 for the real-backend proxy (mean abs error 0.0016 nats).
The fake-quantizer proxy has slope 0.51 and treats int8 as free (summed single-layer cost 0.0004 vs 0.0265 on the real backend; measured uniform int8 costs 0.0188). The earlier r = 0.940 (n = 5) is superseded.
Caveat: the proxy sums single-layer effects, so it is calibrated where those effects are small. All 30 genomes lose under 0.05 nats.

## 2. VRAM model

Fit to measured memory of the 30 genomes. Selected by leave-one-out RMSE (form B beat form A: 2.97 vs 8.40 MiB on static weights).

| quantity | value | 95% bootstrap CI |
|---|---|---|
| intercept for peak memory (batch 1 x 512) | 307 MiB (old guess: 1024) | 304-312 |
| effective bits per 4-bit weight (NF4 + scales) | 4.33 | 4.29-4.37 |
| effective bits per 8-bit weight | 8.09 | 8.02-8.19 |
| leave-one-out RMSE, peak memory | 3.9 MiB | |

The old constant overestimated peak memory by 53% on average (RMSE 700 MiB).

## 3. Latency (warm-up 20, 100 timed iterations x 5 interleaved rounds, CUDA events, batch 1, context 512)

| config | prefill ms (median, 95% CI) | decode ms / token |
|---|---|---|
| FP16 | 40.5 (40.45-40.58) | 10.90 (10.90-10.90) |
| uniform 8-bit | 28.9 (28.87-28.98) | 22.44 (22.40-22.48) |
| uniform 4-bit | 49.9 (49.85-49.91) | 14.19 (14.16-14.22) |
| exact-DP genome, budget 6 | 41.7 | 17.65 |
| exact-DP genome, budget 8 | 36.1 | 19.07 |

The earlier "2-3x lower latency" was a cold FP16 run (486 ms) against warm quantized runs and does not hold.
With warm-up, decode is slower on every quantized configuration (1.3x for 4-bit, 2.1x for 8-bit): bitsandbytes dequantizes on the fly and TinyLlama decode is launch-bound.
Prefill is faster for int8 and slower for int4. GPU clocks moved between 2145 and 2520 MHz across rounds (laptop, not lockable), so absolute values carry that drift; FP16 anchors at start and end agree within 0.4%.
Proxy latency score (bits/16) vs measured decode latency over 30 genomes: Pearson -0.66 (CI -0.87 to -0.35). Fewer bits means slower decode here, so the score had the wrong sign and was removed from the search objectives (see Deviations in the protocol).

## 4. GA vs baselines at matched bit budget (test split, evaluated once)

Test delta-loss vs FP16, in millinats (1e-3 nats). GA and random rows are means over 10 seeds with a bootstrap 95% CI over seeds.
Random search gets the same number of unique proxy evaluations as the paired GA run (about 2,000).

| budget (avg bits <=) | uniform 8-bit | greedy | exact DP | random search | NSGA-II |
|---|---|---|---|---|---|
| 5 | n/a (uniform 4-bit: 50.6) | 42.2 | 35.0 | 36.2 (35.7-36.9) | 35.0 (35.0-35.0) |
| 6 | n/a | 28.1 | 22.8 | 26.0 (25.4-26.6) | 22.6 (22.3-22.9) |
| 7 | n/a | 17.0 | 15.1 | 20.2 (19.3-21.1) | 15.9 (15.5-16.5) |
| 8 | 18.0 | 9.8 | 9.1 | 15.1 (14.2-16.0) | 10.3 (10.0-10.7) |
| 10 | 18.0 | 4.6 | 3.6 | 8.8 (8.2-9.3) | 4.8 (4.6-5.0) |
| 12 | 18.0 | 1.7 | 2.3 | 4.9 (4.1-5.7) | 3.0 (2.5-3.4) |

Paired bootstrap over test chunks, NSGA-II minus other (negative favours NSGA-II), millinats:
- vs random search: better at all six budgets (-1.3 to -4.7, every CI excludes 0).
- vs greedy: better at 5, 6, 7 (-7.2, -5.5, -1.1), worse at 8 (+0.5) and 12 (+1.3), tie at 10 (+0.2, CI -0.1 to 0.5).
- vs exact DP: equal at 5, better at 6 (-0.24, CI -0.47 to -0.01), worse at 7, 8, 10, 12 (+0.8, +1.3, +1.2, +0.7).

Reading: mixed precision beats uniform at the same budget (at 8 bits, the DP genome loses 9.1 millinats against 18.0 for uniform int8). Among searchers, the exact DP on the additive proxy is the one to beat and the GA does not beat it, so the GA is not justified by this problem. The GA beats random search and is close to the proxy optimum. The DP is not best on hardware at budget 12 (greedy 1.7 vs DP 2.3): the proxy ignores layer interactions.
Front quality on the proxy (2 objectives): GA reaches 95.8% of the exact front's hypervolume (worst seed 94.7%), random search 80.3%.

## What these results do not show

- One model, one backend, one hardware platform. TinyLlama has 22 layers with equal parameter counts, so the budget is a pure weight-bit budget.
- The comparison is on WikiText-2 perplexity only. No downstream task accuracy was measured.
- Differences between GA, greedy and DP are 0.2 to 1.3 millinats (<0.06% relative loss); they are statistically resolved on 340,992 test tokens but small in practice.
- The latency result says nothing about other kernels (for example fused int4 kernels), larger batches, or larger models.
- The GA was only run with 10 seeds, one population setting, and pop 50 x 40 generations; no hyperparameter sweep.

## Deviations from the protocol

Both were made before any test-split data was used (full text in `docs/phase3_protocol.md`):
1. NSGA-II objectives reduced from (loss, memory, latency score) to (loss, memory), for the reason in section 3.
2. The 30-genome latency sweep is sequential rather than interleaved; FP16 anchors bound drift.

## Reproduce

```bash
uv run python experiments/phase3/p3_01_profile.py      # sensitivity profiles (~8 min)
uv run python experiments/phase3/p3_02_calibrate.py    # 30 genomes on validation (~14 min)
uv run python experiments/phase3/p3_02b_analyze.py     # correlations, VRAM fit
uv run python experiments/phase3/p3_03_latency.py      # latency (~25 min)
uv run python experiments/phase3/p3_04_compare.py      # test split, refuses to re-run without --force
uv run python experiments/phase3/p3_05_figures.py
```
