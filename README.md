# EMPAS: Evolutionary Mixed-Precision Architecture Search

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red?logo=pytorch) ![Transformers](https://img.shields.io/badge/HuggingFace-4.57.3-yellow?logo=huggingface) ![CUDA](https://img.shields.io/badge/NVIDIA-CUDA%2012.8-green?logo=nvidia) ![Tests](https://img.shields.io/badge/Tests-74/74-brightgreen) ![Status](https://img.shields.io/badge/Status-Stabilized%20at%20Phase%206-brightgreen)

## Executive Summary

**EMPAS** is a research platform for discovering layer-wise mixed-precision quantization configurations for Large Language Models using evolutionary algorithms (NSGA-II).

**Project status:** the original modernization plan for this codebase specified 14 phases. Phases 1-6 were executed, tested, and verified on real GPU hardware; the project was then deliberately concluded and stabilized at that point. Phases 7-14 (measured Pareto optimization, strong baselines, hardware-aware analysis reports, provenance schemas, CI, serving/dashboard hardening, and advanced optimization) were scoped but are **not implemented**; see [Out of Scope: Phases 7-14](#out-of-scope-phases-7-14) below.

### Evaluation results (Phase 3, 2026-11)

Full tables, protocol and deviations: [`results/phase3/SUMMARY.md`](results/phase3/SUMMARY.md) and [`docs/phase3_protocol.md`](docs/phase3_protocol.md). This section supersedes the earlier single-run numbers further down.

- **Proxy vs hardware:** rebuilt from the real bitsandbytes backend and checked on 30 genomes over the full validation split (299,520 tokens): Pearson r = 0.988 (95% CI 0.975-0.997), Spearman 0.988. The original fake-quantizer proxy gets r = 0.933 on the same genomes. The earlier r = 0.940 (n = 5) is superseded.
- **Same bit budget, held-out test split (340,992 tokens, evaluated once):** the exact DP optimum of the additive proxy loses 9.1 millinats vs FP16 at an 8-bit budget, uniform int8 loses 18.0. NSGA-II beats random search at every budget and is within about 1 millinat of the exact DP, but it does **not** beat the DP or greedy in general. For an additive proxy the DP solves the problem exactly; the GA is not justified by this problem alone.
- **VRAM:** the 1024 MB guess overestimated peak memory by 53%. Fitted intercept 307 MiB, 4.33 effective bits per 4-bit weight, leave-one-out RMSE 3.9 MiB.
- **Latency:** with warm-up and interleaved repeats, quantized decode is slower than FP16 (4-bit 14.2 ms vs 10.9 ms per token; 8-bit 22.4 ms). The earlier "2-3x lower latency" was a cold-start artefact and is withdrawn. The search's latency score had the wrong sign and was dropped from the objectives.

### What's Implemented

**Phases 1-6:** Complete reproducible workflow from configuration authority through real GPU execution, proxy calibration, and measurement caching.

| Phase | Capability | Status |
|:------|:-----------|:-------|
| **1** | Configuration authority, deterministic seeding, checkpoint/resume | Complete (8 tests) |
| **2** | Named dataset protocol (calibration/validation/test) with fingerprints | Complete (3 tests) |
| **3** | NSGA-II multi-objective evolutionary search | Complete |
| **4** | Real bitsandbytes backend (4-bit & 8-bit GPU execution) | Complete (5 tests) |
| **5** | Hardware benchmark harness (CUDA timing, memory measurement) | Complete (7 tests) |
| **6** | Proxy calibration against held-out validation data | Complete (16 tests) |
| **7-14** | Measured Pareto optimization, strong baselines, hardware-aware analysis, provenance schemas, CI, serving/dashboard hardening, advanced optimization | Not started (out of scope) |

### Key Achievements

- **74 unit tests**, all passing (39 original + 17 for NSGA-II + 18 for the Phase 3 evaluation tooling)
- **Real GPU execution** via bitsandbytes (4-bit, 8-bit, FP16 passthrough)
- **Hardware measurement** with CUDA synchronization, percentile statistics, memory tracking
- **Reproducible experiments** with configuration authority, RNG checkpointing, dataset fingerprints
- **Proxy calibration** against validation split with correlation metrics
- **Measurement caching** by genome/backend/hardware with automatic versioning

---

## Problem Statement

Modern LLMs are fundamentally memory-bound. Uniform quantization ignores architectural reality:

- **High-Sensitivity Layers:** Attention heads and output projections are load-bearing. Indiscriminate compression destroys downstream quality.
- **Low-Sensitivity Redundancy:** Deep transformer blocks tolerate aggressive compression (2-4 bit) with minimal signal loss.

**The Challenge:** Hand-tuning a 22-layer model with 3 bitwidth choices per layer (4/8/16-bit — the set the real quantization backend actually supports) yields $3^{22} \approx 3.1 \times 10^{10}$ configurations. EMPAS automates discovery via evolutionary optimization.

---

## System Architecture

```
Configuration Authority (Hydra)
        ↓
Deterministic Seeding + Dataset Protocol
        ↓
NSGA-II Evolutionary Search (fake quantization proxy)
        ↓
Pareto Frontier Candidates
        ↓
Real GPU Execution (bitsandbytes backend)
        ↓
Hardware Measurements (CUDA timing, memory)
        ↓
Proxy Calibration (correlation vs measured)
        ↓
Measurement Cache (versioned by genome/hardware/backend)
        ↓
Measured Pareto Frontier + Analysis Reports
```

---

## Project Structure

```
.
├── conf/                       # Hydra configuration authority
│   ├── config.yaml             # Root config (model, backend, seeds, paths)
│   ├── algorithm/              # Evolution algorithm configs
│   │   ├── ga.yaml             # Genetic algorithm parameters
│   │   └── nsga2.yaml          # NSGA-II settings
│   ├── search_space/           # Genome constraints
│   │   └── llama_mixed_precision.yaml  # [4, 8, 16] bit choices
│   └── tracking/               # Experiment tracking
│       └── wandb.yaml          # W&B integration
├── src/
│   ├── core/                   # Search and validation
│   │   ├── evaluator.py        # Fitness metrics (loss, memory, latency)
│   │   ├── proxy_evaluator.py  # Fake quantization proxy
│   │   ├── search_space.py     # Genome constraints
│   │   └── validator.py        # Model evaluation (simulation or bitsandbytes)
│   ├── engine/                 # Evolutionary algorithm (NSGA-II)
│   │   ├── ga.py               # NSGA-II implementation with checkpoint/resume
│   │   ├── operators.py        # Crossover, mutation operators
│   │   └── pareto.py           # Nondominated sorting, crowding distance
│   ├── models/                 # Quantization
│   │   ├── quantizer.py        # Fake quantization kernels
│   │   ├── quantization_backend.py  # Real bitsandbytes backend
│   │   ├── sensitivity.py      # Layer sensitivity profiling
│   │   └── wrapper.py          # Model abstraction
│   ├── benchmark/              # Hardware measurement (Phases 5-6)
│   │   ├── harness.py          # CUDA timing, memory measurement primitives
│   │   ├── cache.py            # Measurement persistence and cache keys
│   │   ├── proxy_calibration.py# Proxy vs measured correlation
│   │   └── proxy_validator.py  # Empirical proxy model
│   ├── hardware/               # GPU profiling
│   │   └── profiler.py         # CUDA device introspection
│   ├── serving/                # Inference API
│   │   └── main.py             # FastAPI HTTP endpoints
│   ├── utils/                  # Utilities
│   │   ├── data.py             # Dataset protocol with fingerprints
│   │   ├── logger.py           # Structured logging
│   │   ├── manifest.py         # Experiment provenance
│   │   └── reproducibility.py  # Deterministic seeding
│   └── demo/                   # Interactive dashboard
│       └── app.py              # Streamlit UI
├── scripts/
│   ├── run_search.py           # NSGA-II search entry point (Hydra)
│   ├── benchmark.py            # Hardware benchmark CLI (Phase 5-6)
│   ├── validate_candidate.py   # Genome evaluation utility
│   ├── profile_sensitivity.py  # Layer sensitivity analysis
│   └── verify_*.py             # Development verification scripts
├── tests/
│   ├── test_phase1.py          # Config, seeding, checkpointing (8 tests)
│   ├── test_phase2.py          # Dataset protocol (3 tests)
│   ├── test_phase3.py          # Nondominated sorting, crowding distance, GA (17 tests)
│   ├── test_phase4.py          # bitsandbytes backend (5 tests)
│   └── test_phase5_6.py        # Benchmark harness, proxy calibration (23 tests)
├── data/
│   ├── logs/                   # Search checkpoints and artifacts
│   ├── profiles/                # Sensitivity profiles
│   └── calibration/             # Proxy-vs-measured calibration reports
├── deployment/
│   └── artifacts/              # Exported quantization configs
│       ├── balanced.json
│       ├── max_accuracy.json
│       └── max_compression.json
└── outputs/                    # Timestamped experiment results
```

---

## Phase Details

### Phase 1: Configuration Authority & Reproducibility

**Objective:** Establish single source of truth for hyperparameters with deterministic execution.

**Key Components:**
- `src/utils/reproducibility.py`: Seed management across Python, NumPy, PyTorch, CUDA
- `src/utils/manifest.py`: Experiment provenance with config hash, git state, hardware ID
- `src/core/configuration.py`: Config validation and integrity checks

**Tests:** 8 passing (config validation, seed reproducibility, checkpoint/resume equivalence)

---

### Phase 2: Dataset Protocol with Fingerprints

**Objective:** Implement named, reproducible splits with SHA-256 fingerprints and overlap detection.

**Key Components:**
- `src/utils/data.py`: DatasetProtocol, DatasetSplit with immutable fingerprints
- Three-tier protocol: calibration (4 samples × 128 tokens), validation (4 × 128), test (4 × 128)
- Explicit overlap rejection across splits

**Tests:** 3 passing (fingerprint reproducibility, overlap rejection, seed sensitivity)

---

### Phase 3: NSGA-II Evolutionary Search

**Objective:** Genuine multi-objective optimization with nondominated sorting and crowding distance.

**Key Components:**
- `src/engine/ga.py`: NSGA-II with step(), run(), checkpoint/resume
- `src/engine/pareto.py`: O(N²) nondominated sorting
- `src/engine/operators.py`: Uniform crossover, random mutation
- Fitness caching to avoid redundant evaluation

**Algorithm (Deb et al., 2002 — implemented in full, not just named after it):**
1. Initialize random population
2. For each generation:
   - Evaluate parent fitness (loss, memory, latency)
   - `fast_nondominated_sort`: partition parents into ranked fronts (O(N²))
   - `crowding_distance`: assign per-front diversity distance (boundary points = infinite)
   - Crowded tournament selection (lower front rank wins; ties broken by larger crowding distance) to breed offspring
   - Uniform crossover (50% gene swap) + per-gene mutation
   - μ+λ environmental selection: combine parents+offspring, re-sort into fronts, fill the next generation front-by-front, trimming the last partial front by crowding distance
3. Checkpoint and resume from saved state

**Tests:** 17 passing (`tests/test_phase3.py` — dominance, multi-front non-dominated sorting, crowding-distance correctness including boundary/tie cases, crossover/mutation operators, and end-to-end GA population-size/validity/convergence behavior)

---

### Phase 4: Real bitsandbytes Backend

**Objective:** Replace fake quantization with genuine GPU execution.

**Key Components:**
- `src/models/quantization_backend.py`: BitsAndBytesBackend wrapper
- Automatic Linear → Linear4bit/Linear8bitLt replacement
- BackendCapabilities with metadata (precision support, group size, compute dtype)
- Immutable artifact persistence with base model versioning

**Tests:** 5 passing on RTX 4070 (4-bit execution, 8-bit execution, 2-bit rejection, artifact round-trip)

**Hardware Validation:**
- GPU: NVIDIA RTX 4070 (compute capability 8.9, 8GB VRAM)
- CUDA: 12.8, PyTorch 2.9.1, bitsandbytes 0.50.1
- Verified: Finite CUDA outputs, correct module replacement, artifact load/save

---

### Phase 5: Hardware Benchmark Harness

**Objective:** Measure actual GPU behavior with controlled, repeatable protocols.

**Key Components:**
- `src/benchmark/harness.py`: CUDA event timing, memory measurement primitives
- `BenchmarkResult`: Serializable dataclass with latency/memory/quality stats
- Measurement protocol:
  1. Clear cache, record baseline memory
  2. Load model, record peak allocated/reserved
  3. Warmups with synchronization
  4. Collect latency samples with CUDA events
  5. Compute percentiles (p50, p95, p99)

**Tests:** 7 passing (latency computation, memory tracking, percentile stats)

---

### Phase 6: Proxy Calibration & Measurement Cache

**Objective:** Calibrate proxy metrics against measured hardware behavior.

**Key Components:**
- `src/benchmark/cache.py`: MeasurementCache with versioned cache keys
- `src/benchmark/proxy_calibration.py`: ProxyCalibration, Pearson correlation reporting
- `src/benchmark/proxy_validator.py`: ProxyValidator, initial design-of-experiments
- Cache keys by: (genome, backend, hardware_id, batch_size, seq_len, protocol_version)

**Calibration Workflow:**
1. Measure initial diverse design (FP16, INT8, INT4, alternating patterns)
2. Collect proxy predictions (additive quality loss from bitwidth)
3. Compute measured metrics on validation split
4. Record Pearson correlations per objective
5. Cache all measurements for reuse

**Tests:** 16 passing (proxy calibration, cache uniqueness, workflow integration, schema separation)

**Executed for real** (`scripts/benchmark.py`, RTX 4070, bitsandbytes backend — not just unit-tested): measured all 5 design-of-experiments genomes (`uniform_fp16`, `uniform_int8`, `uniform_int4`, `alternating_4_8`, `alternating_8_16`) on real GPU hardware and computed the correlation between the additive-sensitivity proxy and measured behavior. Report: `data/calibration/tinyllama_calibration.json`; raw per-design measurements cached under `benchmark_cache/`.

| Correlation (proxy quality-loss delta vs. measured) | Pearson r |
|:---|:---:|
| → measured validation loss | **0.940** (n = 5, superseded by the 30-genome result above) |
| → measured peak memory | -0.789 |
| → measured latency | -0.282 |

Read carefully: this is 5 data points, not a statistically powered study, and the memory/latency correlations are *negative* because the proxy predicts higher quality loss exactly where fewer bits are used — which is also where measured memory is lower, so an inverse relationship there is expected, not a proxy failure. The **0.940 loss correlation is the number that matters**: it's the first real evidence that the additive sensitivity proxy driving the entire evolutionary search actually tracks measured model quality on this hardware. The weak latency correlation reflects the same single-run, unsynchronized timing caveat that applies everywhere else in this README (see Example Run) — the FP16 design's 486ms latency in this run includes CUDA/model warm-up noise from being the first design measured.

---

## Running the System

### Setup

Requires Python >= 3.10 (developed and verified on 3.12) and, for GPU-backed scripts, an NVIDIA GPU with CUDA available to PyTorch.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configuration

`conf/config.yaml` is the Hydra root config (composed with `conf/algorithm/*.yaml`, `conf/search_space/*.yaml`, and `conf/tracking/*.yaml`). Key fields:
```yaml
model_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

quantization:
  backend: "bitsandbytes"  # or "simulation" for numerical fake-quantization only

defaults:
  - algorithm: nsga2        # conf/algorithm/nsga2.yaml: pop_size, n_generations, ...
  - search_space: llama_mixed_precision
  - tracking: wandb         # tracking.enabled: true by default (syncs to W&B)
```
Override any value from the CLI, e.g. `algorithm.pop_size=8 algorithm.n_generations=5`, or `tracking.enabled=false` to run fully offline without a W&B account.

### Run Search

```bash
python scripts/run_search.py
```
Outputs:
- `outputs/{timestamp}/`: Hydra run directory (logs, resolved config)
- `<paths.search_output_dir>/checkpoint_gen_*.json`: per-generation checkpoints (resumable via `paths.resume_from`)
- `<paths.manifest_path>`: experiment provenance manifest

### Export Pareto Artifacts

```bash
python scripts/export_artifacts.py
```
Reads the latest search checkpoint and writes `deployment/artifacts/{balanced,max_accuracy,max_compression}.json`.

### Benchmark Hardware

```bash
python scripts/benchmark.py
```
Requires CUDA. Reports hardware identity and initializes the measurement cache (`benchmark_cache/` by default).

### Validate a Candidate

```bash
python scripts/validate_candidate.py "validation.genome=[4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4]"
```
The genome length must equal `search_space.num_layers` (22 for TinyLlama-1.1B).

### Serve an Artifact

```bash
python -m uvicorn src.serving.main:app --host 0.0.0.0 --port 8000
```
Loads `deployment/artifacts/balanced.json` on startup and exposes `GET /health` and `POST /generate`. This path still applies numerical fake-quantization to the base model (see [Design Principles](#design-principles)); it is not a demonstration of packed low-bit deployment.

### Launch Dashboard

```bash
streamlit run src/demo/app.py
```

---

## Example Run (Single Runs, TinyLlama-1.1B, RTX 4070 Laptop GPU)

These are actual outputs from the commands above on this hardware, not a systematic study: each is one run, and neither uses a synchronized/warmed-up latency protocol (that harness exists as of Phase 5 but isn't wired into these two legacy comparison paths). These two tables are the original single runs on 4 x 128 tokens; the loss/perplexity and latency figures in them are superseded by the Phase 3 results above (the latency claim in particular does not survive a warmed-up, repeated measurement).

**`validate_candidate.py`** (bitsandbytes backend, held-out validation split, genome `[4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4,8,16,4]`):

| Config | Mean Loss | Perplexity | Latency |
|:-------|:---------:|:----------:|:-------:|
| FP16 baseline (all 16-bit) | 2.5714 | 13.0838 | ~150-315ms |
| Evolved genome (4/8/16 mix) | 2.6097 | 13.5953 | ~60-155ms |

Model compression (theoretical weight-storage ratio, `1 - avg_bits/16`): **43.2%**.

**`benchmark_vs_baseline.py`** (numerical fake-quantization simulation path, not bitsandbytes; see [Design Principles](#design-principles)):

| Model | Loss (proxy) | VRAM (MB) | Throughput (tok/s) | Avg Bit-Width |
|:------|:------------:|:---------:|:-------------------:|:-------------:|
| Baseline (FP16) | 2.4161 | 2447 | ~3970 | 16 |
| Naive uniform 4-bit | 2.5016 | 2447 | ~7970 | 4 |
| EMPAS (balanced genome) | 2.4895 | 2447 | ~7290 | 4.4 |

VRAM is identical across rows here because this script's quantizer dequantizes back to full-precision tensors after simulating the numerical error (Design Principle 1); it demonstrates the quality tradeoff of the discovered precision map, not a memory reduction. Real memory savings require the bitsandbytes backend path (`validate_candidate.py`, `benchmark.py`), which Phase 5-6 measured but Phase 7 (measured Pareto frontier across baselines) never assembled into a comparison table like this one.

---

## Test Suite

**Total: 74 tests, all passing**

```bash
pytest tests/ -v

# Phase 1: Config & Reproducibility (8 tests)
pytest tests/test_phase1.py -v

# Phase 2: Dataset Protocol (3 tests)
pytest tests/test_phase2.py -v

# Phase 3: NSGA-II engine — sorting, crowding distance, operators, GA (17 tests)
pytest tests/test_phase3.py -v

# Phase 4: bitsandbytes Backend (5 tests, GPU required)
pytest tests/test_phase4.py -v

# Phase 5-6: Benchmark & Proxy (23 tests)
pytest tests/test_phase5_6.py -v
```

### Linting and Type Checking

```bash
ruff check .
mypy src/ scripts/
```

---

## Design Principles

### 1. Backend Abstraction

Quantization logic decoupled via `QuantizationBackend` protocol enables swapping fake for real execution:

```python
class QuantizationBackend(Protocol):
    def quantize(self, model: nn.Module, genome: list[int]) -> nn.Module: ...
    def save(self, model: nn.Module, path: Path) -> None: ...
    def load(self, path: Path) -> nn.Module: ...
    def capabilities(self) -> BackendCapabilities: ...
```

### 2. Immutable Artifact Persistence

Models saved with immutable references:
- Base model name and revision
- Precision map (layer → bitwidth)
- Backend metadata

On load, artifact rebuilds quantization state from immutable base, avoiding bitsandbytes runtime buffer serialization issues.

### 3. Dataset Fingerprinting

Each split fingerprinted by SHA-256(samples, ids, token_count):
- Ensures reproducibility
- Prevents silent data drift
- Enables provenance tracking

### 4. Predicted vs. Measured Separation

Schema explicitly separates cheap proxy from ground truth:

```python
@dataclass
class CalibrationPoint:
    proxy: ProxyMetrics  # predicted_quality_loss, model_version
    measured: MeasuredMetrics  # mean_loss, perplexity, memory, latency
```

Prevents accidental mixing of heuristics with measured data.

### 5. Measurement Cache by Hardware

Cache keys incorporate hardware fingerprint → measurements never reused across different GPUs or CUDA versions.

---

## Dependencies

**Core:**
- PyTorch 2.9.1 (CUDA 12.8)
- Transformers 4.57.3
- bitsandbytes 0.50.1
- Hydra 1.3.2
- Datasets 3.0+

**Development:**
- pytest 9.1.1
- ruff (linting)
- mypy (type checking)

See `requirements.txt` for complete list.

---

## Out of Scope: Phases 7-14

The original modernization plan for this codebase continued through Phase 14: measured Pareto optimization, strong baselines (FP16/INT8/INT4/published mixed-precision comparisons), hardware-aware analysis reports, experiment provenance schemas, CI, serving/dashboard hardening, and advanced optimization (learned surrogates, custom kernels, multi-GPU search). **That work was not started.** The project was deliberately stabilized and concluded at the end of Phase 6: the reproducible search, real GPU backend, benchmark harness, and proxy calibration all work and are tested, but as of the Phase 3 evaluation (2026-11) the repository does contain a matched-budget baseline comparison against uniform, random, greedy and exact-DP methods (`results/phase3/`), but still no measured (hardware-in-the-loop) Pareto frontier, no hardware-aware analysis report, no CI, and no comparison against published mixed-precision methods. Numbers in this README (loss, latency, compression) come from individual example runs of the existing scripts, not from a systematic measured study across baselines.

---

## Key Results Summary

**Status:** All 74 tests passing. Phases 1-6 complete; phases 7-14 out of scope (see above).

| Component | Status | Evidence |
|:----------|:-------|:---------|
| Configuration Authority | Done | config.yaml + Hydra validation + manifest provenance |
| Deterministic Seeding | Done | reproducibility tests (seed → RNG state equivalence) |
| Dataset Protocol | Done | SHA-256 fingerprints, overlap rejection, 3-way split |
| NSGA-II Search | Done | Nondominated sorting, crowding distance, tournament selection |
| Real GPU Backend | Done | bitsandbytes 4-bit/8-bit execution on RTX 4070 |
| Hardware Measurement | Done | CUDA event timing, memory tracking, percentile stats |
| Proxy Calibration | Done | Correlation reporting, schema separation, cache versioning — executed against real hardware (r=0.940 proxy↔measured loss, `data/calibration/tinyllama_calibration.json`) |

---

## Reproducibility & Validation

**Deterministic Guarantees:**
1. Configuration seeded via Hydra
2. Dataset splits fingerprinted by content
3. RNG state checkpointed at every generation
4. GPU execution synchronized via CUDA events
5. All artifacts include provenance manifest

**Verification:**
- Run twice with identical config → identical results
- Resume from checkpoint → numerically equivalent to uninterrupted run
- Proxy predictions reproducible from training candidates
- Cache keys change only when backend/hardware/shape changes

---

## Citation

```bibtex
@software{empas2025,
  title={EMPAS: Evolutionary Mixed-Precision Architecture Search},
  year={2025}
}
```

---

## License

MIT License. See LICENSE file for details.
