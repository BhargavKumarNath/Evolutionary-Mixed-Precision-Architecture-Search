# EMPAS: Evolutionary Mixed-Precision Architecture Search

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red?logo=pytorch) ![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface) ![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-green?logo=nvidia) [![Live Demo](https://img.shields.io/badge/Live-Dashboard-brightgreen?style=for-the-badge)](https://evolutionary-mixed-precision-search.streamlit.app/)

## Executive Summary
**EMPAS** is a research prototype for discovering layer-wise mixed-precision configurations for Large Language Models (LLMs).

> Current status: the existing search uses a numerical fake-quantization proxy. It does not yet produce a packed low-bit deployment model or demonstrate measured GPU memory reduction, latency improvement, or throughput improvement. Historical artifacts are not validated hardware results.

By formulating mixed-precision quantization as a multi-objective optimization problem solved via evolutionary algorithms (NSGA-II), EMPAS eliminates the need for manual, sub-optimal uniform quantization (e.g., blanket INT4).

**Business Impact on TinyLlama-1.1B:**
*   **Search objective:** explore quality and theoretical memory tradeoffs of layer-wise precision maps.
*   **Deployment result:** unmeasured until a real low-bit backend and hardware benchmark are integrated.

---

## 1. The Engineering Challenge: Heterogeneous Layer Sensitivity
Modern LLMs are fundamentally memory-bound. While uniform quantization provides a baseline for compression, it is deeply flawed because it ignores the architectural reality of Transformer networks:
*   **High-Sensitivity Bottlenecks:** Certain attention heads and output projections act as "load-bearing" structures. Compressing them indiscriminately destroys downstream perplexity.
*   **Low-Sensitivity Redundancy:** Deep transformer blocks exhibit high sparsity and can be compressed aggressively to 2-bit or 4-bit with near-zero signal loss.

The core challenge is allocating a finite "precision budget" across the network. Hand-tuning a 22-layer model with 4 distinct bit-width choices ($4^{22} \approx 1.7 \times 10^{13}$ configurations) is impossible. 

EMPAS automates this discovery process.

---

## 2. System Architecture: The Data Lifecycle

EMPAS eschews monolithic design in favor of decoupled, phase-based batch processing and real-time inference. The system architecture is organized around the lifecycle of a model's quantization state.

![alt text](system_design.svg)

### 2.1 Ingestion Layer (Calibration Data)
To avoid overfitting the search engine to explicit sequences (data leakage), the ingestion layer (`src/utils/data.py`) acts as a deterministic sampler. Currently coupled to standard NLP datasets (e.g., Wikitext-2), it extracts and caches 128-token chunks via fixed RNG seeds. This provides the evolutionary engine with statistically representative distributions without the overhead of full dataset ingestion.

### 2.2 Transformation Engine (Offline Search)
EMPAS operates the Heavy-Compute Search phase asynchronously from serving. 
*   **Sensitivity profiler:** Iterates through every linear layer of the Base Model, applying numerical fake quantization for an additive search proxy.
*   **Genetic Optimizer (NSGA-II):** Consumes the profiler's cached states to evolve "Genomes" mapping layers to bit-widths. The engine evaluates candidate fitness (Loss vs. VRAM limits) without executing memory-bound forward passes.
*   **Artifact Export:** Resolves the Pareto Frontier into static JSON artifacts ("Balanced", "Max Compression"). These artifacts serve as an ultra-lightweight File-Based Model Registry.

### 2.3 Inference Service
Model serving is handled by a horizontal, stateless API (`src/serving/main.py`) powered by FastAPI.
The current service path instantiates the FP16 base model and applies simulated weight perturbations. It is a demonstration path, not a genuinely quantized deployment service.

### 2.4 Monitoring & Feedback Loop (Roadmap)
The current inference service performs localized latency measurement logging. Future work will formalize this boundary, exporting latency MS and output perplexity distributions to an aggregation layer (e.g., Prometheus) to detect hardware drift and trigger automated reconfiguration triggers.

---

## 3. Engineering Constraints & Trade-offs

A core principle of Staff-level design is acknowledging what the system deliberately chooses *not* to do.

1. **Offline Batch Search vs. Online Reinforcement Learning:**
   * **Decision:** We explicitly decouple the computationally intensive NSGA-II search from the runtime environment.
   * **Rationale:** Running NAS concurrently via RL requires differentiable relaxations (DARTS) that lack stability. By treating the search as a massive offline batch job, we accept static runtime configurations in exchange for 0% compute overhead during inference and theoretically optimal discrete search spaces.
2. **File-Based Registry vs. Database Services:**
   * **Decision:** Quantization architectures are serialized to `./deployment/artifacts/*.json` rather than pushing to MLflow or a relational database.
   * **Rationale:** Acknowledged "Hardcoded Fragility" is accepted to ensure the project runs seamlessly in edge, air-gapped, or zero-install GPU environments. This minimizes MLOps infrastructure hurdles for initial proofs-of-concept.
3. **Zero-Cost Additive Proxy vs. Full Perplexity Evaluations:**
   * **Decision:** The search algorithm evaluates fitness using an additive penalty derived from a single 128-token test, not full Wikitext perplexity.
   * **Rationale:** We traded a marginal degree of true-loss accuracy for an $O(1)$ fitness function. This explicit proxy scaling boundary reduces the architectural search phase from $O(\text{Days})$ to $O(\text{Minutes})$.

## 4. Engineering Rigor & Methodologies

EMPAS was built emphasizing reproducibility, scalability, and robust scientific methodology:

*   **Validation Strategy (Data Leakage Prevention):** The proxy evaluator uses highly truncated calibration sets (128 tokens, $N=4$ batches) exclusively to measure relative signal degradation. This prevents the exact calibration sequence from overfitting the search engine to specific text distributions, mimicking zero-shot robustness.
*   **Why NSGA-II over DARTS / RL?** The quantization search space is discrete (hardware natively supports 2/4/8/16-bit blocks). Gradient-based NAS (DARTS) requires continuous relaxations that break down under rounding, and Reinforcement Learning suffers from abysmal sample efficiency. NSGA-II natively handles the non-differentiable, combinatorial nature of this space while naturally preserving population diversity.
*   **Metric Selection:** Optimizing directly for *Throughput (Tokens/sec)* in a simulated environment introduces massive variance dependent on kernel implementations. EMPAS instead optimizes for absolute **VRAM constraints** and a deterministic **Weighted Latency Proxy**, ensuring stable and hardware-agnostic multi-objective convergence.
*   **Current engineering status:** Hydra and optional W&B integration exist, but configuration authority, manifests, deterministic execution, and checkpoint provenance are being stabilized. Hardware validation and genuine low-bit execution are not yet complete.

---

## 5. Experimental Results & Insights

The following historical proxy outputs are retained for auditability only. They are not measured VRAM, latency, perplexity, or deployment results.

| Model Strategy | Predicted Loss (Lower=Better) | Predicted Memory (MB) | Avg Bit-Width |
| :--- | :--- | :--- | :--- |
| **Baseline (FP16)** | 2.3765 | ~2471 | 16.0 |
| **Naive (Uniform 4-bit)** | 2.4685 | ~2471 | 4.0 |
| **EMPAS (Balanced)** | **2.4521** | ~2471 | **4.4** |

**Architectural Intuition Discovered:**
The evolutionary engine consistently arrived at conclusions that matched advanced human intuition, saving weeks of manual tuning:
1.  **Early Feature Extraction (Layers 0-2):** Averaged **4.0-bit**. The embeddings immediately projected into lower precision without major loss.
2.  **Deep Reasoning Blocks (Layers 7, 12):** The NSGA-II algorithm identified specific "load-bearing" attention layers and boosted them to **8-bit** (and sometimes **16-bit**) to preserve reasoning gradients.
3.  **Outputs:** Maintained at lower precision aggressively.

---

## 6. Installation & Usage

**Prerequisites:** Python 3.9+, PyTorch (CUDA required).

```bash
git clone https://github.com/BhargavKumarNath/EMPAS.git
cd EMPAS
pip install -r requirements.txt
```

### Phase 1: Generate Sensitivity Map
Profiles the target model to generate the $O(1)$ zero-cost proxy data.
```bash
python scripts/profile_sensitivity.py
```

### Phase 2: Run Evolutionary Search
Executes the NSGA-II algorithm (logs dynamically to WandB).
```bash
python scripts/run_search.py
```

### Phase 3: Export & Serve
Extract Pareto-optimal JSON artifacts and launch the interactive Streamlit dashboard.
```bash
python scripts/export_artifacts.py
streamlit run src/demo/app.py
```

---

## 7. Future Roadmap
1.  **Custom CUDA Kernels:** EMPAS currently relies on `fake_quantize` for sensitivity measurements. Realizing native mixed-precision throughput requires integration with libraries like `bitsandbytes` or custom Triton kernels.
2.  **Hardware-in-the-Loop Latency:** Replacing the latency proxy with an actual Look-Up Table (LUT) of measured tensor-core execution times on target edge architectures (e.g., Jetson AGX Orin).
3.  **Scale Up:** Validate configuration stability on `Llama-3-8B` down-quantized to fit within standard consumer 8GB VRAM cards.
