![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.9.1-red?logo=pytorch)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface)
![CUDA](https://img.shields.io/badge/NVIDIA-CUDA-green?logo=nvidia)
![NumPy](https://img.shields.io/badge/NumPy-Scientific-blue?logo=numpy)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

[🚀 Live Demo](https://evolutionary-mixed-precision-search.streamlit.app/)

# 1. Overview
EMPAS (Evolutionary Mixed-Precision Architecture Search) is a production grade Neural Architecture Search (NAS) framework that automatically discovers Pareto-optimal quantisation strategies for Large Language Models. By formulating mixed-precision quantisations as a multi-objective optimisation problem and solving it via evolutionary algorithms, EMPAS achieves $35-45\%$ **memory reduction** with minimal accuracy degradation (<0.08 perplexity increase) on TinyLlama-1.1B.

# 2. Mixed Precision Challange
Modern LLMs are predominantly memory-bound. While uniform quantization (e.g., INT4 or INT8 across all layers) provides a baseline for compression, it fails to account for the non-uniform sensitivity of transformer architectures. Specific layers often the first, the last, and specific attention projections exhibit high sensitivity to quantization noise, whereas others are highly redundant.

## Why Evolutionary Search?
1. **Discrete & Non-Differentiable Space:** Quantisation but widths (2, 3, 4, 8) are discrete. Gradient-based Neural Architecture Search (NAS) requires continuous relaxations (like Gumbel-Softmax) which often introduce optimisation instability and architectural collapses.

2. **Combinatorial Explosion:** For a 32-layer model with 4 bit width choices per layer, the search space is $4^{32} \approx 1.8 \times 10^{19}$. Brute-force evaluation is impossible.

3. **Multi_Objective Nature:** The goal is not a single model, but a Pareto Frontier of models representing the optimal trade-off between accuracy (Perplexity) and efficiency (Memory/Latency). Evolutionary algorithms like NSGA-II are natively designed to handle these non-convex frontiers.

# 3. Project Objectives
The system is designed to achieve the following measurable goals on a target baseline (TinyLlama-1.1B) and hardware (8GB VRAM Limit)
1. **Memory Reduction:** Reduce VRAM usage by $>30\%$ compared to FP16 baselines.
2. **Accuracy Preservation:** Maintain validation perplexity within $<2\%$ degradation of the baseline for the "Balanced" archetype.
3. Search Efficiency: Complete the architecture search in <5 minutes on a single GPU using zero-cost proxy evaluation (avoiding full fine-tuning loops).
4. Pareto Optimality: Output a set of non-dominated solutions, allowing MLOps engineers to select the optimal trade-off for specific SLA requirements (e.g., max accuracy vs. min latency)

# 4. System Design
The EMPAS pipeline operates in three distinct phases

![Evolutionary Fitness Curve](system_design.svg)

**Phase 1: Profiling & Search Space Definition**

EMPAS avoids "blind" searching. A one-shot **Sensitivity Profiler** calculates the Hessian-based degradation or PPL impact of quantising indicidual layers to $\{2, 3, 4, 8\}$ bits. This creates a Sensitivity Map that acts as a "Zero-Cost" proxy, allowing for $O(1)$ fitness estimates and informing the search engine which layers requiure higher precision.

**Phase 2: Evolutionary Search Loop**

The heart of the system is the **NSGA-II**. It evolves a population of "Genomes" (architectural configurations).

* **Engine:** Manages population diversity and applies Genetic Operations (Crossover/Mutation).
* **Evaluator:** A dynamic sub-system that wraps the base model in mixed-precision masks, calculates the loss/perplexity on a calibration set, and estimates resource costs (BitOps/VRAM)
* **Multi-Objective:** The `Selection Strategy` sorts candidates based on Pareto dominance, ensuring we optimise for both Accuracy and Compression.

**Phase 3: Selection & Serving**

Once the search converges, the **Pareto Frontier** is analysed to extract distinct archtypes (Max Accuracy, Balanced, Max Compression). These configurations are exported as lightweight JSON artifacts. The **Serving Engine** (FastAPI) then loads the base model and applies the chosen configuration at runtime to serve inference requests.

# 5. Algorithmic Methodology
## 5.1 Evolutionary Strategy
I have employed NSGA-II (Non-dominated Sorting Genetic Algorithm II) instead of Reinforcement Learning (RL) or differentiable NAS (DARTS).
* *Justification:* The search space is discrete and non-differentiable (but-widths cannot easily be relaxed to continuous space without significant proxy error). RL methods often suffer from high sample complexity. NSGA-II is robust for multi-objective problems where maintaining population diversity is important.

## 5.2 Genome Representation
An architecture is represented as a discrete integer vector $G \in \mathbb{Z}^L$, where $L$ is the number of linear layers in the model.

$G = [ g_0, g_1, \ldots, g_{L-1} ]$

Where each gene $g_i \in \{2, 4, 8, 16\}$ represents the bit-width of layer $i$.

## 5.3 Multi-Objective Fitness Function
I have minimised a vector of two objectives:

$G_{\min}\bigl(L_{\text{est}}(G), M(G)\bigr)$

### 1. Momory Proxy ($M$)
Calculated analytically based on parameter counts ($P_i$) per layer.

$M(G) = C_{\text{overhead}} + \sum_{i=0}^{L-1} P_i \cdot \frac{g_i}{8} \; [\text{MB}]$

### 2. Accuracy Proxy ($L_{\text{est}}$)

To avoid expensive forward passes during evolution, I have used an additive sensitivity model derived from the profilling phase:

$L_{\text{est}}(G) = L_{\text{base}} + \sum_{i=0}^{L-1} S(i, g_i)$

Where $S(i, b)$ is the pre-computed degradation of layer $i$ at bit-width $b$.

## 5.4 Genetic Operators
* Selection: Tournament Selection with size $k$ = 3. Comparisons use domination rank (lower is better) and crowding distance (higher is better) to preserve diversity.

* Crossover: Uniform Crossover probability $P_c = 0.9$. This allows independent mixing of layer decisions, preserving locality.

* Mutation: But-flip mutation with probability $P_m = 0.1$. A gene $g_i$ is replaced by a random choice from $\{2, 4, 8, 16\} \setminus \{g_i\}$

# 6. Technical Design Rationale
* **Layer-wise vs. Global:** Global quantization ignores the "bottleneck" effect of sensitive layers. Layer-wise mixed precision allows the search engine to "spend" the bit-budget where it matters most for signal retention.
* **Proxy Evaluation:** Full model evaluation on massive datasets in the bottleneck. EMPAS uses a "Proxy Evaluator" that computes Perplexity on a truncated validation set (e.g., 128-256 tokens) to provide a high-correlation signal for the Genetic Algorithmic (GA) engine.
* **Hydra-based Configuration:** Every aspect of the GA (population size, mutation rate, search space) is managed via Hydra, enabling rapid experimentation without code changes.

# 7. Experimental Results & Insights

* **Early Phase:** The GA quickly discards configurations that quantise early layers (Embeddings/initial blocks) to low bits, as these cause loss.
* **Mid Phase:** The engine discovers "sandwich" structures that is alternating high and low precision in middle layers to balance noise.
* **Late Phase:** Fine-tuning of the Pareto frontier occurs, where marginal gains in memory are traded for small stability improvements.

| Archetype       | Avg Precision | VRAM (MB) | Loss Proxy | Insight                                                                                  |
|-----------------|---------------|-----------|------------|------------------------------------------------------------------------------------------|
| **Baseline**        | 16.0-bit      | ~2200     | 0.886      | Reference                                                                                |
| **Max Accuracy**    | 7.8-bit       | ~2032     | 0.824      | **Counter intuitive:** 8-bit quantization acted as a regularizer, slightly improving PPL on the validation subset. |
| **Balanced**        | 4.6-bit       | ~1528     | 0.887      | **Optimal:** 30% smaller than baseline with identical accuracy.                              |
| **Max Compression** | 3.2-bit       | ~1423     | 1.201      | Diminishing returns. 2-bit layers degrade performance significantly.                     |

**Discovery:** The algorithm consistently learned to keep the first and last layers at higher precision (8/16-bit) where aggressively compressing middle transformer blocks to 4-bit. This aligns with recent literature on LLM quantization sensitivity.

```text
empas/
├── conf/                     # Hydra configuration files
│   ├── config.yaml           # Main entry point
│   ├── search_space/         # Layer definitions
│   └── algorithm/            # GA hyperparameters
├── src/
│   ├── core/
│   │   ├── search_space.py    # Genome & SearchSpace abstractions
│   │   └── proxy_evaluator.py # Zero-cost fitness evaluation
│   ├── engine/
│   │   ├── ga.py              # NSGA-II implementation
│   │   └── pareto.py          # Non-dominated sorting logic
│   ├── models/
│   │   ├── sensitivity.py     # Hessian/Loss profiling logic
│   │   └── quantizer.py       # FakeQuantization simulation
│   └── serving/               # FastAPI inference engine
├── scripts/
│   ├── profile_sensitivity.py # Step 1: Generate proxy data
│   ├── run_search.py          # Step 2: Run evolution
│   └── export_artifacts.py    # Step 3: Extract Pareto optimal models
└── deployment/                # Exported JSON artifacts
```

## 8. Installation and Usage
**Prerequisites:** Python 3.9+, PyTorch with CUDA support

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```
### 2. Phase 2: Profiling
Generate the sensitivity map for the target model.

```bash
python scripts/profile_sensitivity.py
```
### 3. Phase 3: Visualisation & Demo
Launch the Streamlit dashboard to explore the Pareto frontier and run inference.

```bash
streamlit run src/demo/app.py
```

### 9. Future Work
* **True Mixed-Precision Kernels:** Currently, EMPAS uses simulated quantization. Future work involves integrating `bitsandbytes` or writing custom CUDA kernels to realize the latency gains of mixed bit-width computations.
* **Latency-Aware Search:** Integrating a hardware lookup table for latency (measuring actual kernel execution time) rather than using bit-width as a proxy.
* **Large Scale Validation:** Applying EMPAS to Llama-3-70B to fit it onto dual-3090 setups (48GB VRAM), where mixed precision is essential.
