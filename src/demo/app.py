import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import numpy as np
from pathlib import Path

st.set_page_config(
    page_title="EMPAS: Evolutionary Mixed-Precision Architecture Search",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "EMPAS: Hardware-aware neural architecture search using genetic algorithms"
    }
)

# --- ROBUST PATH SETUP ---
CURRENT_DIR = Path(__file__).resolve().parent
# Get the repository root (EMPAS/...)
ROOT_DIR = CURRENT_DIR.parent.parent

# Add root to sys.path so we can import src.core
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

# Define paths relative to ROOT
CHECKPOINT_PATH = ROOT_DIR / "data/logs/search_tinyllama_wandb/checkpoint_gen_50.json"
ARTIFACTS_DIR = ROOT_DIR / "deployment/artifacts"
PROFILE_PATH = ROOT_DIR / "data/profiles/tinyllama_sensitivity.json"

# --- ENVIRONMENT DETECTION ---
MOCK_MODE = os.environ.get("EMPAS_MOCK_MODE", "False").lower() == "true"
GPU_AVAILABLE = False

if not MOCK_MODE:
    try:
        import torch
        # Check if we are on a CPU-only cloud instance
        if not torch.cuda.is_available():
            GPU_AVAILABLE = False
        else:
            GPU_AVAILABLE = True
            
        from src.core.validator import Validator
        from src.core.search_space import Genome
        from src.core.proxy_evaluator import ProxyEvaluator
        
        # Verify critical files exist
        if not PROFILE_PATH.exists():
            print(f"Warning: Profile not found at {PROFILE_PATH}. Switching to Mock Mode.")
            MOCK_MODE = True
            
    except ImportError as e:
        print(f"Import Error (Switching to Mock Mode): {e}")
        MOCK_MODE = True
    except Exception as e:
        print(f"Initialization Error (Switching to Mock Mode): {e}")
        MOCK_MODE = True

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
        margin: 0.5rem 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Generate mock data if needed
def generate_mock_search_data():
    """Generate realistic mock data for demo purposes"""
    np.random.seed(42)
    n_samples = 100
    
    # Generate realistic loss-VRAM trade-off
    vram_range = np.linspace(1400, 2100, n_samples)
    base_loss = 3.3 + (2100 - vram_range) / 700 * 0.9
    noise = np.random.normal(0, 0.05, n_samples)
    loss = base_loss + noise
    
    latency = (vram_range - 1400) / 700 * 35 + 20 + np.random.normal(0, 2, n_samples)
    
    records = []
    for i in range(n_samples):
        # Create fake genes just for visualization
        genes = [np.random.choice([2, 4, 8, 16], p=[0.2, 0.5, 0.2, 0.1]) for _ in range(22)]
        records.append({
            "id": i,
            "loss": loss[i],
            "vram": vram_range[i],
            "latency": latency[i],
            "genes": str(genes)
        })
    
    return pd.DataFrame(records)

def generate_mock_artifacts():
    """Generate mock artifacts for demo"""
    return {
        "max_accuracy": {
            "archetype": "max_accuracy",
            "metrics": {
                "predicted_loss": 3.3059,
                "predicted_vram_mb": 2032.0,
                "predicted_latency_score": 54.55
            },
            "config": {
                "quantization_map": {f"layer_{i}": int(np.random.choice([8, 16])) for i in range(22)}
            }
        },
        "balanced": {
            "archetype": "balanced",
            "metrics": {
                "predicted_loss": 3.3804,
                "predicted_vram_mb": 1528.0,
                "predicted_latency_score": 27.27
            },
            "config": {
                "quantization_map": {f"layer_{i}": int(np.random.choice([4, 8])) for i in range(22)}
            }
        },
        "max_compression": {
            "archetype": "max_compression",
            "metrics": {
                "predicted_loss": 4.2714,
                "predicted_vram_mb": 1423.0,
                "predicted_latency_score": 21.59
            },
            "config": {
                "quantization_map": {f"layer_{i}": int(np.random.choice([2, 4])) for i in range(22)}
            }
        }
    }

@st.cache_data
def load_search_data():
    """Load or generate search history data"""
    # Use path.exists() from pathlib
    if MOCK_MODE or not CHECKPOINT_PATH.exists():
        if not MOCK_MODE:
            st.warning(f"Checkpoint not found at {CHECKPOINT_PATH}. Switching to synthetic data.")
        return generate_mock_search_data()
    
    try:
        with open(CHECKPOINT_PATH, 'r') as f:
            data = json.load(f)
        
        # We need to import these inside the function to avoid top-level crashes
        from src.core.proxy_evaluator import ProxyEvaluator
        from src.core.search_space import Genome
        
        evaluator = ProxyEvaluator(str(PROFILE_PATH))
        
        records = []
        for i, genes in enumerate(data['population']):
            genome = Genome(genes=genes)
            fit = evaluator.evaluate(genome)
            records.append({
                "id": i,
                "loss": fit.validation_loss,
                "vram": fit.vram_peak_mb,
                "latency": fit.latency_ms,
                "genes": str(genes)
            })
        return pd.DataFrame(records)
    except Exception as e:
        st.error(f"Error loading search data: {e}")
        return generate_mock_search_data()

@st.cache_data
def load_artifacts():
    """Load or generate architecture artifacts"""
    if MOCK_MODE:
        return generate_mock_artifacts()
    
    artifacts = {}
    for name in ["max_accuracy", "balanced", "max_compression"]:
        path = ARTIFACTS_DIR / f"{name}.json"
        if path.exists():
            with open(path, 'r') as f:
                artifacts[name] = json.load(f)
        else:
            artifacts = generate_mock_artifacts()
            break
    return artifacts

def calculate_pareto_front(df):
    """Identify Pareto-optimal solutions"""
    pareto_mask = []
    # Simple inefficient pareto for small N
    for i, row in df.iterrows():
        dominated = False
        for j, other in df.iterrows():
            if i != j:
                if (other['loss'] <= row['loss'] and other['vram'] <= row['vram'] and 
                    (other['loss'] < row['loss'] or other['vram'] < row['vram'])):
                    dominated = True
                    break
        pareto_mask.append(not dominated)
    return pareto_mask

# Header
st.markdown('<p class="main-header">🧬 EMPAS: Evolutionary Mixed-Precision Architecture Search</p>', unsafe_allow_html=True)
st.markdown("**Hardware-Aware Neural Network Optimization via Genetic Algorithms**")

if MOCK_MODE:
    st.markdown("""
    <div class="warning-box">
        ⚠️ <strong>Demo Mode Active</strong><br>
        Running with synthetic data. 
        Possible reasons: Running on Cloud without GPU, missing data files, or Torch not installed.
    </div>
    """, unsafe_allow_html=True)

# Load data
artifacts = load_artifacts()
search_df = load_search_data()

# Sidebar
st.sidebar.title("🎛️ Configuration")
st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📖 About EMPAS

EMPAS uses **evolutionary algorithms** to discover optimal mixed-precision quantization strategies for Large Language Models.

**Key Objectives:**
- 🎯 Minimize model size (VRAM)
- 🚀 Maximize inference speed
- 📊 Preserve accuracy
""")

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Select Architecture")

selected_archetype = st.sidebar.radio(
    "Choose a pre-optimized configuration:",
    ["balanced", "max_accuracy", "max_compression", "Baseline (FP16)"],
    help="Each archetype optimizes for different constraints"
)

# Architecture descriptions
arch_descriptions = {
    "balanced": "⚖️ **Balanced**: Best trade-off between accuracy, speed, and memory",
    "max_accuracy": "🎯 **Max Accuracy**: Prioritizes model quality over compression",
    "max_compression": "📦 **Max Compression**: Smallest model size, fastest inference",
    "Baseline (FP16)": "📏 **Baseline**: Standard 16-bit precision (reference)"
}

st.sidebar.info(arch_descriptions[selected_archetype])

# Get configuration
if selected_archetype == "Baseline (FP16)":
    current_genes = [16] * 22
    metrics = {"predicted_loss": 3.28, "predicted_vram_mb": 2200, "predicted_latency_score": 60}
else:
    if selected_archetype in artifacts:
        data = artifacts[selected_archetype]
        qmap = data['config']['quantization_map']
        # Sort keys to ensure correct layer order
        sorted_keys = sorted(qmap.keys(), key=lambda x: int(x.split('_')[1]))
        current_genes = [qmap[k] for k in sorted_keys]
        metrics = data['metrics']
    else:
        st.error("Configuration not found")
        st.stop()

# Calculate derived metrics
avg_bits = np.mean(current_genes)
compression_ratio = 1 - (sum(current_genes) / (16 * 22))
memory_savings_mb = 2200 - metrics.get('predicted_vram_mb', 0)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Quick Stats")

col1, col2 = st.sidebar.columns(2)
col1.metric("Avg Bit-Width", f"{avg_bits:.1f}b")
col2.metric("Compression", f"{compression_ratio*100:.1f}%")

col3, col4 = st.sidebar.columns(2)
col3.metric("Memory Saved", f"{memory_savings_mb:.0f}MB")
col4.metric("Layers", "22")

# Main content tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Search Analysis", 
    "🔬 Architecture Inspector", 
    "📈 Performance Metrics",
    "💬 Inference Demo"
])

# TAB 1: Search Analysis
with tab1:
    st.header("Evolutionary Search Results")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Calculate Pareto front
        search_df['is_pareto'] = calculate_pareto_front(search_df)
        
        # Create interactive scatter plot
        fig = px.scatter(
            search_df,
            x="vram",
            y="loss",
            color="latency",
            size=[15 if p else 8 for p in search_df['is_pareto']],
            hover_data={"loss": ":.4f", "vram": ":.0f", "latency": ":.2f"},
            color_continuous_scale="Viridis",
            labels={
                "vram": "VRAM Usage (MB)",
                "loss": "Validation Loss",
                "latency": "Latency (ms)"
            },
            title="🎯 Pareto Frontier: Loss vs VRAM"
        )
        
        # Highlight selected architecture
        selected_vram = metrics.get('predicted_vram_mb', 0)
        selected_loss = metrics['predicted_loss']
        
        fig.add_scatter(
            x=[selected_vram],
            y=[selected_loss],
            mode='markers',
            marker=dict(size=20, color='red', symbol='star', line=dict(width=2, color='white')),
            name=f'Selected: {selected_archetype}',
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        st.subheader("🏆 Elite Architectures")
        pareto_solutions = search_df[search_df['is_pareto']].nsmallest(5, 'loss')
        
        for idx, row in pareto_solutions.iterrows():
            st.markdown(f"""
            <div class="metric-card">
                <strong>Architecture #{row['id']}</strong><br>
                Loss: {row['loss']:.4f} | VRAM: {row['vram']:.0f} MB
            </div>
            """, unsafe_allow_html=True)

# TAB 2: Architecture Inspector
with tab2:
    st.header("🔬 Quantization Architecture Analysis")
    col1, col2 = st.columns([3, 2])
    
    with col1:
        layer_df = pd.DataFrame({
            "Layer": [f"L{i}" for i in range(len(current_genes))],
            "Bit-Width": current_genes,
            "Layer_ID": list(range(len(current_genes)))
        })
        
        fig3 = px.bar(
            layer_df,
            x="Layer_ID",
            y="Bit-Width",
            color="Bit-Width",
            color_continuous_scale=[[0, '#e74c3c'], [0.33, '#f39c12'], [0.66, '#f1c40f'], [1, '#2ecc71']],
            title=f"📊 Layer-wise Quantization Map: {selected_archetype.upper()}"
        )
        fig3.add_hline(y=avg_bits, line_dash="dash", line_color="red", annotation_text=f"Avg: {avg_bits:.1f}b")
        st.plotly_chart(fig3, use_container_width=True)
    
    with col2:
        st.subheader("📋 Configuration Details")
        st.json({
            "archetype": selected_archetype,
            "avg_bits": f"{avg_bits:.2f}",
            "layers": len(current_genes)
        })

# TAB 3: Performance Metrics
with tab3:
    st.header("📈 Performance Analysis")
    
    # Comparison table code (simplified for robustness)
    comparison_data = []
    baseline_metrics = {"predicted_loss": 3.28, "predicted_vram_mb": 2200, "predicted_latency_score": 60}
    
    # Add artifacts
    for arch_name, arch_data in artifacts.items():
        m = arch_data['metrics']
        comparison_data.append({
            "Architecture": arch_name,
            "Loss": m['predicted_loss'],
            "VRAM (MB)": m['predicted_vram_mb'],
            "Latency": m['predicted_latency_score']
        })
        
    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, use_container_width=True)

# TAB 4: Inference Demo
with tab4:
    st.header("💬 Interactive Inference Playground")
    
    if not GPU_AVAILABLE:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>GPU Not Detected</strong><br>
        Using simulated inference output. To run actual inference, deploy this on a machine with a GPU (e.g., local PC, AWS g4dn).
        </div>
        """, unsafe_allow_html=True)
        
    prompt = st.text_area("Enter your prompt:", "The future of Artificial Intelligence is")
    
    if st.button("🚀 Generate"):
        with st.spinner("Generating..."):
            if GPU_AVAILABLE and not MOCK_MODE:
                try:
                    # Heavy imports only when needed
                    from src.core.validator import Validator
                    from src.core.search_space import Genome
                    
                    # ... (Load model code would go here)
                    # For safety in this demo fix, we still mock if model files are missing
                    raise NotImplementedError("Live inference requires model weights")
                except Exception as e:
                     st.warning(f"Could not run live inference: {e}")
                     st.markdown("### 📝 Generated Output (Simulated)")
                     st.code(prompt + " [Simulated completion: transforming the world via hardware-aware optimization...]")
            else:
                import time
                time.sleep(1.5)
                st.markdown("### 📝 Generated Output (Simulated)")
                st.code(prompt + " [Simulated completion: transforming the world via hardware-aware optimization...]")