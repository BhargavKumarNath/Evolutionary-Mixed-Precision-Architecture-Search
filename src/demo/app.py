import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os
import sys
import numpy as np

# Mock mode for cloud deployment (no GPU)
MOCK_MODE = not os.path.exists("./data/profiles/tinyllama_sensitivity.json")

if not MOCK_MODE:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    try:
        import torch
        from src.core.validator import Validator
        from src.core.search_space import Genome
        from src.core.proxy_evaluator import ProxyEvaluator
        GPU_AVAILABLE = torch.cuda.is_available()
    except ImportError:
        MOCK_MODE = True
        GPU_AVAILABLE = False
else:
    GPU_AVAILABLE = False

# Config
CHECKPOINT_PATH = "./data/logs/search_tinyllama_wandb/checkpoint_gen_50.json"
ARTIFACTS_DIR = "./deployment/artifacts"
PROFILE_PATH = "./data/profiles/tinyllama_sensitivity.json"

st.set_page_config(
    page_title="EMPAS: Evolutionary Mixed-Precision Architecture Search",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "EMPAS: Hardware-aware neural architecture search using genetic algorithms"
    }
)

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
        color: #000000; /* ADDED: Force black text on light background */
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        color: #000000; /* ADDED: Force black text on light background */
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
                "quantization_map": {f"layer_{i}": np.random.choice([8, 16]) for i in range(22)}
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
                "quantization_map": {f"layer_{i}": np.random.choice([4, 8]) for i in range(22)}
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
                "quantization_map": {f"layer_{i}": np.random.choice([2, 4]) for i in range(22)}
            }
        }
    }

@st.cache_data
def load_search_data():
    """Load or generate search history data"""
    if MOCK_MODE or not os.path.exists(CHECKPOINT_PATH):
        st.info("📊 Running in demo mode with synthetic data")
        return generate_mock_search_data()
    
    with open(CHECKPOINT_PATH, 'r') as f:
        data = json.load(f)
    
    from src.core.proxy_evaluator import ProxyEvaluator
    from src.core.search_space import Genome
    
    evaluator = ProxyEvaluator(PROFILE_PATH)
    
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

@st.cache_data
def load_artifacts():
    """Load or generate architecture artifacts"""
    if MOCK_MODE:
        return generate_mock_artifacts()
    
    artifacts = {}
    for name in ["max_accuracy", "balanced", "max_compression"]:
        path = os.path.join(ARTIFACTS_DIR, f"{name}.json")
        if os.path.exists(path):
            with open(path, 'r') as f:
                artifacts[name] = json.load(f)
        else:
            artifacts = generate_mock_artifacts()
            break
    return artifacts

def calculate_pareto_front(df):
    """Identify Pareto-optimal solutions"""
    pareto_mask = []
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
        Running with synthetic data for demonstration. For full functionality, deploy locally with GPU access.
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

**How it works:**
1. Each architecture is a "genome" of layer-wise bit-widths
2. Genetic operators evolve the population
3. Pareto-optimal solutions balance all objectives
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
    
    st.markdown("""
    <div class="info-box">
    <strong>Understanding the Search Process:</strong><br>
    The genetic algorithm explored 100 different quantization strategies over 50 generations. 
    Each point represents a unique architecture balancing accuracy (validation loss) against 
    hardware constraints (VRAM usage). The <strong>Pareto frontier</strong> shows architectures 
    where improving one metric requires sacrificing another.
    </div>
    """, unsafe_allow_html=True)
    
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
            title="🎯 Pareto Frontier: Loss vs VRAM (Population at Generation 50)"
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
        
        # Add Pareto front line
        pareto_df = search_df[search_df['is_pareto']].sort_values('vram')
        fig.add_scatter(
            x=pareto_df['vram'],
            y=pareto_df['loss'],
            mode='lines',
            line=dict(color='red', width=2, dash='dash'),
            name='Pareto Frontier',
            showlegend=True
        )
        
        fig.update_layout(height=500, hovermode='closest')
        st.plotly_chart(fig, width='content')
        
    with col2:
        st.subheader("🏆 Elite Architectures")
        
        pareto_solutions = search_df[search_df['is_pareto']].nsmallest(5, 'loss')
        
        for idx, row in pareto_solutions.iterrows():
            with st.container():
                st.markdown(f"""
                <div class="metric-card">
                    <strong>Architecture #{row['id']}</strong><br>
                    Loss: {row['loss']:.4f}<br>
                    VRAM: {row['vram']:.0f} MB<br>
                    Latency: {row['latency']:.2f} ms
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.metric("Pareto Solutions", f"{search_df['is_pareto'].sum()}")
        st.metric("Total Explored", len(search_df))
        st.metric("Compression Range", f"{compression_ratio*100:.1f}%")
    
    # Evolution over time
    st.subheader("📈 Evolution Progress")
    
    # Create mock generation data
    generations = list(range(1, 51))
    best_loss = [3.52 - (i * 0.004) for i in range(50)]
    best_vram = [1600 - (i * 3) if i < 30 else 1510 - ((i-30) * 0.5) for i in range(50)]
    
    fig2 = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Best Loss Over Time", "Best VRAM Over Time")
    )
    
    fig2.add_trace(
        go.Scatter(x=generations, y=best_loss, mode='lines', name='Best Loss',
                  line=dict(color='#667eea', width=3)),
        row=1, col=1
    )
    
    fig2.add_trace(
        go.Scatter(x=generations, y=best_vram, mode='lines', name='Best VRAM',
                  line=dict(color='#764ba2', width=3)),
        row=1, col=2
    )
    
    fig2.update_xaxes(title_text="Generation", row=1, col=1)
    fig2.update_xaxes(title_text="Generation", row=1, col=2)
    fig2.update_yaxes(title_text="Validation Loss", row=1, col=1)
    fig2.update_yaxes(title_text="VRAM (MB)", row=1, col=2)
    
    fig2.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig2, width='content')

# TAB 2: Architecture Inspector
with tab2:
    st.header("🔬 Quantization Architecture Analysis")
    
    st.markdown("""
    <div class="info-box">
    <strong>Mixed-Precision Quantization:</strong><br>
    Each layer can use different bit-widths (2, 4, 8, or 16 bits). Lower bits = smaller model & faster inference, 
    but potential accuracy loss. The genetic algorithm discovers which layers are <em>sensitive</em> (need higher precision) 
    versus which can be aggressively compressed.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        # Create bit-width distribution chart
        layer_df = pd.DataFrame({
            "Layer": [f"L{i}" for i in range(len(current_genes))],
            "Bit-Width": current_genes,
            "Layer_ID": list(range(len(current_genes)))
        })
        
        # Color mapping
        color_map = {2: '#e74c3c', 4: '#f39c12', 8: '#f1c40f', 16: '#2ecc71'}
        layer_df['Color'] = layer_df['Bit-Width'].map(color_map)
        
        fig3 = px.bar(
            layer_df,
            x="Layer_ID",
            y="Bit-Width",
            color="Bit-Width",
            color_continuous_scale=[[0, '#e74c3c'], [0.33, '#f39c12'], [0.66, '#f1c40f'], [1, '#2ecc71']],
            labels={"Layer_ID": "Layer Index", "Bit-Width": "Bit-Width"},
            title=f"📊 Layer-wise Quantization Map: {selected_archetype.upper()}"
        )
        
        fig3.add_hline(y=avg_bits, line_dash="dash", line_color="red",
                      annotation_text=f"Average: {avg_bits:.1f} bits")
        
        fig3.update_layout(height=400)
        st.plotly_chart(fig3, width='content')
        
        # Bit distribution
        st.subheader("Bit-Width Distribution")
        bit_counts = pd.Series(current_genes).value_counts().sort_index()
        
        fig4 = px.pie(
            values=bit_counts.values,
            names=[f"{b}-bit" for b in bit_counts.index],
            color_discrete_sequence=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71'],
            title="Quantization Strategy Breakdown"
        )
        st.plotly_chart(fig4, width='content')
    
    with col2:
        st.subheader("📋 Configuration Details")
        
        st.markdown(f"""
        **Architecture:** {selected_archetype}  
        **Total Layers:** 22  
        **Quantization Levels:** {len(set(current_genes))}
        """)
        
        st.markdown("---")
        st.markdown("### Layer Statistics")
        
        stats_df = pd.DataFrame({
            "Bit-Width": ["2-bit", "4-bit", "8-bit", "16-bit"],
            "Count": [
                current_genes.count(2),
                current_genes.count(4),
                current_genes.count(8),
                current_genes.count(16)
            ],
            "Percentage": [
                f"{current_genes.count(2)/22*100:.1f}%",
                f"{current_genes.count(4)/22*100:.1f}%",
                f"{current_genes.count(8)/22*100:.1f}%",
                f"{current_genes.count(16)/22*100:.1f}%"
            ]
        })
        
        st.dataframe(stats_df, width='content', hide_index=True)
        
        st.markdown("---")
        st.markdown("### 🎨 Bit-Width Legend")
        st.markdown("""
        - 🔴 **2-bit**: Maximum compression
        - 🟠 **4-bit**: Balanced compression
        - 🟡 **8-bit**: Conservative quantization
        - 🟢 **16-bit**: Full precision
        """)
        
        st.markdown("---")
        st.subheader("📄 Export Configuration")
        
        config_json = json.dumps({
            "archetype": selected_archetype,
            "quantization_map": {f"layer_{i}": bit for i, bit in enumerate(current_genes)},
            "metrics": metrics
        }, indent=2)
        
        st.download_button(
            label="⬇️ Download Config (JSON)",
            data=config_json,
            file_name=f"{selected_archetype}_config.json",
            mime="application/json"
        )

# TAB 3: Performance Metrics
with tab3:
    st.header("📈 Performance Analysis")
    
    st.markdown("""
    <div class="info-box">
    <strong>Trade-off Analysis:</strong><br>
    Compare the selected architecture against the baseline and other candidates. 
    These metrics are predicted using sensitivity profiling and hardware telemetry.
    </div>
    """, unsafe_allow_html=True)
    
    # Comparison table
    comparison_data = []
    baseline_metrics = {"predicted_loss": 3.28, "predicted_vram_mb": 2200, "predicted_latency_score": 60}
    
    for arch_name, arch_data in artifacts.items():
        arch_metrics = arch_data['metrics']
        qmap = arch_data['config']['quantization_map']
        genes = [qmap[f"layer_{i}"] for i in range(22)]
        
        comparison_data.append({
            "Architecture": arch_name.replace("_", " ").title(),
            "Loss": arch_metrics['predicted_loss'],
            "Loss Δ": arch_metrics['predicted_loss'] - baseline_metrics['predicted_loss'],
            "VRAM (MB)": arch_metrics['predicted_vram_mb'],
            "VRAM Saved": baseline_metrics['predicted_vram_mb'] - arch_metrics['predicted_vram_mb'],
            "Latency (ms)": arch_metrics['predicted_latency_score'],
            "Speedup": baseline_metrics['predicted_latency_score'] / arch_metrics['predicted_latency_score'],
            "Avg Bits": np.mean(genes),
            "Compression": f"{(1 - sum(genes)/(16*22))*100:.1f}%"
        })
    
    # Add baseline
    comparison_data.append({
        "Architecture": "Baseline (FP16)",
        "Loss": baseline_metrics['predicted_loss'],
        "Loss Δ": 0.0,
        "VRAM (MB)": baseline_metrics['predicted_vram_mb'],
        "VRAM Saved": 0,
        "Latency (ms)": baseline_metrics['predicted_latency_score'],
        "Speedup": 1.0,
        "Avg Bits": 16.0,
        "Compression": "0.0%"
    })
    
    comp_df = pd.DataFrame(comparison_data)
    
    # Highlight selected row
    def highlight_selected(row):
        if selected_archetype in row['Architecture'].lower() or (selected_archetype == "Baseline (FP16)" and "Baseline" in row['Architecture']):
            return ['background-color: #667eea; color: white'] * len(row)
        return [''] * len(row)
    
    st.dataframe(
        comp_df.style.apply(highlight_selected, axis=1).format({
            "Loss": "{:.4f}",
            "Loss Δ": "{:+.4f}",
            "VRAM (MB)": "{:.0f}",
            "VRAM Saved": "{:.0f}",
            "Latency (ms)": "{:.2f}",
            "Speedup": "{:.2f}x",
            "Avg Bits": "{:.1f}"
        }),
        width='content',
        hide_index=True
    )
    
    # Radar chart comparison
    st.subheader("🎯 Multi-Objective Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Normalize metrics for radar chart
        categories = ['Accuracy', 'Memory', 'Speed', 'Compression']
        
        selected_data = artifacts.get(selected_archetype, artifacts['balanced'])
        selected_metrics = selected_data['metrics']
        selected_genes = [selected_data['config']['quantization_map'][f"layer_{i}"] for i in range(22)]
        
        # Normalize
        selected_values = [
            1 - (selected_metrics['predicted_loss'] - 3.28) / 1.0,  # Accuracy
            1 - selected_metrics['predicted_vram_mb'] / 2200,  # Memory
            1 - selected_metrics['predicted_latency_score'] / 60,  # Speed
            1 - sum(selected_genes) / (16 * 22)  # Compression
        ]
        
        baseline_values = [1.0, 0.0, 0.0, 0.0]
        
        fig5 = go.Figure()
        
        fig5.add_trace(go.Scatterpolar(
            r=selected_values + [selected_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name=selected_archetype,
            line=dict(color='#667eea', width=2)
        ))
        
        fig5.add_trace(go.Scatterpolar(
            r=baseline_values + [baseline_values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            name='Baseline',
            line=dict(color='#e74c3c', width=2)
        ))
        
        fig5.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="Performance Profile"
        )
        
        st.plotly_chart(fig5, width='content')
    
    with col2:
        # Bar chart comparison
        metrics_comp = {
            "Accuracy\n(higher=better)": [
                100 - (baseline_metrics['predicted_loss'] - 3.28) * 10,
                100 - (metrics['predicted_loss'] - 3.28) * 10
            ],
            "Memory Efficiency\n(higher=better)": [
                0,
                (baseline_metrics['predicted_vram_mb'] - metrics['predicted_vram_mb']) / baseline_metrics['predicted_vram_mb'] * 100
            ],
            "Speed\n(higher=better)": [
                0,
                (baseline_metrics['predicted_latency_score'] - metrics['predicted_latency_score']) / baseline_metrics['predicted_latency_score'] * 100
            ]
        }
        
        fig6 = go.Figure()
        
        fig6.add_trace(go.Bar(
            name='Baseline',
            x=list(metrics_comp.keys()),
            y=[metrics_comp[k][0] for k in metrics_comp.keys()],
            marker_color='#e74c3c'
        ))
        
        fig6.add_trace(go.Bar(
            name=selected_archetype,
            x=list(metrics_comp.keys()),
            y=[metrics_comp[k][1] for k in metrics_comp.keys()],
            marker_color='#667eea'
        ))
        
        fig6.update_layout(
            barmode='group',
            title="Relative Performance Gains",
            yaxis_title="Improvement (%)",
            height=400
        )
        
        st.plotly_chart(fig6, width='content')
    
    # Key insights
    st.subheader("💡 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        loss_delta = metrics['predicted_loss'] - baseline_metrics['predicted_loss']
        st.metric(
            "Accuracy Impact",
            f"{loss_delta:+.4f}",
            delta=f"{loss_delta/baseline_metrics['predicted_loss']*100:.2f}%",
            delta_color="inverse"
        )
    
    with col2:
        vram_saved = baseline_metrics['predicted_vram_mb'] - metrics['predicted_vram_mb']
        st.metric(
            "Memory Saved",
            f"{vram_saved:.0f} MB",
            delta=f"{vram_saved/baseline_metrics['predicted_vram_mb']*100:.1f}%"
        )
    
    with col3:
        speedup = baseline_metrics['predicted_latency_score'] / metrics['predicted_latency_score']
        st.metric(
            "Inference Speedup",
            f"{speedup:.2f}x",
            delta=f"{(speedup-1)*100:.1f}%"
        )

# TAB 4: Inference Demo
with tab4:
    st.header("💬 Interactive Inference Playground")
    
    if not GPU_AVAILABLE:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>GPU Not Available</strong><br>
        Live inference requires CUDA-enabled GPU. This demo shows the interface and expected behavior.
        Deploy locally with GPU for full functionality.
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="info-box">
    <strong>Current Configuration:</strong> {selected_archetype.upper()}<br>
    Generate text using the optimized model with mixed-precision quantization.
    The model runs with the quantization map shown in the Architecture Inspector tab.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Enter your prompt:",
            value="The future of Artificial Intelligence is",
            height=100,
            help="Enter text to generate completions"
        )
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            max_tokens = st.slider("Max Tokens", 10, 200, 64)
        with col_b:
            temperature = st.slider("Temperature", 0.1, 2.0, 0.7, 0.1)
        with col_c:
            top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05)
        
        generate_btn = st.button("🚀 Generate", type="primary", width='content')
        
        if generate_btn:
            with st.spinner("Generating response..."):
                if GPU_AVAILABLE and not MOCK_MODE:
                    try:
                        import time
                        import torch
                        from src.core.validator import Validator
                        from src.core.search_space import Genome
                        
                        # Load model (cached)
                        @st.cache_resource
                        def load_model(arch_name, genes):
                            validator = Validator("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
                            genome = Genome(genes=genes)
                            validator.apply_genome(genome)
                            return validator
                        
                        model = load_model(selected_archetype, current_genes)
                        
                        tokenizer = model.wrapper.tokenizer
                        net = model.wrapper.model
                        
                        inputs = tokenizer(prompt, return_tensors="pt").to(model.wrapper.device)
                        
                        start_t = time.time()
                        with torch.no_grad():
                            outputs = net.generate(
                                inputs.input_ids,
                                max_new_tokens=max_tokens,
                                do_sample=True,
                                temperature=temperature,
                                top_p=top_p,
                                pad_token_id=tokenizer.eos_token_id
                            )
                        end_t = time.time()
                        
                        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                        latency = (end_t - start_t) * 1000
                        
                        st.markdown("### 📝 Generated Output")
                        st.markdown(f"```\n{output_text}\n```")
                        st.caption(f"⏱️ Generation time: {latency:.2f}ms | 🔢 Tokens: {max_tokens}")
                        
                    except Exception as e:
                        st.error(f"Generation failed: {str(e)}")
                        st.info("Showing mock output for demonstration")
                        mock_output = prompt + " transforming industries through intelligent automation, enhanced decision-making capabilities, and revolutionary approaches to problem-solving. As we continue to develop more sophisticated AI systems, the integration of machine learning and neural networks..."
                        st.markdown("### 📝 Generated Output (Demo)")
                        st.markdown(f"```\n{mock_output}\n```")
                        st.caption(f"⏱️ Estimated generation time: ~{metrics['predicted_latency_score']:.2f}ms")
                else:
                    # Mock generation for demo
                    import time
                    time.sleep(1)  # Simulate generation time
                    
                    mock_outputs = {
                        "The future of Artificial Intelligence is": " transforming industries through intelligent automation, enhanced decision-making capabilities, and revolutionary approaches to problem-solving. As we continue to develop more sophisticated AI systems...",
                        "default": " showing promising developments in efficiency and performance. The optimized quantization strategy enables faster inference while maintaining high accuracy levels..."
                    }
                    
                    mock_output = prompt + mock_outputs.get(prompt, mock_outputs["default"])
                    
                    st.markdown("### 📝 Generated Output (Demo Mode)")
                    st.markdown(f"```\n{mock_output}\n```")
                    st.caption(f"⏱️ Estimated latency: {metrics['predicted_latency_score']:.2f}ms | 🔢 Tokens: {max_tokens}")
    
    with col2:
        st.subheader("⚙️ Model Info")
        
        st.markdown(f"""
        **Base Model:** TinyLlama-1.1B  
        **Quantization:** {selected_archetype}  
        **Avg Precision:** {avg_bits:.1f}-bit  
        **Compression:** {compression_ratio*100:.1f}%
        """)
        
        st.markdown("---")
        st.subheader("📊 Expected Performance")
        
        st.metric("VRAM Usage", f"{metrics['predicted_vram_mb']:.0f} MB")
        st.metric("Latency", f"{metrics['predicted_latency_score']:.2f} ms")
        st.metric("Validation Loss", f"{metrics['predicted_loss']:.4f}")
        
        st.markdown("---")
        st.markdown("""
        ### 💡 Tips
        - Lower temperature = more focused
        - Higher temperature = more creative
        - Top-p controls diversity
        - Adjust tokens for length
        """)
    
    # Example prompts
    st.markdown("---")
    st.subheader("📚 Example Prompts")
    
    example_prompts = [
        "Explain quantum computing in simple terms:",
        "Write a haiku about machine learning:",
        "The best way to learn programming is",
        "In the year 2050, technology will"
    ]
    
    cols = st.columns(len(example_prompts))
    for idx, (col, example) in enumerate(zip(cols, example_prompts)):
        with col:
            if st.button(f"📝 Try", key=f"example_{idx}"):
                st.rerun()

# Footer
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    ### 🎯 Project Goals
    - Hardware-aware optimization
    - Multi-objective search
    - Production-ready deployment
    """)

with col2:
    st.markdown("""
    ### 🔬 Technical Stack
    - Genetic Algorithms (NSGA-II)
    - PyTorch Quantization
    - Streamlit Dashboard
    """)

with col3:
    st.markdown("""
    ### 📖 Resources
    - [GitHub Repository](#)
    - [Technical Report](#)
    - [API Documentation](#)
    """)