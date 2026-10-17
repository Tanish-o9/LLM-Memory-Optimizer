import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------
# Page Configuration & Aesthetics
# ---------------------------------------------------------
st.set_page_config(
    page_title="LLM Memory Optimizer | Intelligence Hub",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Premium Custom CSS (Glassmorphism, Dark Mode, Glow Effects)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 15% 15%, #111827 0%, #080C14 100%);
        color: #F3F4F6;
    }
    
    /* Header hero section */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px 30px;
        margin-bottom: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.4);
    }
    
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-right: 8px;
    }
    
    .badge-fp16 { background: rgba(56, 189, 248, 0.15); color: #38BDF8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .badge-int8 { background: rgba(52, 211, 153, 0.15); color: #34D399; border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge-int4 { background: rgba(168, 85, 247, 0.15); color: #C084FC; border: 1px solid rgba(168, 85, 247, 0.3); }
    
    /* Custom metric card */
    .metric-card {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 20px;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 8px 25px rgba(56, 189, 248, 0.12);
    }
    
    .metric-label {
        font-size: 0.82rem;
        font-weight: 500;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.2;
    }
    .metric-delta-pos {
        font-size: 0.85rem;
        font-weight: 600;
        color: #34D399;
        margin-top: 6px;
    }
    .metric-delta-neg {
        font-size: 0.85rem;
        font-weight: 600;
        color: #F87171;
        margin-top: 6px;
    }
    
    /* Model output container */
    .output-box {
        background: #0D131F;
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        font-size: 0.9rem;
        line-height: 1.6;
        color: #D1D5DB;
        min-height: 120px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Demo / Fallback Data Generator
# ---------------------------------------------------------
def get_demo_data():
    """Fallback realistic data based on Qwen2.5-0.5B-Instruct on RTX GPU."""
    return pd.DataFrame([
        {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "gpu": "NVIDIA GeForce RTX 2050 Laptop GPU (4GB)",
            "torch": "2.11.0+cu124",
            "mode": "fp16",
            "load_time_s": 2.14,
            "weights_mb": 980.5,
            "peak_gpu_mb": 1420.0,
            "first_token_ms": 28.4,
            "latency_s_mean": 1.152,
            "latency_s_std": 0.042,
            "tokens_per_s": 55.56,
            "perplexity": 12.45,
            "sample_output": "Quantization in machine learning is the process of mapping continuous or high-precision numbers (like 16-bit floats) to a smaller set of discrete values (like 8-bit integers) to compress memory.",
            "memory_reduction_pct": 0.0,
            "speedup_x": 1.0,
            "ppl_change": 0.0,
        },
        {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "gpu": "NVIDIA GeForce RTX 2050 Laptop GPU (4GB)",
            "torch": "2.11.0+cu124",
            "mode": "int8",
            "load_time_s": 3.82,
            "weights_mb": 512.2,
            "peak_gpu_mb": 760.4,
            "first_token_ms": 32.1,
            "latency_s_mean": 1.280,
            "latency_s_std": 0.051,
            "tokens_per_s": 50.00,
            "perplexity": 12.51,
            "sample_output": "Quantization in machine learning refers to reducing the precision of the model weights and activations from 16-bit to 8-bit numbers, significantly reducing VRAM consumption with minimal quality loss.",
            "memory_reduction_pct": 46.5,
            "speedup_x": 0.90,
            "ppl_change": 0.06,
        },
        {
            "model": "Qwen/Qwen2.5-0.5B-Instruct",
            "gpu": "NVIDIA GeForce RTX 2050 Laptop GPU (4GB)",
            "torch": "2.11.0+cu124",
            "mode": "int4",
            "load_time_s": 4.10,
            "weights_mb": 298.0,
            "peak_gpu_mb": 420.8,
            "first_token_ms": 35.6,
            "latency_s_mean": 1.085,
            "latency_s_std": 0.038,
            "tokens_per_s": 58.98,
            "perplexity": 12.88,
            "sample_output": "Quantization is a technique in deep learning to convert model parameters from high-precision floating point representations into compact 4-bit formats, saving up to 75% memory on consumer devices.",
            "memory_reduction_pct": 70.4,
            "speedup_x": 1.06,
            "ppl_change": 0.43,
        }
    ])

# ---------------------------------------------------------
# Sidebar: Data Source & Settings
# ---------------------------------------------------------
st.sidebar.markdown("### ⚡ Control Deck")
st.sidebar.markdown("Configure your benchmark source & target model.")

csv_path = "results.csv"
data_source = st.sidebar.radio(
    "Data Source",
    options=["Live Benchmark (results.csv)", "Upload Custom CSV", "Demo / Reference Benchmark"],
    index=0 if os.path.exists(csv_path) else 2
)

df = None
is_demo = False

if data_source == "Live Benchmark (results.csv)":
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            st.sidebar.success(f"Loaded live `{csv_path}` ({len(df)} records)")
        except Exception as e:
            st.sidebar.error(f"Error reading {csv_path}: {e}")
            df = get_demo_data()
            is_demo = True
    else:
        st.sidebar.warning(f"File `{csv_path}` not found yet. Showing demo data.")
        df = get_demo_data()
        is_demo = True

elif data_source == "Upload Custom CSV":
    uploaded = st.sidebar.file_uploader("Upload Benchmark CSV", type=["csv"])
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.sidebar.success("Custom CSV loaded successfully!")
        except Exception as e:
            st.sidebar.error(f"Failed to parse CSV: {e}")
            df = get_demo_data()
            is_demo = True
    else:
        st.sidebar.info("Upload a file or toggle Demo mode.")
        df = get_demo_data()
        is_demo = True
else:
    df = get_demo_data()
    is_demo = True

# Mode Color Palette Mapping
MODE_COLORS = {
    "fp16": "#38BDF8",  # Ice Sky Blue
    "int8": "#34D399",  # Emerald Mint
    "int4": "#C084FC",  # Neon Purple
}

# ---------------------------------------------------------
# Top Header Banner
# ---------------------------------------------------------
model_name = df["model"].iloc[0] if "model" in df.columns else "Custom Model"
gpu_name = df["gpu"].iloc[0] if "gpu" in df.columns else "NVIDIA GPU"
torch_ver = df["torch"].iloc[0] if "torch" in df.columns else "PyTorch 2.x"

status_badge = "🟢 LIVE DATA" if not is_demo else "🟡 DEMO / REFERENCE BENCHMARK"

st.markdown(f"""
<div class="hero-banner">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <span class="badge-pill" style="background: rgba(99, 102, 241, 0.2); color: #818CF8; border: 1px solid rgba(99, 102, 241, 0.4);">{status_badge}</span>
            <span class="badge-pill badge-fp16">FP16 Baseline</span>
            <span class="badge-pill badge-int8">INT8 Quantized</span>
            <span class="badge-pill badge-int4">INT4 NF4 Quantized</span>
            <h1 style="margin: 8px 0 4px 0; font-size: 2.1rem; font-weight: 800; color: #FFFFFF; letter-spacing: -0.02em;">
                LLM Memory & Efficiency Optimizer
            </h1>
            <p style="margin: 0; color: #9CA3AF; font-size: 0.95rem;">
                Scientific profiling of model weight compression, VRAM footprint, latency, throughput, and perplexity drift.
            </p>
        </div>
        <div style="text-align: right; background: rgba(0,0,0,0.3); padding: 12px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.78rem; color: #9CA3AF; text-transform: uppercase;">Evaluated Model</div>
            <div style="font-weight: 700; color: #38BDF8; font-size: 1.05rem;">{model_name}</div>
            <div style="font-size: 0.75rem; color: #6B7280; margin-top: 2px;">{gpu_name} • PyTorch {torch_ver}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Executive KPI Row
# ---------------------------------------------------------
fp16_row = df[df["mode"] == "fp16"].iloc[0] if len(df[df["mode"] == "fp16"]) > 0 else df.iloc[0]
int4_row = df[df["mode"] == "int4"].iloc[0] if len(df[df["mode"] == "int4"]) > 0 else df.iloc[-1]
int8_row = df[df["mode"] == "int8"].iloc[0] if len(df[df["mode"] == "int8"]) > 0 else df.iloc[0]

vram_saved = int4_row.get("memory_reduction_pct", round((fp16_row["peak_gpu_mb"] - int4_row["peak_gpu_mb"])/fp16_row["peak_gpu_mb"]*100, 1))
speedup = int4_row.get("speedup_x", round(fp16_row["latency_s_mean"] / int4_row["latency_s_mean"], 2))
ppl_diff = int4_row.get("ppl_change", round(int4_row["perplexity"] - fp16_row["perplexity"], 2))

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Max VRAM Reduction (INT4)</div>
        <div class="metric-value">{vram_saved}%</div>
        <div class="metric-delta-pos">↓ {round(fp16_row['peak_gpu_mb'] - int4_row['peak_gpu_mb'], 1)} MB Saved</div>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">INT4 Throughput</div>
        <div class="metric-value">{int4_row['tokens_per_s']} <span style="font-size: 1rem; color: #9CA3AF;">tps</span></div>
        <div class="metric-delta-pos">{speedup}× vs FP16 baseline</div>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">First-Token Latency (TTFT)</div>
        <div class="metric-value">{int4_row['first_token_ms']} <span style="font-size: 1rem; color: #9CA3AF;">ms</span></div>
        <div class="metric-delta-neg">FP16: {fp16_row['first_token_ms']} ms</div>
    </div>
    """, unsafe_allow_html=True)

with c4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Perplexity Drift (Quality Loss)</div>
        <div class="metric-value">{int4_row['perplexity']} <span style="font-size: 1rem; color: #9CA3AF;">PPL</span></div>
        <div class="{'metric-delta-pos' if ppl_diff < 1.0 else 'metric-delta-neg'}">Δ {ppl_diff:+0.2f} (WikiText-2)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs Navigation
# ---------------------------------------------------------
tab_overview, tab_memory, tab_speed, tab_quality, tab_calculator, tab_code = st.tabs([
    "📊 Executive Summary",
    "💾 Memory & VRAM Footprint",
    "⚡ Latency & Throughput",
    "🎯 Output Quality & Perplexity",
    "💻 Laptop VRAM Fitment Check",
    "🛠️ CLI Benchmark Command"
])

# ---------------------------------------------------------
# TAB 1: EXECUTIVE SUMMARY
# ---------------------------------------------------------
with tab_overview:
    col_chart, col_summary = st.columns([7, 5])
    
    with col_chart:
        st.markdown("#### ⚖️ The Core Trade-off: Memory vs Throughput vs Quality")
        
        fig_tradeoff = go.Figure()
        
        for idx, row in df.iterrows():
            mode = row["mode"]
            color = MODE_COLORS.get(mode, "#94A3B8")
            
            fig_tradeoff.add_trace(go.Scatter(
                x=[row["peak_gpu_mb"]],
                y=[row["tokens_per_s"]],
                mode="markers+text",
                name=mode.upper(),
                text=[f"<b>{mode.upper()}</b><br>{row['tokens_per_s']} tps<br>{row['peak_gpu_mb']} MB"],
                textposition="top center",
                marker=dict(
                    size=row["perplexity"] * 2.2,
                    color=color,
                    line=dict(width=2, color="#FFFFFF"),
                    opacity=0.85
                ),
                hovertemplate=(
                    f"<b>Mode:</b> {mode.upper()}<br>"
                    f"<b>Peak VRAM:</b> %{{x}} MB<br>"
                    f"<b>Tokens/Sec:</b> %{{y}}<br>"
                    f"<b>Perplexity:</b> {row['perplexity']}<extra></extra>"
                )
            ))
            
        fig_tradeoff.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            xaxis=dict(title="Peak GPU Memory (MB) [Lower = Better]", gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(title="Generation Speed (Tokens/s) [Higher = Better]", gridcolor="rgba(255,255,255,0.07)"),
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_tradeoff, use_container_width=True)
        st.caption("💡 *Bubble size represents Perplexity score (smaller bubble = better language modeling quality).*")

    with col_summary:
        st.markdown("#### 📋 Precision Comparison Matrix")
        
        display_cols = ["mode", "weights_mb", "peak_gpu_mb", "tokens_per_s", "perplexity", "speedup_x"]
        clean_df = df[[c for c in display_cols if c in df.columns]].copy()
        clean_df.columns = ["Precision", "Static (MB)", "Peak VRAM (MB)", "Speed (TPS)", "Perplexity", "Speedup"]
        clean_df["Precision"] = clean_df["Precision"].str.upper()
        
        st.dataframe(
            clean_df.style.format({
                "Static (MB)": "{:.1f}",
                "Peak VRAM (MB)": "{:.1f}",
                "Speed (TPS)": "{:.2f}",
                "Perplexity": "{:.2f}",
                "Speedup": "{:.2f}×"
            }).highlight_min(subset=["Peak VRAM (MB)", "Perplexity"], color="#064E3B")
              .highlight_max(subset=["Speed (TPS)"], color="#1E3A8A"),
            use_container_width=True,
            hide_index=True
        )
        
        st.markdown("""
        **Key Takeaways:**
        - **INT4 Precision**: Cuts down GPU VRAM by **~70%**, enabling models to fit directly onto entry-level 4GB GPUs (like RTX 2050/3050).
        - **INT8 Precision**: Offers the best safety net — near-zero loss in perplexity, with roughly **45-50% memory savings**.
        - **Quality Retention**: Perplexity degradation in 4-bit NormalFloat (`nf4`) is very low, making it safe for production chat and coding tasks.
        """)

# ---------------------------------------------------------
# TAB 2: MEMORY & VRAM DEEP DIVE
# ---------------------------------------------------------
with tab_memory:
    st.markdown("### 💾 GPU VRAM Consumption & Memory Optimization Breakdown")
    m_col1, m_col2 = st.columns(2)
    
    with m_col1:
        fig_mem = go.Figure()
        
        fig_mem.add_trace(go.Bar(
            name="Static Model Weights",
            x=[m.upper() for m in df["mode"]],
            y=df["weights_mb"],
            marker_color="rgba(99, 102, 241, 0.7)",
            text=[f"{v:.1f} MB" for v in df["weights_mb"]],
            textposition="auto"
        ))
        
        fig_mem.add_trace(go.Bar(
            name="Peak Inference VRAM (with KV Cache)",
            x=[m.upper() for m in df["mode"]],
            y=df["peak_gpu_mb"],
            marker_color="rgba(236, 72, 153, 0.7)",
            text=[f"{v:.1f} MB" for v in df["peak_gpu_mb"]],
            textposition="auto"
        ))
        
        fig_mem.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            xaxis=dict(title="Precision Mode", gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(title="VRAM in MB", gridcolor="rgba(255,255,255,0.07)"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=380,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_mem, use_container_width=True)

    with m_col2:
        fig_waterfall = go.Figure()
        
        pct_savings = [row.get("memory_reduction_pct", 0.0) for _, row in df.iterrows()]
        colors = ["#38BDF8", "#34D399", "#C084FC"]
        
        fig_waterfall.add_trace(go.Bar(
            x=[m.upper() for m in df["mode"]],
            y=pct_savings,
            marker_color=colors,
            text=[f"{v:.1f}% Saved" if v > 0 else "0% (Baseline)" for v in pct_savings],
            textposition="outside"
        ))
        
        fig_waterfall.update_layout(
            title="Percentage of VRAM Saved vs Baseline FP16",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            xaxis=dict(title="Precision Mode", gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(title="% Memory Saved", range=[0, 100], gridcolor="rgba(255,255,255,0.07)"),
            height=380,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_waterfall, use_container_width=True)

    st.markdown("#### ⏱️ Model Cold Load Time (Disk / RAM to GPU VRAM)")
    l_cols = st.columns(len(df))
    for i, (_, row) in enumerate(df.iterrows()):
        with l_cols[i]:
            st.metric(
                label=f"Cold Load: {row['mode'].upper()}",
                value=f"{row['load_time_s']} s",
                delta=f"{round(row['load_time_s'] - fp16_row['load_time_s'], 2)}s vs FP16",
                delta_color="inverse"
            )

# ---------------------------------------------------------
# TAB 3: LATENCY & THROUGHPUT
# ---------------------------------------------------------
with tab_speed:
    st.markdown("### ⚡ Generation Speed, First-Token Latency (TTFT) & Stability")
    s_col1, s_col2 = st.columns(2)
    
    with s_col1:
        fig_tps = go.Figure()
        fig_tps.add_trace(go.Bar(
            x=[m.upper() for m in df["mode"]],
            y=df["tokens_per_s"],
            marker_color=["#38BDF8", "#34D399", "#C084FC"],
            text=[f"{v:.1f} tps" for v in df["tokens_per_s"]],
            textposition="auto"
        ))
        
        fig_tps.update_layout(
            title="Throughput (Tokens Per Second) [Higher is Better]",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            xaxis=dict(title="Mode", gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(title="Tokens / Sec", gridcolor="rgba(255,255,255,0.07)"),
            height=380,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_tps, use_container_width=True)

    with s_col2:
        fig_ttft = go.Figure()
        fig_ttft.add_trace(go.Scatter(
            x=[m.upper() for m in df["mode"]],
            y=df["first_token_ms"],
            mode="lines+markers",
            line=dict(color="#F59E0B", width=3),
            marker=dict(size=12, color="#FBBF24"),
            text=[f"{v:.1f} ms" for v in df["first_token_ms"]],
            textposition="top center"
        ))
        
        fig_ttft.update_layout(
            title="First-Token Latency (TTFT) in Milliseconds [Lower is Better]",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            xaxis=dict(title="Mode", gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(title="Time to First Token (ms)", gridcolor="rgba(255,255,255,0.07)"),
            height=380,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_ttft, use_container_width=True)

    if "latency_s_std" in df.columns:
        st.markdown("#### 📈 End-to-End Latency Variance (Mean ± Standard Deviation)")
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(
            x=[m.upper() for m in df["mode"]],
            y=df["latency_s_mean"],
            error_y=dict(type="data", array=df["latency_s_std"], visible=True),
            marker_color="#818CF8",
            text=[f"{v:.3f}s" for v in df["latency_s_mean"]],
            textposition="auto"
        ))
        fig_err.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            xaxis=dict(title="Mode", gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(title="Mean Latency (s)", gridcolor="rgba(255,255,255,0.07)"),
            height=300,
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig_err, use_container_width=True)

# ---------------------------------------------------------
# TAB 4: QUALITY & PERPLEXITY
# ---------------------------------------------------------
with tab_quality:
    st.markdown("### 🎯 Language Modeling Quality & Qualitative Drift")
    q_col1, q_col2 = st.columns([5, 7])
    
    with q_col1:
        fig_ppl = go.Figure()
        fig_ppl.add_trace(go.Bar(
            x=[m.upper() for m in df["mode"]],
            y=df["perplexity"],
            marker_color=["#38BDF8", "#34D399", "#C084FC"],
            text=[f"{v:.2f}" for v in df["perplexity"]],
            textposition="auto"
        ))
        fig_ppl.update_layout(
            title="WikiText-2 Perplexity (Lower = Better Quality)",
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(17, 24, 39, 0.4)",
            xaxis=dict(title="Mode", gridcolor="rgba(255,255,255,0.07)"),
            yaxis=dict(title="Perplexity Score", gridcolor="rgba(255,255,255,0.07)"),
            height=360,
            margin=dict(l=20, r=20, t=50, b=20)
        )
        st.plotly_chart(fig_ppl, use_container_width=True)
        
    with q_col2:
        st.markdown("#### 🧪 Side-by-Side Response Inspection")
        st.caption("Compare how each quantized model answered the exact same benchmark prompt:")
        
        sample_tabs = st.tabs([m.upper() for m in df["mode"]])
        for i, (_, row) in enumerate(df.iterrows()):
            with sample_tabs[i]:
                output_txt = row.get("sample_output", "No sample output recorded.")
                st.markdown(f"""
                <div class="output-box">
                    <span style="color: {MODE_COLORS.get(row['mode'], '#9CA3AF')}; font-weight: 700;">{row['mode'].upper()} Generation:</span><br><br>
                    {output_txt}
                </div>
                """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 5: LAPTOP VRAM FITMENT CALCULATOR
# ---------------------------------------------------------
with tab_calculator:
    st.markdown("### 💻 Consumer Laptop & Edge Device Fitment Checker")
    st.markdown("Select your available GPU VRAM (e.g., RTX 2050 4GB or RTX 3060 6GB) to see whether each mode will safely run without crashing into CUDA Out-Of-Memory (OOM).")
    
    col_vram_slider, col_fit_res = st.columns([5, 7])
    
    with col_vram_slider:
        user_vram_gb = st.slider("Select Available GPU VRAM (in GB)", min_value=1.0, max_value=24.0, value=4.0, step=0.5)
        os_overhead_gb = st.slider("OS / Display System Overhead (GB)", min_value=0.2, max_value=1.5, value=0.6, step=0.1)
        
        usable_vram_mb = (user_vram_gb - os_overhead_gb) * 1024
        st.info(f"💡 Total Usable GPU VRAM: **{usable_vram_mb:.0f} MB** ({user_vram_gb - os_overhead_gb:.2f} GB)")
        
    with col_fit_res:
        st.markdown("#### Fitment Status for Evaluated Modes:")
        
        for _, row in df.iterrows():
            req_mb = row["peak_gpu_mb"]
            mode_str = row["mode"].upper()
            pct_used = (req_mb / usable_vram_mb) * 100
            
            if req_mb <= usable_vram_mb * 0.85:
                status_html = f"<span style='color: #34D399; font-weight: 700;'>✅ Fits Safely (Uses {pct_used:.1f}% VRAM)</span>"
            elif req_mb <= usable_vram_mb:
                status_html = f"<span style='color: #F59E0B; font-weight: 700;'>⚠️ Tight Fit (Uses {pct_used:.1f}% VRAM - risk of OOM on large context)</span>"
            else:
                status_html = f"<span style='color: #EF4444; font-weight: 700;'>❌ CUDA OOM Crash (Requires {req_mb:.0f} MB, only {usable_vram_mb:.0f} MB available)</span>"
                
            st.markdown(f"""
            <div style="background: rgba(17, 24, 39, 0.6); padding: 14px 18px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-bottom: 10px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span class="badge-pill" style="background: rgba(255,255,255,0.08);">{mode_str}</span>
                        <span style="font-weight: 600; font-size: 0.95rem;">Peak Required: {req_mb:.1f} MB</span>
                    </div>
                    <div>{status_html}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ---------------------------------------------------------
# TAB 6: CLI BENCHMARK COMMAND GENERATOR
# ---------------------------------------------------------
with tab_code:
    st.markdown("### 🛠️ Run the Benchmark Script (`LLM_Memory_Optimizer.py`)")
    st.markdown("Customize your execution parameters below to generate the exact terminal command:")
    
    cmd_c1, cmd_c2 = st.columns(2)
    with cmd_c1:
        chosen_model = st.text_input("Target Model ID", value="Qwen/Qwen2.5-0.5B-Instruct")
        chosen_runs = st.number_input("Repetitions per Prompt (Runs)", min_value=1, max_value=10, value=3)
    with cmd_c2:
        chosen_tokens = st.number_input("Max New Tokens", min_value=16, max_value=512, value=64, step=16)
        chosen_modes = st.multiselect("Modes to Test", options=["fp16", "int8", "int4"], default=["fp16", "int8", "int4"])
        
    modes_arg = " ".join(chosen_modes) if chosen_modes else "fp16 int8 int4"
    generated_cmd = f"python LLM_Memory_Optimizer.py --model {chosen_model} --modes {modes_arg} --runs {chosen_runs} --max_new_tokens {chosen_tokens} --out results.csv"
    
    st.code(generated_cmd, language="bash")
    st.caption("👉 Run this command inside your terminal. It will benchmark each mode and automatically generate `results.csv` which refreshes this dashboard!")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.8rem; padding: 10px;'>"
    "⚡ LLM Memory Optimizer • Powered by PyTorch, BitsAndBytes & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
