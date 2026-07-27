import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import re

# ---------------------------------------------------------
# PAGE CONFIGURATION & METADATA
# ---------------------------------------------------------
st.set_page_config(
    page_title="RobustLens AI | Adversarial Model Benchmarking",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# HIGH-CONTRAST CSS STYLING & GLASSMORPHISM AESTHETICS
# ---------------------------------------------------------
st.markdown(r"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700;800&display=swap');
    
    /* Universal Typography & High-Contrast Colors */
    html, body, [class*="css"], .stMarkdown, p, div, label, span, li, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif !important;
        color: #F8FAFC !important;
    }
    
    /* Main Background */
    .stApp {
        background-color: #0F172A !important;
        background: radial-gradient(circle at 50% -20%, #1e1b4b 0%, #0f172a 60%, #020617 100%) !important;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    [data-testid="stSidebar"] * {
        color: #F8FAFC !important;
    }

    /* Radio Button Labels */
    [data-testid="stRadioButton"] label p {
        color: #F8FAFC !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
    }
    
    /* Header Gradient Title */
    .header-title {
        background: linear-gradient(135deg, #38BDF8 0%, #818CF8 50%, #C084FC 100%);
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 2.8rem !important;
        letter-spacing: -0.02em;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #94A3B8 !important;
        font-size: 1.15rem !important;
        font-weight: 400 !important;
        margin-bottom: 1.5rem;
    }
    
    /* Glass Containers */
    .glass-card {
        background: rgba(30, 41, 59, 0.75) !important;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    
    .glass-card p, .glass-card li, .glass-card code {
        color: #E2E8F0 !important;
    }

    /* Structured Section Cards */
    .report-card-summary {
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.1), rgba(30, 41, 59, 0.8)) !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .report-card-tradeoff {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(30, 41, 59, 0.8)) !important;
        border: 1px solid rgba(245, 158, 11, 0.4) !important;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .report-card-recommendation {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(30, 41, 59, 0.8)) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .report-card-limitation {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(30, 41, 59, 0.8)) !important;
        border: 1px solid rgba(239, 68, 68, 0.4) !important;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }
    
    /* Metric Cards */
    .metric-box {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.9), rgba(15, 23, 42, 0.95)) !important;
        border: 1px solid rgba(56, 189, 248, 0.3) !important;
        border-radius: 14px;
        padding: 18px 12px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .metric-val {
        font-family: 'Outfit', sans-serif !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        color: #38BDF8 !important;
        margin: 4px 0;
    }
    
    .metric-lbl {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-sub {
        font-size: 0.85rem !important;
        color: #F8FAFC !important;
        font-weight: 500 !important;
    }

    /* Input Fields & Forms Contrast */
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div {
        background-color: #1E293B !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: #F8FAFC !important;
    }
    
    input {
        color: #F8FAFC !important;
    }

    /* Form Label Styling */
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stFileUploader > label {
        color: #F8FAFC !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* Dataframe Table Headers & Cell Text */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
    }

    /* Code Blocks */
    pre, code {
        background-color: #1E293B !important;
        color: #38BDF8 !important;
        border-radius: 8px;
    }
    
    /* Button Customization */
    .stButton > button {
        background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 8px 16px !important;
        transition: all 0.2s ease !important;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #60A5FA 0%, #3B82F6 100%) !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4) !important;
    }

    .stDownloadButton > button {
        background: rgba(30, 41, 59, 0.8) !important;
        color: #38BDF8 !important;
        border: 1px solid #38BDF8 !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# SYSTEM PROMPT DEFINITION FOR GEMINI
# ---------------------------------------------------------
SYSTEM_PROMPT = """You are an AI research assistant specialising in adversarial robustness
and Vision Transformers.

Analyse only the experimental results provided by the user.

Your tasks are:
1. Summarise the main result in simple language.
2. Compare clean, FGSM and PGD accuracy.
3. Explain whether the fine-tuning method improved robustness.
4. Identify any clean-accuracy versus robust-accuracy trade-off.
5. Recommend one practical next experiment.
6. Mention limitations such as small test sets, too few epochs,
   weak attacks or unfair model comparisons.
7. Never invent results, papers, model settings or numerical values.
8. Clearly state when there is not enough information.

Keep the explanation clear and suitable for a master's student."""

# ---------------------------------------------------------
# SESSION STATE INITIALIZATION & DEFAULT BENCHMARKS
# ---------------------------------------------------------
if "experiments" not in st.session_state:
    default_df = pd.DataFrame([
        {
            "Experiment Name": "ViT-B/16 Baseline",
            "Model Name": "ViT-B/16",
            "Dataset": "CIFAR-10",
            "Fine-tuning Method": "Standard Fine-Tuning",
            "Clean Accuracy (%)": 92.4,
            "FGSM Accuracy (%)": 24.1,
            "PGD Accuracy (%)": 2.8,
            "Epsilon": "8/255",
            "Epochs": 10
        },
        {
            "Experiment Name": "ViT-B/16 FGSM-AT",
            "Model Name": "ViT-B/16",
            "Dataset": "CIFAR-10",
            "Fine-tuning Method": "FGSM Adversarial Training",
            "Clean Accuracy (%)": 88.2,
            "FGSM Accuracy (%)": 65.4,
            "PGD Accuracy (%)": 18.2,
            "Epsilon": "8/255",
            "Epochs": 15
        },
        {
            "Experiment Name": "ViT-B/16 PGD-7 AT",
            "Model Name": "ViT-B/16",
            "Dataset": "CIFAR-10",
            "Fine-tuning Method": "PGD-7 Adversarial Training",
            "Clean Accuracy (%)": 84.6,
            "FGSM Accuracy (%)": 61.2,
            "PGD Accuracy (%)": 48.5,
            "Epsilon": "8/255",
            "Epochs": 20
        },
        {
            "Experiment Name": "ViT-B/16 TRADES",
            "Model Name": "ViT-B/16",
            "Dataset": "CIFAR-10",
            "Fine-tuning Method": "TRADES (beta=6.0)",
            "Clean Accuracy (%)": 83.1,
            "FGSM Accuracy (%)": 59.8,
            "PGD Accuracy (%)": 51.4,
            "Epsilon": "8/255",
            "Epochs": 25
        },
        {
            "Experiment Name": "ResNet-50 PGD-10",
            "Model Name": "ResNet-50",
            "Dataset": "CIFAR-10",
            "Fine-tuning Method": "PGD-10 Adversarial Training",
            "Clean Accuracy (%)": 82.0,
            "FGSM Accuracy (%)": 56.3,
            "PGD Accuracy (%)": 44.7,
            "Epsilon": "8/255",
            "Epochs": 20
        }
    ])
    st.session_state.experiments = default_df

# ---------------------------------------------------------
# CORE CALCULATION ENGINE
# ---------------------------------------------------------
def calculate_metrics(df):
    if df.empty:
        return df
    
    calc = df.copy()
    calc["FGSM Drop (%)"] = (calc["Clean Accuracy (%)"] - calc["FGSM Accuracy (%)"]).round(2)
    calc["PGD Drop (%)"] = (calc["Clean Accuracy (%)"] - calc["PGD Accuracy (%)"]).round(2)
    
    # Robustness Score Formula:
    # 0.20 * Clean + 0.35 * FGSM + 0.45 * PGD
    calc["Robustness Score"] = (
        0.20 * calc["Clean Accuracy (%)"] +
        0.35 * calc["FGSM Accuracy (%)"] +
        0.45 * calc["PGD Accuracy (%)"]
    ).round(2)
    
    # Epoch Efficiency
    calc["Epoch Efficiency"] = (calc["Robustness Score"] / calc["Epochs"].astype(float)).round(2)
    
    return calc

def get_gemini_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------
# SIDEBAR NAVIGATION & QUICK CONTROLS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 5px;">
        <span style="font-size: 2.4rem;">🛡️</span>
        <div>
            <h2 style="margin:0; font-size: 1.45rem; font-weight: 800; color: #F8FAFC !important;">RobustLens AI</h2>
            <span style="font-size: 0.75rem; color: #94A3B8 !important;">v1.2.0 • Pro Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Adversarial Robustness Analytics & AI Assistant")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🧪 Experiment Analyzer",
            "📊 Model Comparison",
            "📈 Pareto Trade-Off Frontier",
            "🎛️ Epsilon Simulator",
            "🤖 AI Research Assistant"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### **Workspace Manager**")
    exp_count = len(st.session_state.experiments)
    st.info(f"📁 **{exp_count}** Experiment(s) loaded")
    
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        if st.button("🔄 Reset Data", use_container_width=True):
            st.session_state.experiments = default_df.copy()
            st.rerun()
    with col_s2:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.experiments = pd.DataFrame(columns=[
                "Experiment Name", "Model Name", "Dataset", "Fine-tuning Method",
                "Clean Accuracy (%)", "FGSM Accuracy (%)", "PGD Accuracy (%)", "Epsilon", "Epochs"
            ])
            st.rerun()

# ---------------------------------------------------------
# PAGE 1: HOME
# ---------------------------------------------------------
if page == "🏠 Home":
    st.markdown('<div class="header-title">RobustLens AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Adversarial Robustness Evaluation & Diagnostic Assistant for Vision Transformers & CNNs</div>', unsafe_allow_html=True)
    
    c1, c2 = st.columns([3, 2])
    
    with c1:
        st.markdown(r"""
        <div class="glass-card">
            <h3>📌 The Core Research Challenge</h3>
            <p>
                Deep Neural Networks—especially modern <b>Vision Transformers (ViTs)</b>—achieve remarkable clean classification performance. However, they are fragile under tiny, adversarial perturbations.
            </p>
            <p>
                Evaluating defensive fine-tuning methods like <b>FGSM-AT</b>, <b>PGD-AT</b>, or <b>TRADES</b> introduces complex trade-offs between clean accuracy retention and adversarial defense strength.
            </p>
            <h4 style="margin-top: 15px;">💡 How RobustLens AI Helps:</h4>
            <ul>
                <li><b>Automated Drop Metrics</b>: Computes Clean - FGSM and Clean - PGD accuracy degradation instantly.</li>
                <li><b>Weighted Robustness Score</b>: Combines clean retention and attack defenses (0.20·Clean + 0.35·FGSM + 0.45·PGD).</li>
                <li><b>Interactive Visualizations</b>: Grouped multi-bar charts, Radar charts, and Pareto Frontier trade-off analysis.</li>
                <li><b>Epsilon Attack Simulator</b>: Simulates model accuracy breakdown across increasing attack strengths (ε).</li>
                <li><b>Gemini AI Diagnostics</b>: Custom system instruction evaluates trade-offs and suggests next steps.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown("""
        <div class="glass-card">
            <h3>⚡ Quickstart Workflow</h3>
            <ol style="padding-left: 20px;">
                <li><b>Experiment Analyzer</b>: Add custom experiment data or import CSV logs.</li>
                <li><b>Model Comparison</b>: View performance leaderboards, radar charts, & best-in-class highlights.</li>
                <li><b>Pareto Frontier</b>: Analyze clean vs. robust accuracy trade-offs.</li>
                <li><b>Epsilon Simulator</b>: Model performance degradation as attack strength grows.</li>
                <li><b>AI Assistant</b>: Generate paper-ready commentary with Google Gemini.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📚 Adversarial Attack Reference")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(r"""
        <div class="glass-card">
            <h4>⚡ FGSM (Fast Gradient Sign Method)</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">Single-step linear gradient perturbation attack.</p>
            <code>x_adv = x + ε · sign(∇_x L(θ, x, y))</code>
            <p style="margin-top: 10px; font-size: 0.95rem;">
                Fast baseline evaluation for first-order gradient alignment vulnerabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_b:
        st.markdown(r"""
        <div class="glass-card">
            <h4>🔄 PGD (Projected Gradient Descent)</h4>
            <p style="color: #94A3B8; font-size: 0.9rem;">Multi-step iterative projected gradient attack.</p>
            <code>x^{t+1} = Π_{x+S}(x^t + α · sign(∇_x L(θ, x^t, y)))</code>
            <p style="margin-top: 10px; font-size: 0.95rem;">
                Considered the standard benchmark for non-obfuscated multi-step attack resistance.
            </p>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 2: EXPERIMENT ANALYZER
# ---------------------------------------------------------
elif page == "🧪 Experiment Analyzer":
    st.markdown("## 🧪 Experiment Analyzer & Input Logger")
    st.caption("Log single experiment results or batch import CSV files.")
    
    tab1, tab2 = st.columns(2)
    
    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ➕ Add Single Experiment")
        with st.form("add_experiment_form", clear_on_submit=True):
            exp_name = st.text_input("Experiment Name", placeholder="e.g. ViT-B/16 TRADES-beta6")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                model_name = st.text_input("Model Architecture", value="ViT-B/16")
                dataset = st.text_input("Dataset", value="CIFAR-10")
            with col_m2:
                finetune_method = st.text_input("Fine-tuning Method", value="TRADES Fine-Tuning")
                epsilon = st.text_input("Attack Epsilon (ε)", value="8/255")
            
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                clean_acc = st.number_input("Clean Acc (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
            with col_a2:
                fgsm_acc = st.number_input("FGSM Acc (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
            with col_a3:
                pgd_acc = st.number_input("PGD Acc (%)", min_value=0.0, max_value=100.0, value=48.0, step=0.1)
            
            epochs = st.number_input("Training Epochs", min_value=1, max_value=1000, value=20, step=1)
            
            submitted = st.form_submit_button("📥 Save Experiment to Workspace", use_container_width=True)
            
            if submitted:
                if not exp_name.strip():
                    st.error("⚠️ Please enter an Experiment Name.")
                else:
                    if fgsm_acc > clean_acc or pgd_acc > clean_acc:
                        st.warning("⚠️ Warning: FGSM/PGD accuracy exceeds Clean Accuracy. Please verify inputs.")
                    
                    new_row = pd.DataFrame([{
                        "Experiment Name": exp_name.strip(),
                        "Model Name": model_name.strip(),
                        "Dataset": dataset.strip(),
                        "Fine-tuning Method": finetune_method.strip(),
                        "Clean Accuracy (%)": clean_acc,
                        "FGSM Accuracy (%)": fgsm_acc,
                        "PGD Accuracy (%)": pgd_acc,
                        "Epsilon": epsilon.strip(),
                        "Epochs": int(epochs)
                    }])
                    st.session_state.experiments = pd.concat([st.session_state.experiments, new_row], ignore_index=True)
                    st.success(f"✅ Added '{exp_name}' successfully!")
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📁 Batch CSV Import / Export")
        uploaded_file = st.file_uploader("Upload Experiment Log (.csv)", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                req_cols = ["Experiment Name", "Clean Accuracy (%)", "FGSM Accuracy (%)", "PGD Accuracy (%)"]
                missing = [c for c in req_cols if c not in uploaded_df.columns]
                if missing:
                    st.error(f"❌ Missing required CSV columns: {missing}")
                else:
                    st.session_state.experiments = pd.concat([st.session_state.experiments, uploaded_df], ignore_index=True).drop_duplicates(subset=["Experiment Name"])
                    st.success("✅ Batch dataset imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error loading CSV: {e}")

        st.markdown("---")
        sample_csv_data = """Experiment Name,Model Name,Dataset,Fine-tuning Method,Clean Accuracy (%),FGSM Accuracy (%),PGD Accuracy (%),Epsilon,Epochs
ViT-B/16 Baseline,ViT-B/16,CIFAR-10,Standard Fine-Tuning,92.4,24.1,2.8,8/255,10
ViT-B/16 FGSM-AT,ViT-B/16,CIFAR-10,FGSM Adversarial Training,88.2,65.4,18.2,8/255,15
ViT-B/16 PGD-7 AT,ViT-B/16,CIFAR-10,PGD-7 Adversarial Training,84.6,61.2,48.5,8/255,20
ViT-B/16 TRADES,ViT-B/16,CIFAR-10,TRADES (beta=6.0),83.1,59.8,51.4,8/255,25"""
        
        st.download_button(
            label="📄 Download Sample CSV Template",
            data=sample_csv_data,
            file_name="sample_results.csv",
            mime="text/csv",
            use_container_width=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📊 Active Experiments & Accuracy Drop Table")
    
    if st.session_state.experiments.empty:
        st.info("No experiments in workspace. Enter an experiment above or reset sample data.")
    else:
        calc_df = calculate_metrics(st.session_state.experiments)
        st.dataframe(
            calc_df[[
                "Experiment Name", "Model Name", "Fine-tuning Method",
                "Clean Accuracy (%)", "FGSM Accuracy (%)", "FGSM Drop (%)", 
                "PGD Accuracy (%)", "PGD Drop (%)", "Robustness Score"
            ]],
            use_container_width=True
        )

# ---------------------------------------------------------
# PAGE 3: MODEL COMPARISON
# ---------------------------------------------------------
elif page == "📊 Model Comparison":
    st.markdown("## 📊 Model Comparison & Leaderboard")
    st.caption("Comprehensive comparative metrics, best-in-class callouts, and multi-axis charts.")
    
    if st.session_state.experiments.empty:
        st.warning("⚠️ Workspace is empty. Add experiments in the Experiment Analyzer.")
    else:
        calc_df = calculate_metrics(st.session_state.experiments)
        
        # Best Models Indicators
        best_clean = calc_df.loc[calc_df["Clean Accuracy (%)"].idxmax()]
        best_fgsm = calc_df.loc[calc_df["FGSM Accuracy (%)"].idxmax()]
        best_pgd = calc_df.loc[calc_df["PGD Accuracy (%)"].idxmax()]
        best_overall = calc_df.loc[calc_df["Robustness Score"].idxmax()]
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">Best Clean Acc</div>
                <div class="metric-val">{best_clean["Clean Accuracy (%)"]:.1f}%</div>
                <div class="metric-sub">{best_clean["Experiment Name"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">Best FGSM Defense</div>
                <div class="metric-val">{best_fgsm["FGSM Accuracy (%)"]:.1f}%</div>
                <div class="metric-sub">{best_fgsm["Experiment Name"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">Best PGD Defense</div>
                <div class="metric-val">{best_pgd["PGD Accuracy (%)"]:.1f}%</div>
                <div class="metric-sub">{best_pgd["Experiment Name"]}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with c4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-lbl">Top Robustness Score</div>
                <div class="metric-val">{best_overall["Robustness Score"]:.2f}</div>
                <div class="metric-sub">{best_overall["Experiment Name"]}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        
        col_g1, col_g2 = st.columns([3, 2])
        
        with col_g1:
            st.markdown("### 📈 Grouped Performance Bar Chart")
            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(x=calc_df["Experiment Name"], y=calc_df["Clean Accuracy (%)"], name="Clean Accuracy", marker_color="#38BDF8"))
            fig_bar.add_trace(go.Bar(x=calc_df["Experiment Name"], y=calc_df["FGSM Accuracy (%)"], name="FGSM Accuracy", marker_color="#F59E0B"))
            fig_bar.add_trace(go.Bar(x=calc_df["Experiment Name"], y=calc_df["PGD Accuracy (%)"], name="PGD Accuracy", marker_color="#EF4444"))
            
            fig_bar.update_layout(
                barmode="group",
                yaxis=dict(title="Accuracy (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_g2:
            st.markdown("### 🕸️ Multi-Axis Radar Chart")
            radar_df = calc_df.head(4)
            fig_radar = go.Figure()
            
            categories = ['Clean Acc', 'FGSM Acc', 'PGD Acc', 'Robustness Score']
            for _, row in radar_df.iterrows():
                fig_radar.add_trace(go.Scatterpolar(
                    r=[row['Clean Accuracy (%)'], row['FGSM Accuracy (%)'], row['PGD Accuracy (%)'], row['Robustness Score']],
                    theta=categories,
                    fill='toself',
                    name=row['Experiment Name']
                ))
            
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(255,255,255,0.15)"),
                    angularaxis=dict(gridcolor="rgba(255,255,255,0.15)")
                ),
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#F8FAFC"),
                showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.markdown("---")
        st.markdown("### 🏆 Robustness Leaderboard Table")
        
        leader_df = calc_df.sort_values(by="Robustness Score", ascending=False).reset_index(drop=True)
        leader_df.index += 1
        
        st.dataframe(
            leader_df[[
                "Experiment Name", "Model Name", "Fine-tuning Method",
                "Clean Accuracy (%)", "FGSM Accuracy (%)", "PGD Accuracy (%)",
                "FGSM Drop (%)", "PGD Drop (%)", "Robustness Score"
            ]],
            use_container_width=True
        )
        
        csv_bytes = leader_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Leaderboard CSV Report",
            data=csv_bytes,
            file_name="robustlens_leaderboard.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------
# PAGE 4: PARETO TRADE-OFF FRONTIER
# ---------------------------------------------------------
elif page == "📈 Pareto Trade-Off Frontier":
    st.markdown("## 📈 Clean vs. Robust Accuracy Trade-Off Frontier")
    st.caption("Analyze the Pareto Frontier: Clean Accuracy retention vs. PGD multi-step defense.")
    
    if st.session_state.experiments.empty:
        st.warning("⚠️ Workspace is empty. Add experiments in Experiment Analyzer.")
    else:
        calc_df = calculate_metrics(st.session_state.experiments)
        
        fig_scatter = px.scatter(
            calc_df,
            x="Clean Accuracy (%)",
            y="PGD Accuracy (%)",
            size="Robustness Score",
            color="Model Name",
            hover_name="Experiment Name",
            hover_data=["Fine-tuning Method", "FGSM Accuracy (%)", "Epochs"],
            text="Experiment Name",
            title="Pareto Trade-Off Frontier (Clean vs. PGD Defense)"
        )
        
        fig_scatter.update_traces(textposition='top center', marker=dict(line=dict(width=1, color='White')))
        fig_scatter.update_layout(
            xaxis=dict(title="Clean Accuracy (%)", range=[50, 100], gridcolor="rgba(255,255,255,0.1)"),
            yaxis=dict(title="PGD Robust Accuracy (%)", range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            height=550
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("""
        <div class="glass-card">
            <h4>💡 How to Interpret the Pareto Frontier:</h4>
            <ul>
                <li><b>Top-Right Quadrant</b>: Ideal models (High Clean Accuracy + High PGD Defense).</li>
                <li><b>Top-Left Quadrant</b>: High Defense, but trade-off lost clean performance.</li>
                <li><b>Bottom-Right Quadrant</b>: High Clean performance, but vulnerable under PGD attacks.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 5: EPSILON SIMULATOR
# ---------------------------------------------------------
elif page == "🎛️ Epsilon Simulator":
    st.markdown("## 🎛️ Epsilon (ε) Attack Breakdown Simulator")
    st.caption("Simulate accuracy breakdown across varying perturbation strengths ε.")
    
    if st.session_state.experiments.empty:
        st.warning("⚠️ Workspace is empty. Add experiments in Experiment Analyzer.")
    else:
        calc_df = calculate_metrics(st.session_state.experiments)
        
        st.markdown("### ⚙️ Simulation Settings")
        selected_model = st.selectbox("Select Model for Simulation", calc_df["Experiment Name"].unique())
        model_row = calc_df[calc_df["Experiment Name"] == selected_model].iloc[0]
        
        eps_range = np.linspace(0, 16/255, 20)
        eps_labels = [f"{int(e*255)}/255" for e in eps_range]
        
        clean = model_row["Clean Accuracy (%)"]
        fgsm = model_row["FGSM Accuracy (%)"]
        pgd = model_row["PGD Accuracy (%)"]
        
        fgsm_curve = [clean * np.exp(- (e / (8/255)) * np.log(clean/max(fgsm, 0.1))) for e in eps_range]
        pgd_curve = [clean * np.exp(- (e / (8/255)) * np.log(clean/max(pgd, 0.1))) for e in eps_range]
        
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Scatter(x=eps_labels, y=fgsm_curve, mode='lines+markers', name='Simulated FGSM Decay', line=dict(color='#F59E0B', width=3)))
        fig_sim.add_trace(go.Scatter(x=eps_labels, y=pgd_curve, mode='lines+markers', name='Simulated PGD Decay', line=dict(color='#EF4444', width=3)))
        
        fig_sim.update_layout(
            title=f"Attack Breakdown Curve: {selected_model}",
            xaxis_title="Epsilon Perturbation Strength (ε)",
            yaxis_title="Retained Accuracy (%)",
            yaxis=dict(range=[0, 100], gridcolor="rgba(255,255,255,0.1)"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.1)"),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#F8FAFC"),
            height=480
        )
        
        st.plotly_chart(fig_sim, use_container_width=True)

# ---------------------------------------------------------
# PAGE 6: AI RESEARCH ASSISTANT (HIGHLY ORGANIZED UI)
# ---------------------------------------------------------
elif page == "🤖 AI Research Assistant":
    st.markdown("## 🤖 AI Research Assistant")
    st.caption("Paper-ready diagnostics, trade-off evaluation, and experiment recommendations powered by Google Gemini AI.")
    
    if st.session_state.experiments.empty:
        st.warning("⚠️ Workspace is empty. Add experiments to generate AI research insights.")
    else:
        calc_df = calculate_metrics(st.session_state.experiments)
        api_key = get_gemini_key()
        
        # Header Controls & Key Status Card
        col_k1, col_k2 = st.columns([3, 2])
        with col_k1:
            st.markdown('<div class="glass-card" style="padding:16px;">', unsafe_allow_html=True)
            user_key_input = st.text_input(
                "🔑 Google Gemini API Key",
                value=api_key,
                type="password",
                placeholder="AIzaSy...",
                help="Key can be set in GEMINI_API_KEY environment variable or Streamlit secrets."
            )
            final_api_key = user_key_input.strip() if user_key_input else api_key
            st.markdown('</div>', unsafe_allow_html=True)
            
        with col_k2:
            st.markdown('<div class="glass-card" style="padding:16px; text-align:center;">', unsafe_allow_html=True)
            if final_api_key:
                st.markdown("#### API Status: <span style='color:#34D399;'>🟢 Connected</span>", unsafe_allow_html=True)
                st.caption("Gemini 2.5 Flash / 1.5 Flash Model Ready")
            else:
                st.markdown("#### API Status: <span style='color:#F87171;'>🔴 Key Required</span>", unsafe_allow_html=True)
                st.caption("Enter API key to generate insights")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Generation Button & Inspector
        col_b1, col_b2 = st.columns([2, 3])
        with col_b1:
            run_ai = st.button("🚀 Run AI Research Analysis", type="primary", use_container_width=True)
        with col_b2:
            with st.expander("🔍 Inspect System Prompt & Payload Sent to Gemini"):
                st.json(calc_df.to_dict(orient="records"))
                st.code(SYSTEM_PROMPT, language="text")

        if run_ai:
            if not final_api_key:
                st.error("❌ Gemini API Key is missing. Please enter your API key above.")
            else:
                with st.spinner("🧠 Gemini AI is synthesizing model performance and trade-offs..."):
                    payload_summary = calc_df.to_string(index=False)
                    user_query = f"""Here are the experimental results for evaluation:

{payload_summary}

Please analyze these results strictly according to your instructions."""

                    ai_response_text = ""
                    try:
                        from google import genai
                        from google.genai import types
                        
                        client = genai.Client(api_key=final_api_key)
                        try:
                            res = client.models.generate_content(
                                model="gemini-2.5-flash",
                                contents=user_query,
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_PROMPT,
                                    temperature=0.3
                                )
                            )
                            ai_response_text = res.text
                        except Exception:
                            res = client.models.generate_content(
                                model="gemini-1.5-flash",
                                contents=user_query,
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_PROMPT,
                                    temperature=0.3
                                )
                            )
                            ai_response_text = res.text
                    except Exception as err:
                        st.error(f"❌ Gemini Generation Error: {err}")
                    
                    if ai_response_text:
                        st.session_state["last_ai_analysis"] = ai_response_text

        # ORGANIZED DISPLAY OF AI RESULTS
        if "last_ai_analysis" in st.session_state:
            raw_text = st.session_state["last_ai_analysis"]
            
            st.markdown("---")
            st.markdown("### 📋 Structured Academic Diagnostic Report")
            
            tab_report1, tab_report2 = st.tabs(["📑 Organized Visual Dashboard", "📄 Raw Paper Report & Export"])
            
            with tab_report1:
                # Helper function to extract sections from AI response text
                def parse_sections(text):
                    sections = {
                        "summary": "",
                        "comparison": "",
                        "finetuning": "",
                        "tradeoff": "",
                        "next_experiment": "",
                        "limitations": ""
                    }
                    
                    # Splitting text into paragraphs or numbered sections
                    blocks = re.split(r'\n(?=\d+\.|\#\#|\*\*Task|\*\*1|\*\*2|\*\*3|\*\*4|\*\*5|\*\*6)', text)
                    
                    for block in blocks:
                        lower = block.lower()
                        if "main result" in lower or "summarise" in lower or "summary" in lower or "1." in block[:4]:
                            sections["summary"] += block + "\n"
                        elif "trade" in lower or "clean-accuracy versus" in lower or "4." in block[:4]:
                            sections["tradeoff"] += block + "\n"
                        elif "recommend" in lower or "next experiment" in lower or "5." in block[:4]:
                            sections["next_experiment"] += block + "\n"
                        elif "limitation" in lower or "caution" in lower or "6." in block[:4]:
                            sections["limitations"] += block + "\n"
                        elif "fine-tuning" in lower or "improved robustness" in lower or "3." in block[:4]:
                            sections["finetuning"] += block + "\n"
                        else:
                            sections["comparison"] += block + "\n"
                    return sections

                parsed = parse_sections(raw_text)
                
                # Card 1: Main Finding & Executive Summary
                st.markdown(f"""
                <div class="report-card-summary">
                    <h3 style="color:#38BDF8 !important; margin-top:0;">📌 1. Main Findings & Executive Summary</h3>
                    <div style="font-size:1.05rem; line-height:1.7;">
                        {parsed['summary'] if parsed['summary'].strip() else raw_text[:300] + '...'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Card 2 & 3 Dual Columns
                col_r1, col_r2 = st.columns(2)
                
                with col_r1:
                    st.markdown(f"""
                    <div class="report-card-tradeoff">
                        <h4 style="color:#FBBF24 !important; margin-top:0;">⚖️ 2. Clean vs. Robust Accuracy Trade-Off</h4>
                        <div style="font-size:0.98rem; line-height:1.7;">
                            {parsed['tradeoff'] if parsed['tradeoff'].strip() else 'Analyzing trade-off between clean classification and attack resistance...'}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with col_r2:
                    st.markdown(f"""
                    <div class="glass-card" style="border-color:rgba(129, 140, 248, 0.4);">
                        <h4 style="color:#818CF8 !important; margin-top:0;">🛡️ 3. Fine-Tuning & Attack Defense Breakdown</h4>
                        <div style="font-size:0.98rem; line-height:1.7;">
                            {parsed['finetuning'] if parsed['finetuning'].strip() else parsed['comparison']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # Card 4: Recommended Next Experiment
                st.markdown(f"""
                <div class="report-card-recommendation">
                    <h4 style="color:#34D399 !important; margin-top:0;">🧪 4. Recommended Next Experiment</h4>
                    <div style="font-size:1.02rem; line-height:1.7;">
                        {parsed['next_experiment'] if parsed['next_experiment'].strip() else 'Perform evaluation with AutoAttack ensemble to rule out gradient obfuscation.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Card 5: Research Limitations & Caveats
                if parsed['limitations'].strip():
                    st.markdown(f"""
                    <div class="report-card-limitation">
                        <h4 style="color:#F87171 !important; margin-top:0;">⚠️ 5. Research Limitations & Caveats</h4>
                        <div style="font-size:0.98rem; line-height:1.7;">
                            {parsed['limitations']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with tab_report2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.markdown(raw_text)
                st.markdown('</div>', unsafe_allow_html=True)
                
                report_md = f"""# RobustLens AI - Academic Research Report

## Experimental Benchmark Logs Evaluated:
```
{calc_df.to_string(index=False)}
```

## AI Generated Diagnostic Analysis:
{raw_text}
"""
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.download_button(
                        label="📥 Download Research Report (.md)",
                        data=report_md,
                        file_name="robustlens_ai_analysis_report.md",
                        mime="text/markdown",
                        use_container_width=True
                    )
                with col_d2:
                    st.download_button(
                        label="📥 Download Benchmark Metrics (.csv)",
                        data=calc_df.to_csv(index=False).encode('utf-8'),
                        file_name="robustlens_calculated_metrics.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
