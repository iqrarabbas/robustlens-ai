import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import io
import re
import json

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
    html, body, p, div, label, li, h1, h2, h3, h4, h5, h6 {
        font-family: 'Inter', sans-serif;
        color: #F8FAFC !important;
    }
    
    /* Exclude & Protect Streamlit Icon Fonts from Font Overrides */
    [data-testid="stIcon"], 
    [class*="material-symbols"], 
    [class*="material-icons"], 
    [class*="StreamlitIcon"], 
    i, 
    summary span:first-child,
    button [class*="css"] {
        font-family: inherit !important;
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
    div[data-baseweb="input"] > div, div[data-baseweb="select"] > div, div[data-baseweb="textarea"] > div {
        background-color: #1E293B !important;
        border-color: rgba(255, 255, 255, 0.2) !important;
        color: #F8FAFC !important;
    }
    
    input, textarea {
        color: #F8FAFC !important;
    }

    /* Form Label Styling */
    .stTextInput > label, .stNumberInput > label, .stSelectbox > label, .stFileUploader > label, .stTextArea > label {
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

PAPER_EXTRACTION_PROMPT = """You are an expert Machine Learning paper parser. 
Extract all model performance metrics mentioned in the user's text into a clean JSON array.

Return ONLY a valid JSON array of objects with the following keys:
- "Experiment Name": (String, short name e.g. "ViT-B/16 TRADES")
- "Model Name": (String e.g. "ViT-B/16" or "ResNet-50")
- "Dataset": (String e.g. "CIFAR-10" or "ImageNet-1k")
- "Fine-tuning Method": (String e.g. "TRADES" or "Adversarial Training")
- "Clean Accuracy (%)": (Float number e.g. 84.5)
- "FGSM Accuracy (%)": (Float number e.g. 62.1)
- "PGD Accuracy (%)": (Float number e.g. 52.3)
- "Epsilon": (String e.g. "8/255")
- "Epochs": (Integer number e.g. 20)

If FGSM or PGD accuracy is missing in text, estimate reasonable values or use 0.0 based strictly on context.
Do NOT include markdown formatting wrappers like ```json. Return raw JSON string."""

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
            <span style="font-size: 0.75rem; color: #94A3B8 !important;">v1.3.0 • Pro Analytics</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.caption("Adversarial Robustness Analytics & AI Assistant")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📄 Paper & Abstract Extractor",
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
    st.markdown('<div class="sub-title">Empowering Machine Learning Researchers & Students to Benchmark, Analyze, and Defend Models Against Adversarial Threats</div>', unsafe_allow_html=True)
    
    # PROMINENT "WHAT THIS APP IS ABOUT" SUMMARY CARD
    st.markdown("""
    <div class="glass-card" style="border-left: 5px solid #38BDF8; background: linear-gradient(135deg, rgba(56, 189, 248, 0.12), rgba(30, 41, 59, 0.85)); padding: 24px; margin-bottom: 25px;">
        <h3 style="color:#38BDF8 !important; margin-top:0;">🎯 What is RobustLens AI & What Does It Do?</h3>
        <p style="font-size: 1.08rem; line-height: 1.7; color: #F8FAFC !important; margin-bottom: 12px;">
            <b>RobustLens AI</b> is an all-in-one diagnostic & benchmarking platform built for <b>students, researchers, and AI engineers</b> testing Vision Transformers (ViTs) and CNNs under adversarial attacks.
        </p>
        <p style="font-size: 1.02rem; line-height: 1.7; color: #CBD5E1 !important; margin-bottom: 16px;">
            <b>The Problem:</b> When machine learning models are attacked with imperceptible noise (like <b>FGSM</b> or <b>PGD</b>), their performance can drop from <b>92% down to 2%</b>. Evaluating defensive fine-tuning methods requires analyzing complex trade-offs between clean accuracy retention and defense strength.
        </p>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; font-size: 0.95rem;">
            <div style="background:rgba(15,23,42,0.6); padding:10px 14px; border-radius:8px; border:1px solid rgba(255,255,255,0.08);">📥 <b>1. Log & Extract Data</b><br><span style="color:#94A3B8; font-size:0.85rem;">Manual entry, CSV uploads, or AI paper parser.</span></div>
            <div style="background:rgba(15,23,42,0.6); padding:10px 14px; border-radius:8px; border:1px solid rgba(255,255,255,0.08);">📊 <b>2. Auto-Calculate Drop</b><br><span style="color:#94A3B8; font-size:0.85rem;">Computes FGSM/PGD drop & Weighted Score.</span></div>
            <div style="background:rgba(15,23,42,0.6); padding:10px 14px; border-radius:8px; border:1px solid rgba(255,255,255,0.08);">📈 <b>3. Visual Leaderboards</b><br><span style="color:#94A3B8; font-size:0.85rem;">Bar graphs, radar charts, & Pareto frontiers.</span></div>
            <div style="background:rgba(15,23,42,0.6); padding:10px 14px; border-radius:8px; border:1px solid rgba(255,255,255,0.08);">🤖 <b>4. AI Research Reports</b><br><span style="color:#94A3B8; font-size:0.85rem;">Gemini AI generates paper-ready analysis.</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Status Summary Row
    calc_df = calculate_metrics(st.session_state.experiments)
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-lbl">Active Experiments</div>
            <div class="metric-val">{len(calc_df)}</div>
            <div class="metric-sub">Loaded in Workspace</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-lbl">Evaluated Models</div>
            <div class="metric-val">{calc_df['Model Name'].nunique() if not calc_df.empty else 0}</div>
            <div class="metric-sub">Architectures</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-lbl">Attacks Benchmark</div>
            <div class="metric-val">FGSM & PGD</div>
            <div class="metric-sub">Single & Multi-step</div>
        </div>
        """, unsafe_allow_html=True)
    with c_m4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-lbl">AI Diagnostic Engine</div>
            <div class="metric-val" style="color:#34D399;">Gemini AI</div>
            <div class="metric-sub">Custom Research Prompt</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Interactive Beginner-Friendly Explanation Guide
    st.markdown("### 🎓 Adversarial ML 101: Simple Explanations & Real-World Analogies")
    
    tab_exp1, tab_exp2, tab_exp3 = st.tabs(["🚗 Real-World Analogy", "⚡ FGSM vs PGD Explained", "💡 The Clean vs. Robust Trade-Off"])
    
    with tab_exp1:
        st.markdown("""
        <div class="glass-card">
            <h4>🚗 The "Stormy Obstacle Course" Analogy</h4>
            <p>Imagine two different athletes:</p>
            <ul>
                <li><b>Clean Accuracy Athlete</b>: Performs exceptionally fast on a dry, sunny running track (e.g. <b>92% score</b> on standard clean images). However, put them in a muddy, stormy obstacle course, and they slip immediately (dropping to <b>2.8%</b>).</li>
                <li><b>Robust Accuracy Athlete</b>: Trains heavily in heavy rain and mud (<b>Adversarial Training</b>). They run a little slower on the sunny track (e.g. <b>83% clean score</b>), but easily navigate the stormy obstacle course (scoring <b>51.4% robust score</b> under PGD attack).</li>
            </ul>
            <p style="color:#38BDF8;"><b>RobustLens AI helps you find the perfect balance between track speed and obstacle resilience!</b></p>
        </div>
        """, unsafe_allow_html=True)

    with tab_exp2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(r"""
            <div class="glass-card">
                <h4 style="color:#F59E0B !important;">⚡ FGSM (Fast Gradient Sign Method)</h4>
                <p style="color: #94A3B8; font-size: 0.85rem;">Single-step linear perturbation attack.</p>
                <code>x_adv = x + ε · sign(∇_x L(θ, x, y))</code>
                <p style="margin-top: 10px; font-size: 0.9rem;">
                    <b>Analogy</b>: Pushing a car once down a hill. It tests if a model breaks under a single quick nudge.
                </p>
            </div>
            """, unsafe_allow_html=True)
        with col_b:
            st.markdown(r"""
            <div class="glass-card">
                <h4 style="color:#EF4444 !important;">🔄 PGD (Projected Gradient Descent)</h4>
                <p style="color: #94A3B8; font-size: 0.85rem;">Multi-step iterative projected gradient attack.</p>
                <code>x^{t+1} = Π_{x+S}(x^t + α · sign(∇_x L(θ, x^t, y)))</code>
                <p style="margin-top: 10px; font-size: 0.9rem;">
                    <b>Analogy</b>: Repeatedly steering the car off course step-by-step. PGD gets 45% weight in our Robustness Score.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with tab_exp3:
        st.markdown("""
        <div class="glass-card">
            <h4>⚖️ Why Can't We Have 100% Clean AND 100% Robust Accuracy?</h4>
            <p>
                Standard machine learning models rely on <i>non-robust features</i> (subtle background patterns in images) to achieve ultra-high clean accuracy. 
                Adversarial defense fine-tuning forces the model to ignore non-robust patterns and focus strictly on <i>robust core shape features</i>. 
                This causes a minor drop in clean accuracy in exchange for strong defense under attack!
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # 4 Core Modules Navigation Overview
    st.markdown("### ⚡ App Feature Modules")
    mod1, mod2, mod3, mod4 = st.columns(4)
    
    with mod1:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <h3>📄</h3>
            <h4>Paper Parser</h4>
            <p style="font-size:0.85rem; color:#94A3B8;">Paste abstracts or upload research paper files to auto-extract logs.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with mod2:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <h3>📊</h3>
            <h4>Model Leaderboard</h4>
            <p style="font-size:0.85rem; color:#94A3B8;">Leaderboards, grouped bar charts, & 4-axis radar charts.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with mod3:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <h3>📈</h3>
            <h4>Pareto & Simulator</h4>
            <p style="font-size:0.85rem; color:#94A3B8;">Clean vs. Robust trade-off frontiers & attack decay curves.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with mod4:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <h3>🤖</h3>
            <h4>AI Co-Pilot</h4>
            <p style="font-size:0.85rem; color:#94A3B8;">Academic research reports & experiment recommendations.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    
    # Sample Benchmark Preview Table
    st.markdown("### 📋 Sample Benchmark Dataset Preview")
    st.caption("Pre-loaded research evaluation logs on CIFAR-10 benchmark:")
    st.dataframe(
        calc_df[[
            "Experiment Name", "Model Name", "Fine-tuning Method",
            "Clean Accuracy (%)", "FGSM Accuracy (%)", "PGD Accuracy (%)", "Robustness Score"
        ]],
        use_container_width=True
    )

# ---------------------------------------------------------
# PAGE 2: RESEARCH PAPER & ABSTRACT PARSER (NEW FEATURE)
# ---------------------------------------------------------
elif page == "📄 Paper & Abstract Extractor":
    st.markdown("## 📄 Research Paper & Abstract Extractor")
    st.caption("Paste an academic paper abstract or text log—Gemini AI will extract experimental parameters automatically.")
    
    api_key = get_gemini_key()
    
    # Pre-fill sample abstract handler
    sample_abstract = """We evaluated Vision Transformer (ViT-B/16) fine-tuned with TRADES (beta=6.0) on the CIFAR-10 benchmark. Our baseline ViT-B/16 model achieved 92.4% clean accuracy, but dropped to 24.1% under FGSM attack (eps=8/255) and 2.8% under PGD-20 attack. When fine-tuned with TRADES for 25 epochs, clean accuracy was 83.1%, while robust accuracy improved significantly to 59.8% under FGSM and 51.4% under PGD multi-step attack."""
    
    col_p1, col_p2 = st.columns([3, 2])
    
    with col_p1:
        st.markdown("### ✍️ Paste Paper Abstract or Table Text")
        
        col_btn1, col_btn2 = st.columns([2, 3])
        with col_btn1:
            if st.button("📄 Load Sample Abstract", use_container_width=True):
                st.session_state["paper_input_text"] = sample_abstract
                st.rerun()
                
        default_text_val = st.session_state.get("paper_input_text", sample_abstract)
        
        paper_text_input = st.text_area(
            "Paper Abstract / Experimental Results Text",
            value=default_text_val,
            height=210,
            placeholder="Paste research text here..."
        )
        
        uploaded_paper = st.file_uploader("Or Upload Paper Text File (.txt, .md)", type=["txt", "md"])
        if uploaded_paper is not None:
            try:
                paper_text_input = uploaded_paper.read().decode("utf-8")
                st.session_state["paper_input_text"] = paper_text_input
                st.success("✅ Paper file uploaded successfully!")
            except Exception as pe:
                st.error(f"❌ Error reading file: {pe}")

    with col_p2:
        st.markdown("### 🔑 API Key & Extraction Control")
        user_key = st.text_input("Gemini API Key", value=api_key, type="password", placeholder="AIzaSy...")
        active_key = user_key.strip() if user_key else api_key
        
        st.markdown("""
        <div style="background: rgba(30, 41, 59, 0.6); padding: 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); margin-top: 15px; margin-bottom: 20px;">
            <p style="margin: 0; font-size: 0.9rem; color: #CBD5E1;">
                💡 <b>How it works:</b> Gemini AI will analyze the text above, extract model names, datasets, clean accuracy, FGSM/PGD accuracies, attack epsilons, and epoch counts, and format them into an interactive table ready for workspace import.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        extract_btn = st.button("🧠 Extract Experiments with Gemini AI", type="primary", use_container_width=True)

    if extract_btn:
        if not active_key:
            st.error("❌ Gemini API Key is required to extract paper metrics.")
        elif not paper_text_input.strip():
            st.error("⚠️ Please paste text or upload a paper text file.")
        else:
            with st.spinner("🧠 Gemini AI is extracting model architecture and attack metrics..."):
                try:
                    from google import genai
                    from google.genai import types
                    
                    client = genai.Client(api_key=active_key)
                    
                    prompt = f"""Extract experimental results from this text:
{paper_text_input}"""

                    try:
                        res = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=PAPER_EXTRACTION_PROMPT,
                                temperature=0.1
                            )
                        )
                        raw_json = res.text
                    except Exception:
                        res = client.models.generate_content(
                            model="gemini-1.5-flash",
                            contents=prompt,
                            config=types.GenerateContentConfig(
                                system_instruction=PAPER_EXTRACTION_PROMPT,
                                temperature=0.1
                            )
                        )
                        raw_json = res.text
                        
                    # Clean up JSON wrappers if any
                    raw_json = re.sub(r'```json\s*', '', raw_json)
                    raw_json = re.sub(r'```\s*', '', raw_json).strip()
                    
                    extracted_list = json.loads(raw_json)
                    extracted_df = pd.DataFrame(extracted_list)
                    st.session_state["extracted_paper_df"] = extracted_df
                    st.success(f"✅ Extracted {len(extracted_df)} experiment(s) from paper!")
                    
                except Exception as ex:
                    st.error(f"❌ Extraction Error: {ex}. Ensure the text contains accuracy metrics.")

    if "extracted_paper_df" in st.session_state:
        st.markdown("---")
        st.markdown("### 📋 Preview Extracted Paper Experiments")
        extracted_df = st.session_state["extracted_paper_df"]
        st.dataframe(extracted_df, use_container_width=True)
        
        if st.button("📥 Import Extracted Experiments into Main Workspace", type="primary", use_container_width=True):
            st.session_state.experiments = pd.concat([st.session_state.experiments, extracted_df], ignore_index=True).drop_duplicates(subset=["Experiment Name"])
            st.success("✅ Extracted experiments added to workspace!")
            st.rerun()

# ---------------------------------------------------------
# PAGE 3: EXPERIMENT ANALYZER
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
# PAGE 4: MODEL COMPARISON
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
# PAGE 5: PARETO TRADE-OFF FRONTIER
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
# PAGE 6: EPSILON SIMULATOR
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
# PAGE 7: AI RESEARCH ASSISTANT (HIGHLY ORGANIZED UI)
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
        st.markdown('<div class="glass-card" style="padding:20px;">', unsafe_allow_html=True)
        col_k1, col_k2 = st.columns([3, 1.2])
        with col_k1:
            user_key_input = st.text_input(
                "🔑 Google Gemini API Key",
                value=api_key,
                type="password",
                placeholder="AIzaSy...",
                help="Key can be set in GEMINI_API_KEY environment variable or Streamlit secrets."
            )
            final_api_key = user_key_input.strip() if user_key_input else api_key
        with col_k2:
            st.markdown("<div style='margin-top: 32px; text-align: right;'>", unsafe_allow_html=True)
            if final_api_key:
                st.markdown("<span style='background:rgba(52,211,153,0.15); color:#34D399; padding:8px 14px; border-radius:8px; border:1px solid rgba(52,211,153,0.3); font-weight:600; font-size:0.9rem;'>🟢 API Connected</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='background:rgba(248,113,113,0.15); color:#F87171; padding:8px 14px; border-radius:8px; border:1px solid rgba(248,113,113,0.3); font-weight:600; font-size:0.9rem;'>🔴 Key Required</span>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")
        
        # Generation Action Button
        run_ai = st.button("🚀 Run AI Research Analysis", type="primary", use_container_width=True)
        
        # System Prompt & Payload Inspector
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
                def parse_sections(text):
                    sections = {
                        "summary": "",
                        "comparison": "",
                        "finetuning": "",
                        "tradeoff": "",
                        "next_experiment": "",
                        "limitations": ""
                    }
                    
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
                
                st.markdown(f"""
                <div class="report-card-summary">
                    <h3 style="color:#38BDF8 !important; margin-top:0;">📌 1. Main Findings & Executive Summary</h3>
                    <div style="font-size:1.05rem; line-height:1.7;">
                        {parsed['summary'] if parsed['summary'].strip() else raw_text[:300] + '...'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
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

                st.markdown(f"""
                <div class="report-card-recommendation">
                    <h4 style="color:#34D399 !important; margin-top:0;">🧪 4. Recommended Next Experiment</h4>
                    <div style="font-size:1.02rem; line-height:1.7;">
                        {parsed['next_experiment'] if parsed['next_experiment'].strip() else 'Perform evaluation with AutoAttack ensemble to rule out gradient obfuscation.'}
                    </div>
                </div>
                """, unsafe_allow_html=True)

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
