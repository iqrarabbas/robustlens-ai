import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import os
import io

# Page configuration
st.set_page_config(
    page_title="RobustLens AI | Adversarial Robustness Analytics",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling & Glassmorphism Aesthetic
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Global Glass Cards */
    .stCard {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
    }
    
    /* Header Gradient */
    .header-title {
        background: linear-gradient(135deg, #3B82F6 0%, #8B5CF6 50%, #EC4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 0.2rem;
    }
    
    .sub-title {
        color: #9CA3AF;
        font-size: 1.1rem;
        margin-bottom: 1.5rem;
    }
    
    /* Metric Card Customization */
    .metric-container {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38BDF8;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Status Badges */
    .badge-success {
        background-color: rgba(16, 185, 129, 0.2);
        color: #10B981;
        border: 1px solid #10B981;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .badge-warning {
        background-color: rgba(245, 158, 11, 0.2);
        color: #F59E0B;
        border: 1px solid #F59E0B;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Define System Prompt for Gemini
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

# Session State Initialization
if "experiments" not in st.session_state:
    # Load default sample benchmark dataset
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
        }
    ])
    st.session_state.experiments = default_df

# Utility Functions
def calculate_metrics(df):
    if df.empty:
        return df
    
    df_calc = df.copy()
    df_calc["FGSM Accuracy Drop (%)"] = (df_calc["Clean Accuracy (%)"] - df_calc["FGSM Accuracy (%)"]).round(2)
    df_calc["PGD Accuracy Drop (%)"] = (df_calc["Clean Accuracy (%)"] - df_calc["PGD Accuracy (%)"]).round(2)
    
    # Robustness Score calculation:
    # 0.20 * Clean + 0.35 * FGSM + 0.45 * PGD
    df_calc["Robustness Score"] = (
        0.20 * df_calc["Clean Accuracy (%)"] +
        0.35 * df_calc["FGSM Accuracy (%)"] +
        0.45 * df_calc["PGD Accuracy (%)"]
    ).round(2)
    
    return df_calc

def get_gemini_key():
    # 1. Try Streamlit secrets
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    # 2. Try OS environment variable
    return os.environ.get("GEMINI_API_KEY", "")

# Sidebar Navigation
with st.sidebar:
    st.image("https://img.icons8.com/isometric-line/100/shield-protection.png", width=64)
    st.markdown("### **RobustLens AI**")
    st.caption("Adversarial Robustness Evaluation & AI Insights")
    st.markdown("---")
    
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🧪 Experiment Analyzer",
            "📊 Model Comparison",
            "🤖 AI Research Assistant"
        ],
        index=0
    )
    
    st.markdown("---")
    st.markdown("#### **Active Workspace**")
    exp_count = len(st.session_state.experiments)
    st.info(f"📁 **{exp_count}** Experiment(s) loaded")
    
    if st.button("🔄 Reset to Default Benchmark", use_container_width=True):
        st.session_state.experiments = pd.DataFrame([
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
            }
        ])
        st.rerun()

    if st.button("🗑️ Clear All Experiments", use_container_width=True):
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
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown(r"""
        ### 📌 **The Challenge**
        Deep learning models—especially Vision Transformers (ViTs)—achieve state-of-the-art clean accuracy, yet remain highly vulnerable to small, imperceptible adversarial perturbations.
        
        When researchers evaluate adversarial training methods like **FGSM (Fast Gradient Sign Method)** or multi-step **PGD (Projected Gradient Descent)**, understanding the trade-offs between clean accuracy retention and adversarial defense strength can be non-trivial.
        
        ### 💡 **How RobustLens AI Helps**
        - **Calculates Drop Metrics**: Instantly computes accuracy degradation under FGSM single-step and PGD iterative attack regimes.
        - **Computes Weighted Robustness Score**: Evaluates models using a weighted formulation giving higher priority to multi-step attacks ($0.20 \cdot \text{Clean} + 0.35 \cdot \text{FGSM} + 0.45 \cdot \text{PGD}$).
        - **Generates Comparative Visualizations**: Side-by-side grouped bar charts for rapid model benchmarking.
        - **AI-Powered Diagnostics**: Leverages Gemini AI to interpret clean vs. robust trade-offs, explain PGD vulnerabilities, and recommend optimal next steps.
        - **Exportable Reports**: Seamless CSV downloads for inclusion in research papers and technical reports.
        """)
    
    with col2:
        st.markdown("""
        ### 🛠️ **Quick Workflow**
        1. **Input Experiment Data**: Enter metrics via interactive form or upload a CSV in **Experiment Analyzer**.
        2. **Compare Models**: Analyze clean vs. attack performance, inspect ranking tables & dynamic charts in **Model Comparison**.
        3. **Get AI Research Insights**: Generate structured paper-ready commentary in **AI Research Assistant**.
        """)
        
        st.info("💡 **Example Dataset Pre-loaded**: Navigate directly to **Model Comparison** or **AI Research Assistant** to inspect standard ViT-B/16 baseline vs. Adversarially Fine-tuned models on CIFAR-10!")

    st.markdown("---")
    st.markdown("### 📚 **Adversarial Attack Quick Reference**")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(r"""
        #### ⚡ **FGSM (Fast Gradient Sign Method)**
        - **Type**: Single-step gradient attack.
        - **Formulation**: $x_{adv} = x + \epsilon \cdot \text{sign}(\nabla_x L(\theta, x, y))$
        - **Characteristics**: Fast computation, assesses first-order linear gradient alignment vulnerability.
        """)
    with col_b:
        st.markdown(r"""
        #### 🔄 **PGD (Projected Gradient Descent)**
        - **Type**: Multi-step iterative projected gradient attack.
        - **Formulation**: $x^{t+1} = \Pi_{x+S} (x^t + \alpha \cdot \text{sign}(\nabla_x L(\theta, x^t, y)))$
        - **Characteristics**: Considered the standard first-order attack strength benchmark.
        """)

# ---------------------------------------------------------
# PAGE 2: EXPERIMENT ANALYZER
# ---------------------------------------------------------
elif page == "🧪 Experiment Analyzer":
    st.markdown("## 🧪 Experiment Analyzer & Entry")
    st.caption("Add single experiment results or batch import CSV research logs.")
    
    tab1, tab2 = st.columns(2)
    
    with tab1:
        st.markdown("### ➕ Manual Experiment Entry")
        with st.form("add_experiment_form", clear_on_submit=True):
            exp_name = st.text_input("Experiment Name", placeholder="e.g. ViT-B/16 TRADES-5")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                model_name = st.text_input("Model Name", value="ViT-B/16")
                dataset = st.text_input("Dataset", value="CIFAR-10")
            with col_m2:
                finetune_method = st.text_input("Fine-tuning Method", value="Adversarial Training")
                epsilon = st.text_input("Epsilon (Attack Strength)", value="8/255")
            
            col_a1, col_a2, col_a3 = st.columns(3)
            with col_a1:
                clean_acc = st.number_input("Clean Accuracy (%)", min_value=0.0, max_value=100.0, value=85.0, step=0.1)
            with col_a2:
                fgsm_acc = st.number_input("FGSM Accuracy (%)", min_value=0.0, max_value=100.0, value=60.0, step=0.1)
            with col_a3:
                pgd_acc = st.number_input("PGD Accuracy (%)", min_value=0.0, max_value=100.0, value=45.0, step=0.1)
            
            epochs = st.number_input("Epochs", min_value=1, max_value=1000, value=20, step=1)
            
            submitted = st.form_submit_button("📥 Save Experiment to Workspace", use_container_width=True)
            
            if submitted:
                if not exp_name.strip():
                    st.error("⚠️ Please provide an Experiment Name.")
                elif fgsm_acc > clean_acc or pgd_acc > clean_acc:
                    st.warning("⚠️ Warning: FGSM or PGD accuracy exceeds Clean Accuracy. Please verify values.")
                    new_entry = pd.DataFrame([{
                        "Experiment Name": exp_name,
                        "Model Name": model_name,
                        "Dataset": dataset,
                        "Fine-tuning Method": finetune_method,
                        "Clean Accuracy (%)": clean_acc,
                        "FGSM Accuracy (%)": fgsm_acc,
                        "PGD Accuracy (%)": pgd_acc,
                        "Epsilon": epsilon,
                        "Epochs": epochs
                    }])
                    st.session_state.experiments = pd.concat([st.session_state.experiments, new_entry], ignore_index=True)
                    st.success(f"✅ Added '{exp_name}' to workspace!")
                    st.rerun()
                else:
                    new_entry = pd.DataFrame([{
                        "Experiment Name": exp_name,
                        "Model Name": model_name,
                        "Dataset": dataset,
                        "Fine-tuning Method": finetune_method,
                        "Clean Accuracy (%)": clean_acc,
                        "FGSM Accuracy (%)": fgsm_acc,
                        "PGD Accuracy (%)": pgd_acc,
                        "Epsilon": epsilon,
                        "Epochs": epochs
                    }])
                    st.session_state.experiments = pd.concat([st.session_state.experiments, new_entry], ignore_index=True)
                    st.success(f"✅ Added '{exp_name}' to workspace!")
                    st.rerun()

    with tab2:
        st.markdown("### 📁 Batch Import / Export CSV")
        st.markdown("Upload a pre-formatted CSV file containing experiment logs.")
        
        uploaded_file = st.file_uploader("Upload Experiment CSV", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                required_cols = ["Experiment Name", "Clean Accuracy (%)", "FGSM Accuracy (%)", "PGD Accuracy (%)"]
                missing = [c for c in required_cols if c not in uploaded_df.columns]
                if missing:
                    st.error(f"❌ Uploaded CSV is missing required columns: {missing}")
                else:
                    st.session_state.experiments = pd.concat([st.session_state.experiments, uploaded_df], ignore_index=True).drop_duplicates(subset=["Experiment Name"])
                    st.success("✅ Batch experiments imported successfully!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Error reading CSV file: {e}")

        st.markdown("---")
        st.markdown("#### 📥 Sample Data Download")
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

    st.markdown("---")
    st.markdown("### 📊 Active Experiments & Drop Calculations")
    
    if st.session_state.experiments.empty:
        st.info("No experiments currently loaded. Enter an experiment above or load sample data.")
    else:
        calc_df = calculate_metrics(st.session_state.experiments)
        st.dataframe(
            calc_df[[
                "Experiment Name", "Model Name", "Clean Accuracy (%)", 
                "FGSM Accuracy (%)", "FGSM Accuracy Drop (%)", 
                "PGD Accuracy (%)", "PGD Accuracy Drop (%)", 
                "Robustness Score"
            ]],
            use_container_width=True
        )

# ---------------------------------------------------------
# PAGE 3: MODEL COMPARISON
# ---------------------------------------------------------
elif page == "📊 Model Comparison":
    st.markdown("## 📊 Model Comparison & Robustness Leaderboard")
    st.caption("Side-by-side performance evaluation, weighted robustness ranking, and comparative charts.")
    
    if st.session_state.experiments.empty:
        st.warning("⚠️ No experiment data available. Please add experiments in the Experiment Analyzer.")
    else:
        calc_df = calculate_metrics(st.session_state.experiments)
        
        # Top Metrics & Best Models Callout
        best_clean = calc_df.loc[calc_df["Clean Accuracy (%)"].idxmax()]
        best_fgsm = calc_df.loc[calc_df["FGSM Accuracy (%)"].idxmax()]
        best_pgd = calc_df.loc[calc_df["PGD Accuracy (%)"].idxmax()]
        best_overall = calc_df.loc[calc_df["Robustness Score"].idxmax()]
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Best Clean Accuracy</div>
                <div class="metric-value">{:.1f}%</div>
                <div style="font-size:0.8rem; color:#E2E8F0;">{}</div>
            </div>
            """.format(best_clean["Clean Accuracy (%)"], best_clean["Experiment Name"]), unsafe_allow_html=True)
            
        with m_col2:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Best FGSM Defense</div>
                <div class="metric-value">{:.1f}%</div>
                <div style="font-size:0.8rem; color:#E2E8F0;">{}</div>
            </div>
            """.format(best_fgsm["FGSM Accuracy (%)"], best_fgsm["Experiment Name"]), unsafe_allow_html=True)
            
        with m_col3:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Best PGD Defense</div>
                <div class="metric-value">{:.1f}%</div>
                <div style="font-size:0.8rem; color:#E2E8F0;">{}</div>
            </div>
            """.format(best_pgd["PGD Accuracy (%)"], best_pgd["Experiment Name"]), unsafe_allow_html=True)
            
        with m_col4:
            st.markdown("""
            <div class="metric-container">
                <div class="metric-label">Top Robustness Score</div>
                <div class="metric-value">{:.2f}</div>
                <div style="font-size:0.8rem; color:#E2E8F0;">{}</div>
            </div>
            """.format(best_overall["Robustness Score"], best_overall["Experiment Name"]), unsafe_allow_html=True)

        st.markdown("---")
        
        # Interactive Plotly Multi-Bar Chart
        st.markdown("### 📈 Clean vs. FGSM vs. PGD Accuracy Comparison")
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=calc_df["Experiment Name"],
            y=calc_df["Clean Accuracy (%)"],
            name="Clean Accuracy",
            marker_color="#3B82F6"
        ))
        
        fig.add_trace(go.Bar(
            x=calc_df["Experiment Name"],
            y=calc_df["FGSM Accuracy (%)"],
            name="FGSM Accuracy",
            marker_color="#F59E0B"
        ))
        
        fig.add_trace(go.Bar(
            x=calc_df["Experiment Name"],
            y=calc_df["PGD Accuracy (%)"],
            name="PGD Accuracy",
            marker_color="#EF4444"
        ))
        
        fig.update_layout(
            barmode="group",
            xaxis_title="Experiment",
            yaxis_title="Accuracy (%)",
            yaxis=dict(range=[0, 100]),
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=40, t=60, b=40)
        )
        
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        
        # Leaderboard Table
        st.markdown("### 🏆 Comprehensive Leaderboard")
        st.caption("Sorted by Robustness Score = 0.20 × Clean + 0.35 × FGSM + 0.45 × PGD")
        
        leaderboard_df = calc_df.sort_values(by="Robustness Score", ascending=False).reset_index(drop=True)
        leaderboard_df.index += 1
        
        st.dataframe(
            leaderboard_df[[
                "Experiment Name", "Model Name", "Fine-tuning Method",
                "Clean Accuracy (%)", "FGSM Accuracy (%)", "PGD Accuracy (%)",
                "FGSM Accuracy Drop (%)", "PGD Accuracy Drop (%)", "Robustness Score"
            ]],
            use_container_width=True
        )
        
        # Export CSV Button
        csv_buffer = leaderboard_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Comparison Report (CSV)",
            data=csv_buffer,
            file_name="robustlens_model_comparison.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------
# PAGE 4: AI RESEARCH ASSISTANT
# ---------------------------------------------------------
elif page == "🤖 AI Research Assistant":
    st.markdown("## 🤖 AI Research Assistant")
    st.caption("Automated insights, trade-off analysis, and experiment recommendations powered by Google Gemini AI.")
    
    if st.session_state.experiments.empty:
        st.warning("⚠️ Workspace is empty. Please add experiments or reset default data to generate AI insights.")
    else:
        calc_df = calculate_metrics(st.session_state.experiments)
        
        # Gemini API Key Setup
        api_key = get_gemini_key()
        
        st.markdown("### 🔑 API Key Configuration")
        user_key_input = st.text_input(
            "Google Gemini API Key",
            value=api_key,
            type="password",
            help="Stored in GEMINI_API_KEY environment variable or Streamlit secrets."
        )
        
        final_api_key = user_key_input.strip() if user_key_input else api_key
        
        st.markdown("---")
        st.markdown("### 🧠 Generate Experimental Analysis")
        
        # Display data summary payload preview
        with st.expander("🔍 Preview Data Payload Sent to Gemini AI"):
            st.json(calc_df.to_dict(orient="records"))
            st.markdown("**System Prompt Instruction:**")
            st.code(SYSTEM_PROMPT, language="text")

        if st.button("🚀 Analyze Experiments with Gemini AI", type="primary", use_container_width=True):
            if not final_api_key:
                st.error("❌ Google Gemini API Key is missing. Please set GEMINI_API_KEY environment variable, Streamlit secrets, or input it above.")
            else:
                with st.spinner("🧠 Gemini AI is analyzing model trade-offs and robustness metrics..."):
                    # Format prompt payload
                    payload_summary = calc_df.to_string(index=False)
                    user_query = f"""Here are the experimental results for evaluation:

{payload_summary}

Please analyze these results according to your instructions."""

                    ai_response_text = ""
                    try:
                        # Official google-genai SDK call
                        from google import genai
                        from google.genai import types
                        
                        client = genai.Client(api_key=final_api_key)
                        
                        # Try recommended models in sequence
                        model_name = "gemini-2.5-flash"
                        try:
                            response = client.models.generate_content(
                                model=model_name,
                                contents=user_query,
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_PROMPT,
                                    temperature=0.3
                                )
                            )
                            ai_response_text = response.text
                        except Exception as inner_e:
                            # Fallback to gemini-1.5-flash or gemini-2.0-flash if 2.5 is unavailable
                            response = client.models.generate_content(
                                model="gemini-1.5-flash",
                                contents=user_query,
                                config=types.GenerateContentConfig(
                                    system_instruction=SYSTEM_PROMPT,
                                    temperature=0.3
                                )
                            )
                            ai_response_text = response.text
                            
                    except ImportError:
                        # Fallback for google-generativeai package if installed
                        try:
                            import google.generativeai as genai_old
                            genai_old.configure(api_key=final_api_key)
                            model = genai_old.GenerativeModel(
                                model_name="gemini-1.5-flash",
                                system_instruction=SYSTEM_PROMPT
                            )
                            response = model.generate_content(user_query)
                            ai_response_text = response.text
                        except Exception as e_old:
                            st.error(f"❌ API Call Error: {e_old}")
                    except Exception as gen_err:
                        st.error(f"❌ Gemini Generation Error: {gen_err}")
                    
                    if ai_response_text:
                        st.session_state["last_ai_analysis"] = ai_response_text

        # Display AI Result if available
        if "last_ai_analysis" in st.session_state:
            st.markdown("---")
            st.markdown("### 📋 AI Analysis & Research Report")
            st.markdown(st.session_state["last_ai_analysis"])
            
            st.markdown("---")
            # Download Markdown Report
            report_markdown = f"""# RobustLens AI - Research Analysis Report

## Experimental Data Evaluated:
```
{calc_df.to_string(index=False)}
```

## AI Generated Analysis:
{st.session_state["last_ai_analysis"]}
"""
            st.download_button(
                label="📥 Download Research Report (.md)",
                data=report_markdown,
                file_name="robustlens_ai_research_report.md",
                mime="text/markdown"
            )
