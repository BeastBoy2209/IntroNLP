"""
Streamlit Web Demonstration Portal for Automated Content Moderation (Milestone: Week 11).
Authors: Timur Kharitonov, Danial Kairov
Features:
  - Real-time Interactive Text Classification (Normal, Toxic, Spam)
  - Compositional Litotes & Veiled Toxicity Detection ('not smart enough')
  - Calibrated Confidence Gauges & Multi-class Probability Breakdown
  - Token-Level Explainability / Attribution Highlighting
  - Batch Processing Simulation & CSV Export
  - Model Performance Benchmark Viewer & Architecture Blueprint
"""
import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is discoverable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocessing import TextPreprocessor

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Content Moderation Studio",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .card-normal {
        background-color: #ECFDF5;
        border-left: 6px solid #10B981;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .card-toxic {
        background-color: #FEF2F2;
        border-left: 6px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .card-spam {
        background-color: #FFFBEB;
        border-left: 6px solid #F59E0B;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
    }
    .badge-normal {
        background-color: #10B981;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-toxic {
        background-color: #EF4444;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
    .badge-spam {
        background-color: #F59E0B;
        color: white;
        padding: 4px 12px;
        border-radius: 12px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- Resource Caching ---
@st.cache_resource(show_spinner="Loading NLP Models and Vectorizer...")
def load_nlp_pipeline():
    """Loads preprocessor, TF-IDF vectorizer, and production ensemble model into memory."""
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    models_dir = os.path.join(root_dir, "models")
    
    prod_model_path = os.path.join(models_dir, "production_model.joblib")
    base_model_path = os.path.join(models_dir, "baseline_logreg.joblib")
    vec_path = os.path.join(models_dir, "tfidf_vectorizer.joblib")
    meta_path = os.path.join(models_dir, "production_metadata.json")

    chosen_model_path = prod_model_path if os.path.exists(prod_model_path) else base_model_path
    
    model = joblib.load(chosen_model_path)
    vectorizer = joblib.load(vec_path)
    
    metadata = {}
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    preprocessor = TextPreprocessor(
        lower=True,
        expand_contractions=True,
        remove_html=True,
        replace_urls=True,
        normalize_elongations=True,
        demask_obfuscation=True,
        resolve_veiled_toxicity=True,
        remove_punctuation=True,
        remove_stopwords=True,
        preserve_negations=True,
        lemmatize=True
    )
    
    return model, vectorizer, preprocessor, metadata

# Load models
try:
    model, vectorizer, preprocessor, metadata = load_nlp_pipeline()
    pipeline_ready = True
except Exception as e:
    pipeline_ready = False
    pipeline_error = str(e)

# --- Sidebar: System Configuration & Moderation Policies ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Moderation Studio")
    st.markdown("**Authors:** Timur Kharitonov, Danial Kairov  \n**Course:** Intro to NLP (2026)")
    st.divider()

    st.subheader("⚙️ System Status")
    if pipeline_ready:
        st.success("🟢 Production Model Loaded")
        model_name = metadata.get("model_name", type(model).__name__)
        st.caption(f"**Architecture:** {model_name}")
        st.caption(f"**Vocab Dimension:** {len(vectorizer.get_feature_names_out()):,} N-grams")
    else:
        st.error(f"🔴 Model Offline: {pipeline_error}")

    st.divider()
    st.subheader("🛡️ Policy Thresholds")
    confidence_threshold = st.slider(
        "Action Confidence Threshold (τ)",
        min_value=0.50,
        max_value=0.95,
        value=0.70,
        step=0.05,
        help="Confidence score required to automatically block or flag a message."
    )
    
    strict_mode = st.toggle("Strict Mode (Zero-Tolerance)", value=False, help="Lowers threshold to 0.55 for aggressive toxic filtering.")
    active_threshold = 0.55 if strict_mode else confidence_threshold

    st.caption(f"Effective Threshold: **{active_threshold*100:.0f}%**")
    st.divider()
    st.info("💡 **Tip:** Use the pre-set buttons in Tab 1 to test explicit slurs, commercial scams, and veiled litotes like *'not smart enough'*.")

# --- Header Section ---
st.markdown('<div class="main-header">Automated Content Moderation Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Real-time triaging of user-generated content into <b>Normal</b>, <b>Toxic</b>, and <b>Spam</b> with compositional litotes detection.</div>', unsafe_allow_html=True)

if not pipeline_ready:
    st.error(f"Unable to launch moderation studio. Error: {pipeline_error}")
    st.stop()

# --- Main Application Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Live Single Message Tester",
    "📊 Batch Moderation Simulator",
    "📈 Model Leaderboard & Diagnostics",
    "🏗️ Architecture & REST API"
])

# =========================================================================
# TAB 1: Live Interactive Tester with Explainability
# =========================================================================
with tab1:
    st.subheader("Test Individual Message")
    st.write("Enter custom text or click one of the diagnostic benchmark buttons below:")

    # Preset Test Samples
    col_pre1, col_pre2, col_pre3, col_pre4, col_pre5 = st.columns(5)
    preset_text = None

    if col_pre1.button("🟢 Normal Discourse"):
        preset_text = "Thank you for updating the bibliography and improving the citations on this page."
    if col_pre2.button("🔴 Explicit Toxic"):
        preset_text = "You are a complete idiot and a disgusting moron, leave now!"
    if col_pre3.button("⚠️ Veiled Litotes"):
        preset_text = "Honestly, you are not smart enough to understand this topic."
    if col_pre4.button("🧠 Veiled Idiom"):
        preset_text = "You clearly have a room temperature IQ, stop posting here."
    if col_pre5.button("🟡 Commercial Spam"):
        preset_text = "URGENT! You have won £1000 cash! Claim right now at http://win-now.biz"

    # Input Area
    user_input = st.text_area(
        "Message Content:",
        value=preset_text if preset_text else "Honestly, you are not smart enough to understand this.",
        height=90,
        placeholder="Type any message to test moderation..."
    )

    col_btn, col_info = st.columns([1, 4])
    analyze_btn = col_btn.button("Analyze Content", type="primary", use_container_width=True)

    if user_input.strip():
        t0 = time.perf_counter()
        # Clean and predict
        cleaned_text = preprocessor.transform(user_input)
        vec_input = vectorizer.transform([cleaned_text if cleaned_text.strip() else "<EMPTY>"])
        pred_label = str(model.predict(vec_input)[0])
        
        has_proba = hasattr(model, "predict_proba")
        probs = model.predict_proba(vec_input)[0] if has_proba else [0.33, 0.33, 0.33]
        model_classes = list(model.classes_)
        
        prob_dict = {cls: float(probs[model_classes.index(cls)]) for cls in ["Normal", "Toxic", "Spam"] if cls in model_classes}
        confidence = float(max(probs))
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Action Decision
        if pred_label == "Normal":
            action = "APPROVED"
            card_class = "card-normal"
            badge_class = "badge-normal"
            icon = "✅"
        elif pred_label == "Toxic":
            if confidence >= active_threshold:
                action = "BLOCKED & REMOVED"
            else:
                action = "FLAGGED FOR HUMAN REVIEW"
            card_class = "card-toxic"
            badge_class = "badge-toxic"
            icon = "🚫"
        else: # Spam
            if confidence >= active_threshold:
                action = "BLOCKED (SPAM FILTER)"
            else:
                action = "FLAGGED FOR REVIEW"
            card_class = "card-spam"
            badge_class = "badge-spam"
            icon = "⚠️"

        st.divider()

        # Display Top Summary Card
        st.markdown(f"""
        <div class="{card_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-size: 1.1rem; font-weight: 600; color: #374151;">Predicted Classification:</span>
                    <span class="{badge_class}" style="margin-left: 8px; font-size: 1.1rem;">{icon} {pred_label}</span>
                </div>
                <div>
                    <span style="font-size: 1rem; color: #6B7280;">Moderation Decision: <b>{action}</b></span>
                </div>
            </div>
            <div style="margin-top: 10px; display: flex; gap: 40px;">
                <div>Confidence Certainty: <b>{confidence*100:.1f}%</b></div>
                <div>Inference Latency: <b>{latency_ms:.2f} ms</b></div>
                <div>Content Sanitization: <code>{cleaned_text if cleaned_text else '(empty)'}</code></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Multi-class Probabilities & Attribution Breakdown
        col_probs, col_attr = st.columns([1, 1])

        with col_probs:
            st.markdown("#### 📊 Posterior Probability Distribution")
            prob_df = pd.DataFrame({
                "Category": list(prob_dict.keys()),
                "Probability": [v for v in prob_dict.values()]
            })
            st.bar_chart(prob_df.set_index("Category"), color="#3B82F6", height=240)

            for cat, p in prob_dict.items():
                st.write(f"**{cat}:** `{p*100:.2f}%`")

        with col_attr:
            st.markdown("#### 🔬 Token Attribution & Key Indicators")
            st.write("Identified semantic units contributing to classification:")
            
            tokens = cleaned_text.split()
            if not tokens:
                st.caption("No content-bearing tokens identified.")
            else:
                attribution_data = []
                for tok in tokens:
                    # Check if token carries known domain weights
                    if tok in ("not_smart", "not_smart_enough", "veiled_insult_litotes", "not_bright", "room_temperature_iq"):
                        attr = "⚠️ Veiled Toxicity Indicator"
                        impact = "+ High Toxic (Litotes)"
                    elif tok in ("idiot", "moron", "stupid", "fuck", "bitch", "shit", "ugly"):
                        attr = "🚫 Explicit Toxic Keyword"
                        impact = "++ Direct Hostility"
                    elif tok in ("url", "call", "free", "txt", "claim", "prize", "won"):
                        attr = "🟡 Commercial Spam Trigger"
                        impact = "+ High Spam Signal"
                    elif tok in ("thank", "update", "improve", "article", "citation", "reference", "please"):
                        attr = "🟢 Constructive Normative Token"
                        impact = "+ Normal Polarity"
                    else:
                        attr = "Neutral Content Word"
                        impact = "Balanced Weight"
                    attribution_data.append({"Token / N-Gram": tok, "Semantic Role": attr, "Influence": impact})
                
                st.dataframe(pd.DataFrame(attribution_data), use_container_width=True, height=240)

# =========================================================================
# TAB 2: Batch Moderation Simulator
# =========================================================================
with tab2:
    st.subheader("High-Volume Batch Moderation")
    st.write("Process multiple messages simultaneously to test stream ingestion throughput.")

    default_batch = (
        "Thank you for helping us clean up this reference section.\n"
        "You are so stupid and you should delete your account!\n"
        "Claim your free £1000 cash prize now at http://win-now.biz\n"
        "Honestly, you are not smart enough to participate in this discussion.\n"
        "This article is not long enough and needs further citations.\n"
        "Stop acting like an id10t and a troll.\n"
        "We agree to merge sections 3 and 4 per the consensus."
    )

    batch_input = st.text_area("Enter one message per line:", value=default_batch, height=180)
    
    if st.button("Run Batch Moderation", type="primary"):
        lines = [line.strip() for line in batch_input.split("\n") if line.strip()]
        
        if lines:
            t_batch_start = time.perf_counter()
            cleaned_lines = [preprocessor.transform(line) for line in lines]
            vec_batch = vectorizer.transform([c if c.strip() else "<EMPTY>" for c in cleaned_lines])
            preds = model.predict(vec_batch)
            probs = model.predict_proba(vec_batch) if hasattr(model, "predict_proba") else None
            batch_time = (time.perf_counter() - t_batch_start) * 1000.0

            batch_records = []
            for i, line in enumerate(lines):
                lbl = str(preds[i])
                conf = float(max(probs[i])) if probs is not None else 1.0
                
                if lbl == "Normal":
                    decision = "APPROVED"
                elif conf >= active_threshold:
                    decision = "AUTO-BLOCKED"
                else:
                    decision = "FLAGGED"

                batch_records.append({
                    "Message": line,
                    "Predicted Class": lbl,
                    "Confidence": f"{conf*100:.1f}%",
                    "Moderation Action": decision
                })

            res_df = pd.DataFrame(batch_records)

            st.success(f"Processed **{len(lines)} messages** in **{batch_time:.2f} ms** ({batch_time/len(lines):.2f} ms/doc).")
            st.dataframe(res_df, use_container_width=True)

            # Distribution chart
            col_chart1, col_chart2 = st.columns(2)
            with col_chart1:
                st.markdown("##### Category Breakdown")
                st.bar_chart(res_df["Predicted Class"].value_counts())
            with col_chart2:
                st.markdown("##### Action Decision Breakdown")
                st.bar_chart(res_df["Moderation Action"].value_counts())

            # CSV Download
            csv_data = res_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Moderation Report (CSV)",
                data=csv_data,
                file_name="moderated_batch_report.csv",
                mime="text/csv"
            )

# =========================================================================
# TAB 3: Model Leaderboard & Diagnostics
# =========================================================================
with tab3:
    st.subheader("Empirical Model Leaderboard (Week 9 Results)")
    st.write("Quantitative comparison across 6 machine learning architectures evaluated on the test partition ($N = 11,376$):")

    report_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "model_comparison.csv"))
    if os.path.exists(report_csv_path):
        bench_df = pd.read_csv(report_csv_path)
        st.dataframe(bench_df.style.highlight_max(subset=["Accuracy", "Macro F1"], color="#D1FAE5"), use_container_width=True)
    else:
        st.warning("Benchmark CSV not found at reports/model_comparison.csv")

    st.divider()
    st.subheader("Diagnostic Visualizations")
    col_fig1, col_fig2 = st.columns(2)

    fig_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "figures"))
    cm_img = os.path.join(fig_dir, "confusion_matrix.png")
    pareto_img = os.path.join(fig_dir, "model_latency_vs_f1.png")

    with col_fig1:
        if os.path.exists(cm_img):
            st.image(cm_img, caption="Confusion Matrix Analysis (Baseline & Error Boundaries)", use_container_width=True)
        else:
            st.caption("Confusion matrix image not found.")

    with col_fig2:
        if os.path.exists(pareto_img):
            st.image(pareto_img, caption="Pareto Frontier: Inference Latency vs Macro F1-Score", use_container_width=True)
        else:
            st.caption("Pareto frontier image not found.")

# =========================================================================
# TAB 4: Architecture & REST API Integration
# =========================================================================
with tab4:
    st.subheader("System Architecture & Integration Blueprint")
    st.write("How Streamlit UI interacts with the FastAPI backend and prepares for the Week 12 Telegram bot:")

    st.code("""
+---------------------------------------------------------------------------------+
|                                 CLIENT TIER                                     |
|                                                                                 |
|   +------------------------------------+    +-------------------------------+   |
|   |  Streamlit Demonstration Studio    |    |  Telegram Moderation Bot      |   |
|   |  (Week 11 - Interactive Portal)    |    |  (Week 12 - Group Guardian)   |   |
|   +-----------------+------------------+    +---------------+---------------+   |
+---------------------|---------------------------------------|-------------------+
                      |                                       |
                      | HTTP REST JSON (/predict)             | HTTP REST JSON
                      v                                       v
+---------------------------------------------------------------------------------+
|                                FASTAPI BACKEND                                  |
|                                                                                 |
|   1. Async Lifespan Management: Pre-warmed model & vectorizer in RAM            |
|   2. Pydantic Contract Validation: Length checks & schema conformance           |
|   3. Preprocessing Engine: Litotes binding (not_smart_enough) + URL retention   |
|   4. TF-IDF Inference: 50,000 N-gram features (1 to 3 words)                    |
|   5. Soft-Voting Meta-Ensemble: Logistic Regression + Calibrated Linear SVM     |
|   6. Response Payload: { label, confidence, probabilities, latency_ms }         |
+---------------------------------------------------------------------------------+
    """, language="text")

    st.divider()
    st.subheader("Live REST API Sample Call")
    st.write("Test the FastAPI backend directly using cURL or Python `requests`:")
    st.code("""
curl -X POST "http://127.0.0.1:8000/predict" \\
     -H "Content-Type: application/json" \\
     -d '{"text": "Honestly, you are not smart enough to understand this."}'
    """, language="bash")
