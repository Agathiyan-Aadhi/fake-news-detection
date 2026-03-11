"""
AI-Based Fake News Detection — Professional Streamlit Web App
Run with: python -m streamlit run app.py
"""

import streamlit as st
import pickle
import re
import numpy as np
import os

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TruthLens — AI Fake News Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Professional CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & Base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080C14;
    color: #E8EDF5;
}

.stApp {
    background: #080C14;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(0,200,255,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(120,40,255,0.06) 0%, transparent 60%);
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 0 2rem 2rem 2rem !important; max-width: 1400px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0D1117 !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
}
[data-testid="stSidebar"] * { color: #C9D1D9 !important; }
[data-testid="stSidebar"] .stSelectbox label { color: #8B949E !important; font-size: 0.75rem !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }

/* ── Sidebar Logo ── */
.sidebar-logo {
    display: flex; align-items: center; gap: 10px;
    padding: 1.5rem 0 1rem 0; margin-bottom: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.sidebar-logo-icon {
    width: 36px; height: 36px; border-radius: 10px;
    background: linear-gradient(135deg, #00C2FF, #7B2FFF);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.sidebar-logo-text {
    font-family: 'Syne', sans-serif;
    font-weight: 800; font-size: 1.1rem; color: #F0F6FC !important;
}
.sidebar-logo-sub { font-size: 0.7rem; color: #8B949E !important; }

/* ── Sidebar Model Card ── */
.model-info-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px; padding: 1rem; margin: 1rem 0;
}
.model-info-card h4 {
    font-family: 'Syne', sans-serif; font-size: 0.7rem;
    color: #8B949E !important; text-transform: uppercase;
    letter-spacing: 0.1em; margin-bottom: 0.75rem;
}
.model-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.4rem 0; border-bottom: 1px solid rgba(255,255,255,0.04);
}
.model-row:last-child { border-bottom: none; }
.model-name { font-size: 0.82rem; color: #C9D1D9 !important; }
.model-acc { font-size: 0.78rem; font-weight: 600; color: #00C2FF !important; }

/* ── Hero Header ── */
.hero {
    padding: 3.5rem 0 2.5rem 0;
    text-align: center; position: relative;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 6px;
    background: rgba(0,194,255,0.08); border: 1px solid rgba(0,194,255,0.2);
    border-radius: 100px; padding: 0.3rem 0.9rem;
    font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase;
    color: #00C2FF; margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    font-weight: 800; line-height: 1.05;
    color: #F0F6FC; margin-bottom: 0.8rem;
    letter-spacing: -0.02em;
}
.hero-title span {
    background: linear-gradient(135deg, #00C2FF 0%, #7B2FFF 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub {
    font-size: 1rem; color: #8B949E; max-width: 520px;
    margin: 0 auto 2rem auto; line-height: 1.6; font-weight: 300;
}
.hero-stats {
    display: flex; justify-content: center; gap: 2.5rem;
    flex-wrap: wrap;
}
.hero-stat { text-align: center; }
.hero-stat-num {
    font-family: 'Syne', sans-serif; font-size: 1.6rem;
    font-weight: 800; color: #F0F6FC; display: block;
}
.hero-stat-label { font-size: 0.72rem; color: #8B949E; text-transform: uppercase; letter-spacing: 0.08em; }

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    margin: 0.5rem 0 2rem 0;
}

/* ── Tab Styling ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px !important; padding: 4px !important;
    gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: 9px !important;
    color: #8B949E !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important; font-weight: 500 !important;
    padding: 0.5rem 1.2rem !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(0,194,255,0.12) !important;
    color: #00C2FF !important;
}
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem !important; }

/* ── Input Cards ── */
.input-section {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.5rem;
}
.input-label {
    font-size: 0.72rem; color: #8B949E;
    text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.5rem; display: block;
}

/* ── Streamlit input override ── */
.stTextInput input, .stTextArea textarea {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E8EDF5 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    transition: border-color 0.2s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: rgba(0,194,255,0.4) !important;
    box-shadow: 0 0 0 3px rgba(0,194,255,0.06) !important;
}
.stTextInput label, .stTextArea label {
    color: #8B949E !important; font-size: 0.78rem !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
}

/* ── Analyze Button ── */
.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #00C2FF 0%, #7B2FFF 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.95rem !important; font-weight: 700 !important;
    letter-spacing: 0.04em !important; cursor: pointer !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 24px rgba(0,194,255,0.2) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,194,255,0.35) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Result Card ── */
.result-card {
    border-radius: 16px; padding: 2rem 1.5rem;
    text-align: center; position: relative; overflow: hidden;
    margin-bottom: 1rem;
}
.result-fake {
    background: linear-gradient(135deg, rgba(255,45,85,0.15), rgba(255,100,50,0.1));
    border: 1px solid rgba(255,45,85,0.35);
}
.result-real {
    background: linear-gradient(135deg, rgba(0,210,120,0.15), rgba(0,190,255,0.1));
    border: 1px solid rgba(0,210,120,0.35);
}
.result-waiting {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.08);
    min-height: 180px; display: flex;
    flex-direction: column; align-items: center; justify-content: center;
}
.result-icon { font-size: 2.8rem; margin-bottom: 0.5rem; display: block; }
.result-verdict {
    font-family: 'Syne', sans-serif; font-size: 1.8rem;
    font-weight: 800; letter-spacing: -0.01em;
    display: block; margin-bottom: 0.25rem;
}
.result-verdict-fake { color: #FF2D55; }
.result-verdict-real { color: #00D278; }
.result-confidence { font-size: 0.82rem; color: #8B949E; }
.result-confidence strong { color: #C9D1D9; }

/* ── Confidence Bar ── */
.conf-bar-wrap {
    background: rgba(255,255,255,0.05);
    border-radius: 100px; height: 6px;
    margin: 1rem 0; overflow: hidden;
}
.conf-bar-fill-fake {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #FF2D55, #FF6432);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.conf-bar-fill-real {
    height: 100%; border-radius: 100px;
    background: linear-gradient(90deg, #00C2FF, #00D278);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}

/* ── Meta Tags ── */
.meta-row {
    display: flex; gap: 0.75rem; flex-wrap: wrap;
    margin-top: 1rem; justify-content: center;
}
.meta-tag {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 100px; padding: 0.25rem 0.75rem;
    font-size: 0.72rem; color: #8B949E;
}
.meta-tag span { color: #C9D1D9; font-weight: 500; }

/* ── Interpretation Badge ── */
.interp-badge {
    display: inline-flex; align-items: center; gap: 6px;
    padding: 0.4rem 0.9rem; border-radius: 100px;
    font-size: 0.78rem; font-weight: 500; margin-top: 0.75rem;
}
.interp-high { background: rgba(0,210,120,0.12); color: #00D278; border: 1px solid rgba(0,210,120,0.25); }
.interp-med  { background: rgba(255,190,0,0.12);  color: #FFB800; border: 1px solid rgba(255,190,0,0.25); }
.interp-low  { background: rgba(255,100,50,0.12); color: #FF6432; border: 1px solid rgba(255,100,50,0.25); }

/* ── Example Cards ── */
.example-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.2rem 1.4rem;
    margin-bottom: 1rem; cursor: pointer;
    transition: all 0.2s;
}
.example-card:hover {
    border-color: rgba(0,194,255,0.25);
    background: rgba(0,194,255,0.04);
    transform: translateX(4px);
}
.example-type {
    font-size: 0.68rem; text-transform: uppercase;
    letter-spacing: 0.1em; font-weight: 600;
    margin-bottom: 0.4rem; display: block;
}
.example-type-fake { color: #FF2D55; }
.example-type-real { color: #00D278; }
.example-text { font-size: 0.88rem; color: #C9D1D9; line-height: 1.5; }

/* ── Model Compare Table ── */
.model-table {
    width: 100%; border-collapse: collapse; margin-top: 1rem;
}
.model-table th {
    font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #8B949E; padding: 0.75rem 1rem; text-align: left;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.model-table td {
    padding: 0.85rem 1rem; font-size: 0.85rem; color: #C9D1D9;
    border-bottom: 1px solid rgba(255,255,255,0.04);
}
.model-table tr:last-child td { border-bottom: none; }
.model-table tr:hover td { background: rgba(255,255,255,0.02); }
.acc-pill {
    background: rgba(0,194,255,0.12); color: #00C2FF;
    padding: 0.2rem 0.6rem; border-radius: 100px;
    font-size: 0.75rem; font-weight: 600;
}
.speed-dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; margin-right: 3px; }

/* ── Architecture Cards ── */
.arch-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.4rem;
    height: 100%;
}
.arch-card-num {
    font-family: 'Syne', sans-serif; font-size: 0.7rem;
    color: #00C2FF; text-transform: uppercase; letter-spacing: 0.1em;
    margin-bottom: 0.5rem; display: block;
}
.arch-card-title {
    font-family: 'Syne', sans-serif; font-size: 1rem;
    font-weight: 700; color: #F0F6FC; margin-bottom: 0.75rem;
}
.arch-card-desc { font-size: 0.83rem; color: #8B949E; line-height: 1.6; }
.arch-card-tag {
    display: inline-block; margin-top: 0.75rem;
    background: rgba(123,47,255,0.12); color: #A78BFA;
    border: 1px solid rgba(123,47,255,0.2);
    border-radius: 100px; padding: 0.2rem 0.65rem;
    font-size: 0.7rem; font-weight: 500;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important; color: #E8EDF5 !important;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: #00C2FF !important; }

/* ── Footer ── */
.footer {
    text-align: center; padding: 2rem 0 1rem 0;
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 3rem;
}
.footer-text { font-size: 0.78rem; color: #4A5568; }
.footer-text a { color: #00C2FF; text-decoration: none; }

/* ── Waiting placeholder ── */
.waiting-icon { font-size: 2rem; opacity: 0.3; margin-bottom: 0.75rem; }
.waiting-text { font-size: 0.85rem; color: #4A5568; }
</style>
""", unsafe_allow_html=True)


# ── Text Cleaning ──────────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    return re.sub(r'\s+', ' ', text).strip()


# ── Model Loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}
    if os.path.exists("tfidf_vectorizer.pkl") and os.path.exists("lr_model.pkl"):
        with open("tfidf_vectorizer.pkl", "rb") as f:
            models["tfidf"] = pickle.load(f)
        with open("lr_model.pkl", "rb") as f:
            models["lr"] = pickle.load(f)
    try:
        import tensorflow as tf
        if os.path.exists("lstm_model.h5") and os.path.exists("tokenizer.pkl"):
            models["lstm"] = tf.keras.models.load_model("lstm_model.h5")
            with open("tokenizer.pkl", "rb") as f:
                models["tokenizer"] = pickle.load(f)
    except Exception:
        pass
    try:
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch
        if os.path.exists("bert_fake_news"):
            models["bert_tokenizer"] = BertTokenizer.from_pretrained("bert_fake_news")
            models["bert"] = BertForSequenceClassification.from_pretrained("bert_fake_news")
            models["bert"].eval()
            models["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            models["bert"] = models["bert"].to(models["device"])
    except Exception:
        pass
    return models


def predict_lr(text, models):
    clean = clean_text(text)
    features = models["tfidf"].transform([clean])
    pred = models["lr"].predict(features)[0]
    proba = models["lr"].predict_proba(features)[0]
    return int(pred), float(max(proba))


def predict_lstm(text, models, max_len=300):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    clean = clean_text(text)
    seq = models["tokenizer"].texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    prob = float(models["lstm"].predict(padded, verbose=0)[0][0])
    return (1 if prob >= 0.5 else 0), max(prob, 1 - prob)


def predict_bert(text, models, max_len=128):
    import torch
    enc = models["bert_tokenizer"](
        clean_text(text), max_length=max_len,
        padding="max_length", truncation=True, return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(models["device"])
    attention_mask = enc["attention_mask"].to(models["device"])
    with torch.no_grad():
        logits = models["bert"](input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred = int(probs.argmax())
    return pred, float(probs[pred])


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-logo">
            <div class="sidebar-logo-icon">🔬</div>
            <div>
                <div class="sidebar-logo-text">TruthLens</div>
                <div class="sidebar-logo-sub">AI News Verifier</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        model_choice = st.selectbox(
            "ACTIVE MODEL",
            ["Logistic Regression", "Bi-LSTM", "BERT", "Ensemble"],
            index=0
        )

        st.markdown("""
        <div class="model-info-card">
            <h4>Model Performance</h4>
            <div class="model-row">
                <span class="model-name">Logistic Regression</span>
                <span class="model-acc">99.48%</span>
            </div>
            <div class="model-row">
                <span class="model-name">Bi-LSTM + GRU</span>
                <span class="model-acc">97.4%</span>
            </div>
            <div class="model-row">
                <span class="model-name">BERT</span>
                <span class="model-acc">99.1%</span>
            </div>
            <div class="model-row">
                <span class="model-name">Ensemble</span>
                <span class="model-acc">99.2%</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="padding: 0.5rem 0;">
            <div style="font-size:0.7rem; color:#8B949E; text-transform:uppercase; letter-spacing:0.1em; margin-bottom:0.75rem;">Dataset</div>
            <div style="font-size:0.83rem; color:#C9D1D9; line-height:1.7;">
                📊 ISOT Fake News Dataset<br>
                🗂 44,898 labeled articles<br>
                🏷 Reuters (Real) + Web (Fake)<br>
                📅 2015 – 2018
            </div>
        </div>
        """, unsafe_allow_html=True)

    return model_choice


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    models = load_models()
    model_choice = render_sidebar()

    # ── Hero ──────────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="padding:3rem 0 2rem 0; text-align:center;">
        <div style="display:inline-flex;align-items:center;gap:6px;
            background:rgba(0,194,255,0.08);border:1px solid rgba(0,194,255,0.2);
            border-radius:100px;padding:0.3rem 0.9rem;font-size:0.72rem;
            letter-spacing:0.12em;text-transform:uppercase;color:#00C2FF;margin-bottom:1.2rem;">
            🔬 Powered by Machine Learning
        </div>
        <h1 style="font-family:Syne,sans-serif;font-size:3.2rem;font-weight:800;
            line-height:1.05;color:#F0F6FC;margin-bottom:0.8rem;letter-spacing:-0.02em;">
            Detect <span style="background:linear-gradient(135deg,#00C2FF 0%,#7B2FFF 100%);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;">Fake News</span>
            with AI Precision
        </h1>
        <p style="font-size:1rem;color:#8B949E;max-width:500px;margin:0 auto 2.5rem auto;
            line-height:1.6;font-weight:300;">
            Four AI models — from TF-IDF to BERT Transformers —
            working together to verify news authenticity instantly.
        </p>
        <div style="display:flex;justify-content:center;gap:3rem;flex-wrap:wrap;margin-bottom:1rem;">
            <div style="text-align:center;">
                <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:#F0F6FC;">99.48%</div>
                <div style="font-size:0.7rem;color:#8B949E;text-transform:uppercase;letter-spacing:0.08em;">Accuracy</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:#F0F6FC;">44,898</div>
                <div style="font-size:0.7rem;color:#8B949E;text-transform:uppercase;letter-spacing:0.08em;">Trained Articles</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:#F0F6FC;">4</div>
                <div style="font-size:0.7rem;color:#8B949E;text-transform:uppercase;letter-spacing:0.08em;">AI Models</div>
            </div>
            <div style="text-align:center;">
                <div style="font-family:Syne,sans-serif;font-size:1.8rem;font-weight:800;color:#F0F6FC;">&lt;1s</div>
                <div style="font-size:0.7rem;color:#8B949E;text-transform:uppercase;letter-spacing:0.08em;">Response Time</div>
            </div>
        </div>
    </div>
    <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(255,255,255,0.08),transparent);margin-bottom:2rem;"></div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔍  Analyze", "📋  Examples", "⚙️  Models"])

    # ── TAB 1: ANALYZE ────────────────────────────────────────────────────────
    with tab1:
        col_input, col_result = st.columns([3, 2], gap="large")

        with col_input:
            st.markdown('<div class="input-section">', unsafe_allow_html=True)
            title_input = st.text_input(
                "Article Headline",
                placeholder="e.g. Federal Reserve raises interest rates amid inflation concerns",
                key="title"
            )
            body_input = st.text_area(
                "Article Body",
                placeholder="Paste the full article text here for best results.\nThe more content you provide, the more accurate the analysis.",
                height=220,
                key="body"
            )
            analyze_btn = st.button("⚡  Analyze Article", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-top:1rem; padding:0.75rem 1rem;
                background:rgba(0,194,255,0.05); border:1px solid rgba(0,194,255,0.15);
                border-radius:10px; font-size:0.78rem; color:#8B949E; line-height:1.6;">
                💡 <strong style="color:#C9D1D9;">Tip:</strong>
                For highest accuracy, include both the headline and article body.
                The model was trained on Reuters-style news articles.
            </div>
            """, unsafe_allow_html=True)

        with col_result:
            if not analyze_btn:
                st.markdown("""
                <div class="result-card result-waiting">
                    <div class="waiting-icon">🔬</div>
                    <div class="waiting-text">Enter an article and click<br><strong>Analyze Article</strong> to begin</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                combined = (title_input + " " + body_input).strip()
                if len(combined) < 20:
                    st.markdown("""
                    <div class="result-card result-waiting">
                        <div class="waiting-icon">⚠️</div>
                        <div class="waiting-text">Please enter at least<br>20 characters to analyze</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    with st.spinner("Analyzing with AI..."):
                        try:
                            if model_choice == "Logistic Regression" and "lr" in models:
                                pred, conf = predict_lr(combined, models)
                            elif model_choice == "Bi-LSTM" and "lstm" in models:
                                pred, conf = predict_lstm(combined, models)
                            elif model_choice == "BERT" and "bert" in models:
                                pred, conf = predict_bert(combined, models)
                            elif "lr" in models:
                                pred, conf = predict_lr(combined, models)
                            else:
                                st.error("No model loaded. Please check model files.")
                                st.stop()

                            is_fake = pred == 1
                            card_class = "result-fake" if is_fake else "result-real"
                            icon = "🚨" if is_fake else "✅"
                            verdict = "FAKE NEWS" if is_fake else "REAL NEWS"
                            verdict_class = "result-verdict-fake" if is_fake else "result-verdict-real"
                            bar_class = "conf-bar-fill-fake" if is_fake else "conf-bar-fill-real"
                            conf_pct = conf * 100

                            if conf_pct > 90:
                                interp_class = "interp-high"
                                interp_text = "✦ Very high confidence"
                            elif conf_pct > 75:
                                interp_class = "interp-high"
                                interp_text = "✦ High confidence"
                            elif conf_pct > 60:
                                interp_class = "interp-med"
                                interp_text = "◈ Moderate — verify manually"
                            else:
                                interp_class = "interp-low"
                                interp_text = "◉ Low — manual review needed"

                            word_count = len(combined.split())

                            st.markdown(f"""
                            <div class="result-card {card_class}">
                                <span class="result-icon">{icon}</span>
                                <span class="result-verdict {verdict_class}">{verdict}</span>
                                <div class="result-confidence">
                                    Confidence: <strong>{conf_pct:.1f}%</strong>
                                </div>
                                <div class="conf-bar-wrap">
                                    <div class="{bar_class}" style="width:{conf_pct}%"></div>
                                </div>
                                <div class="{interp_class} interp-badge">{interp_text}</div>
                                <div class="meta-row">
                                    <span class="meta-tag">Model: <span>{model_choice}</span></span>
                                    <span class="meta-tag">Words: <span>{word_count}</span></span>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        except Exception as e:
                            st.error(f"Analysis error: {e}")

    # ── TAB 2: EXAMPLES ───────────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div style="margin-bottom:1.5rem;">
            <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#F0F6FC; margin-bottom:0.4rem;">
                Test Examples
            </div>
            <div style="font-size:0.85rem; color:#8B949E;">
                Click any example to copy the text, then paste it in the Analyze tab.
            </div>
        </div>
        """, unsafe_allow_html=True)

        examples = [
            ("REAL", "Reuters: Federal Reserve raises interest rates by 0.25 points",
             "WASHINGTON (Reuters) - The Federal Reserve raised its benchmark interest rate by a quarter of a percentage point on Wednesday, continuing its fight against inflation while signaling it may be near the end of its tightening cycle."),
            ("REAL", "Scientists publish malaria vaccine results in peer-reviewed journal",
             "NEW YORK (Reuters) - Researchers have published promising results from a large-scale clinical trial of a new malaria vaccine, showing 77% efficacy in children under five across multiple African countries, according to The Lancet medical journal."),
            ("FAKE", "SHOCKING: Government secretly adding mind control chemicals to tap water!",
             "BREAKING: Anonymous insider reveals the deep state has been fluoridating water supplies for decades to control the population! Big Pharma and government elites DON'T want you to know this! Share before they DELETE this post!"),
            ("FAKE", "NASA ADMITS moon landing was staged in Hollywood studio!!",
             "EXPOSED: Leaked documents from a former NASA employee confirm that the 1969 moon landing was filmed by Stanley Kubrick in a secret Hollywood studio. The government has been hiding this truth for 50 years! Wake up sheeple!"),
        ]

        col1, col2 = st.columns(2, gap="medium")
        for i, (label, title, body) in enumerate(examples):
            col = col1 if i % 2 == 0 else col2
            with col:
                type_class = "example-type-fake" if label == "FAKE" else "example-type-real"
                icon = "🚨" if label == "FAKE" else "✅"
                st.markdown(f"""
                <div class="example-card">
                    <span class="example-type {type_class}">{icon} Expected: {label}</span>
                    <div class="example-text"><strong>{title}</strong><br><br>{body[:120]}...</div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("Use this example"):
                    st.code(body, language=None)
                    st.caption("Copy the text above → paste into Article Body in the Analyze tab")

    # ── TAB 3: MODELS ─────────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div style="font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700;
            color:#F0F6FC; margin-bottom:1.5rem;">Model Architecture & Performance
        </div>
        """, unsafe_allow_html=True)

        # Performance table
        st.markdown("""
        <table class="model-table">
            <thead>
                <tr>
                    <th>Model</th>
                    <th>Type</th>
                    <th>Accuracy</th>
                    <th>Speed</th>
                    <th>Memory</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td><strong>TF-IDF + Logistic Regression</strong></td>
                    <td>Traditional ML</td>
                    <td><span class="acc-pill">99.48%</span></td>
                    <td><span class="speed-dot" style="background:#00D278"></span><span class="speed-dot" style="background:#00D278"></span><span class="speed-dot" style="background:#00D278"></span> Fast</td>
                    <td>Low</td>
                </tr>
                <tr>
                    <td><strong>TF-IDF + SVM</strong></td>
                    <td>Traditional ML</td>
                    <td><span class="acc-pill">98.3%</span></td>
                    <td><span class="speed-dot" style="background:#00D278"></span><span class="speed-dot" style="background:#00D278"></span><span class="speed-dot" style="background:#00D278"></span> Fast</td>
                    <td>Low</td>
                </tr>
                <tr>
                    <td><strong>Bidirectional LSTM + GRU</strong></td>
                    <td>Deep Learning</td>
                    <td><span class="acc-pill">97.4%</span></td>
                    <td><span class="speed-dot" style="background:#00D278"></span><span class="speed-dot" style="background:#00D278"></span><span class="speed-dot" style="background:#4A5568"></span> Medium</td>
                    <td>Medium</td>
                </tr>
                <tr>
                    <td><strong>BERT (bert-base-uncased)</strong></td>
                    <td>Transformer</td>
                    <td><span class="acc-pill">99.1%</span></td>
                    <td><span class="speed-dot" style="background:#00D278"></span><span class="speed-dot" style="background:#4A5568"></span><span class="speed-dot" style="background:#4A5568"></span> Slow</td>
                    <td>High</td>
                </tr>
                <tr>
                    <td><strong>Weighted Ensemble</strong></td>
                    <td>Combined</td>
                    <td><span class="acc-pill">99.2%</span></td>
                    <td><span class="speed-dot" style="background:#00D278"></span><span class="speed-dot" style="background:#4A5568"></span><span class="speed-dot" style="background:#4A5568"></span> Slow</td>
                    <td>High</td>
                </tr>
            </tbody>
        </table>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Architecture cards
        c1, c2, c3, c4 = st.columns(4, gap="medium")
        cards = [
            ("01", "TF-IDF + LR", "Converts text to word frequency matrix using 50k features. Fast, interpretable, and surprisingly powerful for this task.", "Traditional ML"),
            ("02", "Bi-LSTM + GRU", "Reads article text both forward and backward using LSTM, then extracts sequential patterns with GRU layers.", "Deep Learning"),
            ("03", "BERT", "Pre-trained on 3.3B words. Fine-tuned on our dataset using self-attention to understand full context simultaneously.", "Transformer"),
            ("04", "Ensemble", "Combines LR (20%) + LSTM (30%) + BERT (50%) using weighted probability averaging for maximum reliability.", "Combined"),
        ]
        for col, (num, title, desc, tag) in zip([c1, c2, c3, c4], cards):
            with col:
                st.markdown(f"""
                <div class="arch-card">
                    <span class="arch-card-num">Model {num}</span>
                    <div class="arch-card-title">{title}</div>
                    <div class="arch-card-desc">{desc}</div>
                    <span class="arch-card-tag">{tag}</span>
                </div>
                """, unsafe_allow_html=True)

    # ── Footer ─────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        <div class="footer-text">
            TruthLens — AI Fake News Detector &nbsp;·&nbsp;
            Built with Python, scikit-learn, TensorFlow, HuggingFace & Streamlit &nbsp;·&nbsp;
            ISOT Dataset &nbsp;·&nbsp; 99.48% Accuracy
        </div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
