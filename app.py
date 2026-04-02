"""
app.py — ToxPredict v2 · Drug Toxicity Prediction Interface
CodeCure AI Hackathon · Track A

Launch:
    streamlit run app.py
"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import sys, os, json, io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Obscura — Drug Toxicity AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #E9F2F2; }
.main  { background: #EEF2F7; }

/* ── Header ── */
.app-header {
    text-align: center;
    padding: 2rem 0 1.2rem;
    position: relative;
}
.app-header::after {
    content: '';
    position: absolute;
    bottom: 0; left: 15%; right: 15%;
    height: 2px;
    background: linear-gradient(90deg, transparent, #1D4ED8 35%, #60A5FA 65%, transparent);
    opacity: 0.5;
}
.app-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    letter-spacing: -0.04em;
    color: #1E3A5F;
    margin: 0;
    line-height: 1.1;
}
.app-title span { color: #2563EB; }
.app-sub {
    font-size: 0.76rem;
    color: black !important;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-top: 0.5rem;
}

/* ── Inputs ── */
div[data-testid="stTextInput"] input {
    background:#FFFFFF !important;
    border:1.5px solid #CBD5E1 !important;
    color:#1E293B !important;
    font-family:'JetBrains Mono',monospace !important;
    font-size:0.85rem !important;
    border-radius:10px !important;
    padding:0.7rem 1rem !important;
}
div[data-testid="stTextInput"] input:focus {
    border-color:#2563EB !important;
    box-shadow:0 0 0 3px rgba(37,99,235,0.12) !important;
}
div[data-testid="stTextInput"] input::placeholder { color:#94A3B8 !important; }

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    background:linear-gradient(135deg,#1D4ED8 0%,#3B82F6 100%) !important;
    color:white !important;
    border:none !important;
    border-radius:10px !important;
    font-family:'Inter',sans-serif !important;
    font-weight:600 !important;
    font-size:0.88rem !important;
    padding:0.65rem 1.4rem !important;
    width:100% !important;
    box-shadow:0 2px 8px rgba(37,99,235,0.25) !important;
    transition:opacity 0.2s,transform 0.1s !important;
}
div[data-testid="stButton"] > button:hover { opacity:0.9 !important; transform:translateY(-1px) !important; }

/* ── Risk panel ── */
.risk-panel {
    border-radius:14px; padding:1.4rem; text-align:center;
    border:1.5px solid var(--bdr); background:var(--bg);
}
.risk-HIGH          { --bg:#FFF1F2; --bdr:#FDA4AF; }
.risk-MODERATE-HIGH { --bg:#FFF7ED; --bdr:#FED7AA; }
.risk-MODERATE      { --bg:#FEFCE8; --bdr:#FEF08A; }
.risk-LOW-MODERATE  { --bg:#F0FDF4; --bdr:#BBF7D0; }
.risk-LOW           { --bg:#EFF6FF; --bdr:#BFDBFE; }
.risk-level-text { font-family:'Inter',sans-serif; font-size:1.4rem; font-weight:700; margin:0.2rem 0; letter-spacing:-0.02em; }
.risk-sub { font-size:0.68rem; color:#94A3B8; letter-spacing:0.12em; text-transform:uppercase; margin-top:4px; }

/* ── Molecular properties ── */
.metric-grid {
    display:grid;
    grid-template-columns:1fr 1fr;
    gap:10px;
    width:100%;
    box-sizing:border-box;
}
.m-card {
    background:#FFFFFF;
    border:1.5px solid #E2E8F0;
    border-radius:12px;
    padding:0.85rem 1rem;
    box-shadow:0 1px 4px rgba(0,0,0,0.04);
    min-width:0;
    overflow:hidden;
}
.m-label { font-size:0.63rem; text-transform:uppercase; letter-spacing:0.1em; color:#94A3B8; margin-bottom:4px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.m-value { font-family:'JetBrains Mono',monospace; font-size:1.05rem; font-weight:700; color:#1E293B; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.m-warn { color:#DC2626 !important; }
.m-ok   { color:#16A34A !important; }

/* ── Alert / pass boxes ── */
.alert-box { background:#FFF1F2; border-left:3px solid #F43F5E; border-radius:0 8px 8px 0; padding:0.5rem 0.9rem; margin:0.3rem 0; font-size:0.78rem; color:#BE123C; line-height:1.5; }
.pass-box  { background:#F0FDF4; border-left:3px solid #22C55E; border-radius:0 8px 8px 0; padding:0.5rem 0.9rem; margin:0.3rem 0; font-size:0.78rem; color:#15803D; line-height:1.5; }

/* ── Section headers ── */
.section-head {
    font-family:'Inter',sans-serif; font-size:0.72rem; text-transform:uppercase;
    letter-spacing:0.18em; color:#64748B; font-weight:600;
    border-bottom:2px solid #E2E8F0; padding-bottom:0.45rem; margin:1rem 0 0.75rem;
}

/* ── Molecule container ── */
.mol-box {
    background:#FFFFFF;
    border:1.5px solid #DBEAFE;
    border-radius:14px;
    overflow:hidden;
    display:flex;
    align-items:center;
    justify-content:center;
    width:100%;
    box-sizing:border-box;
    padding:8px;
    min-height:220px;
    max-height:280px;
}
.mol-box img {
    max-width:100%;
    max-height:264px;
    object-fit:contain;
    display:block;
}

/* ── Assay table ── */
.assay-table { width:100%; border-collapse:collapse; }
.assay-tr { border-bottom:1px solid #F1F5F9; }
.assay-tr:hover { background:#F8FAFC; }
.assay-td { padding:0.4rem 0; font-size:0.78rem; vertical-align:middle; }
.assay-name { font-family:'JetBrains Mono',monospace; color:#475569; width:120px; }
.assay-val  { font-family:'JetBrains Mono',monospace; text-align:right; width:56px; }
.assay-bar-wrap { padding:0 12px; }
.assay-bar-bg   { background:#E2E8F0; border-radius:99px; height:5px; width:100%; overflow:hidden; }
.assay-bar-fill { height:100%; border-radius:99px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background:#FFFFFF !important; border-right:1.5px solid #000000; }
section[data-testid="stSidebar"] * { color:#000000; }
section[data-testid="stSidebar"] .stButton > button {
    background:#F1F5F9 !important; color:#1D4ED8 !important;
    border:1px solid #DBEAFE !important; box-shadow:none !important;
    font-size:0.78rem !important; font-family:'JetBrains Mono',monospace !important;
}
section[data-testid="stSidebar"] .stButton > button:hover { background:#DBEAFE !important; }

/* ── READABILITY FIX ── */
html { font-size: 15px; }
.app-sub, .risk-sub, .perf-lab { font-size: 0.85rem !important; }
.input-label, .section-head, .m-label { font-size: 0.78rem !important; }
.assay-td, .alert-box, .pass-box { font-size: 0.82rem !important; }
section[data-testid="stSidebar"] * { font-size: 0.85rem !important; }
div[data-testid="stTextInput"] input { font-size: 0.9rem !important; }
.m-value { font-size: 1.05rem !important; }

/* ── Expander ── */
div[data-testid="stExpander"] { background:#FFFFFF; border:1.5px solid #E2E8F0 !important; border-radius:12px !important; }

/* ── GREY EXPANDER TITLE FIX ── */
div[data-testid="stExpander"] summary span p { color: #6B7280 !important; }
div[data-testid="stExpander"] summary svg { fill: #6B7280 !important; }
div[data-testid="stExpander"] summary { color: #6B7280 !important; }

/* ── Misc ── */
hr { border:none; border-top:1.5px solid #E2E8F0; margin:1.5rem 0; }
.stSpinner > div { border-top-color:#2563EB !important; }
footer { visibility:hidden; }
#MainMenu { visibility:hidden; }
header { visibility:hidden; }
div[data-testid="stSelectbox"] { display:none; }

/* ── FORCE ALL GREY TEXT TO BLACK ── */
[style*="#6B7280"],
[style*="#9CA3AF"],
[style*="#4B5563"],
[style*="rgb(107,114,128)"],
[style*="rgb(156,163,175)"] { color: black !important; }
.app-sub, .risk-sub, .perf-lab, .section-head, .input-label, .assay-desc, .assay-td { color: black !important; }
div[data-testid="stSpinner"] * { color: black !important; }
div[data-testid="stAlert"] * { color: black !important; }
</style>
""", unsafe_allow_html=True)


# ── Constants ──────────────────────────────────────────────────────────────────
TOX21_TARGETS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
]

ASSAY_DESCRIPTIONS = {
    "NR-AR":         "Androgen Receptor",
    "NR-AR-LBD":     "Androgen Receptor LBD",
    "NR-AhR":        "Aryl Hydrocarbon Receptor",
    "NR-Aromatase":  "Aromatase Inhibition",
    "NR-ER":         "Estrogen Receptor α",
    "NR-ER-LBD":     "Estrogen Receptor LBD",
    "NR-PPAR-gamma": "PPAR-gamma Agonism",
    "SR-ARE":        "Antioxidant Response",
    "SR-ATAD5":      "DNA Damage (ATAD5)",
    "SR-HSE":        "Heat Shock Response",
    "SR-MMP":        "Mitochondrial Toxicity",
    "SR-p53":        "p53 Tumour Suppressor",
}

ASSAY_BENCHMARKS = {
    "NR-AR": 0.745, "NR-AR-LBD": 0.811, "NR-AhR": 0.803,
    "NR-Aromatase": 0.791, "NR-ER": 0.728, "NR-ER-LBD": 0.791,
    "NR-PPAR-gamma": 0.726, "SR-ARE": 0.746, "SR-ATAD5": 0.777,
    "SR-HSE": 0.690, "SR-MMP": 0.836, "SR-p53": 0.766,
}

RISK_COLORS = {
    "HIGH":          "#DC2626",
    "MODERATE-HIGH": "#EA580C",
    "MODERATE":      "#CA8A04",
    "LOW-MODERATE":  "#16A34A",
    "LOW":           "#2563EB",
}

EXAMPLES = {
    "Aspirin":            "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine":           "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Bisphenol A":        "CC(c1ccc(O)cc1)(c1ccc(O)cc1)C",
    "Ibuprofen":          "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "Chloroacetanilide":  "CC1=CC=C(C=C1)NC(=O)CCl",
    "Tamoxifen":          "CCC(=C(c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1",
    "Ethanol":            "CCO",
    "Phthalic anhydride": "O=C1OC(=O)c2ccccc21",
}


# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_predictor():
    try:
        sys.path.insert(0, ".")
        from src.ensemble import EnsembleModel
        from src.predict import ToxicityPredictor
        predictor = ToxicityPredictor.load("models/")
        return predictor, None
    except Exception as e:
        return None, str(e)


# ── Helpers ────────────────────────────────────────────────────────────────────
def risk_emoji(level):
    return {"HIGH": "◉", "MODERATE-HIGH": "◉",
            "MODERATE": "◎", "LOW-MODERATE": "○", "LOW": "○"}.get(level, "○")


def render_molecule_svg(smiles: str) -> str:
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        drawer = rdMolDraw2D.MolDraw2DSVG(420, 260)
        opts = drawer.drawOptions()
        opts.backgroundColour = (1.0, 1.0, 1.0, 0.0)
        opts.atomLabelFontSize = 0.55
        opts.bondLineWidth = 1.8
        opts.addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    except Exception:
        return ""


def render_molecule_png(smiles: str):
    try:
        from rdkit import Chem
        from rdkit.Chem.Draw import rdMolDraw2D
        from PIL import Image
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        drawer = rdMolDraw2D.MolDraw2DCairo(420, 260)
        opts = drawer.drawOptions()
        opts.backgroundColour = (1.0, 1.0, 1.0, 1.0)
        opts.atomLabelFontSize = 0.55
        opts.bondLineWidth = 1.8
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))
    except Exception:
        return None


def assay_color(prob: float) -> str:
    if prob >= 0.65: return "#DC2626"
    if prob >= 0.45: return "#EA580C"
    if prob >= 0.28: return "#CA8A04"
    return "#2563EB"


# ── Plotly theme ───────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Space Grotesk, sans-serif", color="#9CA3AF", size=13),
    margin=dict(l=10, r=10, t=36, b=10),
)


def make_radar_chart(assay_probs: dict) -> go.Figure:
    tasks = TOX21_TARGETS
    vals  = [assay_probs.get(t, 0) for t in tasks]
    vals_closed  = vals + [vals[0]]
    tasks_closed = tasks + [tasks[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=tasks_closed, fill='toself',
        fillcolor='rgba(37,99,235,0.08)',
        line=dict(color='#2563EB', width=2),
        marker=dict(size=5, color=[assay_color(v) for v in vals_closed]),
        hovertemplate='<b>%{theta}</b><br>P(Toxic) = %{r:.3f}<extra></extra>',
    ))
    fig.add_trace(go.Scatterpolar(
        r=[0.5]*len(tasks_closed), theta=tasks_closed,
        line=dict(color='rgba(220,38,38,0.35)', width=1, dash='dot'),
        mode='lines', showlegend=False, hoverinfo='skip',
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        polar=dict(
            bgcolor='rgba(241,245,249,0.6)',
            radialaxis=dict(visible=True, range=[0,1],
                            tickfont=dict(size=9, color='#94A3B8'),
                            gridcolor='rgba(148,163,184,0.3)', color='#94A3B8'),
            angularaxis=dict(tickfont=dict(size=9, color='#475569'),
                             gridcolor='rgba(148,163,184,0.25)'),
        ),
        showlegend=False, height=340,
    )
    return fig


def make_bar_chart(assay_probs: dict) -> go.Figure:
    tasks  = [t for t in TOX21_TARGETS if t in assay_probs]
    vals   = [assay_probs[t] for t in tasks]
    colors = [assay_color(v) for v in vals]
    sorted_pairs = sorted(zip(vals, tasks, colors), key=lambda x: x[0])
    vals_s, tasks_s, colors_s = zip(*sorted_pairs)
    fig = go.Figure(go.Bar(
        x=list(vals_s), y=list(tasks_s), orientation='h',
        marker=dict(color=list(colors_s), opacity=0.85, line=dict(width=0)),
        text=[f"{v:.3f}" for v in vals_s], textposition='outside',
        textfont=dict(size=10, color='black', family='Space Mono'),
        hovertemplate='<b>%{y}</b><br>P(Toxic) = %{x:.3f}<extra></extra>',
    ))
    fig.add_vline(x=0.5, line_color='rgba(220,38,38,0.4)', line_dash='dash', line_width=1.5)
    fig.update_layout(
        **PLOTLY_LAYOUT,
        xaxis=dict(range=[0,1.18], showgrid=True,
                   gridcolor='rgba(148,163,184,0.2)', zeroline=False, tickfont=dict(size=9)),
        yaxis=dict(showgrid=False, tickfont=dict(size=10, family='JetBrains Mono, monospace')),
        height=380,
        title=dict(text="Per-Assay Toxicity Probability", font=dict(size=12, color='#64748B'), x=0),
    )
    return fig


def make_gauge(prob: float, risk_level: str) -> go.Figure:
    color = RISK_COLORS.get(risk_level, "#64748B")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(prob * 100, 1),
        number=dict(suffix="%", font=dict(size=32, color=color, family="JetBrains Mono")),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=0, tickcolor='#CBD5E1',
                      tickfont=dict(size=9, color='#94A3B8')),
            bar=dict(color=color, thickness=0.25),
            bgcolor='rgba(0,0,0,0)', borderwidth=0,
            steps=[
                dict(range=[0,18],   color='rgba(37,99,235,0.06)'),
                dict(range=[18,30],  color='rgba(22,163,74,0.06)'),
                dict(range=[30,45],  color='rgba(202,138,4,0.06)'),
                dict(range=[45,65],  color='rgba(234,88,12,0.06)'),
                dict(range=[65,100], color='rgba(220,38,38,0.07)'),
            ],
            threshold=dict(line=dict(color='rgba(100,116,139,0.4)', width=2), value=50),
        ),
    ))
    fig.update_layout(**PLOTLY_LAYOUT, height=200)
    return fig


def make_benchmark_chart(assay_probs: dict, per_assay_auc: dict = None) -> go.Figure:
    OUR_AUC = {
        "NR-AR": 0.7789, "NR-AR-LBD": 0.8340, "NR-AhR": 0.8141,
        "NR-Aromatase": 0.8167, "NR-ER": 0.7429, "NR-ER-LBD": 0.8073,
        "NR-PPAR-gamma": 0.7759, "SR-ARE": 0.8011, "SR-ATAD5": 0.7999,
        "SR-HSE": 0.7402, "SR-MMP": 0.8818, "SR-p53": 0.7875,
    }
    tasks     = TOX21_TARGETS
    our_vals  = [OUR_AUC[t] for t in tasks]
    base_vals = [ASSAY_BENCHMARKS[t] for t in tasks]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='MolNet Baseline', x=tasks, y=base_vals,
        marker_color='rgba(148,163,184,0.55)', marker_line_width=0,
        hovertemplate='<b>%{x}</b><br>Baseline: %{y:.3f}<extra></extra>',
    ))
    fig.add_trace(go.Bar(
        name='ToxPredict (Ours)', x=tasks, y=our_vals,
        marker_color='#2563EB', marker_line_width=0, opacity=0.85,
        hovertemplate='<b>%{x}</b><br>Our AUC: %{y:.3f}<extra></extra>',
    ))
    fig.add_hline(y=0.8, line_color='rgba(100,116,139,0.3)', line_dash='dot')
    fig.update_layout(
        **PLOTLY_LAYOUT,
        barmode='group',
        xaxis=dict(tickangle=-30, tickfont=dict(size=9, family='JetBrains Mono'),
                   gridcolor='rgba(148,163,184,0.15)'),
        yaxis=dict(range=[0.6,0.95], title="ROC-AUC",
                   gridcolor='rgba(148,163,184,0.2)', tickfont=dict(size=9)),
        legend=dict(font=dict(size=10), bgcolor='rgba(255,255,255,0.9)',
                    bordercolor='#E2E8F0', borderwidth=1, x=0.72, y=1.0),
        height=320,
        title=dict(text="Our AUC vs MolNet Baseline (per assay)",
                   font=dict(size=11, color='#64748B'), x=0),
    )
    return fig


# ── Sidebar ────────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("---")
        st.markdown("""<div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.15em;color:#94A3B8;font-weight:600;margin-bottom:8px;'>Model Performance</div>""", unsafe_allow_html=True)
        for label, val in [("ROC-AUC","0.7716"),("AUPR","0.6927"),("MCC","0.4020"),("Mean Assay AUC","0.7984"),("Sensitivity","0.6125")]:
            st.markdown(f"""<div style='display:flex;justify-content:space-between;padding:5px 0;border-bottom:1px solid #F1F5F9;font-size:0.75rem;'>
                <span style='color:#64748B'>{label}</span>
                <span style='font-family:JetBrains Mono,monospace;color:#1D4ED8;font-weight:600'>{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""<div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.15em;color:#94A3B8;font-weight:600;margin-bottom:8px;'>Architecture</div>""", unsafe_allow_html=True)
        for label, val in [("Models","LightGBM + XGBoost + RF"),("Ensemble","AUPR-weighted"),("Features","ECFP4/6 + MACCS + RDKit"),("Feature dim","~7,000+"),("Dataset","Tox21 (~8k cpds)"),("Assays","12 Tox21 targets"),("Threshold","MCC-optimal (0.43)")]:
            st.markdown(f"""<div style='display:flex;justify-content:space-between;padding:4px 0;border-bottom:1px solid #F8FAFC;font-size:0.72rem;'>
                <span style='color:#64748B'>{label}</span>
                <span style='color:#475569;text-align:right;max-width:160px;word-break:break-word;'>{val}</span>
            </div>""", unsafe_allow_html=True)
        st.markdown("---")
        st.markdown("""<div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.15em;color:#94A3B8;font-weight:600;margin-bottom:8px;'>Quick Examples</div>""", unsafe_allow_html=True)
        for name, smi in EXAMPLES.items():
            if st.button(f"  {name}", key=f"ex_{name}", use_container_width=True):
                st.session_state["smiles_val"] = smi
                st.rerun()
        st.markdown("---")


# ── Landing ────────────────────────────────────────────────────────────────────
def render_landing():
    st.markdown("""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:2rem 0;'>
        <div style='background:#FFFFFF;border:1.5px solid #DBEAFE;border-radius:14px;padding:1.2rem;text-align:center;box-shadow:0 1px 6px rgba(37,99,235,0.07);'>
            <div style='font-family:JetBrains Mono,monospace;font-size:1.6rem;font-weight:700;color:#1D4ED8;'>0.80</div>
            <div style='font-size:0.63rem;color:#000000;text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;'>Mean Assay AUC</div>
        </div>
        <div style='background:#FFFFFF;border:1.5px solid #E0E7FF;border-radius:14px;padding:1.2rem;text-align:center;box-shadow:0 1px 6px rgba(67,56,202,0.07);'>
            <div style='font-family:JetBrains Mono,monospace;font-size:1.6rem;font-weight:700;color:#4338CA;'>12</div>
            <div style='font-size:0.63rem;color:#000000;text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;'>Tox21 Assays</div>
        </div>
        <div style='background:#FFFFFF;border:1.5px solid #FFE4E6;border-radius:14px;padding:1.2rem;text-align:center;box-shadow:0 1px 6px rgba(220,38,38,0.06);'>
            <div style='font-family:JetBrains Mono,monospace;font-size:1.6rem;font-weight:700;color:#DC2626;'>3</div>
            <div style='font-size:0.63rem;color:#000000;text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;'>Model Ensemble</div>
        </div>
        <div style='background:#FFFFFF;border:1.5px solid #FEF9C3;border-radius:14px;padding:1.2rem;text-align:center;box-shadow:0 1px 6px rgba(202,138,4,0.06);'>
            <div style='font-family:JetBrains Mono,monospace;font-size:1.6rem;font-weight:700;color:#CA8A04;'>7K+</div>
            <div style='font-size:0.63rem;color:#000000;text-transform:uppercase;letter-spacing:0.1em;margin-top:4px;'>Molecular Features</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="section-head">Model vs MolNet Baseline</div>', unsafe_allow_html=True)
    st.plotly_chart(make_benchmark_chart({}), use_container_width=True, config={"displayModeBar": False})
    st.markdown("""<div style='text-align:center;padding:2rem;color:#94A3B8;font-size:0.82rem;'>
        Enter a SMILES string above or pick a compound from the sidebar to begin analysis.
    </div>""", unsafe_allow_html=True)


# ── Results ────────────────────────────────────────────────────────────────────
def render_results(result):
    if not result.is_valid_smiles:
        st.markdown(f"""
        <div style='background:#FFF1F2;border:1.5px solid #FECDD3;border-radius:10px;padding:1rem 1.2rem;color:#BE123C;font-size:0.85rem;'>
            ❌ <b>Invalid SMILES</b><br><span style='color:#9F1239;'>{result.error_message}</span>
        </div>""", unsafe_allow_html=True)
        return

    risk     = result.risk_level
    color    = RISK_COLORS.get(risk, "#64748B")
    is_toxic = result.aggregate_label == 1

    # ── ROW 1: Structure | Risk | Properties ──────────────────────────────────
    c1, c2, c3 = st.columns([1.2, 1.2, 1.6])

    with c1:
        st.markdown('<div class="section-head">Molecular Structure</div>', unsafe_allow_html=True)
        svg_str = render_molecule_svg(result.smiles)
        if svg_str:
            import base64
            b64 = base64.b64encode(svg_str.encode()).decode()
            st.markdown(f"""
            <div class="mol-box">
                <img src="data:image/svg+xml;base64,{b64}" alt="Molecular Structure" />
            </div>""", unsafe_allow_html=True)
        else:
            img = render_molecule_png(result.smiles)
            if img:
                st.markdown('<div class="mol-box">', unsafe_allow_html=True)
                st.image(img, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="mol-box" style='flex-direction:column;gap:8px;'>
                    <div style='font-size:2rem;'>🧪</div>
                    <div style='font-size:0.72rem;font-family:JetBrains Mono,monospace;color:#94A3B8;text-align:center;padding:0 8px;word-break:break-all;'>
                        {result.smiles[:60]}{'...' if len(result.smiles)>60 else ''}
                    </div>
                </div>""", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-head">Risk Assessment</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(result.aggregate_prob, risk),
                        use_container_width=True, config={"displayModeBar": False})
        pred_str = "⚠ TOXIC"   if is_toxic else "✓ NON-TOXIC"
        pred_col = "#DC2626"   if is_toxic else "#16A34A"
        pred_bg  = "#FFF1F2"   if is_toxic else "#F0FDF4"
        pred_bdr = "#FECDD3"   if is_toxic else "#BBF7D0"
        st.markdown(f"""
        <div class='risk-panel risk-{risk}' style='margin-top:8px;'>
            <div class='risk-level-text' style='color:{color}'>{risk_emoji(risk)} {risk}</div>
            <div class='risk-sub'>risk level</div>
            <div style='margin-top:0.8rem;font-size:0.95rem;font-weight:700;color:{pred_col};
                        background:{pred_bg};border:1.5px solid {pred_bdr};border-radius:8px;
                        display:inline-block;padding:4px 16px;'>{pred_str}</div>
        </div>""", unsafe_allow_html=True)

    with c3:
        st.markdown('<div class="section-head">Molecular Properties</div>', unsafe_allow_html=True)
        if result.admet:
            a = result.admet
            props = [
                ("Mol. Weight", f"{a.mol_weight:.1f} Da", a.mol_weight > 500),
                ("LogP",        f"{a.logp:.3f}",          a.logp > 5),
                ("TPSA",        f"{a.tpsa:.1f} Å²",       a.tpsa > 140 or a.tpsa < 20),
                ("Arom. Rings", str(a.aromatic_rings),     a.aromatic_rings >= 4),
            ]
            html = "<div class='metric-grid'>"
            for label, val, warn in props:
                vc   = "m-warn" if warn else "m-ok"
                icon = "▲" if warn else "✓"
                html += f"""<div class='m-card'>
                    <div class='m-label'>{icon} {label}</div>
                    <div class='m-value {vc}'>{val}</div>
                </div>"""
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

    # ── ROW 2: Alerts | Radar ─────────────────────────────────────────────────
    st.markdown('<hr>', unsafe_allow_html=True)
    c4, c5 = st.columns([1, 1.6])

    with c4:
        if result.alerts:
            st.markdown('<div class="section-head">Structural Alerts</div>', unsafe_allow_html=True)
            for alert in result.alerts:
                clean = alert.replace("⚠ ", "").replace("⚠", "").strip()
                st.markdown(f'<div class="alert-box">⚠ {clean}</div>', unsafe_allow_html=True)
        if result.pass_criteria:
            st.markdown('<div class="section-head">Favourable Properties</div>', unsafe_allow_html=True)
            for p in result.pass_criteria:
                clean = p.replace("✓ ", "").replace("✓", "").strip()
                st.markdown(f'<div class="pass-box">✓ {clean}</div>', unsafe_allow_html=True)
        if not result.alerts and not result.pass_criteria:
            st.markdown("""<div style='color:#94A3B8;font-size:0.8rem;padding:1rem 0;'>No structural alerts detected.</div>""", unsafe_allow_html=True)

    with c5:
        if result.assay_probs:
            st.markdown('<div class="section-head">Toxicity Radar · 12 Assays</div>', unsafe_allow_html=True)
            st.plotly_chart(make_radar_chart(result.assay_probs),
                            use_container_width=True, config={"displayModeBar": False})

    # ── ROW 3: Bar + Table ────────────────────────────────────────────────────
    if result.assay_probs:
        st.markdown('<hr>', unsafe_allow_html=True)
        c6, c7 = st.columns([1.5, 1])

        with c6:
            st.markdown('<div class="section-head">Per-Assay Probability</div>', unsafe_allow_html=True)
            st.plotly_chart(make_bar_chart(result.assay_probs),
                            use_container_width=True, config={"displayModeBar": False})

        with c7:
            st.markdown('<div class="section-head">Assay Detail</div>', unsafe_allow_html=True)
            if result.max_assay:
                mp = result.max_assay_prob
                mc = assay_color(mp)
                st.markdown(f"""
                <div style='background:#F8FAFC;border:1.5px solid #E2E8F0;
                            border-left:4px solid {mc};border-radius:8px;
                            padding:0.7rem 1rem;margin-bottom:10px;'>
                    <div style='font-size:0.62rem;text-transform:uppercase;letter-spacing:0.12em;color:#94A3B8;'>Highest Risk Assay</div>
                    <div style='font-family:JetBrains Mono,monospace;font-size:1rem;font-weight:700;color:{mc};margin-top:2px;'>{result.max_assay}</div>
                    <div style='font-size:0.75rem;color:#64748B;margin-top:2px;'>{ASSAY_DESCRIPTIONS.get(result.max_assay,'')} · p = {mp:.3f}</div>
                </div>""", unsafe_allow_html=True)

            table_rows = ""
            for t in TOX21_TARGETS:
                p = result.assay_probs.get(t, -1)
                if p < 0: continue
                label = result.assay_labels.get(t, 0) if result.assay_labels else (1 if p >= 0.5 else 0)
                col   = assay_color(p)
                bar_w = int(p * 100)
                flag  = f'<span style="color:{col};font-weight:700;">●</span>' if label == 1 else '<span style="color:#CBD5E1;">○</span>'
                table_rows += f"""<tr class='assay-tr'>
                    <td class='assay-td assay-name'>{t}</td>
                    <td class='assay-td assay-bar-wrap'>
                        <div class='assay-bar-bg'><div class='assay-bar-fill' style='width:{bar_w}%;background:{col};'></div></div>
                    </td>
                    <td class='assay-td assay-val' style='color:{col};'>{p:.3f}</td>
                    <td class='assay-td' style='text-align:center;padding-left:6px;'>{flag}</td>
                </tr>"""
            st.markdown(f"<table class='assay-table'>{table_rows}</table>", unsafe_allow_html=True)

    # ── ROW 4: Benchmark ──────────────────────────────────────────────────────
    st.markdown('<hr>', unsafe_allow_html=True)
    st.markdown('<div class="section-head">Our Model vs MolNet Published Baseline</div>', unsafe_allow_html=True)
    st.plotly_chart(make_benchmark_chart(result.assay_probs),
                    use_container_width=True, config={"displayModeBar": False})

    # ── Disclaimer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style="background:#1A1D27; border:1px solid #2E3250; border-radius:8px;
                padding:0.8rem 1.2rem; font-size:0.75rem; color:#5A6080;
                margin-top:2rem; text-align:center;">
        ⚠️ For research purposes only. Not intended for clinical decision-making.
        Predictions are based on Tox21 assay data and may not reflect all toxicity mechanisms.
    </div>
    """, unsafe_allow_html=True)


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    render_sidebar()

    st.markdown("""
    <div class='app-header'>
        <div class='app-title'> Obs<span>cura</span></div>
        <div class='app-sub'>AI-Powered Drug Toxicity Prediction </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Scope explanation (from v1) ────────────────────────────────────────────
    st.markdown("""
    <div style="background:#1A1D27; border:1px solid #2E3250; border-radius:8px;
                padding:1rem 1.5rem; margin-bottom:1.5rem; font-size:0.85rem; color:#8890A8;">
        <b style="color:#C8D0E0;">How toxicity is determined:</b>
        This model predicts <b style="color:#C8D0E0;">Tox21 assay-based toxicity</b> —
        specifically whether a compound activates <b style="color:#C8D0E0;">nuclear receptors</b>
        (NR-AR, NR-ER, NR-AhR etc.) or triggers
        <b style="color:#C8D0E0;">cellular stress response pathways</b>
        (SR-ARE, SR-MMP, SR-p53 etc.).
        This covers <b style="color:#C8D0E0;">acute receptor-mediated and genotoxic toxicity</b>.
        It does <u>not</u> cover chronic mechanisms like carcinogenicity, bioaccumulation,
        or environmental persistence — compounds toxic via those routes (e.g. benzene, trichlorobenzene)
        may score low here by design.
    </div>
    """, unsafe_allow_html=True)

    # ── PubChem lookup ─────────────────────────────────────────────────────────
    with st.expander("🔍 Look up compound by name", expanded=False):
        col_name, col_btn = st.columns([5, 1])
        with col_name:
            cname = st.text_input("Compound name",
                                   placeholder="e.g. aspirin, tamoxifen, benzo[a]pyrene",
                                   label_visibility="collapsed", key="cname_input")
        with col_btn:
            lookup_btn = st.button("Look up", use_container_width=True, key="lookup_btn")
        if lookup_btn and cname:
            try:
                import pubchempy as pcp
                with st.spinner(f"Searching PubChem for '{cname}'..."):
                    results = pcp.get_compounds(cname, "name")
                if results:
                    found_smiles = results[0].isomeric_smiles
                    st.success(f"✅ Found: `{found_smiles}`")
                    st.session_state["smiles_val"] = found_smiles
                    st.rerun()
                else:
                    st.warning(f"No compound found for '{cname}'.")
            except Exception as e:
                st.error(f"Lookup error: {e}")

    # ── SMILES input ───────────────────────────────────────────────────────────
    col_in, col_btn = st.columns([5, 1])
    with col_in:
        smiles = st.text_input(
            "SMILES", label_visibility="collapsed",
            value=st.session_state.get("smiles_val", ""),
            placeholder="Enter SMILES string  ·  e.g. CC(=O)Oc1ccccc1C(=O)O  (Aspirin)",
            key="smiles_main",
        )
    with col_btn:
        predict_btn = st.button("⚡ Predict", use_container_width=True, key="predict_btn")

    if smiles:
        st.session_state["smiles_val"] = smiles

    # ── Load predictor ─────────────────────────────────────────────────────────
    with st.spinner("Loading model..."):
        predictor, load_error = load_predictor()

    if load_error:
        st.markdown(f"""
        <div style='background:#FFF1F2;border:1.5px solid #FECDD3;
                    border-radius:10px;padding:1rem 1.2rem;margin:1rem 0;'>
            <div style='color:#BE123C;font-size:0.85rem;font-weight:600;'>Model failed to load</div>
            <div style='color:#64748B;font-size:0.78rem;margin-top:4px;font-family:JetBrains Mono,monospace;'>{load_error}</div>
            <div style='color:#94A3B8;font-size:0.75rem;margin-top:8px;'>
                Make sure you have run <code style='background:#F1F5F9;padding:2px 6px;border-radius:4px;color:#475569;'>python run.py</code>
                and the <code style='background:#F1F5F9;padding:2px 6px;border-radius:4px;color:#475569;'>models/</code> directory exists.
            </div>
        </div>""", unsafe_allow_html=True)
        render_landing()
        return

    current_smiles = st.session_state.get("smiles_val", "").strip()
    if not current_smiles:
        render_landing()
        return

    with st.spinner("Analysing molecular structure..."):
        result = predictor.predict(current_smiles)

    render_results(result)


if __name__ == "__main__":
    main()
