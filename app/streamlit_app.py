"""
streamlit_app.py
----------------
SmartHire AI — Dark Mode Recruiter Dashboard
Powered by all-MiniLM-L6-v2 Sentence Transformer Embeddings

Run with:
    streamlit run app/streamlit_app.py

Author: SmartHire AI
"""

import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model import get_model
from src.parser import parse_job_description, parse_resume
from src.preprocess import preprocess_text
from src.ranking import CandidateResult, rank_candidates, results_to_dataframe, summarize_rankings
from src.similarity import batch_similarity
from src.skills import full_skill_analysis
from src.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartHireAI-App")

# Bump this string any time vector_store.py changes — forces Streamlit cache bust
_VS_CACHE_KEY = "v3"

st.set_page_config(
    page_title="SmartHire AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background: #0a0e1a; }
        section[data-testid="stSidebar"] { background: #0f1629 !important; border-right: 1px solid #1e2a45; }
        .block-container { padding-top: 1.5rem; }
        .main-header {
            background: linear-gradient(135deg, #0d1b3e 0%, #0a2a6e 50%, #0d1b3e 100%);
            border: 1px solid #00d4ff33; border-radius: 16px; padding: 2rem 2.5rem;
            margin-bottom: 1.5rem; box-shadow: 0 0 40px #00d4ff22, inset 0 1px 0 #00d4ff33;
            position: relative; overflow: hidden;
        }
        .main-header::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
            background: linear-gradient(90deg, transparent, #00d4ff, #7c3aed, #00d4ff, transparent);
        }
        .main-header h1 {
            font-size: 2.4rem; font-weight: 800; margin: 0;
            background: linear-gradient(90deg, #00d4ff, #7c3aed, #00d4ff);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
        }
        .main-header p { font-size: 1rem; color: #94a3b8; margin-top: 0.5rem; }
        .metric-card {
            background: #0f1629; border: 1px solid #1e2a45; border-radius: 12px;
            padding: 1.2rem 1rem; text-align: center; box-shadow: 0 4px 16px rgba(0,0,0,0.4);
            transition: border-color 0.2s;
        }
        .metric-card:hover { border-color: #00d4ff55; }
        .metric-card .value { font-size: 1.9rem; font-weight: 700; color: #00d4ff; }
        .metric-card .label { font-size: 0.8rem; color: #64748b; margin-top: 4px; letter-spacing: 0.05em; text-transform: uppercase; }
        .badge-hr  { background:#052e16; color:#4ade80; padding:5px 14px; border-radius:20px; font-size:0.82rem; font-weight:700; border:1px solid #166534; }
        .badge-rec { background:#0c1a3d; color:#60a5fa; padding:5px 14px; border-radius:20px; font-size:0.82rem; font-weight:700; border:1px solid #1e40af; }
        .badge-con { background:#2d1a00; color:#fbbf24; padding:5px 14px; border-radius:20px; font-size:0.82rem; font-weight:700; border:1px solid #92400e; }
        .badge-nr  { background:#2d0a0a; color:#f87171; padding:5px 14px; border-radius:20px; font-size:0.82rem; font-weight:700; border:1px solid #991b1b; }
        .skill-chip-match   { background:#052e16; color:#4ade80; border-radius:8px; padding:4px 10px; font-size:0.78rem; display:inline-block; margin:2px; border:1px solid #166534; }
        .skill-chip-missing { background:#2d0a0a; color:#f87171; border-radius:8px; padding:4px 10px; font-size:0.78rem; display:inline-block; margin:2px; border:1px solid #991b1b; }
        .skill-chip-critical{ background:#2d1a00; color:#fbbf24; border-radius:8px; padding:4px 10px; font-size:0.78rem; display:inline-block; margin:2px; border:1px solid #92400e; }
        .section-title {
            font-size: 1.1rem; font-weight: 700; color: #e2e8f0;
            border-left: 3px solid #00d4ff; padding-left: 12px;
            margin: 1.5rem 0 1rem 0; letter-spacing: 0.02em;
        }
        .divider { height: 1px; background: #1e2a45; margin: 1.5rem 0; }
        .sidebar-info {
            background: #0a1628; border: 1px solid #1e2a45; border-radius: 10px;
            padding: 12px 14px; font-size: 0.85rem; color: #94a3b8; line-height: 1.8;
        }
        .sidebar-info b { color: #00d4ff; }
        .skills-box {
            background: #0f1629; border: 1px solid #1e2a45; border-radius: 10px;
            padding: 12px 16px; max-height: 160px; overflow-y: auto;
        }
        .vs-card {
            background: #0a1628; border: 1px solid #1e3a5f; border-radius: 12px;
            padding: 1.2rem 1.4rem; margin-bottom: 1rem;
        }
        .vs-card-title { font-size: 1rem; font-weight: 700; color: #00d4ff; margin-bottom: 0.5rem; }
        .vs-stat { font-size: 0.88rem; color: #94a3b8; margin: 4px 0; }
        .vs-stat b { color: #e2e8f0; }
        .stTextArea textarea { background: #0f1629 !important; color: #e2e8f0 !important; border: 1px solid #1e2a45 !important; }
        .stSelectbox > div > div { background: #0f1629 !important; color: #e2e8f0 !important; }
        div[data-testid="stExpander"] { background: #0f1629; border: 1px solid #1e2a45; border-radius: 10px; }
        .stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #0066cc, #7c3aed) !important;
            border: none !important; color: white !important; font-weight: 700 !important;
            box-shadow: 0 0 20px #00d4ff33 !important;
        }
        .stButton > button[kind="primary"]:hover { box-shadow: 0 0 30px #00d4ff66 !important; }
        .stTabs [data-baseweb="tab"] { color: #64748b !important; }
        .stTabs [aria-selected="true"] { color: #00d4ff !important; border-bottom-color: #00d4ff !important; }
        .stDataFrame { background: #0f1629; }
        .stSuccess { background: #052e16 !important; border: 1px solid #166534 !important; color: #4ade80 !important; }
        .stInfo    { background: #0c1a3d !important; border: 1px solid #1e40af !important; color: #60a5fa !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
#  Cached Resource Loading
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model():
    return get_model()


@st.cache_resource(show_spinner=False)
def load_vector_store(_cache_key: str = "v3"):
    """_cache_key forces Streamlit to reload when incremented."""
    return get_vector_store(persist_dir=str(ROOT / "vector_db"))


# ─────────────────────────────────────────────────────────────
#  Helper Functions
# ─────────────────────────────────────────────────────────────

def badge_html(recommendation: str) -> str:
    class_map = {
        "Highly Recommended": "badge-hr",
        "Recommended":        "badge-rec",
        "Consider":           "badge-con",
        "Not Recommended":    "badge-nr",
    }
    return f'<span class="{class_map.get(recommendation, "badge-nr")}">{recommendation}</span>'


def render_skill_chips(skills: list, chip_class: str) -> str:
    if not skills:
        return "<em style='color:#475569'>None detected</em>"
    return " ".join(f'<span class="{chip_class}">{s}</span>' for s in skills)


def score_color(pct: float) -> str:
    if pct >= 90:   return "#4ade80"
    elif pct >= 80: return "#00d4ff"
    elif pct >= 70: return "#fbbf24"
    else:           return "#f87171"


def render_progress_bar(pct: float, color: str, height: int = 22) -> str:
    return f"""
    <div style="background:#1e2a45;border-radius:8px;height:{height}px;overflow:hidden;">
      <div style="width:{pct}%;background:linear-gradient(90deg,{color}aa,{color});height:100%;
                  border-radius:8px;display:flex;align-items:center;padding-left:8px;
                  font-size:0.78rem;font-weight:700;color:#fff;transition:width 0.6s ease;
                  box-shadow:0 0 8px {color}55;">
        {pct:.1f}%
      </div>
    </div>
    """


def safe_get_all_metadata(vs) -> List[Dict]:
    """Safe wrapper — works even if old cached VectorStore lacks get_all_metadata."""
    if hasattr(vs, "get_all_metadata"):
        return vs.get_all_metadata()
    names = vs.get_all_names() if hasattr(vs, "get_all_names") else []
    return [{"name": n, "text_length": 0, "embedding_dim": "N/A", "indexed_at": "N/A"} for n in names]


# ─────────────────────────────────────────────────────────────
#  Load Resources
# ─────────────────────────────────────────────────────────────

with st.spinner("⚡ Loading SmartHire AI model (first load ~10s, then cached)..."):
    model = load_model()
st.success("✅ SmartHire AI model ready", icon="🤖")

vector_store = load_vector_store(_cache_key=_VS_CACHE_KEY)

# ─────────────────────────────────────────────────────────────
#  Sidebar
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        """
        <div style='text-align:center; padding: 1rem 0 0.5rem 0;'>
            <span style='font-size:2rem;'>🤖</span><br>
            <span style='font-size:1.2rem; font-weight:800;
                background:linear-gradient(90deg,#00d4ff,#7c3aed);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;'>SmartHire AI</span><br>
            <span style='font-size:0.75rem; color:#475569;'>Transformer-Based Hiring</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("### ⚙️ Settings")
    similarity_weight = st.slider(
        "Semantic Similarity Weight",
        min_value=0.5, max_value=0.9, value=0.7, step=0.05,
        help="Weight given to semantic similarity vs skill coverage",
    )
    skill_weight = round(1.0 - similarity_weight, 2)
    st.caption(f"Skill Coverage Weight: **{skill_weight}**")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Model Info")
    try:
        info = model.get_model_info()
        finetuned_badge = " 🎯" if info.get("is_finetuned") else ""
        st.markdown(
            f"""
            <div class='sidebar-info'>
            🤖 <b>Model:</b> {info['model_name']}{finetuned_badge}<br>
            📐 <b>Dim:</b> {info['embedding_dim']}<br>
            🔢 <b>Max tokens:</b> {info['max_tokens']}<br>
            📏 <b>Pooling:</b> {info['pooling']}<br>
            💻 <b>Device:</b> {info['device'].upper()}
            </div>
            """,
            unsafe_allow_html=True,
        )
    except Exception:
        st.markdown(
            """<div class='sidebar-info'>🤖 <b>Model:</b> all-MiniLM-L6-v2<br>📐 <b>Dim:</b> 384</div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🗄️ Vector Index")
    vs_info_sb  = vector_store.get_info()
    vs_count_sb = vs_info_sb["count"]
    vs_color_sb = "#4ade80" if vs_count_sb > 0 else "#64748b"
    vs_dim_sb   = vs_info_sb.get("dim", "N/A")
    try:
        model_dim_sb = model.get_model_info()["embedding_dim"]
    except Exception:
        model_dim_sb = None
    dim_warn = ""
    if vs_count_sb > 0 and model_dim_sb and vs_dim_sb != "N/A":
        if int(vs_dim_sb) != int(model_dim_sb):
            dim_warn = "<br>⚠️ <span style='color:#f87171;'>Dim mismatch — clear & rebuild!</span>"
    st.markdown(
        f"""
        <div class='sidebar-info'>
        🗄️ <b>Backend:</b> {vs_info_sb['backend'].upper()}<br>
        📦 <b>Indexed:</b> <span style='color:{vs_color_sb};font-weight:700;'>{vs_count_sb} resume(s)</span><br>
        📐 <b>Vector dim:</b> {vs_dim_sb}<br>
        💾 <b>Persistent:</b> Yes (vector_db/){dim_warn}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("### 🎯 Score Legend")
    st.markdown(
        """
        <span class='badge-hr'>≥90%&nbsp; Highly Recommended</span><br><br>
        <span class='badge-rec'>≥80%&nbsp; Recommended</span><br><br>
        <span class='badge-con'>≥70%&nbsp; Consider</span><br><br>
        <span class='badge-nr'>&lt;70%&nbsp; Not Recommended</span>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem;color:#334155;text-align:center;'>SmartHire AI · all-MiniLM-L6-v2 · PyTorch</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────
#  Main Header
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class='main-header'>
        <h1>🤖 SmartHire AI</h1>
        <p>Transformer-Based Resume &amp; Job Matching System &nbsp;·&nbsp;
           all-MiniLM-L6-v2 &nbsp;·&nbsp; Semantic NLP &nbsp;·&nbsp;
           Candidate Ranking &nbsp;·&nbsp; Skill Gap Analysis &nbsp;·&nbsp; Vector Index</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_upload, tab_results, tab_skills, tab_ranking, tab_vector = st.tabs(
    ["📤 Upload & Analyze", "📊 Match Results", "🔍 Skill Gap Analysis", "🏆 Candidate Ranking", "🗄️ Vector Index"]
)

# ═════════════════════════════════════════════════════════════
#  TAB 1 — Upload & Analyze
# ═════════════════════════════════════════════════════════════

with tab_upload:
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown("<div class='section-title'>📋 Job Description</div>", unsafe_allow_html=True)
        jd_input_mode = st.radio("Input method", ["Paste text", "Upload file"],
                                  horizontal=True, label_visibility="collapsed")
        jd_text_raw: Optional[str] = None

        if jd_input_mode == "Paste text":
            jd_paste = st.text_area("Paste the job description here", height=280,
                placeholder="e.g.\n\nWe are looking for a Machine Learning Engineer...\n\nRequirements:\n- Python, PyTorch\n- NLP, BERT, Transformers\n- Docker and AWS")
            if jd_paste and jd_paste.strip():
                jd_text_raw = jd_paste
        else:
            jd_file = st.file_uploader("Upload JD (PDF, DOCX, or TXT)", type=["pdf","docx","txt"], key="jd_file")
            if jd_file:
                try:
                    jd_text_raw = parse_job_description(jd_file.read(), filename=jd_file.name)
                    st.success(f"Loaded JD: {jd_file.name} ({len(jd_text_raw):,} chars)")
                except Exception as e:
                    st.error(f"Failed to parse JD: {e}")

    with col_right:
        st.markdown("<div class='section-title'>📄 Candidate Resumes</div>", unsafe_allow_html=True)
        st.caption("Upload one or more resumes (PDF, DOCX, or TXT)")
        resume_files = st.file_uploader("Upload resumes", type=["pdf","docx","txt"],
                                         accept_multiple_files=True, label_visibility="collapsed")
        parsed_resumes: List[dict] = []
        if resume_files:
            for rf in resume_files:
                try:
                    text = parse_resume(rf.read(), filename=rf.name)
                    parsed_resumes.append({"name": Path(rf.name).stem, "raw_text": text})
                    st.success(f"✅ {rf.name}  ({len(text):,} chars)")
                except Exception as e:
                    st.error(f"❌ {rf.name}: {e}")

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    vs_cnt = vector_store.count()
    if vs_cnt > 0 and not parsed_resumes:
        st.info(f"💡 **Vector Index active** — {vs_cnt} resume(s) indexed. Use 🗄️ Vector Index tab for instant search.")

    can_analyze = bool(jd_text_raw) and bool(parsed_resumes)
    analyze_btn = st.button("🚀 Analyze Candidates", type="primary",
                             disabled=not can_analyze, use_container_width=True)

    if not can_analyze:
        if not jd_text_raw:      st.info("👆 Please provide a job description.")
        elif not parsed_resumes: st.info("👆 Please upload at least one resume.")

    if analyze_btn and can_analyze:
        with st.status("⚙️ Running SmartHire AI Pipeline...", expanded=True) as status:
            st.write("📝 Preprocessing text...")
            try:
                jd_clean = preprocess_text(jd_text_raw)
            except Exception as e:
                st.error(f"JD preprocessing failed: {e}"); st.stop()

            clean_resumes = []
            for r in parsed_resumes:
                try:
                    clean_text = preprocess_text(r["raw_text"])
                    clean_resumes.append({**r, "clean_text": clean_text})
                except Exception as e:
                    st.warning(f"Skipping {r['name']}: {e}")

            if not clean_resumes:
                st.error("No valid resumes after preprocessing."); st.stop()

            st.write(f"🤖 Encoding {len(clean_resumes)} resume(s)...")
            t0 = time.time()
            resume_embeddings = model.encode([r["clean_text"] for r in clean_resumes])
            jd_embedding      = model.encode_single(jd_clean)
            encode_time       = time.time() - t0

            st.write("📐 Computing cosine similarities...")
            scores = batch_similarity(resume_embeddings, jd_embedding)
            for r, score in zip(clean_resumes, scores):
                r["score"] = score

            st.write("🏆 Ranking candidates...")
            results = rank_candidates(
                [{"name": r["name"], "text": r["clean_text"], "score": r["score"]} for r in clean_resumes],
                jd_clean, similarity_weight=similarity_weight, skill_weight=skill_weight,
            )
            summary = summarize_rankings(results)
            status.update(label=f"✅ {len(results)} candidate(s) ranked in {encode_time:.1f}s", state="complete")

        st.session_state.update({"results": results, "summary": summary,
                                   "jd_clean": jd_clean, "encode_time": encode_time})
        st.success(f"🎉 Done! Analyzed **{len(results)}** candidate(s) in **{encode_time:.2f}s**. "
                   f"Switch to **Match Results** tab.", icon="✅")


# ═════════════════════════════════════════════════════════════
#  TAB 2 — Match Results
# ═════════════════════════════════════════════════════════════

with tab_results:
    if "results" not in st.session_state:
        st.info("👆 Upload resumes and a JD in the **Upload & Analyze** tab first."); st.stop()

    results: List[CandidateResult] = st.session_state["results"]
    summary: dict = st.session_state["summary"]
    encode_time: float = st.session_state.get("encode_time", 0)

    st.markdown("<div class='section-title'>📊 Pipeline Summary</div>", unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    for col, val, label in [
        (c1, summary["total_candidates"],                              "Candidates"),
        (c2, f"{summary['average_score']:.1f}%",                      "Avg Score"),
        (c3, f"{summary['highest_score']:.1f}%",                      "Top Score"),
        (c4, summary["highly_recommended"]+summary["recommended"],     "Recommended"),
        (c5, f"{encode_time:.1f}s",                                    "Encode Time"),
    ]:
        with col:
            st.markdown(f"<div class='metric-card'><div class='value'>{val}</div>"
                        f"<div class='label'>{label}</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    color_map = {"Highly Recommended":"#4ade80","Recommended":"#00d4ff",
                 "Consider":"#fbbf24","Not Recommended":"#f87171"}

    st.markdown("<div class='section-title'>📈 Match Score Distribution</div>", unsafe_allow_html=True)
    fig_bar = px.bar(
        pd.DataFrame({"Candidate":[r.name for r in results],
                      "Match Score (%)":[r.score_pct for r in results],
                      "Recommendation":[r.recommendation for r in results]}),
        x="Candidate", y="Match Score (%)", color="Recommendation",
        color_discrete_map=color_map, text="Match Score (%)", height=380,
    )
    fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig_bar.update_layout(plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
                          yaxis_range=[0,115], margin=dict(t=50,b=20),
                          xaxis=dict(gridcolor="#1e2a45"), yaxis=dict(gridcolor="#1e2a45"))
    st.plotly_chart(fig_bar, use_container_width=True)

    if len(results) > 1:
        col_pie, col_scatter = st.columns(2)
        with col_pie:
            st.markdown("<div class='section-title'>🥧 Recommendation Split</div>", unsafe_allow_html=True)
            pie_data = {k: v for k, v in {
                "Highly Recommended": summary["highly_recommended"],
                "Recommended": summary["recommended"],
                "Consider": summary["consider"],
                "Not Recommended": summary["not_recommended"],
            }.items() if v > 0}
            fig_pie = px.pie(values=list(pie_data.values()), names=list(pie_data.keys()),
                             color=list(pie_data.keys()), color_discrete_map=color_map, height=320)
            fig_pie.update_layout(paper_bgcolor="#0a0e1a", font_color="#e2e8f0", margin=dict(t=20,b=20))
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_scatter:
            st.markdown("<div class='section-title'>📉 Similarity vs Skill Coverage</div>", unsafe_allow_html=True)
            fig_sc = px.scatter(
                pd.DataFrame({"Candidate":[r.name for r in results],
                              "Similarity (%)":[round(r.similarity_score*100,2) for r in results],
                              "Skill Coverage (%)":[r.skill_coverage_pct for r in results],
                              "Match Score (%)":[r.score_pct for r in results],
                              "Recommendation":[r.recommendation for r in results]}),
                x="Similarity (%)", y="Skill Coverage (%)", size="Match Score (%)",
                color="Recommendation", color_discrete_map=color_map,
                hover_data=["Candidate","Match Score (%)"], height=320,
            )
            fig_sc.update_layout(plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
                                  margin=dict(t=20,b=20), xaxis=dict(gridcolor="#1e2a45"),
                                  yaxis=dict(gridcolor="#1e2a45"))
            st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("<div class='section-title'>🎯 Per-Candidate Results</div>", unsafe_allow_html=True)
    for rank, result in enumerate(results, start=1):
        color = score_color(result.score_pct)
        with st.expander(f"#{rank}  {result.name}  —  {result.score_pct:.1f}%  |  {result.recommendation}",
                         expanded=(rank == 1)):
            r1, r2 = st.columns(2)
            with r1:
                for lbl, val, c in [("Match Score", result.score_pct, color),
                                     ("Skill Coverage", result.skill_coverage_pct, "#7c3aed"),
                                     ("Semantic Similarity", round(result.similarity_score*100,1), "#00d4ff")]:
                    st.markdown(f"**{lbl}**")
                    st.markdown(render_progress_bar(val, c), unsafe_allow_html=True)
            with r2:
                st.markdown("**Recommendation**")
                st.markdown(badge_html(result.recommendation), unsafe_allow_html=True)
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| Match Score | **{result.score_pct:.1f}%** |
| Semantic Similarity | {result.similarity_score*100:.1f}% |
| Skill Coverage | {result.skill_coverage_pct:.1f}% |
| Matched Skills | {len(result.matching_skills)} |
| Missing Skills | {len(result.missing_skills)} |
| Critical Missing | {len(result.critical_missing)} |""")


# ═════════════════════════════════════════════════════════════
#  TAB 3 — Skill Gap Analysis
# ═════════════════════════════════════════════════════════════

with tab_skills:
    if "results" not in st.session_state:
        st.info("👆 Run an analysis first from the **Upload & Analyze** tab."); st.stop()

    results: List[CandidateResult] = st.session_state["results"]
    st.markdown("<div class='section-title'>🔍 Skill Gap Analysis</div>", unsafe_allow_html=True)

    selected_name = st.selectbox("Select Candidate", [r.name for r in results])
    selected = next(r for r in results if r.name == selected_name)
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1: st.metric("Matched Skills",   len(selected.matching_skills))
    with s2: st.metric("Missing Skills",   len(selected.missing_skills))
    with s3: st.metric("Critical Missing", len(selected.critical_missing))
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    col_m, col_mi = st.columns(2)
    with col_m:
        st.markdown("#### ✅ Matching Skills")
        st.markdown(f"<div class='skills-box'>{render_skill_chips(selected.matching_skills,'skill-chip-match')}</div>",
                    unsafe_allow_html=True)
        st.markdown("#### 💼 Additional Resume Skills")
        st.markdown(f"<div class='skills-box'>{render_skill_chips(selected.resume_only_skills[:20],'skill-chip-match')}</div>",
                    unsafe_allow_html=True)
    with col_mi:
        st.markdown("#### ❌ Missing Skills")
        st.markdown(f"<div class='skills-box'>{render_skill_chips(selected.missing_skills,'skill-chip-missing')}</div>",
                    unsafe_allow_html=True)
        st.markdown("#### ⚠️ Critical Missing")
        st.markdown(f"<div class='skills-box'>{render_skill_chips(selected.critical_missing,'skill-chip-critical')}</div>",
                    unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 Skill Coverage Breakdown</div>", unsafe_allow_html=True)
    all_skills = list(set(selected.matching_skills + selected.missing_skills))
    if all_skills:
        skill_df = pd.DataFrame({
            "Skill"  : all_skills,
            "Status" : ["Matched" if s in selected.matching_skills else "Missing" for s in all_skills],
        })
        skill_df["Present"] = skill_df["Status"].apply(lambda x: 1 if x=="Matched" else 0)
        fig_sk = px.bar(skill_df.sort_values("Status").head(30), x="Skill", y="Present", color="Status",
                        color_discrete_map={"Matched":"#4ade80","Missing":"#f87171"},
                        title=f"Skill Matrix — {selected_name}", height=360)
        fig_sk.update_layout(yaxis=dict(tickvals=[0,1],ticktext=["Missing","Matched"],gridcolor="#1e2a45"),
                              plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
                              margin=dict(t=40,b=40), xaxis=dict(gridcolor="#1e2a45"))
        st.plotly_chart(fig_sk, use_container_width=True)

    if len(results) > 1:
        st.markdown("<div class='section-title'>🔄 Cross-Candidate Comparison</div>", unsafe_allow_html=True)
        fig_comp = px.bar(
            pd.DataFrame({"Candidate":[r.name for r in results],
                          "Matched Skills":[len(r.matching_skills) for r in results],
                          "Missing Skills":[len(r.missing_skills) for r in results],
                          "Critical Missing":[len(r.critical_missing) for r in results]}),
            x="Candidate", y=["Matched Skills","Missing Skills","Critical Missing"],
            barmode="group", color_discrete_sequence=["#4ade80","#f87171","#fbbf24"], height=360,
        )
        fig_comp.update_layout(plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a", font_color="#e2e8f0",
                               margin=dict(t=20,b=20), xaxis=dict(gridcolor="#1e2a45"),
                               yaxis=dict(gridcolor="#1e2a45"))
        st.plotly_chart(fig_comp, use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  TAB 4 — Candidate Ranking
# ═════════════════════════════════════════════════════════════

with tab_ranking:
    if "results" not in st.session_state:
        st.info("👆 Run an analysis first from the **Upload & Analyze** tab."); st.stop()

    results: List[CandidateResult] = st.session_state["results"]
    summary: dict = st.session_state["summary"]

    st.markdown("<div class='section-title'>🏆 Candidate Ranking Leaderboard</div>", unsafe_allow_html=True)
    df = results_to_dataframe(results)
    st.dataframe(df[["Candidate","Match Score (%)","Recommendation","Skill Coverage (%)"]],
                 use_container_width=True, height=min(60+len(results)*42, 420))
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    top = results[0]
    st.markdown("<div class='section-title'>🥇 Top Candidate Spotlight</div>", unsafe_allow_html=True)
    col_s1, col_s2 = st.columns([1, 2])
    with col_s1:
        color = score_color(top.score_pct)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=top.score_pct,
            title={"text": top.name, "font": {"size":16,"color":"#e2e8f0"}},
            gauge={"axis":{"range":[0,100],"tickcolor":"#64748b"},"bar":{"color":color},
                   "bgcolor":"#0f1629","bordercolor":"#1e2a45",
                   "steps":[{"range":[0,70],"color":"#1a0a0a"},{"range":[70,80],"color":"#1a1200"},
                             {"range":[80,90],"color":"#0a1628"},{"range":[90,100],"color":"#052e16"}],
                   "threshold":{"line":{"color":color,"width":4},"thickness":0.8,"value":top.score_pct}},
            number={"suffix":"%","font":{"size":36,"color":color}},
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=30,b=0),
                                paper_bgcolor="#0a0e1a", font_color="#e2e8f0")
        st.plotly_chart(fig_gauge, use_container_width=True)
    with col_s2:
        st.markdown(f"**{top.name}** — match score **{top.score_pct:.1f}%**")
        st.markdown(badge_html(top.recommendation), unsafe_allow_html=True)
        st.markdown("**Key Strengths:**")
        st.markdown(render_skill_chips(top.matching_skills[:12],"skill-chip-match"), unsafe_allow_html=True)
        if top.critical_missing:
            st.markdown("**⚠️ Areas to Address:**")
            st.markdown(render_skill_chips(top.critical_missing,"skill-chip-critical"), unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📊 All Candidates — Score Breakdown</div>", unsafe_allow_html=True)
    for rank, result in enumerate(results, start=1):
        cr, cn, cb, cbadge = st.columns([0.5, 2, 4, 2])
        with cr: st.markdown(f"**#{rank}**")
        with cn: st.markdown(f"**{result.name}**")
        with cb:
            c = score_color(result.score_pct)
            st.markdown(render_progress_bar(result.score_pct, c, height=26), unsafe_allow_html=True)
        with cbadge: st.markdown(badge_html(result.recommendation), unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>💾 Export Results</div>", unsafe_allow_html=True)
    csv_bytes = results_to_dataframe(results).to_csv(index=True).encode("utf-8")
    st.download_button("⬇️ Download Results as CSV", data=csv_bytes,
                       file_name="smarthire_ai_results.csv", mime="text/csv", use_container_width=True)


# ═════════════════════════════════════════════════════════════
#  TAB 5 — Vector Index
# ═════════════════════════════════════════════════════════════

with tab_vector:
    st.markdown("<div class='section-title'>🗄️ Resume Vector Index</div>", unsafe_allow_html=True)
    st.caption("Pre-encode and persistently store resume embeddings for instant JD search (sub-100ms).")

    vs_info = vector_store.get_info()
    try:
        current_model_dim = model.get_model_info()["embedding_dim"]
    except Exception:
        current_model_dim = None

    stored_dim = vs_info.get("dim", "N/A")

    # Dimension mismatch banner
    if vs_info["count"] > 0 and current_model_dim and stored_dim != "N/A":
        if int(stored_dim) != int(current_model_dim):
            st.error(
                f"⚠️ **Dimension Mismatch!** Stored vectors are **{stored_dim}-dim** "
                f"but current model outputs **{current_model_dim}-dim**.  \n"
                f"**Fix:** Go to ⚙️ Manage Index → Clear → Rebuild.", icon="🚨"
            )

    # Status cards
    vi_c1, vi_c2, vi_c3, vi_c4 = st.columns(4)
    with vi_c1:
        st.markdown(f"<div class='metric-card'><div class='value' style='color:#4ade80;'>"
                    f"{vs_info['count']}</div><div class='label'>Indexed Resumes</div></div>",
                    unsafe_allow_html=True)
    with vi_c2:
        st.markdown(f"<div class='metric-card'><div class='value' style='color:#00d4ff;'>"
                    f"{vs_info['backend'].upper()}</div><div class='label'>Backend</div></div>",
                    unsafe_allow_html=True)
    with vi_c3:
        sl, sc = ("Ready","#4ade80") if vs_info["count"] > 0 else ("Empty","#f87171")
        st.markdown(f"<div class='metric-card'><div class='value' style='color:{sc};'>"
                    f"{sl}</div><div class='label'>Status</div></div>", unsafe_allow_html=True)
    with vi_c4:
        dd = stored_dim if stored_dim != "N/A" else (current_model_dim or "N/A")
        st.markdown(f"<div class='metric-card'><div class='value' style='color:#7c3aed;'>"
                    f"{dd}</div><div class='label'>Vector Dim</div></div>", unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    # ── Section A: Build Index ────────────────────────────────
    st.markdown("<div class='section-title'>⚡ Build / Update Index</div>", unsafe_allow_html=True)
    index_files = st.file_uploader("Upload resumes to index (PDF, DOCX, TXT)",
                                    type=["pdf","docx","txt"], accept_multiple_files=True, key="index_upload")
    col_b1, col_b2 = st.columns([2, 1])
    with col_b1:
        rebuild_mode = st.radio("Index mode",
                                 ["Add to existing index", "Rebuild from scratch (clear first)"],
                                 horizontal=True)
    with col_b2:
        build_btn = st.button("⚡ Build Index", type="primary",
                               disabled=not bool(index_files), use_container_width=True)

    if not index_files:
        st.info("👆 Upload at least one resume to build or update the index.")

    if build_btn and index_files:
        to_index, parse_errors = [], []
        for rf in index_files:
            try:
                raw   = parse_resume(rf.read(), filename=rf.name)
                clean = preprocess_text(raw)
                to_index.append({"name": Path(rf.name).stem, "text": clean})
            except Exception as e:
                parse_errors.append(f"{rf.name}: {e}")

        for err in parse_errors:
            st.warning(f"⚠️ Skipped — {err}")

        if not to_index:
            st.error("No valid resumes could be parsed.")
        else:
            if "Rebuild" in rebuild_mode:
                vector_store.clear()
                st.info("🗑️ Existing index cleared.")

            progress_bar = st.progress(0, text="Starting...")
            status_txt   = st.empty()

            def update_progress(i, total, name):
                progress_bar.progress(int((i+1)/total*100), text=f"Encoding {i+1}/{total}: {name}")
                status_txt.markdown(f"🤖 Encoding **{name}**...")

            with st.spinner("Building vector index..."):
                t0    = time.time()
                stats = vector_store.build_index(resumes=to_index, model=model,
                                                  progress_callback=update_progress)
                dur   = time.time() - t0

            progress_bar.progress(100, text="Done!")
            status_txt.empty()
            st.success(f"✅ **{stats['indexed']}** resume(s) indexed in **{dur:.1f}s** | "
                       f"Total: **{stats['total']}** | Backend: **{stats['backend'].upper()}**", icon="🗄️")
            if stats["skipped"]:
                st.warning(f"⚠️ {stats['skipped']} resume(s) skipped.")
            st.rerun()

    # ── Section B: Indexed Candidates ────────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>📋 Indexed Candidates</div>", unsafe_allow_html=True)

    all_meta = safe_get_all_metadata(vector_store)
    if all_meta:
        idx_df = pd.DataFrame([
            {
                "#"          : i + 1,
                "Candidate"  : m.get("name", "Unknown"),
                "Text Length": f"{m.get('text_length', 0):,} chars" if m.get("text_length") else "N/A",
                "Vector Dim" : m.get("embedding_dim", "N/A"),
                "Indexed At" : m.get("indexed_at", "N/A"),
                "Status"     : "✅ Indexed",
            }
            for i, m in enumerate(all_meta)
        ])
        st.dataframe(idx_df, use_container_width=True, height=min(60+len(all_meta)*38, 400))
    else:
        st.markdown("<div style='color:#475569;font-style:italic;padding:1rem 0;'>No resumes indexed yet.</div>",
                    unsafe_allow_html=True)

    # ── Section C: Instant JD Search ─────────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>🔎 Instant JD Search</div>", unsafe_allow_html=True)
    st.caption("Search your indexed resume pool against any JD — results in under 100ms.")

    search_jd_text = st.text_area(
        "Paste Job Description to search",
        height=180,
        placeholder="e.g.\n\nWe are looking for a Senior Data Scientist with Python, ML, and AWS experience...",
        key="vs_jd_search",
    )
    max_k = max(1, vs_info["count"])
    top_k = st.slider("Top K results", min_value=1, max_value=min(20, max_k), value=min(5, max_k))

    search_btn = st.button("🔎 Search Vector Index", type="primary",
                            disabled=(not bool(search_jd_text and search_jd_text.strip())
                                      or vector_store.is_empty()),
                            use_container_width=True)

    if vector_store.is_empty():
        st.info("👆 Build the index first.")

    if search_btn and search_jd_text and search_jd_text.strip():
        try:
            with st.spinner("🔎 Searching..."):
                t_s        = time.time()
                jd_emb_vs  = model.encode_single(preprocess_text(search_jd_text))
                vs_results = vector_store.search(jd_emb_vs, top_k=top_k)
                search_ms  = (time.time() - t_s) * 1000

            st.success(f"⚡ Found **{len(vs_results)}** result(s) in **{search_ms:.1f}ms**", icon="🔎")
            st.markdown("<div class='section-title'>🏆 Top Matching Candidates</div>", unsafe_allow_html=True)

            if vs_results:
                fig_vs = px.bar(
                    pd.DataFrame({"Candidate":[r["name"] for r in vs_results],
                                  "Similarity (%)":[round(r["score"]*100,2) for r in vs_results]}),
                    x="Candidate", y="Similarity (%)", text="Similarity (%)", height=320,
                    color="Similarity (%)",
                    color_continuous_scale=["#f87171","#fbbf24","#4ade80"], range_color=[0,100],
                )
                fig_vs.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                fig_vs.update_layout(plot_bgcolor="#0a0e1a", paper_bgcolor="#0a0e1a",
                                     font_color="#e2e8f0", yaxis_range=[0,115],
                                     margin=dict(t=20,b=20), coloraxis_showscale=False,
                                     xaxis=dict(gridcolor="#1e2a45"), yaxis=dict(gridcolor="#1e2a45"))
                st.plotly_chart(fig_vs, use_container_width=True)

                for i, res in enumerate(vs_results, start=1):
                    sim_pct = round(res["score"]*100, 2)
                    with st.expander(f"#{i}  {res['name']}  —  Similarity: {sim_pct:.1f}%", expanded=(i==1)):
                        st.markdown("**Semantic Similarity**")
                        st.markdown(render_progress_bar(sim_pct, score_color(sim_pct)), unsafe_allow_html=True)
                        meta = res.get("metadata", {})
                        tl   = meta.get("text_length", 0)
                        st.markdown(
                            f"<div class='vs-stat'>📅 <b>Indexed at:</b> {meta.get('indexed_at','N/A')}</div>"
                            f"<div class='vs-stat'>📄 <b>Text length:</b> {f'{tl:,} chars' if tl else 'N/A'}</div>"
                            f"<div class='vs-stat'>📐 <b>Vector dim:</b> {meta.get('embedding_dim','N/A')}</div>",
                            unsafe_allow_html=True,
                        )
                        preview = res.get("text") or meta.get("text_preview","")
                        if preview:
                            st.caption(f"Preview: {preview[:300]}...")

        except RuntimeError as e:
            st.error(f"🚨 Search failed: {e}")

    # ── Section D: Manage Index ───────────────────────────────
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>⚙️ Manage Index</div>", unsafe_allow_html=True)

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.markdown(
            f"""
            <div class='vs-card'>
                <div class='vs-card-title'>🗄️ Index Details</div>
                <div class='vs-stat'>📦 <b>Total indexed:</b> {vs_info['count']}</div>
                <div class='vs-stat'>🔧 <b>Backend:</b> {vs_info['backend'].upper()}</div>
                <div class='vs-stat'>📐 <b>Stored vector dim:</b> {stored_dim}</div>
                <div class='vs-stat'>🤖 <b>Current model dim:</b> {current_model_dim or 'N/A'}</div>
                <div class='vs-stat'>💾 <b>Store path:</b> vector_db/</div>
                <div class='vs-stat'>🔒 <b>Persistent:</b> Yes (survives restarts)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_m2:
        st.markdown("**⚠️ Danger Zone**")
        st.caption("Clear index if you changed models or want to start fresh.")
        clear_confirm = st.checkbox("I confirm I want to clear all indexed vectors")
        if st.button("🗑️ Clear Entire Index", disabled=not clear_confirm, use_container_width=True):
            vector_store.clear()
            st.success("✅ Index cleared.")
            st.rerun()


# ─────────────────────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────────────────────

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div style='text-align:center; color:#334155; font-size:0.78rem; padding: 0.5rem 0 1rem 0;'>
        SmartHire AI &nbsp;·&nbsp;
        all-MiniLM-L6-v2 + PyTorch + Hugging Face + ChromaDB + Streamlit &nbsp;·&nbsp;
        Transformer-Based Semantic Resume Screening
    </div>
    """,
    unsafe_allow_html=True,
)
