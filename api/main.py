"""
api/main.py
-----------
SmartHire AI — FastAPI REST API

Exposes the full SmartHire AI pipeline as HTTP endpoints so any
frontend (React, Next.js, Vue, Node.js, etc.) can use it.

Streamlit UI is completely untouched — this runs as a separate server.

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

Run on Hugging Face Spaces (Docker):
    uvicorn api.main:app --host 0.0.0.0 --port 7860

Base URL (local):      http://localhost:8000
Base URL (HF Spaces):  https://vishu200672-smarthire-ai-api.hf.space
API Docs:              <base_url>/docs
Redoc:                 <base_url>/redoc

Author: SmartHire AI
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# ── Make src/ importable when running from project root ──────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model        import get_model
from src.parser       import parse_job_description, parse_resume
from src.preprocess   import preprocess_text
from src.ranking      import rank_candidates, summarize_rankings
from src.similarity   import batch_similarity
from src.skills       import full_skill_analysis
from src.vector_store import get_vector_store

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SmartHireAI-API")

# ── Lazy singletons ───────────────────────────────────────────
_model        = None
_vector_store = None


def get_loaded_model():
    global _model
    if _model is None:
        logger.info("Loading SmartHire AI model...")
        _model = get_model()
        logger.info("Model loaded.")
    return _model


def get_loaded_store():
    global _vector_store
    if _vector_store is None:
        _vector_store = get_vector_store(persist_dir=str(ROOT / "vector_db"))
    return _vector_store


# ── FastAPI App ───────────────────────────────────────────────

app = FastAPI(
    title       = "SmartHire AI API",
    description = (
        "Transformer-based resume & job description matching API.\n\n"
        "Upload resumes + a JD to get semantic similarity scores, "
        "skill gap analysis, candidate rankings, and vector index search.\n\n"
        "**GitHub:** https://github.com/Vishu200672/SmartHire-AI"
    ),
    version  = "1.0.0",
    docs_url = "/docs",
    redoc_url= "/redoc",
)

# ── CORS — open for all origins (public API) ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Fix Swagger UI to show file pickers for UploadFile fields ─
from fastapi.openapi.utils import get_openapi

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    # Force resume1-5 and jd_file to render as file upload in Swagger
    for path_data in schema.get("paths", {}).values():
        for method_data in path_data.values():
            body = method_data.get("requestBody", {})
            content = body.get("content", {})
            form = content.get("multipart/form-data", {})
            props = form.get("schema", {}).get("properties", {})
            for field in ["resume1","resume2","resume3","resume4","resume5","jd_file"]:
                if field in props:
                    props[field] = {"type": "string", "format": "binary"}
    app.openapi_schema = schema
    return app.openapi_schema

app.openapi = custom_openapi


# ═════════════════════════════════════════════════════════════
#  HEALTH & INFO
# ═════════════════════════════════════════════════════════════

@app.get("/", tags=["Health"])
def root():
    """API root — confirms the server is running."""
    return {
        "status"  : "ok",
        "service" : "SmartHire AI API",
        "version" : "1.0.0",
        "docs"    : "/docs",
        "github"  : "https://github.com/Vishu200672/SmartHire-AI",
    }


@app.get("/health", tags=["Health"])
@app.head("/health", tags=["Health"])
def health():
    """Health check — supports both GET and HEAD (for uptime monitors)."""
    return {"status": "healthy", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


@app.get("/model/info", tags=["Model"])
def model_info():
    """Returns metadata about the currently loaded embedding model."""
    try:
        model = get_loaded_model()
        return model.get_model_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info failed: {e}")


# ═════════════════════════════════════════════════════════════
#  CORE MATCHING — POST /match
# ═════════════════════════════════════════════════════════════

@app.post("/match", tags=["Matching"])
async def match_resumes(
    request           : Request,
    resume1           : Optional[UploadFile] = File(None, description="Resume file 1 (PDF, DOCX, TXT)"),
    resume2           : Optional[UploadFile] = File(None, description="Resume file 2"),
    resume3           : Optional[UploadFile] = File(None, description="Resume file 3"),
    resume4           : Optional[UploadFile] = File(None, description="Resume file 4"),
    resume5           : Optional[UploadFile] = File(None, description="Resume file 5"),
    jd_text           : str                  = Form("",   description="Job description as plain text"),
    jd_file           : Optional[UploadFile] = File(None, description="Job description file (use jd_text OR jd_file)"),
    similarity_weight : float                = Form(0.7,  description="Semantic similarity weight 0.5–0.9 (default 0.7)"),
):
    """
    **Main endpoint** — match resumes against a job description.

    Upload up to 5 resume files (PDF/DOCX/TXT) + a JD (text or file).
    Returns ranked candidates with scores, skills, and recommendations.

    **curl example (multiple resumes):**
    ```
    curl -X POST https://your-api-url/match \\
      -F "resume1=@resume1.pdf" \\
      -F "resume2=@resume2.docx" \\
      -F "jd_text=We are looking for a Python ML Engineer..."
    ```
    """
    # Collect whichever resume slots were filled
    resumes = [f for f in [resume1, resume2, resume3, resume4, resume5]
               if f is not None and f.filename]
    t_start = time.time()
    model   = get_loaded_model()

    similarity_weight = round(max(0.5, min(0.9, similarity_weight)), 2)
    skill_weight      = round(1.0 - similarity_weight, 2)

    # Parse JD
    raw_jd = ""
    if jd_file and jd_file.filename:
        try:
            raw_jd = parse_job_description(await jd_file.read(), filename=jd_file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JD file parse failed: {e}")
    elif jd_text and jd_text.strip():
        raw_jd = jd_text.strip()
    else:
        raise HTTPException(status_code=400, detail="Provide either jd_text or jd_file.")

    try:
        jd_clean = preprocess_text(raw_jd)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"JD preprocessing failed: {e}")

    # Parse resumes
    if not resumes:
        raise HTTPException(status_code=400, detail="No resume files provided.")

    parsed = []
    errors = []
    for rf in resumes:
        try:
            raw_text = parse_resume(await rf.read(), filename=rf.filename)
            clean    = preprocess_text(raw_text)
            parsed.append({"name": Path(rf.filename).stem, "clean_text": clean})
        except Exception as e:
            errors.append({"file": rf.filename, "error": str(e)})

    if not parsed:
        raise HTTPException(status_code=400, detail=f"No valid resumes parsed. Errors: {errors}")

    # Encode & score
    try:
        resume_embeddings = model.encode([r["clean_text"] for r in parsed])
        jd_embedding      = model.encode_single(jd_clean)
        scores            = batch_similarity(resume_embeddings, jd_embedding)
        for r, score in zip(parsed, scores):
            r["score"] = score
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Encoding failed: {e}")

    # Rank
    try:
        candidates = [{"name": r["name"], "text": r["clean_text"], "score": r["score"]} for r in parsed]
        results    = rank_candidates(candidates, jd_clean,
                                     similarity_weight=similarity_weight,
                                     skill_weight=skill_weight)
        summary    = summarize_rankings(results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ranking failed: {e}")

    return {
        "status"          : "success",
        "duration_sec"    : round(time.time() - t_start, 3),
        "total_candidates": len(results),
        "parse_errors"    : errors,
        "summary"         : summary,
        "candidates"      : [
            {
                "rank"               : r.rank,
                "name"               : r.name,
                "score_pct"          : r.score_pct,
                "semantic_similarity": round(r.similarity_score * 100, 2),
                "skill_coverage_pct" : r.skill_coverage_pct,
                "recommendation"     : r.recommendation,
                "recommendation_color": r.recommendation_color,
                "confidence"         : r.confidence,
                "percentile_rank"    : r.percentile_rank,
                "matching_skills"    : r.matching_skills,
                "missing_skills"     : r.missing_skills,
                "critical_missing"   : r.critical_missing,
                "important_missing"  : r.important_missing,
                "resume_only_skills" : r.resume_only_skills,
                "skill_coverage_pct" : r.skill_coverage_pct,
                "weighted_coverage_pct": r.weighted_coverage_pct,
                "ai_insight"         : r.ai_insight,
            }
            for r in results
        ],
    }


# ═════════════════════════════════════════════════════════════
#  SKILLS — POST /skills
# ═════════════════════════════════════════════════════════════

@app.post("/skills", tags=["Skills"])
async def extract_skills(
    resume  : UploadFile = File(..., description="Resume file (PDF, DOCX, TXT)"),
    jd_text : str        = Form("",  description="Job description text"),
):
    """
    Extract and compare skills from a resume against a JD.
    Returns matching, missing, critical skills and coverage %.
    """
    try:
        raw_text   = parse_resume(await resume.read(), filename=resume.filename)
        clean      = preprocess_text(raw_text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Resume parse failed: {e}")

    if not jd_text.strip():
        raise HTTPException(status_code=400, detail="jd_text is required.")

    try:
        jd_clean   = preprocess_text(jd_text)
        skill_data = full_skill_analysis(clean, jd_clean)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Skill analysis failed: {e}")

    return {
        "status"               : "success",
        "candidate"            : Path(resume.filename).stem,
        "matching_skills"      : skill_data["matching"],
        "missing_skills"       : skill_data["missing"],
        "critical_missing"     : skill_data["critical_missing"],
        "important_missing"    : skill_data.get("important_missing", []),
        "resume_only_skills"   : skill_data["resume_only"],
        "skill_coverage_pct"   : skill_data["skill_coverage_pct"],
        "weighted_coverage_pct": skill_data["weighted_coverage_pct"],
        "jd_skills"            : skill_data["jd_skills"],
        "resume_skills"        : skill_data["resume_skills"],
        "skills_by_category"   : skill_data.get("skills_by_category", {}),
    }


# ═════════════════════════════════════════════════════════════
#  VECTOR INDEX — /index/*
# ═════════════════════════════════════════════════════════════

@app.get("/index/info", tags=["Vector Index"])
def index_info():
    """Returns current vector index metadata."""
    try:
        return get_loaded_store().get_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/index/candidates", tags=["Vector Index"])
def index_candidates():
    """Returns list of all indexed candidates with metadata."""
    try:
        store = get_loaded_store()
        meta  = store.get_all_metadata() if hasattr(store, "get_all_metadata") else \
                [{"name": n} for n in store.get_all_names()]
        return {"status": "success", "count": len(meta), "candidates": meta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/build", tags=["Vector Index"])
async def index_build(
    resumes : List[UploadFile] = File(...,  description="Resume files to encode and store"),
    rebuild : bool             = Form(False, description="Clear existing index first"),
):
    """
    Encode and store resume embeddings in the persistent vector index.
    Once indexed, use POST /index/search for instant results.
    """
    model = get_loaded_model()
    store = get_loaded_store()

    if rebuild:
        store.clear()

    to_index, errors = [], []
    for rf in resumes:
        try:
            raw   = parse_resume(await rf.read(), filename=rf.filename)
            clean = preprocess_text(raw)
            to_index.append({"name": Path(rf.filename).stem, "text": clean})
        except Exception as e:
            errors.append({"file": rf.filename, "error": str(e)})

    if not to_index:
        raise HTTPException(status_code=400, detail=f"No valid resumes to index. Errors: {errors}")

    try:
        t0    = time.time()
        stats = store.build_index(resumes=to_index, model=model)
        dur   = round(time.time() - t0, 3)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Index build failed: {e}")

    return {
        "status"      : "success",
        "duration_sec": dur,
        "indexed"     : stats["indexed"],
        "skipped"     : stats["skipped"],
        "total"       : stats["total"],
        "backend"     : stats["backend"],
        "parse_errors": errors,
    }


@app.post("/index/search", tags=["Vector Index"])
async def index_search(
    jd_text : str        = Form("",   description="Job description text"),
    jd_file : UploadFile = File(None, description="Job description file (use jd_text OR jd_file)"),
    top_k   : int        = Form(5,    description="Number of top results (max 20)"),
):
    """
    Instantly search the vector index for the best matching resumes.
    Results in milliseconds — no re-encoding needed.
    """
    model = get_loaded_model()
    store = get_loaded_store()

    if store.is_empty():
        raise HTTPException(status_code=400,
                            detail="Index is empty. POST resumes to /index/build first.")

    raw_jd = ""
    if jd_file and jd_file.filename:
        try:
            raw_jd = parse_job_description(await jd_file.read(), filename=jd_file.filename)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"JD parse failed: {e}")
    elif jd_text and jd_text.strip():
        raw_jd = jd_text.strip()
    else:
        raise HTTPException(status_code=400, detail="Provide either jd_text or jd_file.")

    top_k = max(1, min(20, top_k))

    try:
        t0          = time.time()
        jd_emb      = model.encode_single(preprocess_text(raw_jd))
        results     = store.search(jd_emb, top_k=top_k)
        duration_ms = round((time.time() - t0) * 1000, 1)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    return {
        "status"     : "success",
        "duration_ms": duration_ms,
        "total_found": len(results),
        "results"    : [
            {
                "rank"         : i + 1,
                "name"         : r["name"],
                "similarity_pct": round(r["score"] * 100, 2),
                "indexed_at"   : r.get("metadata", {}).get("indexed_at", "N/A"),
                "text_length"  : r.get("metadata", {}).get("text_length", 0),
                "embedding_dim": r.get("metadata", {}).get("embedding_dim", "N/A"),
                "preview"      : (r.get("text") or r.get("metadata", {}).get("text_preview", ""))[:300],
            }
            for i, r in enumerate(results)
        ],
    }


@app.delete("/index/clear", tags=["Vector Index"])
def index_clear():
    """Clear all stored vectors from the index."""
    try:
        get_loaded_store().clear()
        return {"status": "success", "message": "Vector index cleared."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index/add", tags=["Vector Index"])
async def index_add(
    resume : UploadFile = File(..., description="Single resume to add to existing index"),
):
    """Add a single resume to the existing index without rebuilding."""
    model = get_loaded_model()
    store = get_loaded_store()
    try:
        raw     = parse_resume(await resume.read(), filename=resume.filename)
        clean   = preprocess_text(raw)
        name    = Path(resume.filename).stem
        success = store.add_resume(name=name, text=clean, model=model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to add resume: {e}")

    if not success:
        raise HTTPException(status_code=500, detail="Failed to store resume in index.")

    return {"status": "success", "name": name, "total": store.count()}


# ═════════════════════════════════════════════════════════════
#  UTILITIES
# ═════════════════════════════════════════════════════════════

@app.post("/parse", tags=["Utilities"])
async def parse_file(
    file : UploadFile = File(..., description="Resume or JD file to parse (PDF, DOCX, TXT)"),
):
    """Parse a file and return raw + cleaned text preview."""
    try:
        data     = await file.read()
        raw_text = parse_resume(data, filename=file.filename)
        clean    = preprocess_text(raw_text)
        return {
            "status"       : "success",
            "filename"     : file.filename,
            "raw_length"   : len(raw_text),
            "clean_length" : len(clean),
            "raw_preview"  : raw_text[:500],
            "clean_preview": clean[:500],
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Parse failed: {e}")


@app.post("/embed", tags=["Utilities"])
async def embed_text(
    text : str = Form(..., description="Text to embed into a vector"),
):
    """Encode any text and return its embedding vector."""
    try:
        model     = get_loaded_model()
        embedding = model.encode_single(preprocess_text(text))
        vec       = embedding.cpu().numpy().tolist()
        return {"status": "success", "dim": len(vec), "embedding": vec}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embed failed: {e}")
