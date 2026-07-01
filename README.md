<<<<<<< HEAD
# 🤖 SmartHire AI: Transformer-Based Resume & Job Matching System

[![Python](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=flat-square&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c?style=flat-square&logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface)](https://huggingface.co)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B?style=flat-square&logo=streamlit)](https://streamlit.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

> **ATS-inspired AI recruitment system** that matches candidate resumes with job descriptions using fine-tuned Sentence Transformer embeddings and cosine semantic similarity — going far beyond simple keyword matching.

---

## 📌 Project Overview

SmartHire AI is a **production-style HRTech application** that demonstrates:

- **Transformer-based NLP** using `all-MiniLM-L6-v2` (fine-tuned on 127 resume–JD pairs across 41 job roles)
- **Semantic understanding** via mean-pooled sentence embeddings
- **Cosine similarity scoring** — context-aware, not keyword-matching
- **Candidate ranking** from multiple simultaneous resume uploads
- **Skill gap analysis** with critical missing skill detection (300+ skill vocabulary)
- **Persistent vector index** (ChromaDB/NumPy) for instant sub-100ms resume search
- **REST API** (FastAPI) for frontend integration
- **Interactive recruiter dashboard** built with Streamlit + Plotly

---

## 📸 Screenshots

### 🖥️ Upload & Analyze
![Upload & Analyze](assets/Screenshot%202026-06-26%20160129.png)

### 📊 Match Results — Pipeline Summary & Score Distribution
![Match Results](assets/Screenshot%202026-06-26%20160148.png)

### 📊 Match Results — Charts & Per-Candidate Cards
| Pie Chart + Scatter Plot | All Candidate Score Cards |
|--------------------------|--------------------------| 
| ![](assets/Screenshot%202026-06-26%20160204.png) | ![](assets/Screenshot%202026-06-26%20160227.png) |

### 🔍 Skill Gap Analysis
| Skill Chips (Matching / Missing / Critical) | Skill Matrix & Cross-Candidate Comparison |
|--------------------------------------------|------------------------------------------|
| ![](assets/Screenshot%202026-06-26%20160310.png) | ![](assets/Screenshot%202026-06-26%20160324.png) |

### 🏆 Candidate Ranking
| Leaderboard + Top Candidate Gauge | Score Breakdown + CSV Export |
|----------------------------------|------------------------------|
| ![](assets/Screenshot%202026-06-26%20160347.png) | ![](assets/Screenshot%202026-06-26%20160404.png) |

### 📁 CSV Export Result
![CSV Export](assets/Screenshot%202026-06-26%20160431.png)

---

## 📊 Fine-Tuning Results

| Metric | Value |
|--------|-------|
| Pearson r | **0.9733** |
| Spearman ρ | **0.9604** |
| Strong Match Accuracy | **98%** |
| Partial Match Accuracy | **47%** |
| Mismatch Accuracy | **100%** |
| Overall 3-tier Accuracy | **81.25%** |
| Fine-tuning Gain | **+9.4%** |

> Fine-tuned on 127 pairs across 41 job roles: 43 Strong (34%), 40 Partial (31%), 44 Mismatch (35%)

---

## 🏗️ Project Structure

```
SmartHireAI/
│
├── app/
│   └── streamlit_app.py       # Full Streamlit dashboard (dark mode, port 8501)
│
├── api/
│   ├── __init__.py
│   ├── main.py                # FastAPI REST API server (port 8000)
│   └── README.md              # Full API endpoint documentation
│
├── src/
│   ├── __init__.py
│   ├── parser.py              # PDF, DOCX, TXT resume parser
│   ├── preprocess.py          # Text cleaning & normalization pipeline
│   ├── model.py               # Sentence Transformer embedding model
│   ├── similarity.py          # Cosine similarity & calibrated scoring
│   ├── skills.py              # Skill extraction & gap analysis (300+ skills)
│   ├── ranking.py             # Candidate ranking & export
│   └── vector_store.py        # ChromaDB/NumPy persistent vector index
│
├── train/
│   └── training_data.json     # 127 labeled resume–JD pairs (41 job roles)
│
├── datasets/
│   ├── sample_jd.txt
│   ├── candidate_alice.txt
│   ├── candidate_bob.txt
│   ├── candidate_carol.txt
│   └── candidate_david.txt
│
├── finetune.py                # Fine-tuning script (CosineSimilarityLoss, 6 epochs)
├── evaluate.py                # Evaluation (Pearson r, Spearman ρ, 3-tier accuracy)
├── diagnose.py                # Calibration diagnostics
├── requirements.txt
├── RUN_APP.bat                # Windows: launch Streamlit UI
├── RUN_API.bat                # Windows: launch FastAPI server
├── SETUP_AND_RUN.bat          # Windows: first-time setup
└── main.py                    # CLI entry point
```

---

## ⚡ Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Vishu200672/SmartHire-AI.git
cd SmartHire-AI
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# .\venv\Scripts\activate       # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ First run downloads the embedding model (~90 MB). Subsequent runs use the HuggingFace cache.

### 4. Run the Streamlit Dashboard

```bash
streamlit run app/streamlit_app.py
# → http://localhost:8501
```

### 5. Run the REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# → http://localhost:8000
# → http://localhost:8000/docs  (interactive Swagger UI)
```

> Both servers can run simultaneously — they share the same `src/` model.

### 6. Run the CLI Demo

```bash
python main.py --demo
# Or with your own files:
python main.py --resume resume.pdf --jd job_description.txt
```

---

## 🌐 REST API

SmartHire AI includes a full **FastAPI REST API** for integrating the matching engine into any frontend (React, Next.js, Vue, Node.js, etc.).

### Base URL
```
http://localhost:8000
```

### Interactive Docs
```
http://localhost:8000/docs      ← Swagger UI (try all endpoints in browser)
http://localhost:8000/redoc     ← Redoc
```

### Key Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/model/info` | Loaded model metadata |
| **POST** | **`/match`** | **Match resumes vs JD — main endpoint** |
| POST | `/skills` | Skills-only analysis |
| POST | `/index/build` | Build persistent vector index |
| POST | `/index/search` | Instant search against index (<100ms) |
| GET | `/index/info` | Index stats |
| GET | `/index/candidates` | List indexed resumes |
| POST | `/index/add` | Add single resume to index |
| DELETE | `/index/clear` | Clear index |
| POST | `/parse` | Parse file → raw text |
| POST | `/embed` | Get embedding vector for any text |

### Example — Match Resumes (JavaScript)

```javascript
const form = new FormData();
form.append("resumes", resumeFile1);
form.append("resumes", resumeFile2);
form.append("jd_text", "Looking for Python ML Engineer with PyTorch...");
form.append("similarity_weight", "0.7");

const res  = await fetch("http://localhost:8000/match", {
  method: "POST",
  body: form,
});
const data = await res.json();
// data.candidates → ranked list with scores, skills, recommendations
```

### Example Response

```json
{
  "status": "success",
  "duration_sec": 1.23,
  "total_candidates": 2,
  "summary": {
    "average_score": 72.5,
    "highest_score": 85.0,
    "highly_recommended": 1,
    "recommended": 1
  },
  "candidates": [
    {
      "rank": 1,
      "name": "John_Doe",
      "score_pct": 85.0,
      "semantic_similarity": 91.2,
      "skill_coverage_pct": 75.0,
      "recommendation": "Highly Recommended",
      "matching_skills": ["python", "pytorch", "docker"],
      "missing_skills": ["kubernetes"],
      "critical_missing": [],
      "ai_insight": "Strong contextual alignment with the JD..."
    }
  ]
}
```

See [`api/README.md`](api/README.md) for full endpoint documentation.

---

## 🖥️ Streamlit Dashboard Features

| Tab | Features |
|-----|----------|
| **Upload & Analyze** | Upload PDF/DOCX/TXT resumes, paste or upload JD, run pipeline |
| **Match Results** | Score distribution bar chart, scatter plot, per-candidate cards |
| **Skill Gap Analysis** | Matching/missing/critical skill chips, skill matrix chart |
| **Candidate Ranking** | Leaderboard table, gauge chart for top candidate, CSV export |
| **Vector Index** | Build/search persistent resume index, instant JD search |

---

## 🗄️ Vector Index

SmartHire AI includes a **persistent vector index** that pre-encodes resumes so JD search is instant:

```
Normal flow:   Upload resumes → encode each (~0.06s each) → compare → results
Vector index:  Index resumes once → search any JD → results in <100ms
```

**Backends supported:**
- **ChromaDB** (recommended) — `pip install chromadb`
- **NumPy flat-file** (automatic fallback) — no extra install needed

**Usage via API:**
```bash
# Index resumes once
curl -X POST http://localhost:8000/index/build \
  -F "resumes=@resume1.pdf" -F "resumes=@resume2.docx"

# Search instantly for any JD
curl -X POST http://localhost:8000/index/search \
  -F "jd_text=Python ML Engineer with PyTorch experience" \
  -F "top_k=5"
```

---

## 🧠 How It Works

### Architecture Pipeline

```
Resume (PDF/DOCX/TXT)
        │
        ▼
[parser.py] Extract raw text
        │
        ▼
[preprocess.py] Normalize → clean → chunk (400 tokens, 50 overlap)
        │
        ▼
[model.py] Tokenize → forward pass → mean pooling → L2 normalize → embedding
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
[similarity.py]                    [skills.py]
Cosine similarity vs JD            Skill extraction (300+ vocab)
Calibrated score 0–100%           Gap analysis (matching/missing/critical)
        │                                  │
        └──────────────┬───────────────────┘
                       ▼
              [ranking.py]
              Composite score = 70% semantic + 30% skill
              Sort → Recommendation tier → AI insight
```

### Composite Ranking Score

```
Final Score = (Semantic Similarity × 0.70) + (Skill Coverage × 0.30)
```

Weights are configurable via API parameter or Streamlit sidebar slider.

---

## 🎯 Recommendation Tiers

| Score | Recommendation | Action |
|-------|---------------|--------|
| **≥ 60%** | 🟢 Highly Recommended | Fast-track to interview |
| **38–60%** | 🔵 Recommended | Schedule screening call |
| **18–38%** | 🟠 Consider | Review manually |
| **< 18%** | 🔴 Not Recommended | Archive |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Core Model | Fine-tuned DistilBERT / `all-MiniLM-L6-v2` |
| DL Framework | PyTorch 2.0+ |
| NLP Library | Hugging Face Transformers + Sentence-Transformers |
| REST API | FastAPI + Uvicorn |
| Vector Store | ChromaDB / NumPy |
| Web App | Streamlit |
| Charts | Plotly |
| PDF Parsing | pdfplumber + PyPDF2 |
| DOCX Parsing | python-docx |
| Data | Pandas, NumPy |

---

## 🚀 Performance Benchmarks

| Operation | Time (CPU) |
|-----------|-----------|
| Model load (first time) | ~5–10s |
| Encode 1 resume | ~0.06s |
| Encode 60 resumes | ~4–5s |
| Vector index search | **<100ms** |
| Skill gap analysis | <0.01s per candidate |

---

## 📝 Module Documentation

Each module is fully documented with:
- Google-style docstrings
- Python type hints throughout
- `logging` at every pipeline step
- Meaningful error messages

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit changes (`git commit -m "Add: your feature"`)
4. Push to branch (`git push origin feature/your-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [Hugging Face](https://huggingface.co) for Transformers and `all-MiniLM-L6-v2`
- [Sentence Transformers](https://www.sbert.net) for the fine-tuning framework
- [FastAPI](https://fastapi.tiangolo.com) for the API framework
- [Streamlit](https://streamlit.io) for the dashboard framework
- [Plotly](https://plotly.com) for interactive charts
- [ChromaDB](https://www.trychroma.com) for the vector store

---

## 📬 Contact

Built as a portfolio project demonstrating Transformer-based NLP, semantic search, fine-tuning, REST API design, and production ML engineering practices.

**GitHub**: [github.com/Vishu200672/SmartHire-AI](https://github.com/Vishu200672/SmartHire-AI)
=======
---
title: SmartHire AI
emoji: 🐨
colorFrom: yellow
colorTo: yellow
sdk: docker
pinned: false
license: mit
short_description: Smarthire-AI Model
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> baf962354e1d9489fd69e0a72ef89a968b89b38b
