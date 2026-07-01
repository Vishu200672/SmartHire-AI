# SmartHire AI — REST API Reference

Base URL: `http://localhost:8000`  
Interactive Docs: `http://localhost:8000/docs`  
Redoc: `http://localhost:8000/redoc`

---

## Start the API

```bash
# Install new dependencies first (one time)
pip install fastapi uvicorn[standard] python-multipart

# Start the API server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Or double-click RUN_API.bat on Windows
```

The Streamlit UI still runs separately:
```bash
streamlit run app/streamlit_app.py   # port 8501
uvicorn api.main:app --port 8000     # port 8000
```

---

## Endpoints

### Health

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Root — confirms server is running |
| GET | `/health` | Health check with timestamp |
| GET | `/model/info` | Loaded model metadata |

---

### Core Matching

#### `POST /match`
Match resumes against a job description. Returns ranked candidates.

**Form fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `resumes` | File(s) | ✅ | PDF, DOCX, or TXT resume files |
| `jd_text` | string | one of | JD as plain text |
| `jd_file` | File | one of | JD as file |
| `similarity_weight` | float | ❌ | 0.5–0.9, default 0.7 |

**Example (JavaScript fetch):**
```javascript
const form = new FormData();
form.append("resumes", resumeFile1);
form.append("resumes", resumeFile2);
form.append("jd_text", "We are looking for a Python ML Engineer...");
form.append("similarity_weight", "0.7");

const res = await fetch("http://localhost:8000/match", {
  method: "POST",
  body: form,
});
const data = await res.json();
```

**Response:**
```json
{
  "status": "success",
  "duration_sec": 1.23,
  "total_candidates": 2,
  "summary": {
    "total_candidates": 2,
    "average_score": 72.5,
    "highest_score": 85.0,
    "highly_recommended": 1,
    "recommended": 1,
    "consider": 0,
    "not_recommended": 0
  },
  "candidates": [
    {
      "rank": 1,
      "name": "John_Doe",
      "score_pct": 85.0,
      "semantic_similarity": 91.2,
      "skill_coverage_pct": 75.0,
      "recommendation": "Highly Recommended",
      "confidence": "High",
      "percentile_rank": 100.0,
      "matching_skills": ["python", "pytorch", "docker"],
      "missing_skills": ["kubernetes"],
      "critical_missing": [],
      "important_missing": ["kubernetes"],
      "resume_only_skills": ["flask", "pandas"],
      "ai_insight": "Strong contextual alignment..."
    }
  ],
  "parse_errors": []
}
```

---

### Skills

#### `POST /skills`
Extract and compare skills from a single resume vs JD.

**Form fields:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `resume` | File | ✅ | Resume file |
| `jd_text` | string | ✅ | JD text |

**Response:**
```json
{
  "status": "success",
  "candidate": "John_Doe",
  "matching_skills": ["python", "pytorch"],
  "missing_skills": ["kubernetes"],
  "critical_missing": [],
  "skill_coverage_pct": 75.0,
  "weighted_coverage_pct": 80.0,
  "jd_skills": ["python", "pytorch", "kubernetes"],
  "resume_skills": ["python", "pytorch", "flask"]
}
```

---

### Vector Index

#### `POST /index/build`
Encode and store resumes in the persistent vector index.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `resumes` | File(s) | ✅ | Resume files to index |
| `rebuild` | bool | ❌ | Clear index first (default false) |

#### `POST /index/search`
Instantly search the index for the best matching resumes.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `jd_text` | string | one of | JD text |
| `jd_file` | File | one of | JD file |
| `top_k` | int | ❌ | Number of results (default 5, max 20) |

**Response:**
```json
{
  "status": "success",
  "duration_ms": 12.4,
  "total_found": 2,
  "results": [
    {
      "rank": 1,
      "name": "John_Doe",
      "similarity_pct": 95.8,
      "indexed_at": "2026-07-01T20:29:18",
      "text_length": 1763,
      "embedding_dim": 768,
      "preview": "john doe machine learning engineer..."
    }
  ]
}
```

#### `GET /index/info`
Get index stats (count, backend, dim, etc.)

#### `GET /index/candidates`
List all indexed candidates with metadata.

#### `POST /index/add`
Add a single resume to the existing index without rebuilding.

#### `DELETE /index/clear`
Wipe the entire index.

---

### Utilities

#### `POST /parse`
Parse a file and return raw + cleaned text. Good for debugging.

#### `POST /embed`
Encode any text and return its raw embedding vector.

---

## Frontend Integration (React/Next.js example)

```javascript
// api/smarthire.js

const BASE_URL = "http://localhost:8000";

// Match resumes against a JD
export async function matchResumes(resumeFiles, jdText, similarityWeight = 0.7) {
  const form = new FormData();
  resumeFiles.forEach(f => form.append("resumes", f));
  form.append("jd_text", jdText);
  form.append("similarity_weight", similarityWeight);

  const res = await fetch(`${BASE_URL}/match`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Build vector index
export async function buildIndex(resumeFiles, rebuild = false) {
  const form = new FormData();
  resumeFiles.forEach(f => form.append("resumes", f));
  form.append("rebuild", rebuild);

  const res = await fetch(`${BASE_URL}/index/build`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Search the vector index
export async function searchIndex(jdText, topK = 5) {
  const form = new FormData();
  form.append("jd_text", jdText);
  form.append("top_k", topK);

  const res = await fetch(`${BASE_URL}/index/search`, { method: "POST", body: form });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

// Get model info
export async function getModelInfo() {
  const res = await fetch(`${BASE_URL}/model/info`);
  return res.json();
}
```

---

## CORS

By default the API allows all origins (`*`).  
For production, update `allow_origins` in `api/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    ...
)
```
