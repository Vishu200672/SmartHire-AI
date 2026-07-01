# ─────────────────────────────────────────────────────────────
# SmartHire AI — Dockerfile
# Hugging Face Spaces (Docker SDK) compatible
# Port: 7860 (required by HF Spaces)
# ─────────────────────────────────────────────────────────────

FROM python:3.10-slim

# HF Spaces requires user 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

# Install dependencies (API-only, no Streamlit/Plotly)
COPY --chown=user requirements_api.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements_api.txt

# Copy project source
COPY --chown=user src/ ./src/
COPY --chown=user api/ ./api/

# Pre-download embedding model so first request is instant
# Uses all-MiniLM-L6-v2 (smaller, faster — ideal for cloud)
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('Model downloaded successfully')" || echo "Model pre-download skipped"

# HF Spaces requires port 7860
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
