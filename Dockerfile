# ─────────────────────────────────────────────────────────────
# SmartHire AI — Dockerfile
# Hugging Face Spaces (Docker SDK) compatible
# Port: 7860 (required by HF Spaces)
# Optimized: CPU-only torch, pinned deps, model pre-baked
# ─────────────────────────────────────────────────────────────

FROM python:3.10-slim

# HF Spaces requires user 1000
RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/user/.cache/huggingface \
    TRANSFORMERS_CACHE=/home/user/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/home/user/.cache/sentence_transformers

WORKDIR $HOME/app

# Step 1: Install CPU-only torch first (biggest package, cached as own layer)
RUN pip install --no-cache-dir torch==2.1.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Step 2: Install remaining dependencies
COPY --chown=user requirements_api.txt .
RUN pip install --no-cache-dir \
    transformers==4.35.2 \
    sentence-transformers==2.2.2 \
    scikit-learn==1.3.2 \
    pandas==2.1.3 \
    numpy==1.26.2 \
    scipy==1.11.4 \
    fastapi==0.104.1 \
    "uvicorn[standard]==0.24.0" \
    python-multipart==0.0.6 \
    pdfplumber==0.10.3 \
    PyPDF2==3.0.1 \
    python-docx==1.1.0 \
    tqdm==4.66.1 \
    python-dateutil==2.8.2

# Step 3: Copy project source
COPY --chown=user src/ ./src/
COPY --chown=user api/ ./api/

# Step 4: Pre-download and cache embedding model into image
# This means cold starts never need to download the model
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'); \
print('✓ Model cached successfully')" || echo "Model pre-download skipped"

# HF Spaces requires port 7860
EXPOSE 7860

# Start FastAPI
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]
