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
RUN pip install --no-cache-dir -r requirements_api.txt

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
