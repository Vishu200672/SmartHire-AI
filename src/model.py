"""
model.py
--------
Production-grade embedding model with fine-tuning auto-detection.

Priority order:
  1. models/smarthire-finetuned/  (your fine-tuned model -- best accuracy)
  2. sentence-transformers/all-MiniLM-L6-v2  (pretrained -- strong baseline)
  3. distilbert-base-uncased  (legacy fallback if sentence-transformers missing)

Run train/finetune.py once to create the fine-tuned model.
The app auto-detects and loads it on next restart.

Author: SmartHire AI
"""

import logging
import math
from pathlib import Path
from typing import List, Optional, Union

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Fine-tuned model path (auto-used if it exists after running train/finetune.py)
FINETUNED_MODEL = "models/smarthire-finetuned"
# Primary pretrained model -- fast, accurate, 384-dim
PRIMARY_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"
# Fallback model -- larger, slightly more accurate
FALLBACK_MODEL  = "sentence-transformers/all-mpnet-base-v2"
# Legacy fallback if sentence-transformers not installed
LEGACY_MODEL    = "distilbert-base-uncased"

# Chunking config for long documents
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 50    # overlap between chunks


class EmbeddingModel:
    """
    Production embedding model with smart chunking and multi-backend support.

    Tries sentence-transformers first for best accuracy.
    Falls back to raw HuggingFace DistilBERT if not available.
    """

    def __init__(
        self,
        model_name: str = PRIMARY_MODEL,
        device: Optional[str] = None,
        use_chunking: bool = True,
    ) -> None:
        self.model_name   = model_name
        self.use_chunking = use_chunking
        self.device       = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self) -> None:
        """Try sentence-transformers, fall back to raw transformers."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading sentence-transformers model: {self.model_name}")
            self._st_model    = SentenceTransformer(self.model_name, device=self.device)
            self.backend      = "sentence-transformers"
            self._hidden_size = self._st_model.get_sentence_embedding_dimension()
            self._max_tokens  = 512
            logger.info(f"sentence-transformers loaded. dim={self._hidden_size}")
        except ImportError:
            logger.warning("sentence-transformers not found -- falling back to DistilBERT")
            self._load_distilbert()
        except Exception as e:
            logger.warning(f"sentence-transformers load failed ({e}) -- falling back to DistilBERT")
            self._load_distilbert()

    def _load_distilbert(self) -> None:
        from transformers import AutoModel, AutoTokenizer
        logger.info(f"Loading HuggingFace model: {LEGACY_MODEL}")
        self._tokenizer   = AutoTokenizer.from_pretrained(LEGACY_MODEL)
        self._hf_model    = AutoModel.from_pretrained(LEGACY_MODEL)
        self._hf_model.to(self.device)
        self._hf_model.eval()
        self.model_name   = LEGACY_MODEL
        self.backend      = "transformers"
        self._hidden_size = self._hf_model.config.hidden_size
        self._max_tokens  = self._hf_model.config.max_position_embeddings
        logger.info(f"DistilBERT loaded. dim={self._hidden_size}")

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
    ) -> torch.Tensor:
        """
        Encode texts into L2-normalized embedding vectors.
        Automatically applies smart chunking for long documents.

        Returns: Tensor [N, hidden_size], L2-normalized.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            raise ValueError("Cannot encode empty text list.")

        if self.use_chunking:
            embeddings = []
            for text in texts:
                chunks = self._chunk_text(text)
                if len(chunks) == 1:
                    emb = self._encode_batch(chunks, batch_size)
                else:
                    chunk_embs = self._encode_batch(chunks, batch_size)
                    emb = chunk_embs.mean(dim=0, keepdim=True)
                    emb = F.normalize(emb, p=2, dim=1)
                embeddings.append(emb)
            return torch.cat(embeddings, dim=0)
        else:
            return self._encode_batch(texts, batch_size)

    def _encode_batch(self, texts: List[str], batch_size: int) -> torch.Tensor:
        """Encode a flat list of texts -- no chunking."""
        if self.backend == "sentence-transformers":
            vecs = self._st_model.encode(
                texts,
                batch_size=batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return vecs.cpu()
        else:
            return self._hf_encode(texts, batch_size)

    def _hf_encode(self, texts: List[str], batch_size: int) -> torch.Tensor:
        """DistilBERT encoding with attention-weighted mean pooling."""
        all_embeddings = []
        num_batches    = math.ceil(len(texts) / batch_size)

        with torch.no_grad():
            for i in range(num_batches):
                batch          = texts[i * batch_size:(i + 1) * batch_size]
                encoded        = self._tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
                input_ids      = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                output         = self._hf_model(input_ids=input_ids, attention_mask=attention_mask)

                token_embs    = output.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(token_embs.size()).float()
                pooled        = torch.sum(token_embs * mask_expanded, dim=1)
                pooled        = pooled / torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
                normalized    = F.normalize(pooled, p=2, dim=1)
                all_embeddings.append(normalized.cpu())

        return torch.cat(all_embeddings, dim=0)

    def _chunk_text(self, text: str) -> List[str]:
        """Split long text into overlapping chunks."""
        words = text.split()
        if len(words) <= CHUNK_SIZE:
            return [text]

        chunks = []
        start  = 0
        while start < len(words):
            end   = min(start + CHUNK_SIZE, len(words))
            chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += CHUNK_SIZE - CHUNK_OVERLAP

        logger.debug(f"Document split into {len(chunks)} chunks")
        return chunks

    def encode_single(self, text: str) -> torch.Tensor:
        """Encode a single text -- returns 1D tensor [hidden_size]."""
        return self.encode([text])[0]

    def get_model_info(self) -> dict:
        """Return live metadata about the loaded model."""
        finetuned = Path(FINETUNED_MODEL).exists() and any(Path(FINETUNED_MODEL).iterdir())
        return {
            "model_name"    : self.model_name,
            "backend"       : self.backend,
            "embedding_dim" : self._hidden_size,
            "max_tokens"    : self._max_tokens,
            "device"        : self.device,
            "chunking"      : self.use_chunking,
            "chunk_size"    : CHUNK_SIZE,
            "chunk_overlap" : CHUNK_OVERLAP,
            "is_finetuned"  : finetuned,
            "pooling"       : "sentence-transformers mean" if self.backend == "sentence-transformers" else "attention-weighted mean",
            "similarity"    : "Cosine",
            "framework"     : "HuggingFace Transformers + PyTorch",
        }


# -- Singleton ---------------------------------------------------------

_model_instance: Optional[EmbeddingModel] = None


def get_model(model_name: str = None) -> EmbeddingModel:
    """
    Return module-level singleton EmbeddingModel (lazy init).

    Auto-detects fine-tuned model:
      - If models/smarthire-finetuned/ exists and is non-empty -> fine-tuned
      - Otherwise -> pretrained all-MiniLM-L6-v2
    """
    global _model_instance
    if _model_instance is None:
        if model_name is None:
            finetuned_path = Path(FINETUNED_MODEL)
            if finetuned_path.exists() and any(finetuned_path.iterdir()):
                model_name = FINETUNED_MODEL
                logger.info(f"Fine-tuned model detected -- loading: {FINETUNED_MODEL}")
            else:
                model_name = PRIMARY_MODEL
                logger.info(f"No fine-tuned model found -- loading pretrained: {PRIMARY_MODEL}")
        _model_instance = EmbeddingModel(model_name=model_name)
    return _model_instance
