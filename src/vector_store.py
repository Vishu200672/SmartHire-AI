"""
vector_store.py  v3
-------------------
SmartHire AI — ChromaDB / NumPy Vector Store

Pre-encode and persistently store resume embeddings so that:
  - Each resume is encoded ONCE and saved to disk
  - JD matching becomes instant (sub-100ms cosine search)
  - New resumes can be added without re-encoding the whole pool

Author: SmartHire AI
"""

import hashlib
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = "vector_db"

# Version string — bump this to bust Streamlit's @st.cache_resource
VECTOR_STORE_VERSION = "v3"


class VectorStore:
    """
    Local persistent vector store for resume embeddings.
    Uses ChromaDB when available, falls back to NumPy flat-file store.
    """

    def __init__(self, persist_dir: str = DEFAULT_STORE_DIR, collection_name: str = "resumes") -> None:
        self.persist_dir      = Path(persist_dir)
        self.collection_name  = collection_name
        self.backend          = None
        self._chroma_client   = None
        self._collection      = None
        self._np_vectors: Optional[np.ndarray] = None
        self._np_meta: List[Dict] = []
        self._init_backend()

    # ─────────────────────────────────────────────────────────
    #  Initialization
    # ─────────────────────────────────────────────────────────

    def _init_backend(self) -> None:
        try:
            import chromadb
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            self._chroma_client = chromadb.PersistentClient(path=str(self.persist_dir))
            self._collection    = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self.backend = "chromadb"
            logger.info(f"VectorStore: ChromaDB | docs={self._collection.count()}")
        except ImportError:
            logger.warning("chromadb not installed — using NumPy fallback.")
            self._init_numpy_fallback()
        except Exception as e:
            logger.warning(f"ChromaDB init failed ({e}) — using NumPy fallback.")
            self._init_numpy_fallback()

    def _init_numpy_fallback(self) -> None:
        self.backend          = "numpy"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._np_vectors_path = self.persist_dir / "vectors.npy"
        self._np_meta_path    = self.persist_dir / "meta.json"
        self._load_numpy_store()

    def _load_numpy_store(self) -> None:
        if self._np_vectors_path.exists() and self._np_meta_path.exists():
            try:
                self._np_vectors = np.load(str(self._np_vectors_path))
                with open(self._np_meta_path, "r", encoding="utf-8") as f:
                    self._np_meta = json.load(f)
                logger.info(f"NumPy store loaded: {len(self._np_meta)} vectors")
            except Exception as e:
                logger.warning(f"Could not load NumPy store ({e}). Starting fresh.")
                self._np_vectors = None
                self._np_meta    = []
        else:
            self._np_vectors = None
            self._np_meta    = []

    def _save_numpy_store(self) -> None:
        if self._np_vectors is not None:
            np.save(str(self._np_vectors_path), self._np_vectors)
        with open(self._np_meta_path, "w", encoding="utf-8") as f:
            json.dump(self._np_meta, f, indent=2)

    def _check_dim_compat(self, new_vec: np.ndarray) -> bool:
        if self._np_vectors is not None and len(self._np_meta) > 0:
            stored = self._np_vectors.shape[1]
            new    = new_vec.shape[0]
            if stored != new:
                logger.warning(f"Dim mismatch stored={stored} new={new}. Clearing.")
                self._np_vectors = None
                self._np_meta    = []
                self._save_numpy_store()
                return False
        return True

    # ─────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────

    def count(self) -> int:
        if self.backend == "chromadb":
            return self._collection.count()
        return len(self._np_meta)

    def is_empty(self) -> bool:
        return self.count() == 0

    def get_all_names(self) -> List[str]:
        """Return list of all stored candidate names."""
        if self.backend == "chromadb":
            if self._collection.count() == 0:
                return []
            result = self._collection.get(include=["metadatas"])
            return [m.get("name", "Unknown") for m in result["metadatas"]]
        return [m.get("name", "Unknown") for m in self._np_meta]

    def get_all_metadata(self) -> List[Dict]:
        """Return full metadata list for all indexed resumes."""
        if self.backend == "chromadb":
            if self._collection.count() == 0:
                return []
            result = self._collection.get(include=["metadatas"])
            return list(result["metadatas"])
        # NumPy: return a copy so callers can't mutate the store
        return list(self._np_meta)

    def clear(self) -> None:
        if self.backend == "chromadb":
            self._chroma_client.delete_collection(self.collection_name)
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        else:
            self._np_vectors = None
            self._np_meta    = []
            for p in [self._np_vectors_path, self._np_meta_path]:
                if p.exists():
                    p.unlink()
        logger.info("VectorStore cleared.")

    def build_index(
        self,
        resumes: List[Dict],
        model,
        batch_size: int = 32,
        progress_callback=None,
    ) -> Dict:
        """Encode all resumes and store their vectors."""
        if not resumes:
            raise ValueError("No resumes provided.")

        t0      = time.time()
        indexed = 0
        skipped = 0

        for i in range(0, len(resumes), batch_size):
            batch = resumes[i: i + batch_size]
            names = [r["name"] for r in batch]
            texts = [r["text"]  for r in batch]

            if progress_callback:
                progress_callback(i, len(resumes), names[0])

            try:
                embeddings = model.encode(texts)
            except Exception as e:
                logger.warning(f"Encoding failed batch {i}: {e}")
                skipped += len(batch)
                continue

            for name, text, emb in zip(names, texts, embeddings):
                doc_id = _make_id(name, text)
                vec_np = emb.cpu().numpy()

                if self.backend == "numpy":
                    self._check_dim_compat(vec_np)

                metadata = {
                    "name"         : name,
                    "text_preview" : text[:300],
                    "text_length"  : len(text),
                    "indexed_at"   : time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "embedding_dim": int(vec_np.shape[0]),
                }

                try:
                    self._upsert(doc_id, vec_np, metadata, text)
                    indexed += 1
                except Exception as e:
                    logger.warning(f"Store failed '{name}': {e}")
                    skipped += 1

        if self.backend == "numpy":
            self._save_numpy_store()

        stats = {
            "indexed"     : indexed,
            "skipped"     : skipped,
            "total"       : self.count(),
            "duration_sec": round(time.time() - t0, 2),
            "backend"     : self.backend,
        }
        logger.info(f"build_index: {stats}")
        return stats

    def add_resume(self, name: str, text: str, model) -> bool:
        try:
            emb    = model.encode_single(text)
            vec_np = emb.cpu().numpy()
            doc_id = _make_id(name, text)
            meta   = {
                "name"         : name,
                "text_preview" : text[:300],
                "text_length"  : len(text),
                "indexed_at"   : time.strftime("%Y-%m-%dT%H:%M:%S"),
                "embedding_dim": int(vec_np.shape[0]),
            }
            self._upsert(doc_id, vec_np, meta, text)
            if self.backend == "numpy":
                self._save_numpy_store()
            return True
        except Exception as e:
            logger.error(f"add_resume failed '{name}': {e}")
            return False

    def search(self, jd_embedding: torch.Tensor, top_k: int = 10) -> List[Dict]:
        if self.is_empty():
            raise RuntimeError("Vector store is empty. Build index first.")

        jd_vec = jd_embedding.cpu().numpy()

        if self.backend == "chromadb":
            return self._search_chroma(jd_vec, top_k)
        else:
            return self._search_numpy(jd_vec, top_k)

    def get_info(self) -> Dict:
        dim = "N/A"
        if self.backend == "numpy" and self._np_vectors is not None and len(self._np_meta) > 0:
            dim = int(self._np_vectors.shape[1])
        return {
            "backend"    : self.backend,
            "count"      : self.count(),
            "persist_dir": str(self.persist_dir),
            "collection" : self.collection_name,
            "is_empty"   : self.is_empty(),
            "dim"        : dim,
            "version"    : VECTOR_STORE_VERSION,
        }

    # ─────────────────────────────────────────────────────────
    #  Internal Helpers
    # ─────────────────────────────────────────────────────────

    def _upsert(self, doc_id: str, vec_np: np.ndarray, metadata: dict, text: str) -> None:
        if self.backend == "chromadb":
            self._collection.upsert(
                ids=[doc_id],
                embeddings=[vec_np.tolist()],
                metadatas=[metadata],
                documents=[text[:500]],
            )
        else:
            arr   = vec_np.astype(np.float32)
            entry = {
                "id"           : doc_id,
                "name"         : metadata["name"],
                "text_preview" : metadata.get("text_preview", ""),
                "text_length"  : metadata.get("text_length", 0),
                "indexed_at"   : metadata.get("indexed_at", ""),
                "embedding_dim": metadata.get("embedding_dim", int(arr.shape[0])),
            }
            existing_ids = [m["id"] for m in self._np_meta]
            if doc_id in existing_ids:
                idx                   = existing_ids.index(doc_id)
                self._np_vectors[idx] = arr
                self._np_meta[idx]    = entry
            else:
                self._np_vectors = arr.reshape(1, -1) if self._np_vectors is None \
                                   else np.vstack([self._np_vectors, arr.reshape(1, -1)])
                self._np_meta.append(entry)

    def _search_chroma(self, jd_vec: np.ndarray, top_k: int) -> List[Dict]:
        k      = min(top_k, self._collection.count())
        result = self._collection.query(
            query_embeddings=[jd_vec.tolist()],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )
        output = []
        for meta, doc, dist in zip(
            result["metadatas"][0],
            result["documents"][0],
            result["distances"][0],
        ):
            similarity = float(max(0.0, min(1.0, 1.0 - dist)))
            output.append({
                "name"    : meta.get("name", "Unknown"),
                "text"    : doc,
                "score"   : round(similarity, 4),
                "metadata": meta,
            })
        return output

    def _search_numpy(self, jd_vec: np.ndarray, top_k: int) -> List[Dict]:
        if self._np_vectors is None or len(self._np_meta) == 0:
            return []

        if jd_vec.shape[0] != self._np_vectors.shape[1]:
            raise RuntimeError(
                f"Dimension mismatch: JD embedding is {jd_vec.shape[0]}-dim "
                f"but stored vectors are {self._np_vectors.shape[1]}-dim. "
                f"Please clear the index (⚙️ Manage Index → Clear) and rebuild it."
            )

        jd_norm = jd_vec / (np.linalg.norm(jd_vec) + 1e-9)
        norms   = np.linalg.norm(self._np_vectors, axis=1, keepdims=True) + 1e-9
        normed  = self._np_vectors / norms
        sims    = normed @ jd_norm
        k       = min(top_k, len(sims))
        top_idx = np.argsort(sims)[::-1][:k]

        return [
            {
                "name"    : self._np_meta[i].get("name", "Unknown"),
                "text"    : self._np_meta[i].get("text_preview", ""),
                "score"   : round(float(sims[i]), 4),
                "metadata": self._np_meta[i],
            }
            for i in top_idx
        ]


# ─────────────────────────────────────────────────────────────
#  Utility
# ─────────────────────────────────────────────────────────────

def _make_id(name: str, text: str) -> str:
    """Stable unique ID from name + full text hash."""
    full_hash = hashlib.sha256((name + text).encode("utf-8")).hexdigest()[:16]
    safe_name = "".join(c if c.isalnum() else "_" for c in name)[:40]
    return f"{safe_name}_{full_hash}"


# ─────────────────────────────────────────────────────────────
#  Singleton — keyed by version so cache busts on upgrade
# ─────────────────────────────────────────────────────────────

_store_instances: Dict[str, "VectorStore"] = {}


def get_vector_store(persist_dir: str = DEFAULT_STORE_DIR) -> "VectorStore":
    """Return a VectorStore singleton keyed by persist_dir."""
    global _store_instances
    if persist_dir not in _store_instances:
        _store_instances[persist_dir] = VectorStore(persist_dir=persist_dir)
    return _store_instances[persist_dir]
