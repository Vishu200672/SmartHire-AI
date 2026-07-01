import os, logging, hashlib, time
from typing import Dict, List
import numpy as np
import torch

logger = logging.getLogger(__name__)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY", "")
PINECONE_HOST = "https://smarthire-resumes-jqtzind.svc.aped-4627-b74a.pinecone.io"
PINECONE_INDEX = "smarthire-resumes"
EMBEDDING_DIM = 384

class PineconeVectorStore:
    def __init__(self):
        self._index = None
        self._connect()
    def _connect(self):
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key="pcsk_3YJTrB_C2vfzUCyLhm2vxKjXAbmUK38yyXBBVU9r6uCbjeKAiXCusyv9BafYNKprxoagcw")
            self._index = pc.Index(host=PINECONE_HOST)
            stats = self._index.describe_index_stats()
            logger.info(f"Pinecone connected. Vectors: {stats.total_vector_count}")
        except Exception as e:
            logger.error(f"Pinecone connection failed: {e}")
            self._index = None
    @property
    def is_connected(self):
        return self._index is not None
    def _make_id(self, name, text):
        return hashlib.sha256(f"{name}:{text[:200]}".encode()).hexdigest()[:32]
    def add(self, name, text, embedding):
        vec_id = self._make_id(name, text)
        self._index.upsert(vectors=[{"id": vec_id, "values": embedding.cpu().numpy().tolist(), "metadata": {"name": name, "text_preview": text[:500], "text_length": len(text), "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}}])
        return vec_id
    def build_index(self, resumes, model):
        vectors = []
        for r in resumes:
            emb = model.encode_single(r["text"])
            vec_id = self._make_id(r["name"], r["text"])
            vectors.append({"id": vec_id, "values": emb.cpu().numpy().tolist(), "metadata": {"name": r["name"], "text_preview": r["text"][:500], "text_length": len(r["text"]), "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S")}})
        for i in range(0, len(vectors), 100):
            self._index.upsert(vectors=vectors[i:i+100])
        stats = self._index.describe_index_stats()
        return {"indexed": len(vectors), "total_vectors": stats.total_vector_count, "backend": "pinecone"}
    def search(self, query_embedding, top_k=10):
        res = self._index.query(vector=query_embedding.cpu().numpy().tolist(), top_k=top_k, include_metadata=True)
        return [{"name": m.metadata.get("name","Unknown"), "score": round(m.score*100,2), "text_preview": m.metadata.get("text_preview",""), "text_length": m.metadata.get("text_length",0), "indexed_at": m.metadata.get("indexed_at",""), "id": m.id} for m in res.matches]
    def get_stats(self):
        stats = self._index.describe_index_stats()
        return {"backend": "pinecone", "connected": True, "total_vectors": stats.total_vector_count, "index_name": PINECONE_INDEX, "dimension": EMBEDDING_DIM}
    def get_all_names(self):
        return []
    def get_all_metadata(self):
        stats = self._index.describe_index_stats()
        return [{"total_indexed": stats.total_vector_count, "backend": "pinecone"}]
    def clear(self):
        self._index.delete(delete_all=True)

class NumpyVectorStore:
    def __init__(self):
        self._embeddings = []
        self._metadata = []
    def add(self, name, text, embedding):
        vec_id = hashlib.sha256(f"{name}:{text[:200]}".encode()).hexdigest()[:32]
        self._embeddings.append(embedding.cpu().numpy())
        self._metadata.append({"name": name, "text_preview": text[:500], "text_length": len(text), "id": vec_id})
        return vec_id
    def build_index(self, resumes, model):
        for r in resumes:
            emb = model.encode_single(r["text"])
            self.add(r["name"], r["text"], emb)
        return {"indexed": len(resumes), "total_vectors": len(self._embeddings), "backend": "numpy"}
    def search(self, query_embedding, top_k=10):
        if not self._embeddings:
            return []
        q = query_embedding.cpu().numpy()
        mat = np.stack(self._embeddings)
        scores = (mat @ q) / (np.linalg.norm(mat, axis=1) * np.linalg.norm(q) + 1e-9)
        idx = np.argsort(scores)[::-1][:top_k]
        return [{"name": self._metadata[i]["name"], "score": round(float(scores[i])*100,2), "text_preview": self._metadata[i]["text_preview"], "id": self._metadata[i]["id"]} for i in idx]
    def get_stats(self):
        return {"backend": "numpy", "connected": True, "total_vectors": len(self._embeddings), "persistent": False}
    def get_all_names(self):
        return [m["name"] for m in self._metadata]
    def get_all_metadata(self):
        return self._metadata
    def clear(self):
        self._embeddings.clear()
        self._metadata.clear()

_store_instance = None
def get_vector_store():
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    if PINECONE_API_KEY:
        store = PineconeVectorStore()
        if store.is_connected:
            _store_instance = store
            return _store_instance
    _store_instance = NumpyVectorStore()
    return _store_instance
