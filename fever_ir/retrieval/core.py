from __future__ import annotations

import json
from pathlib import Path
import pickle
import re
import os

# FEVER dense retrieval uses PyTorch. Prevent an incompatible globally-installed
# TensorFlow package from being imported by Transformers on Windows.
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import sys
sys.modules.setdefault("tensorflow", None)

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .preprocessing import preprocess_text, preprocess_for_tfidf, clean_text


def get_sentence_transformer(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    from sentence_transformers import SentenceTransformer
    cache_base = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}" / "snapshots"
    if cache_base.exists():
        snapshots = [d for d in cache_base.iterdir() if d.is_dir()]
        if snapshots:
            return SentenceTransformer(str(snapshots[0]))
    return SentenceTransformer(model_name)


def tokenize(text):
    """Backward compatibility tokenize function."""
    return preprocess_text(text)


class EvidenceIndex:
    def __init__(self, passages):
        self.passages = passages
        self.texts = [p["text"] for p in passages]
        self.ids = [p["passage_id"] for p in passages]
        
        # Apply standardized preprocessing: cleaning, tokenization, stop-words, stemming
        self.preprocessed_texts = [preprocess_for_tfidf(t) for t in self.texts]
        self.tokenized_corpus = [preprocess_text(t) for t in self.texts]
        
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.preprocessed_texts)
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def _rows(self, indices, scores, method):
        return [
            {
                "passage_id": self.ids[i],
                "title": self.passages[i].get("title", ""),
                "sentence_id": self.passages[i].get("sentence_id", ""),
                "text": self.texts[i],
                "score": float(scores[j]),
                "rank": j + 1,
                "method": method,
            }
            for j, i in enumerate(indices)
        ]

    def search_tfidf(self, query, k=10):
        prep_query = preprocess_for_tfidf(query)
        scores = linear_kernel(self.tfidf_vectorizer.transform([prep_query]), self.tfidf_matrix).ravel()
        idx = np.argsort(-scores)[:k]
        return self._rows(idx, scores[idx], "tfidf")

    def search_bm25(self, query, k=10):
        tokens = preprocess_text(query)
        scores = np.asarray(self.bm25.get_scores(tokens))
        idx = np.argsort(-scores)[:k]
        return self._rows(idx, scores[idx], "bm25")

    def save_sparse(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_sparse(path):
        with open(path, "rb") as f:
            return pickle.load(f)


class DenseIndex:
    def __init__(self, passages, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.passages = passages
        self.ids = [p["passage_id"] for p in passages]
        self.model_name = model_name
        self.model = get_sentence_transformer(model_name)
        self.embeddings = self.model.encode(
            [clean_text(p["text"]) for p in passages],
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def search(self, query, k=10):
        q = self.model.encode([clean_text(query)], normalize_embeddings=True)[0]
        scores = np.asarray(self.embeddings) @ q
        idx = np.argsort(-scores)[:k]
        return [
            {
                "passage_id": self.ids[i],
                "title": self.passages[i].get("title", ""),
                "sentence_id": self.passages[i].get("sentence_id", ""),
                "text": self.passages[i]["text"],
                "score": float(scores[i]),
                "rank": j + 1,
                "method": "dense",
            }
            for j, i in enumerate(idx)
        ]

    def save(self, path):
        np.savez_compressed(path, embeddings=self.embeddings)
        with open(str(path) + ".meta.json", "w", encoding="utf-8") as f:
            json.dump({"passages": self.passages, "model_name": self.model_name}, f)

    @classmethod
    def load(cls, path):
        p = Path(path)
        meta_path = Path(str(p) + ".meta.json") if not str(p).endswith(".meta.json") else p
        npz_path = p if p.suffix == ".npz" else Path(str(p).replace(".meta.json", ""))
        if not npz_path.exists() and p.with_suffix(".npz").exists():
            npz_path = p.with_suffix(".npz")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        idx = cls.__new__(cls)
        idx.passages = meta["passages"]
        idx.ids = [item["passage_id"] for item in meta["passages"]]
        idx.model_name = meta["model_name"]
        idx.model = get_sentence_transformer(idx.model_name)
        npz = np.load(npz_path)
        idx.embeddings = npz["embeddings"]
        return idx


def reciprocal_rank_fusion(result_lists, k=60, top_k=10):
    """
    Reciprocal Rank Fusion (RRF) Formula:
    RRF_Score(d in D) = sum_{m in M} 1 / (k + r_m(d))
    Combines sparse and dense rankings without requiring score normalization.
    """
    fused = {}
    for results in result_lists:
        for rank, row in enumerate(results, start=1):
            pid = row["passage_id"]
            if pid not in fused:
                fused[pid] = {**row, "score": 0.0, "rank": 0, "method": "rrf"}
            fused[pid]["score"] += 1.0 / (k + rank)
    out = sorted(fused.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    for rank, row in enumerate(out, start=1):
        row["rank"] = rank
    return out


def load_passages(path, limit=None):
    with open(path, encoding="utf-8") as f:
        rows = []
        for line in f:
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
        return rows
