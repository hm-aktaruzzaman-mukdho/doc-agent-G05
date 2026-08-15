"""A2 retrieval proof over the Stage 4 hybrid RAG index."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

import numpy as np
import torch
from rank_bm25 import BM25Okapi

from ..contracts import Chunk
from ..index import store
from . import rerank as rerank_stage

_TOKEN_PATTERN = re.compile(r"[\u0980-\u09ff]+|[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


class Retriever:
    def __init__(self, cfg: dict) -> None:
        self.full_cfg = cfg
        self.cfg = cfg["retrieve"]
        loaded = store.load(cfg)
        self.index = loaded["index"]
        self.records = loaded["records"]
        self.manifest = loaded["manifest"]
        self.device = self._device(str(cfg.get("device", "cuda")))
        self.embedding_model = self.manifest.get("embedding_model", cfg["embed"]["model"])
        self._embedder: Any | None = None
        self._bm25 = BM25Okapi([_tokens(record["text"]) for record in self.records])

    @staticmethod
    def _device(requested: str) -> str:
        if requested == "auto":
            return "cuda:0" if torch.cuda.is_available() else "cpu"
        if requested == "cuda":
            requested = "cuda:0"
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Retrieval is configured for CUDA, but no NVIDIA GPU is available")
        return requested

    def _load_embedder(self) -> Any:
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer

            model_kwargs = {"torch_dtype": torch.float16} if self.device.startswith("cuda") else {}
            self._embedder = SentenceTransformer(
                self.embedding_model,
                device=self.device,
                model_kwargs=model_kwargs,
            )
        return self._embedder

    @staticmethod
    def _add_rrf(scores: dict[int, float], ids: list[int], constant: int = 60) -> None:
        for rank, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (constant + rank)

    def retrieve(self, query: str, k: int | None = None) -> list[Chunk]:
        """Return page-citable hybrid results with a relevance score on each chunk."""
        query = query.strip()
        if not query:
            raise ValueError("Query cannot be empty")
        top_k = int(k or self.cfg["k"])
        if top_k < 1:
            raise ValueError("k must be at least 1")
        candidate_k = min(
            max(int(self.cfg.get("candidate_k", 24)), top_k),
            len(self.records),
        )

        embedder = self._load_embedder()
        encode_query = getattr(embedder, "encode_query", None) or embedder.encode
        query_vector = encode_query(
            [query],
            batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
        dense_scores, dense_ids = self.index.search(query_vector, candidate_k)
        dense_ranked = [int(item) for item in dense_ids[0] if item >= 0]
        dense_score_map = {
            int(chunk_id): float(score)
            for chunk_id, score in zip(dense_ids[0], dense_scores[0], strict=True)
            if chunk_id >= 0
        }

        fusion_scores: dict[int, float] = {}
        self._add_rrf(fusion_scores, dense_ranked)
        if self.cfg.get("hybrid", True):
            lexical_scores = np.asarray(self._bm25.get_scores(_tokens(query)))
            lexical_ranked = (
                np.argsort(-lexical_scores)[:candidate_k].astype(int).tolist()
                if lexical_scores.size and np.max(lexical_scores) > 0
                else []
            )
            self._add_rrf(fusion_scores, lexical_ranked)
        fused_ids = sorted(fusion_scores, key=fusion_scores.get, reverse=True)[:candidate_k]

        candidates: list[Chunk] = []
        for index in fused_ids:
            record = self.records[index]
            doc_id = str(record.get("document_id", "document"))
            page = int(record.get("page", 0))
            page_ids = record.get("page_ids") or [f"{doc_id}:page:{page:04d}"]
            candidates.append(
                Chunk(
                    id=str(record.get("id", f"{doc_id}:page:{page:04d}:chunk:{index}")),
                    doc_id=doc_id,
                    text=record["text"],
                    page_ids=page_ids,
                    score=float(dense_score_map.get(index, 0.0)),
                )
            )
        return rerank_stage.rerank(query, candidates, self.full_cfg)[:top_k]

    def release_models(self) -> None:
        self._embedder = None
        rerank_stage.release_models()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# --- evidence-strength policy: read by agent.decide() for evidence-gated re-search ---
def top_score(chunks: list[Chunk]) -> float:
    """Strength of the current evidence = best chunk score (0.0 if empty)."""
    return max((c.score for c in chunks), default=0.0)


def is_weak(chunks: list[Chunk], cfg: dict) -> bool:
    """Weak evidence = best score below cfg.retrieve.weak_threshold."""
    return top_score(chunks) < cfg["retrieve"]["weak_threshold"]


def next_k(k: int, cfg: dict) -> int | None:
    """Widen the net: k + k_step, or None once it would exceed k_max (signal to ABSTAIN)."""
    nk = k + cfg["retrieve"]["k_step"]
    return nk if nk <= cfg["retrieve"]["k_max"] else None
