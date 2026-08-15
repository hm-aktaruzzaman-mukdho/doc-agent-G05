"""Stage 5 — multilingual cross-encoder reranking."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import torch

from ..contracts import Chunk

_RERANKER_CACHE: dict[tuple[str, str, int], Any] = {}


def _device(cfg: dict) -> str:
    requested = str(cfg.get("device", "cuda"))
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Reranking is configured for CUDA, but no NVIDIA GPU is available")
    return requested


def _load_model(cfg: dict) -> Any:
    from sentence_transformers import CrossEncoder

    retrieve_cfg = cfg["retrieve"]
    device = _device(cfg)
    model_name = str(retrieve_cfg["reranker"])
    max_length = int(retrieve_cfg.get("reranker_max_length", 512))
    cache_key = (model_name, device, max_length)
    if cache_key not in _RERANKER_CACHE:
        model_kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
        _RERANKER_CACHE[cache_key] = CrossEncoder(
            model_name,
            device=device,
            model_kwargs=model_kwargs,
            max_length=max_length,
        )
    return _RERANKER_CACHE[cache_key]


def rerank(query: str, candidates: list[Chunk], cfg: dict) -> list[Chunk]:
    """Score and reorder fused candidates with the configured cross-encoder."""
    if not candidates or not cfg["retrieve"].get("rerank", True):
        return candidates
    query = query.strip()
    if not query:
        raise ValueError("Query cannot be empty")

    pairs = [(query, candidate.text) for candidate in candidates]
    scores = np.asarray(
        _load_model(cfg).predict(
            pairs,
            batch_size=int(cfg["retrieve"].get("batch_size", 16)),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
    ).reshape(-1)
    if len(scores) != len(candidates):
        raise RuntimeError(
            f"Reranker returned {len(scores)} scores for {len(candidates)} candidates"
        )

    scored = [
        candidate.model_copy(update={"score": float(score)})
        for candidate, score in zip(candidates, scores, strict=True)
    ]
    return sorted(scored, key=lambda candidate: candidate.score, reverse=True)


def release_models() -> None:
    """Release cached cross-encoders and unused CUDA allocations."""
    _RERANKER_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
