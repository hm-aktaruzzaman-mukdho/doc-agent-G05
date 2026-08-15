"""Stage 4 — embed chunks"""

from __future__ import annotations

import numpy as np
import torch

from ..contracts import Chunk


def _device(cfg: dict) -> str:
    requested = str(cfg.get("device", "cuda"))
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Embedding is configured for CUDA, but no NVIDIA GPU is available")
    return requested


def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    """Encode chunks with normalized BGE-M3 vectors on the configured device."""
    if not chunks:
        return np.empty((0, int(cfg["embed"]["dim"])), dtype=np.float32)

    from sentence_transformers import SentenceTransformer

    embed_cfg = cfg["embed"]
    device = _device(cfg)
    model_kwargs = {"torch_dtype": torch.float16} if device.startswith("cuda") else {}
    model = SentenceTransformer(
        embed_cfg["model"],
        device=device,
        model_kwargs=model_kwargs,
    )
    encode_document = getattr(model, "encode_document", None) or model.encode
    vectors = encode_document(
        [chunk.text for chunk in chunks],
        batch_size=int(embed_cfg.get("batch_size", 16)),
        convert_to_numpy=True,
        normalize_embeddings=bool(embed_cfg.get("normalize", True)),
        show_progress_bar=bool(embed_cfg.get("show_progress", True)),
    ).astype(np.float32, copy=False)

    expected_dim = int(embed_cfg["dim"])
    if vectors.shape != (len(chunks), expected_dim):
        raise RuntimeError(
            f"Unexpected embedding shape {vectors.shape}; expected ({len(chunks)}, {expected_dim})"
        )
    return vectors
