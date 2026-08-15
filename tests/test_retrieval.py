"""Unit tests for page-citable FAISS retrieval."""

import json

import faiss
import numpy as np
import pytest

from doc_agent.contracts import Chunk
from doc_agent.retrieval import rerank as rerank_stage
from doc_agent.retrieval.retriever import Retriever


class _FakeEmbedder:
    def encode_query(self, texts, **kwargs):
        assert texts == ["স্বাধীনতা"]
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


def test_retrieval_returns_ranked_chunk_with_page_number(tmp_path):
    records = [
        {
            "chunk_id": 0,
            "document_id": "history",
            "page": 163,
            "page_chunk": 1,
            "text": "ছয় দফা ও স্বাধীনতা আন্দোলন",
        },
        {
            "chunk_id": 1,
            "document_id": "geography",
            "page": 42,
            "page_chunk": 1,
            "text": "পৃথিবীর বার্ষিক গতি",
        },
    ]
    with (tmp_path / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"embedding_model": "fake", "embedding_dimension": 2}),
        encoding="utf-8",
    )
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    faiss.write_index(index, str(tmp_path / "index.faiss"))
    cfg = {
        "device": "cpu",
        "embed": {"model": "fake", "dim": 2},
        "index": {"path": str(tmp_path)},
        "retrieve": {
            "k": 1,
            "candidate_k": 2,
            "hybrid": False,
            "rerank": False,
            "weak_threshold": 0.35,
            "k_step": 1,
            "k_max": 2,
        },
    }
    retriever = Retriever(cfg)
    retriever._embedder = _FakeEmbedder()

    results = retriever.retrieve("স্বাধীনতা")

    assert len(results) == 1
    assert results[0].doc_id == "history"
    assert results[0].page_ids == ["history:page:0163"]
    assert results[0].score == 1.0


class _FakeReranker:
    def predict(self, pairs, **kwargs):
        assert pairs == [("query", "first"), ("query", "second")]
        return np.asarray([0.2, 0.9], dtype=np.float32)


def test_cross_encoder_reranking_is_delegated_to_rerank_module(monkeypatch):
    candidates = [
        Chunk(id="c1", doc_id="book", text="first", page_ids=["book:page:0001"]),
        Chunk(id="c2", doc_id="book", text="second", page_ids=["book:page:0002"]),
    ]
    cfg = {
        "device": "cpu",
        "retrieve": {"rerank": True, "reranker": "fake", "batch_size": 2},
    }
    monkeypatch.setattr(rerank_stage, "_load_model", lambda _: _FakeReranker())

    results = rerank_stage.rerank("query", candidates, cfg)

    assert [result.id for result in results] == ["c2", "c1"]
    assert results[0].score == pytest.approx(0.9)
