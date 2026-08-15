"""Stage 4 — vector store"""
from __future__ import annotations
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from ..contracts import Chunk 
import numpy as np

from ..contracts import Chunk

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAGE_NUMBER = re.compile(r":page:(\d+)")
_PART_NUMBER = re.compile(r":part:(\d+)$")


def _index_dir(cfg: dict) -> Path:
    path = Path(cfg["index"].get("path", "rag_index")).expanduser()
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _record(chunk: Chunk, chunk_id: int) -> dict[str, Any]:
    page_match = _PAGE_NUMBER.search(chunk.page_ids[0]) if chunk.page_ids else None
    part_match = _PART_NUMBER.search(chunk.id)
    page = int(page_match.group(1)) if page_match else 0
    page_chunk = int(part_match.group(1)) if part_match else 1
    return {
        "chunk_id": chunk_id,
        "id": chunk.id,
        "document_id": chunk.doc_id,
        "document_title": chunk.doc_id.replace("_", " ").title(),
        "page": page,
        "page_chunk": page_chunk,
        "page_ids": chunk.page_ids,
        "text": chunk.text,
        "token_count": len(chunk.text.split()),
        "source": chunk.doc_id,
    }

def build(chunks, vectors, cfg: dict) -> None:
    """Persist a vector index (cfg['index']['type']). IMPLEMENT."""
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] != len(chunks):
        raise ValueError(f"Vector/chunk mismatch: vectors={matrix.shape}, chunks={len(chunks)}")
    if not chunks:
        raise ValueError("Cannot build an empty vector index")

    index_type = str(cfg["index"]["type"])
    if index_type == "faiss:flat-ip":
        index = faiss.IndexFlatIP(matrix.shape[1])
    elif index_type == "faiss:hnsw":
        index = faiss.IndexHNSWFlat(matrix.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    else:
        raise ValueError(f"Unsupported index.type: {index_type}")
    index.add(matrix)

    index_dir = _index_dir(cfg)
    index_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_dir / "index.faiss"))

    records = [_record(chunk, index_id) for index_id, chunk in enumerate(chunks)]
    with (index_dir / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    documents: dict[str, dict[str, Any]] = {}
    for record in records:
        document = documents.setdefault(
            record["document_id"],
            {
                "document_id": record["document_id"],
                "document_title": record["document_title"],
                "pages": set(),
                "chunks": 0,
            },
        )
        document["pages"].add(record["page"])
        document["chunks"] += 1
    document_list = [
        {**document, "pages": len(document["pages"])} for document in documents.values()
    ]
    manifest = {
        "format_version": 2,
        "created_utc": datetime.now(UTC).isoformat(),
        "documents": document_list,
        "pages": len({page_id for chunk in chunks for page_id in chunk.page_ids}),
        "chunks": len(chunks),
        "chunker": cfg["index"].get("chunker", "chonkie.TokenChunker"),
        "chunk_size": int(cfg["index"]["chunk_tokens"]),
        "chunk_overlap": int(cfg["index"]["overlap"]),
        "embedding_model": cfg["embed"]["model"],
        "embedding_dimension": int(matrix.shape[1]),
        "embedding_normalized": bool(cfg["embed"].get("normalize", True)),
        "faiss_index": type(index).__name__,
        "build_device": cfg.get("device", "cuda"),
    }
    (index_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load(cfg: dict) -> dict[str, Any]:
    """Load and validate the FAISS index, metadata, and manifest."""
    index_dir = _index_dir(cfg)
    required = [index_dir / name for name in ("index.faiss", "chunks.jsonl", "manifest.json")]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing RAG index files: " + ", ".join(map(str, missing)))

    index = faiss.read_index(str(index_dir / "index.faiss"))
    records: list[dict[str, Any]] = []
    with (index_dir / "chunks.jsonl").open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid chunks.jsonl line {line_number}") from exc
    manifest = json.loads((index_dir / "manifest.json").read_text(encoding="utf-8"))
    if index.ntotal != len(records):
        raise ValueError(f"Index/chunk mismatch: {index.ntotal} vectors and {len(records)} records")
    return {"index": index, "records": records, "manifest": manifest, "path": index_dir}
