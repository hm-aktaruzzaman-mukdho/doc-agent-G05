"""Stage 4 — chunk text"""

from __future__ import annotations

from chonkie import TokenChunker

from ..contracts import Chunk


def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Split every OCR page independently, retaining its page provenance."""

    if not chunks:
        return []

    index_cfg = cfg["index"]
    chunk_size = int(index_cfg["chunk_tokens"])
    overlap = int(index_cfg["overlap"])
    if chunk_size < 1:
        raise ValueError("index.chunk_tokens must be at least 1")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("index.overlap must be non-negative and smaller than chunk_tokens")

    chunker = TokenChunker(
        tokenizer=cfg["embed"]["model"],
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    batches = chunker.chunk_batch([chunk.text for chunk in chunks])

    output: list[Chunk] = []
    for source, page_chunks in zip(chunks, batches, strict=True):
        for part_number, page_chunk in enumerate(page_chunks, start=1):
            text = page_chunk.text.strip()
            if not text:
                continue
            output.append(
                Chunk(
                    id=f"{source.id}:part:{part_number:04d}",
                    doc_id=source.doc_id,
                    text=text,
                    page_ids=source.page_ids,
                )
            )
    return output
