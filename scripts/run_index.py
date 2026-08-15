"""Build only the A2 text index from the completed OCR Markdown outputs."""

from doc_agent import config
from doc_agent.index import chunk, embed, store
from doc_agent.vision import ocr


def main() -> None:
    cfg = config.load()
    page_chunks = ocr.transcribe([], cfg)
    chunks = chunk.split(page_chunks, cfg)
    vectors = embed.encode(chunks, cfg)
    store.build(chunks, vectors, cfg)
    print(f"Indexed {len(chunks)} chunks from {len(page_chunks)} OCR pages")


if __name__ == "__main__":
    main()
