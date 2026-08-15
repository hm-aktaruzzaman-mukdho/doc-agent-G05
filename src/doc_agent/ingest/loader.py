"""Stage 1 — discover scanned page images."""

from __future__ import annotations

import re
from pathlib import Path

from ..contracts import Page

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}
_PAGE_NUMBER = re.compile(r"(?:page|p)?[_\- ]*(\d+)$", flags=re.IGNORECASE)


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug or "document"


def _document_id(path: Path, raw_dir: Path) -> str:
    relative = path.relative_to(raw_dir)
    if relative.parent != Path("."):
        return _slug(relative.parts[0])
    stem = _PAGE_NUMBER.sub("", path.stem).rstrip("_- ")
    return _slug(stem)


def load_pages(cfg: dict) -> list[Page]:
    """Read configured image files into stable, document-aware ``Page`` objects."""
    ingest_cfg = cfg.get("ingest", {})
    raw_dir = _resolve_path(ingest_cfg.get("raw_dir", "data/raw"))
    if not raw_dir.exists():
        if ingest_cfg.get("allow_empty", cfg.get("ocr", {}).get("use_precomputed", False)):
            return []
        raise FileNotFoundError(f"Raw page directory does not exist: {raw_dir}")

    extensions = {
        str(extension).casefold()
        for extension in ingest_cfg.get("extensions", sorted(_DEFAULT_EXTENSIONS))
    }
    image_paths = sorted(
        path.resolve()
        for path in raw_dir.rglob("*")
        if path.is_file() and path.suffix.casefold() in extensions
    )
    if not image_paths and not ingest_cfg.get(
        "allow_empty", cfg.get("ocr", {}).get("use_precomputed", False)
    ):
        raise FileNotFoundError(f"No supported page images found under: {raw_dir}")

    pages: list[Page] = []
    document_counts: dict[str, int] = {}
    used_ids: set[str] = set()
    for image_path in image_paths:
        doc_id = _document_id(image_path, raw_dir)
        document_counts[doc_id] = document_counts.get(doc_id, 0) + 1
        match = _PAGE_NUMBER.search(image_path.stem)
        page_number = int(match.group(1)) if match else document_counts[doc_id]
        page_id = f"{doc_id}:page:{page_number:04d}"
        while page_id in used_ids:
            page_number += 1
            page_id = f"{doc_id}:page:{page_number:04d}"
        used_ids.add(page_id)
        pages.append(Page(id=page_id, image_path=str(image_path), doc_id=doc_id))
    return pages
