"""Stage 3 — Bangla OCR/HTR.

The corpus was transcribed with ``dl-ocr-parallel.py``: EasyOCR supplies the
Bangla text and Qwen3.5-4B restores page structure.  Building the knowledge
base normally consumes those completed Markdown files, so an index rebuild
does not repeat an expensive full-book OCR run.  ``Reader`` keeps the same
EasyOCR region path available for unseen page images and held-out evaluation.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Any

from ..contracts import Chunk, Region
from .layout import document_id_for, image_path_for

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PAGE_HEADER = re.compile(r"(?m)^## Page\s+(\d+)(?:\s*\([^\n)]*\))?\s*$")


def _clean_text(text: str) -> str:
    """Apply the same Unicode and blank-line cleanup as dl-ocr-parallel.py."""
    text = unicodedata.normalize("NFKC", text or "")
    lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line:
            lines.append(line)
            previous_blank = False
        elif not previous_blank:
            lines.append("")
            previous_blank = True
    return "\n".join(lines).strip()


def _reconstruct_easyocr(results: list[Any]) -> tuple[str, float]:
    """Turn EasyOCR word boxes into reading-order lines and mean confidence."""
    if not results:
        return "", 0.0

    ordered = sorted(results, key=lambda item: (item[0][0][1], item[0][0][0]))
    lines: list[list[tuple[float, str]]] = []
    current_line: list[tuple[float, str]] = []
    last_y_mid: float | None = None
    current_height = 20.0
    confidences: list[float] = []

    for bbox, text, confidence in ordered:
        y_min = min(point[1] for point in bbox)
        y_max = max(point[1] for point in bbox)
        y_mid = (y_min + y_max) / 2
        height = max(float(y_max - y_min), 1.0)
        confidences.append(float(confidence))

        if last_y_mid is None or abs(y_mid - last_y_mid) < current_height * 0.7:
            current_line.append((float(bbox[0][0]), str(text)))
            last_y_mid = y_mid if last_y_mid is None else (last_y_mid + y_mid) / 2
            current_height = height if len(current_line) == 1 else current_height
        else:
            lines.append(current_line)
            current_line = [(float(bbox[0][0]), str(text))]
            last_y_mid = y_mid
            current_height = height

    if current_line:
        lines.append(current_line)

    text = "\n".join(
        " ".join(word for _, word in sorted(line, key=lambda item: item[0])) for line in lines
    )
    return _clean_text(text), sum(confidences) / len(confidences)


def _resolve_path(value: str | Path) -> Path:
    """Resolve configured paths from the repository root, independent of cwd."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _document_pages(path: Path, doc_id: str) -> list[Chunk]:
    """Parse page-delimited Markdown emitted by dl-ocr-parallel.py."""
    markdown = path.read_text(encoding="utf-8")
    matches = list(_PAGE_HEADER.finditer(markdown))
    if not matches:
        raise ValueError(f"No '## Page N' sections found in OCR output: {path}")

    chunks: list[Chunk] = []
    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        text = markdown[match.end() : end]
        text = re.sub(r"(?:\n\s*---\s*)+$", "", text).strip()
        text = _clean_text(text)
        if not text:
            continue
        page_id = f"{doc_id}:page:{page_number:04d}"
        chunks.append(
            Chunk(
                id=f"{page_id}:ocr",
                doc_id=doc_id,
                text=text,
                page_ids=[page_id],
            )
        )
    return chunks


class Reader:
    """Lazy GPU EasyOCR reader for a single detected page region."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.device = str(cfg.get("device", "cuda"))
        self._reader: Any | None = None

    def _load_reader(self) -> Any:
        if self._reader is not None:
            return self._reader

        import easyocr
        import torch

        wants_gpu = bool(self.cfg.get("gpu", self.device.startswith("cuda")))
        has_gpu = torch.cuda.is_available()
        if wants_gpu and not has_gpu and self.cfg.get("require_gpu", True):
            raise RuntimeError("OCR is configured for CUDA, but no NVIDIA CUDA GPU is available")

        languages = list(self.cfg.get("languages", ["bn"]))
        self._reader = easyocr.Reader(languages, gpu=wants_gpu and has_gpu)
        return self._reader

    def _resolve_image(self, page_id: str) -> Path:
        direct_path = _resolve_path(page_id)
        if direct_path.is_file():
            return direct_path

        configured = self.cfg.get("page_images", {}).get(page_id)
        if configured:
            configured_path = _resolve_path(configured)
            if configured_path.is_file():
                return configured_path

        image_dir = self.cfg.get("page_image_dir")
        if image_dir:
            root = _resolve_path(image_dir)
            for suffix in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                candidate = root / f"{page_id}{suffix}"
                if candidate.is_file():
                    return candidate

        registered = image_path_for(page_id)
        if registered:
            registered_path = Path(registered)
            if registered_path.is_file():
                return registered_path

        raise FileNotFoundError(
            f"No image found for page_id={page_id!r}; use an image path as the page id "
            "or configure ocr.page_images/page_image_dir"
        )

    def transcribe_region(self, region: Region) -> str:
        """Crop a pixel-coordinate region and transcribe it with Bangla EasyOCR."""
        import numpy as np
        from PIL import Image

        image_path = self._resolve_image(region.page_id)
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            left, top, right, bottom = region.bbox
            if right <= left or bottom <= top:
                raise ValueError(f"Invalid OCR region bbox: {region.bbox}")
            crop = np.asarray(rgb.crop((left, top, right, bottom)))

        results = self._load_reader().readtext(
            crop,
            detail=1,
            paragraph=False,
            batch_size=int(self.cfg.get("batch_size", 8)),
        )
        text, _ = _reconstruct_easyocr(results)
        return text


def _precomputed_chunks(cfg: dict) -> list[Chunk]:
    sources = cfg["ocr"].get("precomputed_markdown", {})
    if isinstance(sources, list):
        source_items = [(Path(value).stem, value) for value in sources]
    else:
        source_items = list(sources.items())

    chunks: list[Chunk] = []
    for doc_id, value in source_items:
        path = _resolve_path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Configured OCR Markdown does not exist: {path}")
        chunks.extend(_document_pages(path, str(doc_id)))
    return chunks


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Convert OCR output or detected regions into page-citable text chunks."""
    ocr_cfg = cfg["ocr"]
    if ocr_cfg.get("use_precomputed", False):
        precomputed_chunks = _precomputed_chunks(cfg)
        if not regions:
            return precomputed_chunks

        requested_pages = {region.page_id for region in regions}
        selected = [chunk for chunk in precomputed_chunks if set(chunk.page_ids) & requested_pages]
        return selected or precomputed_chunks

    reader = Reader(cfg)
    ordered_regions = sorted(
        regions,
        key=lambda region: (region.page_id, region.bbox[1], region.bbox[0]),
    )
    chunks: list[Chunk] = []
    for index, region in enumerate(ordered_regions):
        text = reader.transcribe_region(region)
        if not text:
            continue
        doc_id = document_id_for(region.page_id) or str(ocr_cfg.get("doc_id", "document"))
        chunks.append(
            Chunk(
                id=f"{region.page_id}:region:{index:04d}",
                doc_id=doc_id,
                text=text,
                page_ids=[region.page_id],
            )
        )
    return chunks
