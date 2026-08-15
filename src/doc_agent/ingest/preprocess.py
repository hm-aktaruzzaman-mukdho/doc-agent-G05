"""Stage 1 — deterministic deskew, denoise, and binarization."""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

from ..contracts import Page

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (_PROJECT_ROOT / path).resolve()


def _deskew(gray: np.ndarray) -> np.ndarray:
    foreground = np.column_stack(np.where(gray < 220))
    if len(foreground) < 20:
        return gray
    angle = cv2.minAreaRect(foreground[:, ::-1].astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.05 or abs(angle) > 15:
        return gray
    height, width = gray.shape
    rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
    return cv2.warpAffine(
        gray,
        rotation,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _safe_name(page_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", page_id)


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Clean page images without altering their IDs or source documents."""
    preprocess_cfg = cfg.get("preprocess", {})
    if not pages or not preprocess_cfg.get("enabled", True):
        return pages

    output_dir = _resolve_path(preprocess_cfg.get("output_dir", "data/interim/preprocessed"))
    output_dir.mkdir(parents=True, exist_ok=True)
    processed: list[Page] = []
    for page in pages:
        gray = cv2.imread(page.image_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not read page image: {page.image_path}")
        if preprocess_cfg.get("deskew", True):
            gray = _deskew(gray)
        if preprocess_cfg.get("denoise", True):
            gray = cv2.fastNlMeansDenoising(
                gray,
                None,
                h=float(preprocess_cfg.get("denoise_strength", 7)),
                templateWindowSize=7,
                searchWindowSize=21,
            )
        if preprocess_cfg.get("binarize", True):
            gray = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                int(preprocess_cfg.get("threshold_block_size", 31)),
                int(preprocess_cfg.get("threshold_constant", 15)),
            )

        output_path = output_dir / f"{_safe_name(page.id)}.png"
        if not cv2.imwrite(str(output_path), gray):
            raise OSError(f"Could not write preprocessed page: {output_path}")
        processed.append(Page(id=page.id, image_path=str(output_path), doc_id=page.doc_id))
    return processed
