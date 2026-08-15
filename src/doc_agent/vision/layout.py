"""Stage 2 — OpenCV layout detection and reading order."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..contracts import Page, Region

_PAGE_IMAGES: dict[str, str] = {}
_PAGE_DOCUMENTS: dict[str, str] = {}


def image_path_for(page_id: str) -> str | None:
    """Return the image path registered when ``detect`` saw this page."""
    return _PAGE_IMAGES.get(page_id)


def document_id_for(page_id: str) -> str | None:
    """Return the document ID registered when ``detect`` saw this page."""
    return _PAGE_DOCUMENTS.get(page_id)


def _overlap_fraction(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    first_area = max((first[2] - first[0]) * (first[3] - first[1]), 1)
    second_area = max((second[2] - second[0]) * (second[3] - second[1]), 1)
    return intersection / min(first_area, second_area)


def _merge_boxes(
    boxes: list[tuple[int, int, int, int]], threshold: float
) -> list[tuple[int, int, int, int]]:
    merged: list[list[int]] = []
    for box in sorted(boxes, key=lambda item: (item[1], item[0])):
        for current in merged:
            current_tuple = (current[0], current[1], current[2], current[3])
            if _overlap_fraction(box, current_tuple) > threshold:
                current[0] = min(current[0], box[0])
                current[1] = min(current[1], box[1])
                current[2] = max(current[2], box[2])
                current[3] = max(current[3], box[3])
                break
        else:
            merged.append(list(box))
    return [(box[0], box[1], box[2], box[3]) for box in merged]


def _table_boxes(image: np.ndarray, cfg: dict) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        25,
        15,
    )
    height, width = gray.shape
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(width // 25, 20), 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(height // 25, 20)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    grid = cv2.dilate(cv2.bitwise_or(horizontal, vertical), np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    minimum_fraction = float(cfg.get("table_min_area_fraction", 0.02))
    page_area = height * width
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if (
            box_width > 60
            and box_height > 40
            and (box_width * box_height) / page_area >= minimum_fraction
        ):
            boxes.append((x, y, x + box_width, y + box_height))
    return _merge_boxes(boxes, float(cfg.get("merge_threshold", 0.3)))


def _text_boxes(image: np.ndarray, cfg: dict) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    height, width = gray.shape
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(width // 60, 12), max(height // 700, 2)),
    )
    connected = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    minimum_fraction = float(cfg.get("text_min_area_fraction", 0.00015))
    page_area = height * width
    boxes: list[tuple[int, int, int, int]] = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if (
            box_width >= int(cfg.get("text_min_width", 30))
            and box_height >= int(cfg.get("text_min_height", 8))
            and (box_width * box_height) / page_area >= minimum_fraction
        ):
            boxes.append((x, y, x + box_width, y + box_height))
    return _merge_boxes(boxes, float(cfg.get("merge_threshold", 0.3)))


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect page-local text/table regions and emit them in reading order."""
    layout_cfg = cfg.get("layout", {})
    regions: list[Region] = []
    for page in pages:
        image_path = Path(page.image_path).expanduser().resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read page image for layout detection: {image_path}")
        _PAGE_IMAGES[page.id] = str(image_path)
        _PAGE_DOCUMENTS[page.id] = page.doc_id

        tables = _table_boxes(image, layout_cfg)
        text = [
            box
            for box in _text_boxes(image, layout_cfg)
            if not any(_overlap_fraction(box, table) > 0.5 for table in tables)
        ]
        detected = [(box, "text") for box in text] + [(box, "table") for box in tables]
        if not detected:
            height, width = image.shape[:2]
            detected = [((0, 0, width, height), "text")]
        detected.sort(key=lambda item: (item[0][1], item[0][0]))
        regions.extend(Region(page_id=page.id, bbox=box, kind=kind) for box, kind in detected)
    return regions
