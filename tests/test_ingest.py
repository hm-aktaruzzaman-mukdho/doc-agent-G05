"""Unit tests for image ingest, preprocessing, and layout registration."""

from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from doc_agent.ingest import enhance, loader, preprocess
from doc_agent.vision import layout
from doc_agent.vision.ocr import Reader


def _write_page(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", (400, 240), "white")
    draw = ImageDraw.Draw(image)
    for y in (40, 70, 100, 160):
        draw.rectangle((30, y, 350, y + 8), fill="black")
    image.save(path)


def test_ingest_preprocess_layout_and_ocr_path_bridge(tmp_path):
    raw_dir = tmp_path / "raw"
    source_path = raw_dir / "history" / "page_001.png"
    _write_page(source_path)
    cfg = {
        "device": "cpu",
        "ingest": {"raw_dir": str(raw_dir), "allow_empty": False},
        "preprocess": {
            "enabled": True,
            "output_dir": str(tmp_path / "interim"),
            "deskew": True,
            "denoise": True,
            "binarize": True,
            "threshold_block_size": 31,
            "threshold_constant": 15,
        },
        "enhance": {"enabled": False, "type": "none"},
        "layout": {"model": "opencv:morphology"},
        "ocr": {"gpu": False, "use_precomputed": False},
    }

    pages = loader.load_pages(cfg)
    assert len(pages) == 1
    assert pages[0].id == "history:page:0001"

    processed = preprocess.run(pages, cfg)
    assert processed[0].id == pages[0].id
    assert Path(processed[0].image_path).is_file()
    assert cv2.imread(processed[0].image_path, cv2.IMREAD_GRAYSCALE) is not None
    assert enhance.run(processed, cfg) == processed

    regions = layout.detect(processed, cfg)
    assert regions
    assert all(region.page_id == pages[0].id for region in regions)
    assert regions == sorted(regions, key=lambda region: (region.bbox[1], region.bbox[0]))

    reader = Reader(cfg)
    assert reader._resolve_image(pages[0].id) == Path(processed[0].image_path)


def test_empty_precomputed_corpus_is_allowed(tmp_path):
    cfg = {
        "ingest": {"raw_dir": str(tmp_path / "missing"), "allow_empty": True},
        "ocr": {"use_precomputed": True},
    }
    assert loader.load_pages(cfg) == []
