"""Unit tests for the Stage 3 OCR adapter."""

import pytest

from doc_agent.vision.ocr import _reconstruct_easyocr, transcribe


def test_precomputed_markdown_preserves_page_citations(tmp_path):
    markdown = tmp_path / "sample.md"
    markdown.write_text(
        "# Extracted Text\n\n"
        "## Page 1\n\nপ্রথম পাতা\n\n---\n\n"
        "## Page 2 (easyocr)\n\nদ্বিতীয় পাতা\n\n---\n",
        encoding="utf-8",
    )
    cfg = {
        "device": "cpu",
        "ocr": {
            "use_precomputed": True,
            "precomputed_markdown": {"sample_book": str(markdown)},
        },
    }

    chunks = transcribe([], cfg)

    assert [chunk.page_ids for chunk in chunks] == [
        ["sample_book:page:0001"],
        ["sample_book:page:0002"],
    ]
    assert [chunk.text for chunk in chunks] == ["প্রথম পাতা", "দ্বিতীয় পাতা"]


def test_easyocr_words_are_reconstructed_in_reading_order():
    results = [
        ([[50, 40], [80, 40], [80, 60], [50, 60]], "দুই", 0.8),
        ([[10, 10], [40, 10], [40, 30], [10, 30]], "প্রথম", 0.9),
        ([[10, 40], [40, 40], [40, 60], [10, 60]], "লাইন", 0.7),
    ]

    text, confidence = _reconstruct_easyocr(results)

    assert text == "প্রথম\nলাইন দুই"
    assert confidence == pytest.approx(0.8)
