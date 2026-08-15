#!/usr/bin/env bash
# A2 — build the Chonkie/BGE-M3/FAISS index from saved OCR Markdown.
set -euo pipefail
PYTHONPATH=src python scripts/run_index.py
