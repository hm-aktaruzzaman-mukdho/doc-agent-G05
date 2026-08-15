# Knowledge-base pipeline diagram (A2 — fill this)
# SSC Bangla Humanities OCR Pipeline — Full Diagram (Text)

```
                              ┌─────────────────────┐
                              │   PDF book(s)        │
                              │   PDF_DIR             │
                              └──────────┬───────────┘
                                         │
========================================│===============================
 PHASE 1 — EasyOCR pass (all pages, GPU 0)
========================================│===============================
                                         ▼
                        ┌────────────────────────────────┐
                        │ split_pdf_to_page_images         │
                        │ render every page to PNG          │
                        └──────────────┬───────────────────┘
                                       │
                                       ▼
                        ┌────────────────────────────────┐
                        │ easyocr_all_pages                │
                        │ EasyOCR (Bangla) on each page      │
                        └──────────────┬───────────────────┘
                                       │
                                       ▼
                        ┌────────────────────────────────┐
                        │ reconstruct_text_from_easyocr    │
                        │ group boxes → lines by y-midpoint │
                        │ returns: text, mean_conf          │
                        └──────────────┬───────────────────┘
                                       │
                                       ▼
                        ┌────────────────────────────────┐
                        │ Free EasyOCR VRAM                │
                        │ del reader, gc.collect(),         │
                        │ torch.cuda.empty_cache()          │
                        └──────────────┬───────────────────┘
                                       │
                                       ▼
                              ┌───────────────┐
                              │ USE_QWEN?      │
                              └───┬───────┬───┘
                            No    │       │  Yes
                    ┌─────────────┘       └─────────────┐
                    ▼                                     ▼
        ┌────────────────────────┐        ┌───────────────────────────┐
        │ Write raw EasyOCR text  │        │  num_gpus >= 2             │
        │ doc_name.md              │        │  and DUAL_GPU_QWEN?        │
        │ (EasyOCR only)            │        └──────┬─────────────┬─────┘
        └───────────┬─────────────┘             Yes │             │ No
                    │                                 ▼             ▼
                    │                    ┌─────────────────┐  ┌─────────────────────┐
                    │                    │ Split page_jobs   │  │ Single GPU / CPU      │
                    │                    │ at midpoint        │  │ extract_pdf_to_       │
                    │                    └───┬───────────┬──┘  │ markdown_single_gpu   │
                    │                        │           │      └───────────┬──────────┘
                    │                        ▼           ▼                  │
                    │         ┌───────────────────┐ ┌───────────────────┐   │
                    │         │ Thread: cuda:0      │ │ Thread: cuda:1      │   │
                    │         │ _gpu_worker_thread   │ │ _gpu_worker_thread   │   │
                    │         │ (left pages)          │ │ (right pages)         │   │
                    │         └──────────┬─────────┘ └──────────┬─────────┘   │
                    │                    │                       │             │
                    │                    └───────────┬───────────┘             │
                    │                                │                          │
                    │                                ▼                          │
                    │              ┌──────────────────────────────────┐        │
                    │              │ QWEN_PROMPT_STRUCTURAL_REFINE      │◄───────┘
                    │              │ page image + EasyOCR text          │
                    │              │ → restructured Markdown             │
                    │              └───────────────┬────────────────────┘
                    │                              │
                    │                              ▼
                    │                    ┌───────────────────┐
                    │                    │ Qwen call succeeded?│
                    │                    └───┬─────────────┬──┘
                    │                    No   │             │  Yes
                    │              ┌──────────┘             └──────────┐
                    │              ▼                                    ▼
                    │   ┌──────────────────────┐          ┌──────────────────────────┐
                    │   │ Fallback: use raw       │          │ Restructured Markdown      │
                    │   │ EasyOCR text for page    │          │ (headings, tables, lists;   │
                    │   │                            │          │  original Bangla spelling)  │
                    │   └───────────┬────────────┘          └──────────────┬────────────┘
                    │              └────────────────┬──────────────────────┘
                    │                               ▼
                    │                  ┌───────────────────────────┐
                    │                  │ Merge per-page results       │
                    │                  │ sorted by page number         │
                    │                  └──────────────┬────────────────┘
                    │                                 ▼
                    │                  ┌───────────────────────────┐
                    │                  │ Cleanup temp page PNGs       │
                    │                  │ _cleanup_page_images          │
                    │                  └──────────────┬────────────────┘
                    │                                 ▼
                    │                  ┌───────────────────────────┐
                    │                  │ Write final Markdown          │
                    │                  │ doc_name_dual_gpu.md          │
                    │                  │ or doc_name.md                 │
                    │                  └──────────────┬────────────────┘
                    │                                 │
                    └─────────────────┬────────────────┘
                                       ▼
                        ┌────────────────────────────────┐
                        │ Sanity check cell                 │
                        │ read first .md, print preview       │
                        └────────────────────────────────┘
```

---

## 2. Alternate driver: single-pass, region-routed OCR

`extract_pdf_to_markdown()` / `process_scanned_page()` — defined in `ocr_helper.py` but **not** the path `main()` actually calls. Documented since it may be swapped in.

```
                              ┌───────────────────────┐
                              │  Page from PDF          │
                              └───────────┬────────────┘
                                         ▼
                          ┌──────────────────────────────┐
                          │ Native text layer present?      │
                          │ chars > 50 and has fonts?         │
                          └───────┬───────────────┬────────┘
                             Yes  │                │  No
                    ┌─────────────┘                └─────────────┐
                    ▼                                              ▼
      ┌───────────────────────────┐                ┌───────────────────────────┐
      │ Use PDF text layer directly │                │ process_scanned_page        │
      └────────────┬────────────────┘                └────────────┬────────────────┘
                    ▼                                              ▼
      ┌───────────────────────────┐                ┌───────────────────────────────┐
      │ detect_image_regions         │                │ detect_image_regions +          │
      │ (embedded images on page)     │                │ detect_table_regions             │
      └────────────┬────────────────┘                │ (OpenCV ruling-line detection)   │
                    ▼                                 └────────────┬──────────────────────┘
      ┌───────────────────────────┐                                ▼
      │ OCR each embedded image      │                ┌───────────────────────────────┐
      │ with Qwen map prompt           │                │ Drop table regions overlapping   │
      └────────────┬────────────────┘                │ image regions                     │
                    ▼                                 └────────────┬──────────────────────┘
      ┌───────────────────────────┐                                ▼
      │ Append as                    │                ┌───────────────────────────────┐
      │ "Embedded Figure/Map"          │                │ merge_overlapping_boxes           │
      └────────────┬────────────────┘                └────────────┬──────────────────────┘
                    │                                              │
                    │                    ┌─────────────────────────┼─────────────────────────┐
                    │                    ▼                         ▼                          ▼
                    │      ┌──────────────────────┐ ┌──────────────────────┐  ┌──────────────────────────┐
                    │      │ OCR each map region     │ │ OCR each table region  │  │ process_prose_full_page    │
                    │      │ Qwen, dedupe lines        │ │ Qwen Markdown table      │  │ hybrid EasyOCR + escalation │
                    │      └──────────┬─────────────┘ │ or EasyOCR                │  └──────────────┬────────────┘
                    │                │                 └──────────┬─────────────┘                 │
                    │                │                            │                                ▼
                    │                │                            │              ┌───────────────────────────────┐
                    │                │                            │              │ mean_conf < 0.70 or               │
                    │                │                            │              │ garbage_ratio > 0.15 ?             │
                    │                │                            │              └───────┬───────────────┬────────┘
                    │                │                            │                  Yes  │               │  No
                    │                │                            │            ┌───────────┘               └───────────┐
                    │                │                            │            ▼                                       ▼
                    │                │                            │  ┌──────────────────────┐              ┌──────────────────────┐
                    │                │                            │  │ Escalate to Qwen          │              │ Keep EasyOCR text        │
                    │                │                            │  │ (prose prompt)              │              │                            │
                    │                │                            │  └──────────┬─────────────┘              └──────────┬─────────────┘
                    │                │                            │            └────────────────┬───────────────────────┘
                    └────────────────┴────────────────────────────┴─────────────────────────────┼──────────────────────┐
                                                                                                  ▼                      │
                                                                                    ┌───────────────────────────┐       │
                                                                                    │ Assemble page Markdown       │◄──────┘
                                                                                    │ Figure/Map + Table +          │
                                                                                    │ Body Text sections             │
                                                                                    └──────────────┬────────────────┘
                                                                                                   ▼
                                                                                    ┌───────────────────────────┐
                                                                                    │ Append to output .md          │
                                                                                    │ every 5 pages: gc +            │
                                                                                    │ empty_cache                     │
                                                                                    └───────────────────────────┘



# History/Bhugol RAG Pipeline — Full Diagram

Plain-text diagram covering both scripts: `build_rag_index.py` (indexing) and `rag.py` (query time), backed by shared logic in `rag_core.py`.

---

## 1. Index building — `build_rag_index.py`

```
                        ┌───────────────────────────────┐
                        │  OCR Markdown source(s)          │
                        │  --source (default: History +      │
                        │  Bhugol .md files)                  │
                        └───────────────┬─────────────────┘
                                        ▼
                        ┌───────────────────────────────┐
                        │  Validate sources                 │
                        │  - all files exist                 │
                        │  - no duplicate paths                │
                        │  - chunk_overlap < chunk_size        │
                        └───────────────┬─────────────────┘
                                        ▼
                    ┌───────────────────────────────────────┐
                    │  For each source:                        │
                    │  parse_page_markdown (rag_core)             │
                    │  split on "## Page N" headings,              │
                    │  never cross page boundaries                  │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  infer_document_identity (rag_core)       │
                    │  filename → document_id + title             │
                    │  ("bhugol" / "history" / fallback slug)       │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  Build document metadata                  │
                    │  id, title, source path, sha256,             │
                    │  page count, chunk count (init 0)             │
                    │  reject duplicate document_id                 │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  page_entries: (document, page, text)      │
                    │  flattened across all source books           │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  Chonkie TokenChunker                      │
                    │  tokenizer = embedding_model,                │
                    │  chunk_size=384, chunk_overlap=64             │
                    │  chunk_batch() over all page texts             │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  Build chunk records                       │
                    │  chunk_id, document_id/title, page,          │
                    │  page_chunk, text, token_count,                │
                    │  start_index, end_index, source                │
                    │  skip empty text; increment document.chunks    │
                    └───────────────┬───────────────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │  Any records produced?  │
                        └───┬───────────────┬───┘
                        No  │               │  Yes
                            ▼               ▼
                ┌────────────────┐  ┌───────────────────────────────┐
                │  raise            │  │  Load SentenceTransformer         │
                │  RuntimeError      │  │  (BAAI/bge-m3), fp16 on CUDA        │
                └────────────────┘  └───────────────┬───────────────────┘
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │  embedder.encode_document()       │
                                    │  batch_size, normalize_embeddings,  │
                                    │  → float32 numpy array               │
                                    └───────────────┬───────────────────┘
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │  Validate embedding shape          │
                                    │  ndim==2, rows==len(records)         │
                                    └───────────────┬───────────────────┘
                                                    ▼
                                    ┌───────────────────────────────┐
                                    │  faiss.IndexFlatIP                 │
                                    │  add(embeddings)  (exact cosine,     │
                                    │  since vectors are normalized)        │
                                    └───────────────┬───────────────────┘
                                                    ▼
                    ┌────────────────────────────────────────────────┐
                    │  Write outputs to --index-dir                       │
                    │  - index.faiss        (FAISS index)                   │
                    │  - chunks.jsonl       (one JSON record per chunk)       │
                    │  - manifest.json      (format_version, documents,        │
                    │                        chunker/model/device config,       │
                    │                        embedding_dimension, etc.)          │
                    └───────────────────────┬────────────────────────────┘
                                            ▼
                                ┌───────────────────────┐
                                │  del embedder,           │
                                │  release_cuda_memory()     │
                                └───────────────────────┘
```

---

## 2. Query time — `rag.py` + `HistoryRetriever` (`rag_core.py`)

```
                        ┌───────────────────────────────┐
                        │  CLI args                          │
                        │  query, --top-k=5, --candidate-k=24, │
                        │  --device, --reranker-model,           │
                        │  --generator-model, --no-rerank,        │
                        │  --no-generate, --interactive             │
                        └───────────────┬─────────────────┘
                                        ▼
                    ┌───────────────────────────────────────┐
                    │  HistoryRetriever.__init__                │
                    │  - resolve_device (auto/cpu/cuda)            │
                    │  - load_manifest(index_dir)                    │
                    │  - load chunks.jsonl → self.records             │
                    │  - faiss.read_index(index.faiss)                  │
                    │  - verify index.ntotal == len(records)              │
                    │  - build BM25Okapi over lexical_tokens(text)          │
                    │    (self._bm25, in-memory, always built eagerly)      │
                    └───────────────┬───────────────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │  query text supplied?    │
                        └───┬───────────────┬───┘
                     Single │               │  --interactive
                            ▼               ▼
              ┌───────────────────┐  ┌───────────────────────────┐
              │  answer_one(query)   │  │  loop: read line from stdin  │
              └─────────┬─────────┘  │  until blank / EOF, call        │
                        │            │  answer_one() per question         │
                        │            └───────────────┬───────────────────┘
                        └───────────────┬─────────────┘
                                        ▼
========================================│===============================
 HistoryRetriever.search()
========================================│===============================
                                        ▼
                    ┌───────────────────────────────────────┐
                    │  Validate query + top_k                    │
                    │  candidate_k = clamp(candidate_k,             │
                    │                       top_k, len(records))     │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  Dense retrieval                           │
                    │  _load_embedder() (lazy, cached)             │
                    │  embedder.encode_query([query])                │
                    │  faiss index.search → dense_ids, dense_scores    │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  Lexical retrieval                         │
                    │  lexical_tokens(query)                        │
                    │  self._bm25.get_scores()                        │
                    │  top candidate_k by score (if any nonzero)        │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  Reciprocal Rank Fusion (_rrf_add)          │
                    │  combine dense_ranked + bm25_ranked            │
                    │  score += 1 / (60 + rank) per list               │
                    │  fused_ids = top candidate_k by fusion_score      │
                    └───────────────┬───────────────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │  use_reranker?           │
                        └───┬───────────────┬───┘
                        No  │               │  Yes
                            ▼               ▼
              ┌─────────────────────┐ ┌───────────────────────────────┐
              │  final_ids =            │ │  _load_reranker() (lazy, cached)  │
              │  fused_ids[:top_k]        │ │  CrossEncoder                       │
              └─────────┬───────────┘ │  (BAAI/bge-reranker-v2-m3)            │
                        │             │  predict(query, chunk_text) pairs        │
                        │             └───────────────┬───────────────────────┘
                        │                             ▼
                        │             ┌───────────────────────────────┐
                        │             │  final_ids = fused_ids sorted        │
                        │             │  by rerank_score, top_k               │
                        │             └───────────────┬───────────────────┘
                        └─────────────────┬─────────────┘
                                          ▼
                    ┌───────────────────────────────────────┐
                    │  Build SearchResult list                   │
                    │  chunk_id, document_id/title, source,         │
                    │  page, page_chunk, text, token_count,           │
                    │  dense/bm25/fusion/rerank scores                  │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  print_results()                           │
                    │  --json → JSON dump                            │
                    │  else  → ranked text with scores                │
                    └───────────────┬───────────────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │  --no-generate set?      │
                        └───┬───────────────┬───┘
                        Yes │               │  No
                            ▼               ▼
                ┌────────────────┐  ┌───────────────────────────────┐
                │  stop, print       │  │  retriever.release_models()      │
                │  results only       │  │  drop embedder + reranker,          │
                └────────────────┘  │  release_cuda_memory()                │
                                    └───────────────┬───────────────────┘
                                                    ▼
========================================│===============================
 generate_answer() (rag_core)
========================================│===============================
                                        ▼
                    ┌───────────────────────────────────────┐
                    │  resolve_device; require CUDA               │
                    │  (raises if resolved to "cpu")                │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  Load Qwen3.5-4B                           │
                    │  Qwen3_5ForConditionalGeneration              │
                    │  fp16, device_map="auto"                        │
                    │  _tie_generator_weights() (lm_head fix)           │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  build_grounded_prompt(query, results)      │
                    │  numbered [Context N | Title | Page |         │
                    │  Chunk] blocks, citation rules,                 │
                    │  valid citation labels listed                    │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  apply_chat_template                       │
                    │  (enable_thinking=False)                       │
                    │  → processor → inputs (fp16, target device)      │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  model.generate()                          │
                    │  greedy (do_sample=False),                    │
                    │  repetition_penalty=1.05,                       │
                    │  max_new_tokens (default 512)                     │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  Decode + clean                            │
                    │  batch_decode new tokens only                 │
                    │  strip <think>...</think> blocks                │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  normalize_answer_citations()               │
                    │  match [Title?, Page N] patterns,             │
                    │  resolve ambiguous/missing titles using         │
                    │  page → title map from retrieved results         │
                    └───────────────┬───────────────────────────┘
                                    ▼
                    ┌───────────────────────────────────────┐
                    │  del model/inputs/generated,                │
                    │  release_cuda_memory()                        │
                    └───────────────┬───────────────────────────┘
                                    ▼
                        ┌───────────────────────┐
                        │  Print grounded answer    │
                        └───────────────────────┘
