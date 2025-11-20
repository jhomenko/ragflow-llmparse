# VLM Implementation Summary

## Overview
RAGFlow now supports two complementary Vision-Language workflows:

1. **Full-page VLM parsing** – every PDF page is converted to an image and transcribed by the working VLM module. This path is ideal for user guides, flowcharts, or documents whose layout repeatedly confuses classic parsers.
2. **Hybrid DeepDoc + VLM tables/figures** – DeepDoc continues to handle page layout, OCR, and chunk metadata, while only the hard regions (tables, charts, infographics) are re-written by the VLM. This keeps DeepDoc’s speed/positional accuracy and upgrades complex visuals.

Both flows share the same retry logic, prompts, and telemetry so operators get consistent output and diagnostics regardless of which strategy is active.

## Core Improvements

### Working VLM Module
- `rag/llm/working_vlm_module.py`, `rag/app/picture.py`
- Provides the single gateway for all VLM calls (full-page parsing, tables, figures).
- Adds configurable retries with prompt hints via `VLM_PAGE_MAX_ATTEMPTS`.
- Normalizes responses, strips fences, logs suspicious repetitions, and surfaces exceptions up the stack so ingestion either succeeds completely or fails loudly.

### Hybrid Table + Figure Flow
- `deepdoc/parser/pdf_parser.py`
- DeepDoc extracts table regions and, when `USE_VLM_TABLE_PARSING=true`, routes each cropped table through `_vlm_table_parser`.
- Concurrency is governed by `PARALLEL_VLM_TABLE_REQUESTS`; failures fall back to the original `TableStructureRecognizer`.
- VLM responses are wrapped back into DeepDoc’s bbox schema (page number, coordinates, layout type) so downstream chunking, highlighting, and UI rendering continue to work with HTML.
- Figures leverage `VisionFigureParser` with the refreshed `vision_llm_figure_describe_prompt.md`, producing compact Markdown summaries that can be indexed for retrieval.

### Prompt Updates
- `rag/prompts/vision_llm_table_prompt.md` – standalone user prompt guaranteeing compact `<table>` HTML with proper `rowspan/colspan`.
- `rag/prompts/vision_llm_figure_describe_prompt.md` – new instructions for Markdown summaries that capture labels, sequences, and captions with chunk-friendly spacing.

## Files & Modules

| Area | Key Files | Notes |
|------|-----------|-------|
| Hybrid tables | `deepdoc/parser/pdf_parser.py`, `rag/prompts/vision_llm_table_prompt.md` | Enables selective VLM rewriting with fallback + concurrency |
| Figures | `deepdoc/parser/figure_parser.py`, `rag/prompts/vision_llm_figure_describe_prompt.md` | Uses the working VLM prompt for diagram descriptions |
| Full-page parsing | `rag/app/picture.py`, `rag/llm/working_vlm_module.py`, `rag/app/picture.py` | Retry/validation logic shared across entry points |
| Parser wiring | `rag/flow/parser/parser.py` | Creates `LLMBundle` for hybrid mode and attaches it to `RAGFlowPdfParser` |
| Documentation | `FINAL_VLM_IMPLEMENTATION_SUMMARY.md`, `VLM_IMPLEMENTATION_SUMMARY.md` | Updated to describe both flows and configuration |

## Configuration

| Variable | Description | Default |
|----------|-------------|---------|
| `USE_WORKING_VLM` | Enable in-process working VLM module for all requests | `true` |
| `VLM_BASE_URL` | Base URL for VLM server | `http://192.168.68.186:8080/v1` |
| `VLM_PAGE_MAX_ATTEMPTS` | Attempts per page (retry adds repetition hint & warmer temperature) | `2` |
| `USE_VLM_TABLE_PARSING` | Enable hybrid table rewriting inside DeepDoc | `false` |
| `VLM_TABLE_MODEL` | Optional override for the LLMBundle used in hybrid mode (falls back to attached `vision_model`) | unset |
| `VLM_TABLE_TIMEOUT_SEC` | Soft timeout per table when contacting the VLM | unset |
| `PARALLEL_VLM_TABLE_REQUESTS` | Max concurrent hybrid table requests | `1` |
| `VLM_TABLE_OUTPUT_FORMAT` | `html` or `markdown` output for tables (UI expects HTML) | `html` |
| `PARALLEL_VLM_REQUESTS` | Controls VisionParser semaphore for full-page parsing | `1` |

## Data Flow Summary

1. **DeepDoc-only** (`parse_method="deepdoc"`, `USE_VLM_TABLE_PARSING=false`): standard DeepDoc pipeline produces HTML snippets per layout block. HTML is stored as `content_with_weight` and rendered directly in the UI.
2. **Hybrid** (`parse_method="deepdoc"`, `USE_VLM_TABLE_PARSING=true`):
   - DeepDoc detects table coordinates → crops → `_vlm_table_parser` (working VLM) → validated HTML.
   - HTML replaces only the table nodes; everything else (text blocks, headings, positions) remains untouched.
   - If the VLM fails, TableStructureRecognizer’s output is used automatically.
3. **Full VLM** (`parse_method` set to a VLM model): VisionParser renders every page to JPEG and calls `vision_llm_chunk`. Markdown is stored as-is for retrieval; convert to HTML at render time if desired.

## Testing & Validation

- `test/test_vlm_parallel.py` – validates per-page concurrency, failure propagation, and cancellation handling.
- `test/test_vision_llm_chunk_retry.py` – ensures repetition detection triggers retries and raises structured errors when exhausted.
- `test_vlm_table_hybrid.py` – diagnostic harness to run DeepDoc with hybrid tables enabled against real PDFs.

Manual checks:
- Toggle `USE_VLM_TABLE_PARSING` to confirm DeepDoc chunks retain HTML and bounding boxes while tables become VLM HTML.
- Inspect logs for `_vlm_table_parser: table X processed` vs. fallback warnings to verify the retry/fallback flow.

## Operational Guidance

- Stick with DeepDoc-only for extremely high throughput scenarios with clean tables.
- Enable hybrid mode when colored headers, merged cells, or screenshots are common; it adds a small VLM cost only for affected tables.
- Use full VLM parsing when documents are predominantly visual or when DeepDoc frequently misses layout context (e.g., flowcharts).
- Remember UI rendering expects HTML. When storing Markdown (full VLM path), convert to HTML before display or adopt a Markdown renderer in the frontend.
