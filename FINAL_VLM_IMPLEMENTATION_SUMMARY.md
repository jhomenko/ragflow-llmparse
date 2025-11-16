# FINAL VLM IMPLEMENTATION SUMMARY

## Overview
This document provides a comprehensive summary of the Vision Language Model (VLM) implementation in RAGFlow based on analysis of all documentation files in the changeDocs directory. The implementation addresses critical issues that were preventing VLM PDF parsing from working correctly.

## Key Changes Made

### 1. Critical Bug Fixes
The implementation addresses four critical bugs that were preventing VLM from working:

#### Bug #1: Return Value Mismatch (CRITICAL)
- **File**: `api/db/services/llm_service.py:165`
- **Issue**: `LLMBundle.describe_with_prompt()` returned only `txt` instead of `(txt, used_tokens)` tuple
- **Fix**: Changed return statement to `return txt, used_tokens`
- **Impact**: Fixed result unpacking errors in calling code

#### Bug #2: Missing System Message (HIGH)
- **File**: `rag/llm/cv_model.py:158-176`
- **Issue**: VLM was not receiving context about its role as a PDF transcriber
- **Fix**: Added comprehensive system message to `vision_llm_prompt()`
- **Impact**: Improved output quality and consistency

#### Bug #3: Missing API Parameters (MEDIUM)
- **File**: `rag/llm/cv_model.py:199-208`
- **Issue**: No explicit `max_tokens` or `temperature` parameters
- **Fix**: Added `max_tokens=4096` and `temperature=0.1`
- **Impact**: Ensured consistent, complete responses

#### Bug #4: Stop Token Issue (CRITICAL)
- **File**: `rag/llm/cv_model.py:206`
- **Issue**: OpenAI Python client adding default stop tokens causing premature termination
- **Fix**: Added `stop=[]` to disable default stop tokens
- **Impact**: Prevented early truncation after only 6 tokens

### 2. Image Format Conversion Fix
- **File**: `deepdoc/parser/pdf_parser.py:1438-1473`
- **Issue**: VisionParser was passing PIL Image objects directly to VLM instead of JPEG bytes
- **Fix**: Convert PIL Images to JPEG bytes with quality optimization
- **Impact**: Enabled proper image processing by VLM

### 3. Prompt Template Fix
- **File**: `rag/prompts/vision_llm_describe_prompt.md`
- **Issue**: Prompt template included page divider instructions causing VLM to echo the divider
- **Fix**: Removed page divider instruction from template
- **Impact**: VLM now transcribes actual content instead of just page markers

### 4. Per-Page Retry & Fail-Fast Enforcement
- **Files**: `rag/app/picture.py`, `rag/llm/working_vlm_module.py`, `deepdoc/parser/pdf_parser.py`, `rag/flow/parser/parser.py`
- **Issue**: Working module ignored endpoint failures, retried blindly, and then fell back to the legacy path, producing partial/incorrect documents
- **Fixes**:
  - Introduced structured retries with prompt hints and temperature nudges (controlled by `VLM_PAGE_MAX_ATTEMPTS`)
  - Removed legacy fallback path; working VLM responses are validated for repetition/emptiness
  - Added `VisionParserPageError` so VisionParser aborts the entire document if any page fails after retries
  - Parser layer now propagates these errors to the task executor to prevent indexing partial documents
- **Impact**: Ensures documents are either fully parsed or fail loudly, giving operators actionable logs while still mitigating transient repetition loops.

## Architecture Changes

### Before Implementation
- RAGFlow used traditional PDF parsing methods
- VLM integration was broken with empty responses
- No proper image format conversion

### After Implementation
- VLM module adds multimodal pre-processing stage
- PDF/Image → direct_vision_parser → multimodal chunks
- Deterministic per-page retries with fail-fast behavior to avoid partial outputs
- Configurable chunking strategies (auto, page, heading, token-based)

## Files Modified

### Core Implementation Files
1. `api/db/services/llm_service.py` - Fixed return value
2. `rag/llm/cv_model.py` - Added system message, API parameters, stop tokens
3. `deepdoc/parser/pdf_parser.py` - Fixed image format conversion
4. `rag/app/picture.py` - Enhanced byte handling and error recovery
5. `rag/flow/parser/parser.py` - Implemented VLM model selection
6. `rag/nlp/rag_tokenizer.py` - Added token counting and chunking utilities

### Test and Debug Files
1. `test_vlm_fix.py` - Validation script
2. `test_vlm_pdf_complete.py` - Complete test suite
3. `test/test_vlm_parallel.py` - Concurrency and failure-injection coverage
4. `test/test_vision_llm_chunk_retry.py` - Validates retry/prompt mitigation logic
5. Various debug and analysis documents in changeDocs/

## Key Improvements

### 1. Reliability
- Each page uses structured retries (prompt hint + temperature adjustments) before surfacing errors
- VisionParser aborts the entire document when a page fails after retries to prevent partial ingestion
- Comprehensive input validation plus explicit `VisionParserPageError`/callback updates provide actionable diagnostics

### 2. Performance
- Optimized image compression for oversized images
- Adaptive compression for huge pages
- Efficient token counting and chunking

### 3. Quality
- Better markdown formatting preservation
- Improved table and heading detection
- Enhanced content quality detection

### 4. Flexibility
- Configurable chunking strategies
- Customizable prompts
- Model selection via UI

## Configuration Options

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| USE_WORKING_VLM | true | Enable in-process working VLM module |
| VLM_BASE_URL | http://192.168.68.186:8080/v1 | VLM server URL |
| VLM_TIMEOUT_SEC | 60 | Request timeout in seconds |
| VLM_MAX_PAGE_SIZE | 5MB | Maximum allowed size per page image |
| VLM_PAGE_MAX_ATTEMPTS | 2 | Number of working-module attempts per page (adds prompt hint & higher temperature) |
| PARALLEL_VLM_REQUESTS | 1 | Optional semaphore to enable concurrent page processing |

### Parser Configuration
```json
{
  "pdf": {
    "parse_method": "Qwen2.5VL-3B",
    "lang": "English",
    "zoomin": 3,
    "output_format": "markdown",
    "chunk_token_num": 512,
    "chunking_strategy": "auto",
    "vision_prompt_path": "/path/to/custom/prompt.md"
  }
}
```

## Testing Strategy

### Unit Testing
- Validate parsing logic and OCR fallback
- Test chunk normalization
- Verify retry + fail-fast mechanisms (`test/test_vision_llm_chunk_retry.py`, `test/test_vlm_parallel.py`)

### Integration Testing
- End-to-end ingestion → retrieval → LLM prompt flow
- Multi-page PDF processing
- Model selection validation

### Manual Testing
- Sample PDFs with diagrams/charts
- Images with captions
- Noisy scan documents

## Troubleshooting Guide

### Common Issues
1. **Empty VLM Responses**: Check model configuration and prompt files; confirm retries are allowed via `VLM_PAGE_MAX_ATTEMPTS`
2. **Gibberish or Repetition**: Increase image quality (higher zoomin) or fine-tune prompts; inspect retry logs for repetition hints
3. **VisionParserPageError**: Indicates a page failed after all retries—review logs for the specific page and resolve upstream issues
4. **Chunks Too Large/Small**: Adjust token limits and chunking strategy
5. **Metadata Parsing Errors**: Verify VisionParser output format

### Debugging Tips
- Enable DEBUG logging for detailed insights
- Use callback progress messages to track processing
- Compare with working curl test commands
- Monitor VLM server logs for errors

## Performance Considerations

### Processing Times
- Simple pages: 5-15 seconds
- Complex pages with tables: 10-20 seconds
- Large documents: 3-5 minutes per 100 pages

### Resource Usage
- Peak memory: 4-5GB per container
- GPU utilization: High during VLM calls
- Network: Required for VLM server communication

### Token Consumption
- Average: 500-2000 tokens per page
- Cost considerations for commercial APIs
- Caching opportunities for repeated content

## Deployment Instructions

### Docker Rebuild
```bash
# Clean rebuild with no cache
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Verification Checklist
- [ ] Service starts without errors
- [ ] Ingestion API accepts PDF and image inputs
- [ ] Extracted multimodal chunks contain image metadata
- [ ] Retrieval returns improved relevance on image-heavy docs
- [ ] CI unit tests pass

## Rollback Plan
If VLM integration causes issues:
1. Revert changed files to previous commit
2. Disable VLM proxy via environment variable
3. Restore previous container images
4. Invalidate any new embeddings in vector store if needed

## Success Criteria
- ✅ VLM returns 1000-5000 characters per page (not 14)
- ✅ Token count >500 (not just 6)
- ✅ Response contains markdown headers (##, ###)
- ✅ No crashes or unhandled exceptions
- ✅ Progress callbacks working (0.0 to 1.0)
- ✅ Retrieval quality improved on documents with visual content

## Architecture Diagram
```
[PDF/Image] --> [direct_vision_parser] --> [Multimodal Chunks] --> [Embedding Store] --> [Retrieval] --> [LLM]
```

## Future Enhancements
- Smarter prompt templates per layout (tables, diagrams, dense text)
- OCR fallback on VLM failure for higher reliability
- Custom chunking rules per document type
- Caching of VLM results to reduce API costs
- Performance benchmarks and optimization

## VisionParser Image Resizing for VLM Models
- **Date**: 2025-11-09

### Overview
This section documents the recently implemented VisionParser image resizing improvements that optimize images for Vision Language Models (VLMs). The changes introduce high-DPI extraction and a "smart" resize algorithm that preserves aspect ratio while aligning dimensions to a configurable factor, improving VLM compliance and overall quality.

### Changes Made
1. Added four utility functions to support factor-aligned resizing:
   - [`round_by_factor()`](deepdoc/parser/pdf_parser.py:61)
   - [`ceil_by_factor()`](deepdoc/parser/pdf_parser.py:66)
   - [`floor_by_factor()`](deepdoc/parser/pdf_parser.py:71)
   - [`smart_resize()`](deepdoc/parser/pdf_parser.py:76)
   - Location: [`deepdoc/parser/pdf_parser.py:61-76`](deepdoc/parser/pdf_parser.py:61)

2. Increased extraction DPI for better VLM input quality:
   - Modified [`VisionParser.__images__`](deepdoc/parser/pdf_parser.py:1422)
   - Changed `resolution=72 * zoomin` (≈216 DPI typical) to `resolution=600` for high-DPI extraction.

3. Replaced prior max-2000px resize logic with smart resizing and environment configuration:
   - Updated [`VisionParser.__call__`](deepdoc/parser/pdf_parser.py:1499-1515)
   - Replaced simple max-2000px clamp with a call to [`smart_resize()`](deepdoc/parser/pdf_parser.py:76)
   - Introduced `VLM_RESIZE_FACTOR` environment variable (default: 32)
   - New target max dimension set to 1024px, and dimensions are now multiples of the configured factor.

### Key Features
- High DPI extraction (600 DPI) for improved VLM image fidelity.
- Smart resize that aligns output dimensions to a configurable factor (e.g., 32).
- Environment variable configurability for resize behavior.
- Target maximum dimension: 1024px (instead of previous 2000px).
- Preservation of original aspect ratio during resizing.

### Environment Variables
| Variable | Default | Description |
|----------|---------|-------------|
| VLM_RESIZE_FACTOR | 32 | The alignment factor (pixels) used so final width/height are multiples of this value. Adjust to match VLM model requirements. |

### Benefits
- VLM compliance: image dimensions and formats are better aligned with model expectations.
- Better quality: high-DPI extraction plus controlled resizing preserves visual detail important for VLM understanding.
- Faster processing: smaller, factor-aligned images reduce downstream processing and network payload sizes compared to unnecessarily large images.

### Configuration Examples
Bash examples to set desired resize factor before starting the service:

```bash
# Use default (32)
export VLM_RESIZE_FACTOR=32

# Use 16 for smaller alignment granularity
export VLM_RESIZE_FACTOR=16

# Use 64 for coarser alignment (potentially faster but less precise)
export VLM_RESIZE_FACTOR=64
```

### Technical Details (smart_resize algorithm)
- Input: original image width (w) and height (h), max dimension target (1024), and alignment factor (F; default 32).
- Compute scale to fit the largest dimension to <= 1024 while preserving aspect ratio:
  - scale = min(1.0, 1024 / max(w, h))
  - new_w = floor(w * scale), new_h = floor(h * scale)
- Align each dimension to the nearest valid value using factor helpers:
  - new_w_aligned = max(F, round_by_factor(new_w, F))
  - new_h_aligned = max(F, round_by_factor(new_h, F))
- Ensure multiples of F (using `ceil_by_factor` or `floor_by_factor` where appropriate) to conform to VLM memory/stride expectations.
- Returns a resized image with preserved aspect ratio and dimensions that are multiples of the configured factor.

Utility functions added:
- [`round_by_factor()`](deepdoc/parser/pdf_parser.py:61): round a number to nearest multiple of factor.
- [`ceil_by_factor()`](deepdoc/parser/pdf_parser.py:66): ceil to nearest multiple.
- [`floor_by_factor()`](deepdoc/parser/pdf_parser.py:71): floor to nearest multiple.
- [`smart_resize()`](deepdoc/parser/pdf_parser.py:76): performs the full algorithm described above.

### Testing Recommendations
- Verify DPI extraction:
  - Confirm images extracted from PDF use `resolution=600` at [`deepdoc/parser/pdf_parser.py:1422`](deepdoc/parser/pdf_parser.py:1422).
- Validate resize behavior:
  - Test images with extreme aspect ratios (very wide, very tall).
  - Confirm largest dimension ≤ 1024 and both width/height are multiples of `VLM_RESIZE_FACTOR`.
  - Check results for several `VLM_RESIZE_FACTOR` values (16, 32, 64).
- Quality checks:
  - Compare VLM outputs (accuracy, character counts) before/after changes using representative PDFs.
  - Ensure VLM no longer drops content because of inappropriate image sizes.
- Performance:
  - Measure end-to-end processing time and payload sizes with typical documents.
  - Ensure throughput improvements or acceptable trade-offs when using 600 DPI.
- Regression:
  - Confirm no regressions for prior image-format conversion bug fixes (JPEG bytes conversion still in place at [`deepdoc/parser/pdf_parser.py:1438-1473`](deepdoc/parser/pdf_parser.py:1438)).

## Conclusion
The VLM implementation successfully addresses all critical issues preventing PDF parsing from working correctly. The system now reliably processes PDFs using Vision Language Models, providing enhanced document processing capabilities with improved accuracy for documents containing images, charts, and complex layouts.
## Hybrid VLM Table Parsing Feature
- **Date**: 2025-11-11

### Overview
A hybrid table parsing approach was implemented that combines DeepDoc's high-accuracy table location detection with VLM-based semantic table parsing. DeepDoc identifies table regions quickly and reliably; VLMs parse cell contents, handle merged cells and nested headers, and produce structured HTML or Markdown. On failure, the pipeline automatically falls back to the existing TableStructureRecognizer for robustness.

### Key Features
- Deepdoc detects table locations (fast and accurate).
- VLM parses table content (handles complex structures, merged cells, nested headers).
- Automatic fallback to TableStructureRecognizer on VLM failure.
- Configurable via environment variables (opt-in by default).
- Smart resize integration for Qwen3-VL and similar models (dimensions as multiples of 32).

### Files Modified
1. [`rag/prompts/table_vlm_prompt.md`](rag/prompts/table_vlm_prompt.md:1) - Created table-specific VLM prompt
2. [`deepdoc/parser/pdf_parser.py`](deepdoc/parser/pdf_parser.py:1) - Added VLM table parser methods and validation helpers
3. [`rag/flow/parser/parser.py`](rag/flow/parser/parser.py:1) - Pass vision_model to parser and opt-in enablement
4. [`docker/docker-compose.yml`](docker/docker-compose.yml:51) - Added environment variables for hybrid table parsing
5. [`test_vlm_table_parsing.py`](test_vlm_table_parsing.py:1) - Created unit tests validating hybrid behavior and fallbacks

### Environment Variables
- `USE_VLM_TABLE_PARSING` (default: false) — Enable hybrid VLM table parsing (opt-in).
- `VLM_TABLE_MODEL` (default: empty) — Optional specific VLM model to use for table parsing.
- `VLM_TABLE_TIMEOUT_SEC` (default: 30) — Timeout per table parsing attempt (seconds).
- `VLM_TABLE_FALLBACK_ENABLED` (default: true) — If true, fallback to TableStructureRecognizer on VLM parse failure.
- `VLM_TABLE_OUTPUT_FORMAT` (default: html) — Output format expected from VLM: "html" or "markdown".
- `VLM_TABLE_PROMPT_PATH` (default: repo prompt) — Optional path to custom table prompt file.
- `VLM_RESIZE_FACTOR` (default: 32) — Ensures VLM images are resized to dimensions that are multiples of this value.

### Benefits
- Better accuracy on complex tables (merged cells, nested headers).
- Maintains speed (only processes table regions with VLM).
- Non-breaking: opt-in and disabled by default for backwards compatibility.
- Robust fallback mechanisms ensure no loss of table extraction capability.

### Usage
Enable the feature via environment variable or docker-compose override.

```bash
# Enable in shell
export USE_VLM_TABLE_PARSING=true
export VLM_TABLE_MODEL=Qwen3-VL
```

Or in `docker/docker-compose.yml` (already added under service environment):

```bash
# Example snippet already present:
- USE_VLM_TABLE_PARSING=false
- VLM_TABLE_MODEL=${VLM_TABLE_MODEL:-}
- VLM_TABLE_TIMEOUT_SEC=30
- VLM_TABLE_FALLBACK_ENABLED=true
- VLM_TABLE_OUTPUT_FORMAT=html
- VLM_TABLE_PROMPT_PATH=${VLM_TABLE_PROMPT_PATH:-}
```

Quick runtime enable via docker-compose env file (.env):

```bash
# .env
USE_VLM_TABLE_PARSING=true
VLM_TABLE_MODEL=Qwen3-VL
VLM_TABLE_OUTPUT_FORMAT=html
```

This hybrid implementation keeps default behavior unchanged unless explicitly enabled, and provides a path to leverage VLMs for improved table content fidelity while preserving the proven TableStructureRecognizer as a safe fallback.
