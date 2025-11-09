# VLM Implementation Summary

## Overview
This document provides a concise summary of the Vision Language Model (VLM) implementation in the RAGFlow codebase. The implementation enables PDF and image parsing using VLMs, providing enhanced document processing capabilities with improved accuracy for documents containing images, charts, and complex layouts.

## Key Changes Made

### Files Created
- `deepdoc/parser/direct_vision_parser.py` - New parser for processing images and visual inputs, extracting structured metadata and OCR/text regions
- `rag/llm/working_vlm_module.py` - Core VLM orchestration module handling ingestion, pre-processing, and model invocation
- `WORKING_VLM_IMPLEMENTATION_SUMMARY.md` - This summary document

### Files Modified
- `deepdoc/parser/pdf_parser.py` - Updated PDF parsing pipeline to call the new direct vision parser for embedded images and normalize extracted text chunks
- `rag/flow/parser/parser.py` - Adjusted flow parsing to accept multimodal chunk types and propagate image metadata into chunk descriptors
- `rag/nlp/rag_tokenizer.py` - Added tokenization handling for image-derived text and OCR noise normalization
- `api/db/services/llm_service.py` - Exposed new VLM invocation endpoints and wiring for model selection and request tracing
- `rag/app/picture.py` - Enhanced `vision_llm_chunk` with robust byte handling and error recovery
- `rag/llm/cv_model.py` - Added system message support, API parameters, and proper return value handling

## Critical Bugs Fixed

### 1. Return Value Mismatch (Critical)
- **Issue**: `LLMBundle.describe_with_prompt` returned only text string instead of (text, token_count) tuple
- **Location**: `api/db/services/llm_service.py:165`
- **Fix**: Modified to return `txt, used_tokens` tuple as expected by calling code
- **Impact**: Result unpacking now works correctly, preventing empty responses

### 2. Missing System Message (High)
- **Issue**: VLM was not receiving proper context about its role as a PDF transcriber
- **Location**: `rag/llm/cv_model.py:158-164`
- **Fix**: Added comprehensive system message to guide the VLM's behavior
- **Impact**: Improved output quality and consistency

### 3. Missing API Parameters (Medium)
- **Issue**: No explicit `max_tokens` or `temperature` parameters, causing unpredictable behavior
- **Location**: `rag/llm/cv_model.py:199-205`
- **Fix**: Added `max_tokens=4096` and `temperature=0.1` parameters
- **Impact**: Ensured consistent, complete responses

### 4. Stop Token Issue (Critical)
- **Issue**: OpenAI Python client adding default stop tokens causing premature termination
- **Location**: `rag/llm/cv_model.py:206`
- **Fix**: Added `stop=[]` to explicitly disable default stop tokens
- **Impact**: Prevented early truncation after only 6 tokens

### 5. Prompt Template Issue (Medium)
- **Issue**: Prompt template instructed VLM to add page dividers, causing it to output only "— Page 1 —"
- **Location**: `rag/prompts/vision_llm_describe_prompt.md`
- **Fix**: Removed page divider instruction from template
- **Impact**: VLM now transcribes actual content instead of just page markers

## Architecture Changes

### Before
- RAG pipeline treated inputs as text-first: PDF -> text extraction -> chunking -> embedding -> retrieval
- Image content embedded in PDFs was not fully parsed; OCR and visual features were separate utilities

### After
- VLM module adds a multimodal pre-processing stage: PDF/Image -> direct_vision_parser -> multimodal chunks (text + image metadata + visual features) -> unified chunking -> embedding/retrieval
- The pipeline is now:
  1. Ingest (PDF/Image)
  2. Visual/Text extraction via direct_vision_parser
  3. Multimodal chunk normalization
  4. Embedding + RAG retrieval
  5. Downstream LLM consumption

```
[PDF/Image] --> [direct_vision_parser] --> [Multimodal Chunks] --> [Embedding Store] --> [Retrieval] --> [LLM]
```

## Key Improvements and Benefits

- **Higher fidelity extraction**: Improved extraction of image-embedded text and visual context
- **Unified chunk representation**: Simplifies downstream RAG logic
- **Better retrieval relevance**: Enhanced results for documents with figures, charts, or screenshots
- **Clear module boundaries**: Enables independent testing and reuse
- **Configurable chunking strategies**: Auto, page, heading, and token-based strategies
- **Robust error handling**: Per-page error recovery with fallback content
- **Quality detection**: Identifies and flags potential gibberish or low-quality output

## Configuration

The VLM functionality can be configured through environment variables:

- `USE_WORKING_VLM` (default: true) - Enable the in-process working VLM module
- `VLM_BASE_URL` (default: http://192.168.68.186:8080/v1) - Base URL for VLM server
- `VLM_TIMEOUT_SEC` (default: 60) - Request timeout in seconds
- `VLM_MAX_PAGE_SIZE` (default: 5MB) - Maximum allowed size per page image
- `VLM_RETRY_COUNT` (default: 2) - Number of retries for transient errors

## Testing Strategy

- Unit tests: Validate parsing logic, OCR fallback, and chunk normalization
- Integration tests: End-to-end ingestion -> retrieval -> LLM prompt flow
- Manual tests: Sample PDFs with diagrams/charts, images with captions, and noisy scans

## Deployment Instructions

1. Ensure dependencies are installed (see `pyproject.toml` and container requirements)
2. Rebuild the backend container:
   ```bash
   # from repo root
   docker-compose build backend
   docker-compose up -d backend
   ```
3. Run migrations (if applicable) and restart services
4. For GPU/accelerated inference, use the GPU compose files: `docker/docker-compose-gpu.yml`

## Success Criteria

- VLM returns 1000-5000 characters per page (vs previous 14 characters)
- Token count is 500-2000+ (vs previous 6 tokens)
- Response contains full markdown transcription with proper formatting
- Chunks have actual content for RAG retrieval
- No regressions in text-only document handling
- Retrieval relevance improves on documents with visual content

## Rollback Plan

If issues arise, revert the changed files via git to the commit prior to this implementation:

```bash
git checkout -- deepdoc/parser/pdf_parser.py
git checkout -- rag/flow/parser/parser.py
git checkout -- rag/nlp/rag_tokenizer.py
git checkout -- api/db/services/llm_service.py
git checkout -- rag/app/picture.py
git checkout -- rag/llm/cv_model.py
```

Then re-deploy the previous container images:
```bash
docker-compose down
docker-compose pull
docker-compose up -d
```

## Future Enhancements

- Parallel page processing for improved throughput
- OCR fallback on VLM failure for higher reliability
- Custom chunking rules per document type
- Caching of VLM results to reduce API costs
- Performance benchmarks and optimization