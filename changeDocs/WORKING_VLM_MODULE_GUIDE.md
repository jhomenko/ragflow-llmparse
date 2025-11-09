# WORKING VLM MODULE GUIDE

Last updated: 2025-11-08

## What is the Working VLM Module?

The Working VLM (Vision-Language Model) module provides end-to-end support for processing images/visual documents and generating language outputs. It integrates UI, backend API, and a VLM server to allow users to select models, submit pages, and retrieve text/structured outputs. This guide documents configuration, usage, troubleshooting, testing, and architecture details.

## How It Works

High-level flow from UI to VLM server:

```mermaid
flowchart LR
  subgraph UI
    A[User selects model & uploads page]
  end
  A --> B[Frontend API call (/api/vlm/process)]
  B --> C[Backend service (llm_service / VLM client)]
  C --> D[VLM Server / Direct Vision Parser]
  D --> E[Postprocessing & OCR]
  E --> F[Store results / Return to UI]
```

Diagram notes:
- The UI calls the backend API which routes through llm_service to the VLM server or direct parser.
- The VLM server performs multimodal inference and returns structured text or tokens.

## Features

- ✅ Model selection from UI (multiple VLMs)
- ✅ Per-page processing and batching
- ✅ Integrated OCR fallback and postprocessing
- ✅ Configurable environment variables for endpoints and timeouts
- ✅ Unit and integration tests included (see Testing)
- ✅ Clear error reporting and retry logic

## Configuration

Environment variables (add to your docker-compose / .env or system environment):

| Variable | Default | Description |
|---|---:|---|
| USE_WORKING_VLM | true | Enable the in-process working VLM module (default: true). When true, RAGFlow uses the local working module that calls your VLM server using the proven client code. |
| VLM_BASE_URL | http://192.168.68.186:8080/v1 | Base URL for your VLM server. The exact model name selected in the UI is sent to this endpoint (e.g., model="Qwen2.5VL-3B"). |
| VLM_TIMEOUT_SEC | 60 | Request timeout in seconds for VLM calls |
| VLM_MAX_PAGE_SIZE | 5MB | Maximum allowed size per page image sent to the VLM |
| VLM_RETRY_COUNT | 2 | Number of retries for transient network or server errors |
| ENABLE_DIRECT_PARSER | false | If true, use the local direct parser (deepdoc/parser/direct_vision_parser.py) instead of contacting the remote VLM |

Tips:
- Keep VLM_BASE_URL internal (docker network) when running via compose.
- Tune VLM_TIMEOUT_SEC for larger pages or slow GPUs.

## Using Different Models (UI instructions)

1. Open the document processing view in the web UI.
2. Locate the model selector dropdown near the "Process" button.
3. Select the desired model from the list (models are loaded from the backend via a config endpoint).
4. Optional: set per-job parameters (page range, OCR toggle, confidence threshold).
5. Click "Process". The UI will show progress and return results when done.

Notes on model selection:
- Select the exact model name your VLM server expects in the UI (for example: "Qwen2.5VL-3B"). The UI passes that exact string through the backend to the VLM server; no name mapping is required.
- Environment variables control routing and endpoints (e.g. USE_WORKING_VLM, ENABLE_DIRECT_PARSER, VLM_BASE_URL), not the model name. Set `VLM_BASE_URL` to your server and choose the desired model in the UI.
- Set `ENABLE_DIRECT_PARSER=true` only if you intend to route processing to the local direct parser (deepdoc/parser/direct_vision_parser.py) instead of contacting the remote VLM server.
- Example: export VLM_BASE_URL="http://192.168.68.186:8080/v1" and select "Qwen2.5VL-3B" in the UI; the backend will send model="Qwen2.5VL-3B" to the VLM server.

## Troubleshooting

Common issues and solutions:

Issue: "Timeout while contacting VLM server"
- Cause: Slow inference or wrong VLM_BASE_URL.
- Solution: Increase VLM_TIMEOUT_SEC, check VLM_BASE_URL, inspect VLM server logs.

Issue: "Uploaded page rejected — file too large"
- Cause: Exceeding VLM_MAX_PAGE_SIZE.
- Solution: Reduce page size or increase VLM_MAX_PAGE_SIZE and restart services.

Issue: "Model not listed in UI"
- Cause: Backend model registry misconfigured.
- Solution: Confirm VLM_DEFAULT_MODEL and ensure backend config exposes model list. Restart backend.

Issue: "OCR text low quality"
- Cause: Poor image quality or wrong OCR pipeline config.
- Solution: Enable preprocessing (deskew, denoise) or use higher-quality OCR model in config.

Issue: "Server returns 5xx errors"
- Cause: VLM server crash or misconfiguration.
- Solution: Check server logs, ensure correct GPU resources, verify model files present.

## Testing

Running tests locally:

```bash
# run unit tests
pytest test_vision_parser_integration.py -q

# run specific VLM integration tests
pytest test_vlm_pdf_complete.py::test_pdf_process -q
```

- Ensure VLM server is running (or ENABLE_DIRECT_PARSER=true) before integration tests.
- Tests will use test fixtures to upload sample pages and validate outputs.

## Advanced Usage

- Batch processing: call backend endpoint with multiple pages; ensure VLM server supports batching to optimize throughput.
- Custom postprocessing: modify deepdoc/vision/postprocess.py to adjust text cleaning and segmentation rules.
- Switch to direct parser: set ENABLE_DIRECT_PARSER=true and confirm system uses deepdoc/parser/direct_vision_parser.py.

## FAQ

Q: Where are models hosted?
A: Models may be hosted inside the VLM server container or a remote inference service. See VLM_BASE_URL.

Q: How do I add a new model to the UI?
A: Add the model id to backend config that serves the model registry endpoint, then restart the backend.

Q: Can I process multi-page PDFs?
A: Yes — the backend supports per-page extraction and aggregation. Use the page range option in UI.

## Performance Notes

- Page processing time depends on model size, GPU, and page complexity.
- Typical processing times:
  - vlm-base: ~0.5–2s per simple page on GPU
  - vlm-large: ~2–8s per page on GPU
- Use batching and caching for high-throughput workloads.
- Monitor memory/GPU usage and increase VLM_TIMEOUT_SEC accordingly.

## Architecture Details (files modified/unchanged)

Files involved by the Working VLM Module:

- deepdoc/parser/direct_vision_parser.py — direct parser implementation (unchanged)
- deepdoc/parser/pdf_parser.py — PDF parsing glue (unchanged)
- deepdoc/vision/* — OCR and postprocessing modules (unchanged)
- api/db/services/llm_service.py — backend interface to VLM server (may be updated by implementation)
- web UI files (frontend) — model selector and process UI (frontend path depends on project)

Notes:
- This guide documents runtime behavior only. No files were modified by this documentation step.

## Where to find this guide

The guide is saved as [`WORKING_VLM_MODULE_GUIDE.md`](WORKING_VLM_MODULE_GUIDE.md:1) at the project root.

-- End of guide