# Parallel VLM Request Implementation Plan

## Overview
This document describes how to add parallel request handling to the RAG application's Vision Language Model (VLM) processing so multiple PDF pages can be processed concurrently. The primary target for changes is [`deepdoc/parser/pdf_parser.py`](deepdoc/parser/pdf_parser.py:1). Minimal changes are required elsewhere; configuration is via environment variables.

## Files referenced
- [`deepdoc/parser/pdf_parser.py`](deepdoc/parser/pdf_parser.py:1)
- [`rag/llm/working_vlm_module.py`](rag/llm/working_vlm_module.py:1)
- [`parallelImplementationPlan.md`](parallelImplementationPlan.md:1)

## Current State Analysis
The current VisionParser.__call__ method (approx lines 1801-2001 in [`deepdoc/parser/pdf_parser.py`](deepdoc/parser/pdf_parser.py:1)) iterates through pages sequentially:

```python
for idx, img_pil in enumerate(self.page_images or []):
    # Process each page one by one
```

Each page currently undergoes:
- Image preprocessing (RGB conversion)
- smart_resize
- JPEG conversion
- VLM API call via `picture_vision_llm_chunk()` (OpenAI-compatible call)
- Response normalization + robust per-page error handling

This flow is robust but serial. The goal is to add an optional async parallel mode that preserves the exact per-page processing and fallbacks.

## Goals
- Add optional client-side parallelism controlled by environment variables.
- Minimize changes to existing logic; encapsulate per-page logic into helpers.
- Preserve `smart_resize` behavior and per-page fallback semantics.
- Reuse a single async OpenAI client to leverage HTTP connection pooling.
- Provide clear tests and rollout steps.

## Environment variables
- `PARALLEL_VLM_REQUESTS` (int, default `1`) — number of concurrent requests. `1` = current sequential behavior.
- `VLM_CONCURRENT_TIMEOUT` (int, seconds, default `300`) — request timeout per page.
- `VLM_RETRY_COUNT` (int, default `2`) — retry attempts for transient errors.
- `VLM_RESIZE_FACTOR` (int, default `32`) — preserves current resizing behavior.

## Design summary
- Use asyncio and a bounded Semaphore to cap concurrency to `PARALLEL_VLM_REQUESTS`.
- Offload CPU-bound PIL ops (convert/resize/encode) to threads with `asyncio.to_thread` to avoid blocking the event loop.
- Encapsulate the existing per-page logic into `_execute_page_processing(...)` and wrap it with `_process_single_page(...)` that enforces semaphore gating.
- Orchestrate tasks in the synchronous `__call__` entrypoint by creating a temporary event loop when needed to run tasks concurrently; preserve sequential path when concurrency is 1.
- Collect results as `(page_idx, text)` and sort by `page_idx` to preserve ordering.

## Full code examples and context (drop-in-able snippets)

### 1) Imports and lightweight helpers
```python
# language: python
from typing import Tuple, List, Optional
import os
import asyncio
from asyncio import Semaphore
from PIL import Image
import io
from openai import OpenAI  # async client in modern openai package
# Optional: tenacity for retries (if not already a dependency)
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
```

### 2) VisionParser.__init__ additions and client reuse
```python
# language: python
class VisionParser(RAGFlowPdfParser):
    def __init__(self, vision_model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vision_model = vision_model
        # concurrency semaphore (initially None)
        self._vlm_semaphore: Optional[Semaphore] = None
        # single shared async OpenAI client (create lazily)
        self._openai_client: Optional[OpenAI] = None

    def _ensure_openai_client(self):
        if self._openai_client is None:
            # Create one shared async client for reuse (connection pooling)
            self._openai_client = OpenAI()
```

### 3) Preserve smart_resize
```python
# language: python
# Use the existing smart_resize util exactly as today:
# target_h, target_w = smart_resize(h, w, factor=resize_factor, target_max_dimension=1024)
```

### 4) CPU-bound PIL -> JPEG helper
```python
# language: python
def pil_to_jpeg_bytes(img: Image.Image, quality: int = 90) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
```

### 5) Core per-page async execution (_execute_page_processing)
This moves the exact per-page logic into an async-friendly function. CPU work is done via to_thread so the event loop is free to run other tasks.

```python
# language: python
async def _execute_page_processing(self, idx: int, img_pil: Image.Image, pdf_page_num: int, *,
                                    start_page: int, end_page: int, zoomin: int,
                                    prompt_text: str, callback=None) -> Tuple[int, str]:
    """
    Returns (page_idx, text_result)
    """
    # keep same resize factor used today
    resize_factor = int(os.getenv("VLM_RESIZE_FACTOR", "32"))

    # CPU-bound image preparation executed in a thread to avoid blocking loop
    def prepare_image():
        img = img_pil.convert("RGB")
        h, w = img.size[1], img.size[0]  # preserve current ordering if used
        target_h, target_w = smart_resize(h, w, factor=resize_factor, target_max_dimension=1024)
        img_resized = img.resize((target_w, target_h), resample=Image.Resampling.LANCZOS)
        jpeg_bytes = pil_to_jpeg_bytes(img_resized, quality=90)
        return jpeg_bytes

    jpeg_bytes = await asyncio.to_thread(prepare_image)

    # Ensure a shared OpenAI client exists
    self._ensure_openai_client()
    client = self._openai_client

    # Retry wrapper for transient network errors
    @retry(wait=wait_exponential(multiplier=0.5, max=10),
           stop=stop_after_attempt(int(os.getenv("VLM_RETRY_COUNT", "2"))),
           retry=retry_if_exception_type(Exception))
    async def call_vlm():
        # Example: adapt to your llama-serve fields if needed
        return await client.responses.create(
            model=os.getenv("VLM_MODEL_NAME", "vlm"),
            input=prompt_text,
            images=[{"data": jpeg_bytes, "mime": "image/jpeg"}],
            max_tokens=4096,
            temperature=0.1,
            stop=[]
        )

    try:
        vlm_resp = await asyncio.wait_for(call_vlm(), timeout=int(os.getenv("VLM_CONCURRENT_TIMEOUT", "300")))
    except Exception as e:
        # preserve existing fallback behavior: don't raise; return a placeholder per-page
        fallback = f"[VLM_ERROR page {pdf_page_num}]: {str(e)}"
        return idx, fallback

    # Normalize/interpret response the same way current code does
    text = getattr(vlm_resp, "output_text", None) or getattr(vlm_resp, "text", None) or str(vlm_resp)
    # run any existing heuristics / token counting code here
    return idx, text
```

### 6) Semaphore wrapper
```python
# language: python
async def _process_single_page(self, *args, **kwargs):
    if self._vlm_semaphore:
        async with self._vlm_semaphore:
            return await self._execute_page_processing(*args, **kwargs)
    return await self._execute_page_processing(*args, **kwargs)
```

### 7) Orchestration in __call__ (sync entrypoint)
This shows a safe pattern for running coroutines from sync code. It creates a temporary event loop so it is robust for both sync and non-async contexts. Note: if your application already runs inside an asyncio loop (e.g., FastAPI background task), adapt to await directly rather than creating a loop.

```python
# language: python
def __call__(self, filename, from_page=0, to_page=100000, **kwargs):
    # existing setup and validations remain unchanged

    start_page = from_page
    end_page = to_page
    parallel_requests = int(os.getenv("PARALLEL_VLM_REQUESTS", "1"))
    if parallel_requests > 1:
        self._vlm_semaphore = Semaphore(parallel_requests)

    page_tasks = []
    page_indices = []
    for idx, img_pil in enumerate(self.page_images or []):
        pdf_page_num = idx
        if pdf_page_num < start_page or pdf_page_num >= end_page:
            continue
        page_indices.append(idx)
        prompt_text = build_prompt_for_page(idx)  # plug your existing prompt builder
        page_tasks.append(
            self._process_single_page(idx, img_pil, pdf_page_num,
                                      start_page=start_page, end_page=end_page,
                                      zoomin=kwargs.get("zoomin", 1),
                                      prompt_text=prompt_text)
        )

    results = []
    if parallel_requests > 1 and len(page_tasks) > 0:
        # run in a dedicated event loop to avoid interfering with other code
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            results = loop.run_until_complete(asyncio.gather(*page_tasks))
        finally:
            loop.close()
            asyncio.set_event_loop(None)
    else:
        # preserve strictly sequential behavior by running coroutines one-by-one
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            for coro in page_tasks:
                results.append(loop.run_until_complete(coro))
        finally:
            loop.close()
            asyncio.set_event_loop(None)

    # Ensure results are in page order
    results.sort(key=lambda r: r[0])
    texts_in_order = [r[1] for r in results]

    # Convert into existing return structure (replace with your existing doc creation)
    docs = [{"page": idx, "text": txt} for idx, txt in zip(page_indices, texts_in_order)]
    return docs, []
```

## Ordering and idempotence
- Return `(page_idx, text)` from tasks and sort by `page_idx` to preserve original ordering.
- Tasks must be pure with respect to external state; any shared state access must be protected or avoided.

## Testing checklist
- Unit test: `PARALLEL_VLM_REQUESTS=1` must produce identical output to current code.
- Integration test: `PARALLEL_VLM_REQUESTS=4` against a mock/real llama-serve verifying concurrent requests.
- Fault injection: simulate timeouts/errors and ensure other page tasks complete and fallback text is used.
- Memory/CPU profiling for large PDFs.
- Streaming tests if you consume streaming tokens per-page (ensure independent streams per task).

## Rollout steps
1. Implement feature in feature branch.
2. Add/adjust unit and integration tests.
3. Deploy to staging with `PARALLEL_VLM_REQUESTS=1` (default).
4. Benchmark/warm-up and gradually increase `PARALLEL_VLM_REQUESTS` to align with llama-server `-np` slots.
5. Monitor metrics and adjust.

## Monitoring and metrics
- Add per-page latency histogram and success/error counters.
- Track queue depth and semaphore utilization to tune concurrency defaults.

## Dependencies to add to requirements.txt
The following packages need to be added to requirements.txt for the parallel processing functionality:

- `tenacity>=8.2.0,<9.0.0` - For retry logic with exponential backoff (already referenced in the code examples above)

Note: The core asyncio functionality including `asyncio.Semaphore`, `asyncio.gather`, `asyncio.to_thread`, etc. are part of Python's standard library, so no additional packages are needed for basic async/await and concurrency control.

## Notes and caveats
- If the app already runs inside an asyncio event loop (e.g., running under an async web server), adapt orchestration to `await asyncio.gather(...)` rather than creating new event loops.
- CPU-heavy image ops can still contend; consider a bounded ThreadPoolExecutor for image prep or reduce concurrency.
- Keep `PARALLEL_VLM_REQUESTS` conservative by default and align with server `-np` slots to avoid overload.

End of plan