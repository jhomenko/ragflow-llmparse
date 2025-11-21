OV OpenVINO Implementation Plan

Goal: enable Intel/CPU-friendly inference paths (OpenVINO/ORT) while keeping current behavior as fallback. Each step lists the code anchor to change so we stay aligned with the repo.

0) Preconditions
- Add OpenVINO/ORT deps to the environment (openvino-runtime/onnxruntime-openvino) and keep ultralytics/doclayout_yolo versions that support OpenVINO export.
- Export required weights to IR/ONNX and place them alongside existing weights (see ModelPath constants), or wire in a configurable model root.

1) Device/config plumbing
- Extend device selection to understand Intel/OpenVINO: update `mineruxpu/mineru/utils/config_reader.py:get_device` to honor env (e.g., MINERU_DEVICE_MODE=intel:gpu/cpu/auto) and avoid assuming CUDA-only.
- Introduce env/config toggles consumed by model init (e.g., MINERU_USE_OPENVINO=1, MINERU_OPENVINO_DEVICE=intel:gpu/cpu, MINERU_OV_FALLBACK=cpu). Surface these through `MineruPipelineModel` construction in `backend/pipeline/model_init.py:200-270`.
- Allow AtomModelSingleton cache keys to include backend/device where relevant to avoid mixing PT/OV instances (`backend/pipeline/model_init.py:120-198`).

2) Layout (DocLayout-YOLO) OpenVINO branch
- The model currently uses `doclayout_yolo.YOLOv10(...).to(device)` and torch parsing (`model/layout/doclayoutyolo.py:13-82`). Add an OpenVINO path:
  - Detect a flag/use_openvino in __init__; if set, load the exported IR via Ultralytics YOLO OpenVINO loader or openvino.runtime Core (depending on the exported format).
  - Store an OV-compatible device string (intel:gpu/intel:cpu); pass it to predict rather than calling .to().
  - Update `_parse_prediction` to handle numpy outputs without `.cpu()` assumptions while keeping existing behavior for torch.
  - Ensure `batch_predict` works for OpenVINO (keep batching semantics; handle conf tweak when batch_size==1).
- Wire doclayout_yolo_model_init to pass the flag/device and locate IR (e.g., weight path resolution in `backend/pipeline/model_init.py:91-97`).

3) Formula detector (YOLOv8 MFD) OpenVINO branch
- In `model/mfd/yolo_v8.py:12-90`, mirror the layout changes:
  - Support loading exported OpenVINO IR (Ultralytics YOLO supports OpenVINO folders) behind a flag.
  - Route device strings via predict rather than `.to`, and guard `.cpu()` usage in `_run_predict`/visualize for numpy outputs.
  - Expose config via `mfd_model_init` (`backend/pipeline/model_init.py:73-88`) and ensure BatchAnalyze uses the updated model transparently (`backend/pipeline/batch_analyze.py:25-83`).

4) OCR backend migration (Torch Paddle → ORT/OpenVINO)
- Current OCR stack is torch-based (`model/ocr/pytorch_paddle.py:140-239` with detectors/recognizers in `model/utils/tools/infer/predict_det.py` etc.). Add an OpenVINO-capable backend:
  - Prepare ONNX/IR for the detector/recognizer (and optional angle classifier) matching the ModelPath assets.
  - Introduce an alternate OCR class (or a backend switch inside PytorchPaddleOCR) that:
    - Initializes ONNXRuntime sessions with OpenVINOExecutionProvider (configurable device) or openvino.runtime directly.
    - Reuses existing pre/post-processing (DB postprocess, crop/rotate, decoding) from `predict_det.py` and `predict_rec.py` but feeds ONNX/OV outputs instead of torch modules.
    - Preserves API compatibility of `ocr(...)`/`__call__` so pipeline callers do not change (BatchAnalyze uses `.ocr` and raw text_detector/text_recognizer).
  - Add a flag (e.g., MINERU_OCR_BACKEND=ov/torch) and plumb it through `ocr_model_init` (`backend/pipeline/model_init.py:97-118`).
  - Keep a clean fallback to the current torch path if OV assets are missing.

5) Table submodules (already ONNX) — add OV EP + device controls
- Orientation classifier and table cls use onnxruntime CPU (`model/ori_cls/paddle_ori_cls.py:1-200`, `model/table/cls/paddle_table_cls.py:1-190`). Add provider selection to prefer OpenVINOExecutionProvider (with device type) and fall back to CPU.
- Wireless/wired table recognizers: confirm runtime
  - `model/table/rec/slanet_plus/main.py` uses ONNX under the hood via TableStructurer; expose provider/device selection similarly.
  - `model/table/rec/unet_table/main.py` uses TSRUnet (PyTorch/ONNX). Add device/config hooks if ONNXRuntime is available; otherwise leave as CPU-torch.
- Ensure AtomModelSingleton keys include backend/device so mixed providers don’t collide.

6) Formula recognition (Unimernet / pp_formulanet) handling
- Keep models on CPU unless a compatible backend exists. Guard `.to(device)` calls in `model/mfr/unimernet/Unimernet.py:13-86` to avoid XPU/CUDA misuse; allow explicit MINERU_MFR_DEVICE override with safe fallback to CPU.
- Document that this stage remains PT/CPU for Phase 1, and make sure pipeline tolerates slower CPU path (batching already present in `UnimernetModel.batch_predict`).

7) Model asset handling
- Extend `utils/models_download_utils.py` to locate OV/ONNX exports (e.g., prefer `<weight>_openvino/` if present, otherwise download PT). Consider a new env (MINERU_MODEL_FORMAT=pt/ov) to influence selection.
- Update ModelPath or add derivation helpers so IR folders sit next to existing paths without breaking current downloads.

8) Pipeline integration and logging
- In `backend/pipeline/pipeline_analyze.py:1-150` and `batch_analyze.py:1-190`, ensure device strings are propagated (or unnecessary) after backend refactor; remove any assumptions about CUDA tensors (box coords should be numpy-safe).
- Add lightweight timing/logging hooks around each stage (layout, mfd, ocr det/rec, table) to compare PT vs OV performance. Use existing helpers `get_vram/clean_memory` responsibly to avoid GPU thrash.
- Verify batch paths that call `.cpu()` or torch-specific ops (e.g., OCR det batch outputs in `batch_analyze.py:140-240`) and make them backend-agnostic.

9) Validation checklist
- Functional regression: run sample PDFs through pipeline_s (digital, scanned, math-heavy, table-heavy); compare box counts/text against current output.
- Performance: measure per-stage latency on CPU vs intel:gpu with OV; capture device utilization.
- Fallbacks: simulate missing IR models to confirm PT fallback retains behavior and logs clear warnings.

Deliverables
- Code changes per steps above with flags/configs documented.
- Export scripts or instructions for generating OV/ONNX artifacts.
- Brief runbook for testing and enabling OV in deployment.

Implementation notes (current changes)
- Added env/config plumbing: use MINERU_USE_OPENVINO=1 and MINERU_OPENVINO_DEVICE (intel:gpu/intel:cpu/auto) to drive OV branches; ONNXRuntime EP selection derives from these.
- Layout + MFD now accept OV backends (Ultralytics loader path) with device-aware predict; parsing handles numpy/tensor outputs.
- OCR keeps torch backend but now has a backend flag placeholder (MINERU_OCR_BACKEND) warning when non-torch requested.
- ONNX-based table/ori classifiers now prefer OpenVINOExecutionProvider when enabled, falling back safely to CPU defaults.
- OpenVINO model path resolution: when OV is enabled and a sibling `<model>_openvino_model` folder exists next to the `.pt`, it is automatically preferred; otherwise the `.pt` path is used with a warning.
