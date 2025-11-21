# MinerU Intel XPU Enablement Notes

This file records all touch points that were updated so MinerU can run on Intel GPUs
(PyTorch XPU / intel-extension-for-pytorch). Keep it up to date so future rebases
can pick up the same customizations.

## Device Detection & CLI

- `mineru/utils/config_reader.py:get_device()` now returns `"xpu"` when
  `torch.xpu.is_available()`.
- `mineru/cli/client.py` accepts `--device xpu`/`xpu:0` and writes the value into
  `MINERU_DEVICE_MODE`.
- `projects/multi_gpu_v2/server.py`, `projects/mineru_tianshu/start_all.py`,
  `projects/mineru_tianshu/litserve_worker.py` treat device strings starting with
  `"xpu"` the same way as CUDA/NPU (for VRAM estimation and worker launch).

## Runtime Utilities

- `mineru/backend/pipeline/pipeline_analyze.py` dispatches to the transformers
  backend when the device starts with `xpu`.
- `mineru/backend/vlm/utils.py` no longer hard fails when CUDA is absent; it checks
  PyTorch XPU first.
- `mineru/utils/model_utils.py` functions (`clean_memory`, `get_vram`, etc.) support
  XPU via `torch.xpu`.
- `mineru/utils/block_sort.py` reads device properties from `torch.xpu` when running
  on Intel hardware.

## Models / ONNX

- `mineru/model/layout/doclayoutyolo.py` and
  `mineru/model/mfd/yolo_v8.py` accept `device="xpu"` (falls back to CPU if the
  corresponding backend is missing) and call `torch.compile(..., backend="openvino")`
  when available.
- `mineru/model/table/rec/slanet_plus/table_structure_utils.py`,
  `mineru/model/table/rec/unet_table/utils.py`, and the Paddle classifiers now
  instantiate ONNX Runtime sessions with the `OpenVINOExecutionProvider` (GPU)
  whenever it is available, falling back to CPU otherwise.

## Services

- Worker launchers (`litserve_worker.py`) and VRAM estimators now recognize the
  `"xpu"` prefix, so setting `--accelerator xpu` or `--device xpu` routes tasks to
  the Intel GPU without extra hacks.

## Open Items / Future Work

- Multi-GPU orchestration currently normalizes XPU devices but still assumes one
  worker per logical device. If we add true multi-XPU scheduling we should revisit
  `workers_per_device`.
- ONNX Runtime for XPU (when available) can replace the current CPU fallback for
  DocLayoutYolo/table models. Right now those components simply use PyTorch XPU or
  drop to CPU if an XPU kernel is missing.
