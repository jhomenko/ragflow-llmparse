High-level architecture of MinerU’s model stack

MinerU’s PDF extraction pipeline orchestrates multiple specialized models (from the PDF-Extract-Kit toolkit) in a staged fashion
blog.csdn.net
blog.csdn.net
. The pipeline first converts each PDF page to an image, then applies these model components in order:

Layout Detection (DocLayout-YOLO) – Identifies high-level regions like titles, paragraphs, lists, figures, tables, captions, footnotes, and formulas on each page. This uses a YOLO-based object detection model (DocLayout-YOLO) built on the Ultralytics YOLOv10 architecture
blog.csdn.net
blog.csdn.net
.

Multi-Field Detection (MFD) – Additional YOLOv8-based detectors for specialized elements not fully handled by the main layout model. In practice, this is used for tasks like mathematical formula region detection (to precisely localize equations) and possibly other fine-grained fields. (Note: “MFD” stands for multi-field detection; “MFC” may refer to classification tasks like heading classification).

Formula Recognition (MFR) – Transcribes detected formula images into LaTeX. MinerU uses the UniMERNet model for this, a transformer-based image-to-text model specialized for math expression recognition
blog.csdn.net
. (Variants like UniMERNet-tiny/small are provided for different speed-accuracy tradeoffs.)

OCR (Optical Character Recognition) – Recognizes text in regions that require it (e.g. scanned documents or in images). MinerU integrates PaddleOCR (PP-OCR) for multilingual text detection and recognition
github.com
github.com
. This covers 80+ languages and scripts (using PP-OCRv4/v5 models for text detection, recognition, and an optional classifier for image orientation).

Table Structure Recognition – Extracts structured table data (rows, columns, cell contents) from table region images. MinerU uses a hybrid approach: a Wired Table Detector to detect table lines/borders (for bordered tables), and a Table Structure Recognizer (the StructEqTable model) for unruled tables
blog.csdn.net
blog.csdn.net
. The StructEqTable model is a powerful end-to-end table parser that outputs HTML/LaTeX, but it is heavy and currently requires GPU (it’s based on a transformer with sequence generation
pdf-extract-kit.readthedocs.io
pdf-extract-kit.readthedocs.io
).

Reading Order Model – After obtaining layout blocks and content, MinerU uses a learned LayoutReader model or heuristic to sort text and other elements in natural reading order
blog.csdn.net
blog.csdn.net
. This ensures the final output follows human reading sequence (e.g. columns, headings before body, etc.).

Each component is loaded and managed within MinerU’s pipeline engine, typically the pipeline backend (distinct from the end-to-end VLM engine)
blog.csdn.net
blog.csdn.net
. Below is a summary of each component’s current framework, device usage, and potential future backend:

Component	Purpose	Framework (current)	Typical Device (CUDA build)	Candidate Future Backend
DocLayout-YOLO	Layout analysis – detect titles, text blocks, lists, tables, figures, captions, footnotes, formulas
blog.csdn.net
blog.csdn.net
.	PyTorch (Ultralytics YOLOv10)	NVIDIA GPU (if available; else CPU)	OpenVINO (IR model on Intel GPU or CPU)
MFD YOLOv8 Detectors	Fine detection of specific fields (e.g. formula regions, possibly other small objects).	PyTorch (Ultralytics YOLOv8)	NVIDIA GPU (if available; else CPU)	OpenVINO (IR on Intel GPU/CPU) or Torch-XPU (if needed)
MFR – UniMERNet	Formula recognition – image to LaTeX transcription
blog.csdn.net
.	PyTorch (HF Transformer)	NVIDIA GPU (for speed; CPU if no GPU)	PyTorch CPU (keep on CPU for now; consider ONNX->OpenVINO in future)
OCR (Text det/rec)	Text detection & recognition (multilingual PP-OCR v4/v5)
github.com
github.com
.	PaddlePaddle (PPOCR models)	Often CPU; NVIDIA GPU if paddle-gpu installed (for acceleration)	OpenVINO (via ONNX or Paddle-frontend on Intel GPU/CPU)
Table Recognition	Table structure extraction (StructEqTable model + line detection)
blog.csdn.net
blog.csdn.net
.	PyTorch (HF Transformer for structure; OpenCV/algos for lines)	NVIDIA GPU required for StructEqTable
pdf-extract-kit.readthedocs.io
 (flash-attn, etc.); CPU for line-detection	PyTorch CPU (keep heavy model on CPU or skip on Arc; consider lighter SLANet+OpenVINO as future alternative)
Reading Order (LayoutReader)	Sequence model for ordering content (may use spatial+text features).	PyTorch (small sequence model or rule-based)	CPU (minor overhead)	PyTorch CPU (no change; not performance-critical)

Note: In the code, these models are initialized in modules like mineru.backend.pipeline.model_init and under mineru.model.* subpackages. For example, mineru/model/layout/doclayoutyolo.py defines DocLayoutYOLOModel, mineru/model/mfd/yolo_v8.py contains the YOLOv8 detector class, mineru/model/ocr/... contains OCR integration, etc. We will identify exact file paths to inspect in the final section.

Current model-by-model status (device & framework)

Layout (DocLayout-YOLO): Currently uses Ultralytics YOLOv10 weights (e.g. yolov10l_ft.pt) loaded via the Ultralytics API
huggingface.co
huggingface.co
. This is a PyTorch model utilizing Ultralytics’ YOLOv8/v10 code. In a typical installation with CUDA, the model is moved to GPU (.to("cuda")) for inference
blog.csdn.net
. The Ultralytics framework handles preprocessing (image resizing to e.g. 1280px) and outputs predictions as bounding boxes with class labels for the layout classes
blog.csdn.net
. OpenVINO status: Ultralytics officially supports exporting YOLOv8/v10 to OpenVINO format
docs.ultralytics.com
docs.ultralytics.com
, and their docs report up to 3× speedup on CPU and GPU acceleration on Intel hardware
docs.ultralytics.com
. No inherent limitations are noted for running YOLO models on Intel GPUs aside from needing the proper OpenVINO runtime. (One caution: if dynamic input shapes are needed, Ultralytics suggests using dynamic=True during export to handle varying image sizes
github.com
community.ultralytics.com
.)

MFD/MFC YOLOv8 models: MinerU’s “multi-field” detectors are also based on Ultralytics (likely YOLOv8). For example, a YOLOv8MFDModel is referenced for mathematical formula detection
blog.csdn.net
. These use smaller YOLO models (.pt weights) to detect specific targets (e.g. formula bounding boxes in pages). Currently, these load via the Ultralytics YOLO API (PyTorch) and run on the default device (GPU if available). Ultralytics provides official OpenVINO export for YOLOv8 as well – a blog post shows conversion of YOLOv8 to ONNX and OpenVINO IR for Intel devices
ultralytics.com
ultralytics.com
. On Intel Arc or CPU, OpenVINO can significantly speed up YOLOv8 inference while maintaining accuracy
docs.ultralytics.com
docs.ultralytics.com
. Limitations: No major functional limitations are documented; however, one should use the latest Ultralytics version for YOLOv8 export, and note that NMS and confidence thresholds become part of the exported model (adjusting those may require re-export or filtering outputs manually).

Formula Recognition (UniMERNet): The UniMERNet model (from Shanghai AI Lab) is integrated for converting formula images to LaTeX
blog.csdn.net
. This model uses a Swin Transformer encoder and mBART decoder
paddlepaddle.github.io
. Currently, MinerU loads it (likely via a HuggingFace or ModelScope interface) as a PyTorch model (.pth or safetensors weights) and runs on GPU if available, because it’s quite heavy. There’s no official OpenVINO or ONNX support noted for UniMERNet. In practice, to run on Intel, this will likely fall back to CPU (since PyTorch XPU may not support all ops needed for the transformer decoder). Upstream support: None specific to OpenVINO yet – this would require custom export to ONNX which is complex and beyond Phase 2. We plan to keep this on CPU for now.

OCR (PaddleOCR): MinerU uses PaddleOCR for text extraction. In recent versions, it updated to PP-OCRv5 for multilingual recognition
github.com
, supporting ~37 languages and even handwritten text. Under the hood, PaddleOCR includes three models: a text detection model (often based on a Differentiable Binarization or DB algorithm), a text recognition model (CRNN or Transformer-based), and an optional classifier (PP-LCNet) for image orientation
allpcb.com
allpcb.com
. Currently, MinerU likely uses Paddle’s Python API (from the paddleocr package) to load the pretrained detection and recognition models (e.g. PP-OCRv5_det_infer and *_rec_infer models) either automatically or from local cache. These models can run on CPU by default (with MKL acceleration) and on NVIDIA GPU if paddlepaddle-gpu is installed. For Intel GPUs, PaddlePaddle does not natively support oneAPI, so presently MinerU would be using CPU for OCR in an Intel environment. OpenVINO status: OpenVINO 2023+ can directly import Paddle models or use ONNX as an intermediary
allpcb.com
allpcb.com
. In fact, Intel’s docs provide a demo of running PP-OCRv5 with OpenVINO without conversion
docs.openvino.ai
. Alternatively, one can export Paddle models to ONNX (using paddle2onnx as shown in an example) and then use OpenVINO’s Model Optimizer
allpcb.com
 or onnxruntime with OpenVINO EP. There are no accuracy issues reported; performance on CPU can improve up to ~15× with OpenVINO vs stock Paddle in some cases
medium.com
. Limitation: The text detection model uses dynamic shapes (since image sizes vary), which OpenVINO can handle but we must ensure the IR is generated with dynamic dims or use OpenVINO’s Paddle frontend which handles it seamlessly. Also, the PP-OCRv5 recognition model uses attention (LSTM/Transformer) but OpenVINO supports those ops.

Table Recognition: MinerU’s table pipeline uses a two-part approach: (1) a wired-table detector for table borders, likely a lightweight algorithm or CNN (possibly based on Hough lines or a small model like TableNet), and (2) the StructEqTable model for structure recognition
blog.csdn.net
pdf-extract-kit.readthedocs.io
. The StructEqTable model is provided as a HuggingFace/ModelScope artifact (with model.safetensors, config, etc.) and is essentially a sequence-generation model (it generates LaTeX/HTML tokens for the table). It is likely loaded via HuggingFace’s AutoModelForSeq2SeqLM or ModelScope, hence running in PyTorch. This model currently only runs on GPU – the PDF-Extract-Kit notes it “only supports running on GPU devices”
pdf-extract-kit.readthedocs.io
, and indeed it leverages flash-attention and large Transformer layers (on CPU it would be extremely slow or may not run if flash-attn kernels are missing). OpenVINO status: There is no known OpenVINO export for StructEqTable at this time. We will treat this as a special case to leave on CPU (or skip if performance is untenable on Arc). If high table throughput is needed on CPU/Arc, a potential alternative is PaddleOCR’s SLANet (a lightweight table structure model using PP-LCNet, optimized for CPU)
arxiv.org
researchgate.net
 – but integrating that is beyond Phase 2. Thus, currently table parsing remains PyTorch on CPU (or using the Arc GPU via oneAPI if PyTorch XPU supported it, which is unlikely for this model’s ops).

Summary: In the current MinerU (NVIDIA-optimized scenario), most models run on the GPU via CUDA (YOLO detectors, UniMERNet, possibly StructEqTable) and use PyTorch, except PaddleOCR which might run on CPU or a second GPU stream. In an Intel Arc scenario, PyTorch XPU is experimental – indeed, the user found torchvision ops missing. Therefore, our migration will offload YOLO and CNN-based models to OpenVINO (which supports Arc GPUs well), while keeping complex seq2seq models on CPU or trying PyTorch XPU only if fully supported. Ultralytics and Intel’s documentation confirm that YOLOv8/v10 models can be exported to OpenVINO and run on CPU, Arc GPUs, or even VPUs with significant speedups and no accuracy loss
docs.ultralytics.com
docs.ultralytics.com
. PaddleOCR models can similarly be accelerated via OpenVINO’s ONNX or Paddle support
allpcb.com
medium.com
. We found no blocking compatibility issues in upstream docs, aside from ensuring proper driver setup for Arc (discussed later) and acknowledging that some large models (UniMERNet, StructEqTable) will not be migrated in Phase 2 due to lack of existing OpenVINO support.

OpenVINO migration strategy by component
3.1 DocLayout-YOLO (Layout Detection)

Exporting YOLOv10 to OpenVINO: Ultralytics makes this straightforward. We will take the pretrained DocLayout YOLO weight (e.g. yolov10l_ft.pt from HuggingFace) and use Ultralytics’ export utility to produce an OpenVINO IR. This can be done in code or via CLI. For example, using Python API
docs.ultralytics.com
:

from ultralytics import YOLO
model = YOLO("path/to/yolov10l_ft.pt")
model.export(format="openvino", dynamic=True)  # outputs folder like 'yolov10l_ft_openvino_model/'


This converts the PyTorch model to ONNX then to OpenVINO IR (FP32 by default). We set dynamic=True to allow dynamic image shapes
github.com
docs.ultralytics.com
, since pages can vary in size (we typically resize to 1024 or 1280, but dynamic IR adds flexibility for future). Optionally, we could set half=True during export to generate an FP16 IR – on Intel Arc GPUs, FP16 inference is natively fast and uses less memory (Arc and integrated GPUs use FP16 internally for optimal speed). If half=True is not directly supported in export, we can run OpenVINO’s Post-Training Optimization Tool later for FP16 conversion, but this may not be necessary as the runtime can downcast for GPU.

Loading and inference with OpenVINO model: Once exported, we will load the OpenVINO model in MinerU’s code via the Ultralytics YOLO class. Ultralytics is designed to recognize the OpenVINO model folder – we just point YOLO() to the directory:

ov_model = YOLO("path/to/yolov10l_ft_openvino_model/")  # load OpenVINO IR


Under the hood, this wraps OpenVINO’s runtime (using openvino.runtime.Core) to run inference. We can perform inference similarly to before: results = ov_model(image_array, device="intel:gpu"). The device parameter is crucial – for OpenVINO models Ultralytics expects "intel:gpu", "intel:cpu", or "intel:npu" strings
docs.ultralytics.com
docs.ultralytics.com
. We will use intel:gpu for Arc GPUs (or intel:cpu to run on CPU). Ultralytics will handle the image preprocessing and postprocessing as usual, returning results in the same format as PyTorch. Specifically, results will be a list of Ultralytics Results objects, each containing .boxes (with coordinates, confidence, class id) and .names (class name mapping) just like the PyTorch version.

Adapting DocLayoutYOLOModel: In MinerU’s DocLayoutYOLOModel (likely in mineru/model/layout/doclayoutyolo.py), the current initialization does something like:

self.model = YOLOv10(weight_path).to(device)  # uses Ultralytics internals
self.layout_classes = [ "title", "text", "figure", "table", "caption", "footnote", "formula" ]


We will modify this to detect if an OpenVINO variant is to be used. For example, we might add an argument or config flag for use_openvino. If true, we attempt to load the IR:

if use_openvino:
    self.model = YOLO(f"{weight_dir}_openvino_model/")  # assuming IR files are there
else:
    self.model = YOLO(weight_path)
    if device: 
        self.model.to(device)


Note: We don’t call .to("intel:gpu") on the model; instead, we specify the device at inference time (Ultralytics requires device per predict, not a persistent .to for OpenVINO models). We’ll ensure DocLayoutYOLOModel.predict() passes device="intel:gpu" when calling the model. The predict method likely looks like
blog.csdn.net
:

def predict(self, image):
    prediction = self.model.predict(image, imgsz=1280, conf=0.1, iou=0.45)
    return self._parse_prediction(prediction)


For OpenVINO, this self.model.predict will internally use OpenVINO and accept the same parameters. We must verify if Ultralytics honors conf=0.1 and iou=0.45 at runtime for OpenVINO models – often the NMS is already embedded with the thresholds used during export. We might need to re-export with the desired conf if needed, or simply filter the results ourselves. This is a detail to confirm when inspecting the code.

Output format adjustments: After getting prediction, MinerU calls an internal _parse_prediction to convert Ultralytics results to its own format (possibly a list of dicts with keys like “type”, “bbox”, etc.). We expect the output structure from Ultralytics to remain identical for OpenVINO. The boxes will be numpy arrays instead of PyTorch tensors, but Ultralytics abstracts that. We should double-check that _parse_prediction doesn’t assume a torch tensor (if it does, we can cast or adjust it, but likely Ultralytics Results provides standard Python types).

In summary, for DocLayout-YOLO, the migration involves exporting the weights to OpenVINO IR, loading via Ultralytics, and ensuring the predict() call uses the correct device and parsing. After this change, layout detection will run on Intel GPUs efficiently. This addresses the major bottleneck, since layout-YOLO processes every page image (Ultralytics reports ~7ms/image on Arc A770 for a small model vs ~16ms on CPU
docs.ultralytics.com
, with larger models seeing 5–10× speedups
docs.ultralytics.com
).

3.2 MFD / MFC (YOLOv8-style detectors)

For the YOLOv8-based detection models (formula detector, etc.), the approach is similar to above but with some nuances:

Export YOLOv8 to OpenVINO: Using Ultralytics, e.g. model = YOLO("yolov8_formula.pt"); model.export(format="openvino", dynamic=True). We will do this for each YOLOv8 model integrated in MinerU (there might be one for formulas, possibly one for figure image detection or other tasks if present). The export process is the same; we’ll get a <name>_openvino_model/ folder with IR files. We should choose precision based on device: for CPU inference, INT8 could yield huge gains (Ultralytics shows YOLO models can be quantized with minimal mAP drop
docs.ultralytics.com
docs.ultralytics.com
), but quantization would be an extra step. Initially, FP32 IR is fine, and OpenVINO can be configured to use FP16 on Arc GPUs automatically.

Load and integration: In MinerU’s code, these YOLOv8 models are likely wrapped in classes like YOLOv8MFDModel. We will load the IR similarly: self.model = YOLO("..._openvino_model/"). We’ll ensure inference calls specify device. If MinerU currently passes a device string (like "cuda", "cpu") to these models, we might intercept it. For example, if YOLOv8MFDModel accepts a device arg, we can map "cuda" -> "intel:gpu" under the hood when loading OpenVINO.

Batching and image size: YOLOv8 detectors might use smaller input sizes (perhaps 640 or 1024). We should use the same imgsz as in config when exporting for consistency. With OpenVINO, we can still batch images – Ultralytics will batch automatically if we pass a list of images to predict. If dynamic shapes are enabled, batch dimension can also be dynamic. We should verify if MinerU’s pipeline ever sends multiple images at once to these sub-detectors; typically formula detection might be per page, so single at a time. No major changes needed there.

Outputs: As with layout, results will come as Ultralytics Results objects. We must adapt _parse_prediction or equivalent in each model class if needed. Likely the formula detector returns bounding boxes of formulas (class “formula” with confidence). Downstream, these are used to crop formula images for recognition. We need to ensure the coordinates and confidence formatting remain the same. (Ultralytics OpenVINO backend yields the same coordinates in pixel units.)

Ultralytics on Intel best-practices: For YOLOv8 on Intel GPU vs CPU, a few points: On CPU, INT8 models can be 2× faster than FP32
docs.ultralytics.com
. We could consider int8 quantization for the formula detector if CPU-bound. However, since we plan to use Arc GPU, FP16/FP32 is fine. On Arc, the benchmarks show even FP32 OpenVINO is 5–10× faster than PyTorch CPU and even faster than PyTorch (if it were possible on Arc)
docs.ultralytics.com
. We should also note that OpenVINO’s AUTO device could be used (it automatically chooses GPU if free, else CPU). We may allow a config like device="AUTO" which Ultralytics might map appropriately.

In summary, for each YOLOv8 detector, we export to OpenVINO, load via YOLO(...), and ensure the rest of the pipeline (e.g. cropping formulas for recognition) receives the same data as before. This will allow formula region detection (and any similar sub-detection) to utilize Arc GPU.

One gotcha to monitor: if MinerU’s YOLOv8 integration uses any Torch-specific postprocessing or expects a torch.Tensor, we’ll have to adjust that. For example, if they directly accessed prediction[0].boxes.data assuming a torch tensor, we might need to convert it to numpy. We will verify such details in code review.

3.3 OCR & Table models (PaddleOCR and others)

OCR (PaddleOCR) Migration: PaddleOCR models can be served through OpenVINO in two ways:

Via ONNXRuntime with OpenVINO EP: Since MinerU might not have OpenVINO Python usage yet, a quick integration is to use ONNXRuntime’s OpenVINO Execution Provider. We can convert the PaddleOCR detection and recognition models to ONNX using Paddle’s tools (as shown in an example
allpcb.com
). In fact, many pre-converted ONNX models for PP-OCR exist (for example, the monkt/paddleocr-onnx repo on Hugging Face provides multilingual PP-OCR models in ONNX). Once we have det.onnx and rec.onnx, we can use ONNXRuntime like:

import onnxruntime as ort
sess_options = ort.SessionOptions()
providers = [("OpenVINOExecutionProvider", {"device_type": "GPU"})]  # or "CPU"
det_session = ort.InferenceSession("PP-OCRv5_det.onnx", sess_options, providers=providers)
rec_session = ort.InferenceSession("PP-OCRv5_rec.onnx", sess_options, providers=providers)


This will internally load OpenVINO and run the inference on the specified device (Intel GPU or CPU)
allpcb.com
. The benefit of this approach is minimal code changes to MinerU’s OCR logic: wherever images are passed to PaddleOCR, we instead feed them to the ONNXRuntime sessions and get back detection boxes and text strings. The onnxruntime OpenVINO EP will handle all optimizations. We should ensure to set appropriate device_type (likely “GPU” for Arc, or we could even use AUTO to let OpenVINO pick GPU and fallback to CPU if GPU isn’t available).

Via OpenVINO Runtime directly: OpenVINO 2023 introduced a PaddlePaddle frontend that can load .pdmodel files directly, so we could skip ONNX. The flow would be: use openvino.runtime.Core() to read the Paddle model and .pdparams and compile it for GPU. This might yield even better integration (and avoid onnxruntime dependency), but it’s more involved in code. Given our team’s familiarity with OpenVINO, we could try this. The openvino docs even have a ready sample for PaddleOCR on OpenVINO. In code, something like:

core = ov.Core()
det_model = core.read_model("PP-OCRv5_det_infer.pdmodel", "PP-OCRv5_det_infer.pdiparams")
det_compiled = core.compile_model(det_model, "GPU")


and similarly for rec_model. Then use det_compiled([input_blob]) to run inference. We would need to handle preprocessing (scaling image for text detector, etc.) as PaddleOCR normally does. Perhaps leveraging PaddleOCR’s Python code for preprocessing but swapping out the predictor. This is more custom work.

Given time, the ONNXRuntime route is a pragmatic choice for Phase 2. We will thus:

Export or download ONNX versions of the needed OCR models (detection + recognition for whatever languages we need; possibly the default multilingual model covers all). If MinerU uses multiple recognition models depending on lang parameter (e.g. English vs Chinese vs others), we may need ONNX for each chosen model. Alternatively, use one high-accuracy model for all (the PP-OCRv5 server model is multilingual already
github.com
).

In MinerU’s OCR pipeline code (likely MultiLanguageOCR or similar class), replace calls to PaddleOCR’s ocr = PaddleOCR(...); ocr.ocr(image) with our ONNXRuntime inference:

Run det_session on the page image -> get text boxes.

For each box, crop and run rec_session -> get text string.

Assemble results in the same format PaddleOCR would have returned (list of (text, confidence) per box, etc.).

We must be careful to mimic PaddleOCR’s post-processing (it sorts boxes by reading order per page, etc. MinerU might already rely on PaddleOCR’s output structure). We can use PaddleOCR’s output from a sample as a reference to ensure our replacement returns the same type of data.

Performance expectation: With OpenVINO on Arc, text detection should be much faster. Paddle’s detection model is essentially a segmentation-like CNN on the full page; OpenVINO’s optimizations can significantly speed this up on GPU. The text recognition model (an RNN/Transformer per text line) will also benefit, though that runs per text crop (which are small images, so CPU might even be okay). We will benchmark after integration, but we anticipate a noticeable improvement in OCR throughput, freeing the CPU for other tasks or enabling near real-time OCR on many pages.

Table Structure Recognition: Migrating the table model to OpenVINO is challenging due to its nature (seq2seq generation). For Phase 2, we propose to keep using the current approach on CPU and not attempt OpenVINO conversion. The StructEqTable model likely uses HuggingFace Transformers (maybe a variant of GPT or DETR+LM). Converting that to ONNX/OpenVINO would be complex and uncertain. Instead, for Intel deployment, we have two choices:

Run StructEqTable on CPU (it might be slow – possibly several seconds per table).

Skip or limit its use on Arc GPUs. If table parsing is not a priority in Phase 2, we might disable it or require a discrete NVIDIA GPU for that part (with a clear note).

However, if table extraction is needed, one idea is to leverage a lighter model. PaddleOCR’s SLANet (Table Structure Recognition v2) is optimized for CPU and could potentially be run via ONNX/OpenVINO. SLANet uses a CNN+LSTM approach (with PP-LCNet backbone) that is 1-2 orders of magnitude faster on CPU than transformer models
arxiv.org
. It outputs HTML structure similarly. As a future improvement (Phase 3), we could integrate SLANet or a similar lightweight model for tables when on CPU/Arc, instead of StructEqTable. This would drastically improve speed with minimal accuracy drop on simple tables. We will note this as a consideration but not implement it in Phase 2 due to scope.

For now, our strategy by component:

DocLayout-YOLO: Use OpenVINO IR on GPU.

Formula/field YOLOv8: Use OpenVINO IR on GPU.

OCR (text det/rec): Use ONNXRuntime+OpenVINO on GPU (or CPU if GPU busy).

UniMERNet (formula rec): Keep PyTorch on CPU (possibly try PyTorch XPU in future if support matures, but not in Phase 2).

Table (StructEqTable): Keep PyTorch on CPU (consider disabling or limiting for large volumes).

Reading Order & others: Keep as is (these are lightweight).

3.4 MFR (Formula Recognition) and other remaining models

As noted, UniMERNet likely cannot be easily migrated in Phase 2. There are no public examples of converting it to ONNX. It has complex layers (custom attention, MBart decoder) which may not all be supported by ONNX or OpenVINO without custom plugins. The safest plan: run UniMERNet on CPU using PyTorch. Modern CPUs with MKL can handle the small UniMERNet (the “small” variant is presumably a few hundred million params) reasonably for individual formulas. Since formula recognition is not needed for every text, the occasional heavy CPU load is acceptable. We will ensure the device is set to CPU in code (i.e. do not attempt .to("cuda") when on an Intel system, or catch exceptions if .to("xpu") fails and revert to CPU).

If, in the future, we wanted to try migrating MFR: a potential path is to replace UniMERNet with OpenVINO’s own text recognition pipeline for math (if any exists) or attempt an ONNX export of the trained model (likely requires significant effort and verifying the decoding logic). This is beyond our current phase, and we’ll mark it as a research item for later.

Another minor model: Heading Classification (MFC) if implemented (MinerU mentions heading classification feature toggle
github.com
). If this is a learned model (perhaps a small BERT classifier on text or an image-based classifier on layout blocks), we should evaluate its implementation. Possibly it’s rule-based (font size, boldness) rather than a neural model. If it is a model (e.g. a classifier to label a text block as heading vs body), it might be a PyTorch or ONNX model. We can decide to leave it on CPU if trivial, or if it’s an ONNX, use onnxruntime. This is a relatively minor piece and won’t impact performance significantly, so we’ll keep it simple.

Finally, image caption matching (if any algorithmic model is used) is likely heuristic or uses CLIP. If they use a CLIP model to match figure images with captions, that’s another model (PyTorch possibly). There’s no explicit mention of a CLIP model in the pipeline description (just “智能图文关联算法” i.e. smart image-text association
blog.csdn.net
). If they do use a CLIP or similar, it might be a small ViT-based model. We can consider leaving it on CPU or using OpenVINO if it’s ONNX-friendly. This detail is less clear, so we’ll include it as an open question to verify.

Detailed implementation checklist (Phase 2)

Below is a step-by-step implementation plan to migrate MinerU’s pipeline to the new backend setup. We outline which files/functions to modify and what changes to make. Note: Pseudocode is provided for clarity, but we will verify exact class and function names using the code (with context7) before finalizing any code changes.

4.1 Repository modules and configuration mapping

Identify relevant modules:

Model Initialization: mineru/backend/pipeline/model_init.py – likely contains logic to load all models based on config. We expect functions or a class here that instantiate DocLayoutYOLOModel, YOLOv8MFDModel, UnimernetModel, OCR engine, etc., possibly reading a YAML/JSON config of model paths.

Pipeline Orchestration: mineru/backend/pipeline/pipeline_analyze.py (or similar) – orchestrates the steps: layout -> formulas -> tables -> OCR -> merge. We need to adjust device placement logic here if present (for example, it might move images to GPU, etc. which in our case may not apply).

Layout model class: mineru/model/layout/doclayoutyolo.py – contains DocLayoutYOLOModel. We will open this to confirm how it loads the YOLO model and performs prediction. Key functions: __init__, predict (and possibly predict_images or predict_pdfs methods).

Formula detector class: mineru/model/mfd/yolo_v8.py – likely contains YOLOv8MFDModel (the blog reference
blog.csdn.net
 suggests this). We’ll verify its __init__ (should load model weights) and usage.

Formula recognizer: mineru/model/mfr/unimernet.py or similar – contains UnimernetModel or FormulaRecognizer class. We’ll see how it loads weights (HuggingFace model hub or local path) and ensure it stays on CPU.

OCR engine: Possibly mineru/model/ocr/paddleocr.py or mineru/model/ocr/multilang_ocr.py. The blog snippet shows a MultiLanguageOCR class inside a HybridOCRStrategy
blog.csdn.net
. We need to find where MultiLanguageOCR is implemented. It likely loads PaddleOCR (possibly via PaddleOCR API or custom). We will open this to replace with ONNX runtime calls.

Table pipeline: mineru/model/table/... – Perhaps classes for WiredTableDetector, TableStructureRecognizer, CrossPageTableMerger as seen in the blog
blog.csdn.net
blog.csdn.net
. We will locate these. Changes here: possibly skip changes (just ensure it runs on CPU). But we may add a warning or config flag to disable table parsing if no suitable GPU (Arc won’t help here).

Config files: MinerU likely uses a JSON or YAML config (mineru.template.json or similar) that specifies which model weights to use. There might be entries like "layout_model": "layout_detection_yolo" or paths. We should ensure these point to our local weight paths (or IR paths) if we’re not relying on auto-download. For offline deployment, we’ll likely mount the converted models in a known directory and use the weight_dir parameters (the CSDN article
blog.csdn.net
 indicates we can pass weight_dir to these model classes).

For each module above, we will do the following when we inspect the code with context7:

Confirm the class/function signatures and internal logic.

Insert conditional logic for OpenVINO vs PyTorch as needed.

Ensure any hardcoded device strings (“cuda”, “cpu”) are abstracted. We might introduce an environment variable MINERU_OPENVINO_DEVICE or config entries for each model’s backend.

Configuration approach: We propose adding a global config or environment flags such as:

MINERU_USE_OPENVINO – if set, use OpenVINO for supported models (YOLO, OCR).

MINERU_OPENVINO_DEVICE – e.g. "GPU" or "CPU" or "AUTO". This would translate to Ultralytics device string "intel:gpu" etc., and for ONNXRuntime EP config.

Alternatively, a per-model backend option: e.g. layout_backend: "openvino" in the config file. This is more granular but requires config changes. We might opt for a simpler env var to force OpenVINO for all that can use it.

We will decide based on how easily we can detect the environment in code (the team can set env vars in the deployment).

Now, module-specific steps:

4.2 DocLayout-YOLO migration steps

Export the layout model to OpenVINO IR: Outside the code (one-time setup), run the Ultralytics export as described earlier. Use the same image size as configured (the PDF-Extract-Kit config uses 1024 as long-edge
pdf-extract-kit.readthedocs.io
, but code snippet uses 1280 in predict
blog.csdn.net
 – we should confirm which is correct. Possibly 1280 for YOLOv10 large model). We’ll export with dynamic=True. This yields a directory (e.g. doclayout_yolo_openvino_model/) containing model.xml and model.bin (IR files), and likely model.mapping and model.yaml.

We should choose a precision: default FP32 is fine; OpenVINO will internally use FP16 on GPU. (If we see a need, we can run pot (Post-Training Optimizer) for int8 on CPU later.)

INT8 consideration: If CPU performance is a concern, OpenVINO’s POT could quantize the model. But since we target Arc GPU, which doesn’t benefit from int8 much (Arc GPU core is optimized for FP16/FP32), we skip int8 for now.

Integrate loading in DocLayoutYOLOModel:

In __init__, add a parameter or detect via env that OpenVINO should be used. For example:

use_openvino = (os.getenv("MINERU_USE_OPENVINO", "0") == "1")
if use_openvino and weight_dir:
    # weight_dir might be path containing original .pt; our IR could be in weight_dir + "_openvino_model"
    ov_path = os.path.join(weight_dir, "openvino", "")  # depending on how we store it
    self.model = YOLO(ov_path)
else:
    self.model = YOLO(weight)
    if device:
        self.model.to(device)


We will verify the actual parameters (weight vs weight_dir) when we see the code. The CSDN snippet
blog.csdn.net
 suggests DocLayoutYOLOModel(weight_dir="...") is possible, so they may internally append the filename. We’ll adjust accordingly.

Ensure layout_classes remains the same (we might even get class names from self.model.names which Ultralytics populates from the model’s YAML; but since we know them, keeping the static list is fine as long as it matches).

Adjust predict_images/predict_pdfs: The PDF-Extract-Kit docs mention using predict_images vs predict_pdfs
pdf-extract-kit.readthedocs.io
. These likely just call predict internally in a loop. No major changes needed there, except to pass device for OpenVINO.

In predict(self, image): Ultralytics YOLO.predict accepts a device argument. If OpenVINO model was loaded, calling self.model.predict(image, device="intel:gpu") will run on GPU. If device is not passed, Ultralytics might default to CPU even if model is compiled for GPU (not entirely sure), so to be safe we pass it. We can obtain the desired device string from an env or a class attribute. Perhaps set self.device = "intel:gpu" in init if openvino. Then:

if self.device:
    prediction = self.model.predict(image, imgsz=self.img_size, conf=self.conf_thres, iou=self.iou_thres, device=self.device)
else:
    prediction = self.model.predict(image, imgsz=..., conf=..., iou=...)


_parse_prediction: likely converts Ultralytics prediction to a nicer format. We will open it to ensure it works with OpenVINO results. Possibly it iterates over prediction[0].boxes and maps class indices to names via layout_classes. That should still work. Just ensure if it expects a torch tensor (e.g. box.xyxy property works regardless of backend, but if not, we can use box.data which might be numpy now). We’ll confirm. This function is critical for downstream merging.

Testing this component: After changes, test DocLayoutYOLOModel.predict on a sample page image in isolation: ensure we get roughly the same boxes as before. We will compare results from PyTorch model vs OpenVINO model on the same image to verify equivalence (within a small tolerance).

Batch processing: The snippet
blog.csdn.net
 shows an adaptive batch strategy. This might feed a list of images to self.model.predict for speed. The Ultralytics OpenVINO model can batch if the IR was exported with a fixed shape and perhaps a fixed batch size. If dynamic=True, it likely supports dynamic batch as well. We should confirm by experiment if ov_model([img1, img2]) yields 2 results. Assuming yes, we should ensure our code doesn’t break batching. Possibly no changes needed, but keep an eye on any Torch-specific concatenation (shouldn’t be an issue if using Ultralytics API).

4.3 MFD/MFC YOLOv8 migration steps

Export YOLOv8 models: For each YOLOv8-based weight in use (e.g. yolo8_formula.pt, maybe a model for other field detection), run model.export(format="openvino", dynamic=True). Use an input size according to the config (perhaps 640 or 1024). The export process is similar to YOLOv10. Ensure the exported folder is accessible to the code.

Modify YOLOv8MFDModel (or similar class):

Similar to layout, adjust __init__ to load OpenVINO model if enabled. Use YOLO("path/to/onnxdir/").

The CSDN snippet gave model = YOLOv8MFDModel(weight="/local/path/yolov8.pt")
blog.csdn.net
. So the class likely wraps Ultralytics YOLO too. We’ll open it and apply analogous modifications.

Ensure the model’s predict method passes device="intel:gpu".

If there’s any custom postprocessing (e.g. perhaps combining results or filtering), keep it same.

Adjust usage in pipeline: Where the formula detector is called. Possibly in pipeline_analyze, after layout, something like:

if formula_enable:
    formulas = formula_detector.detect(page_images)


If their code was moving tensors to a specific device, that might need changing. Ideally, after our changes, formula_detector internally handles device via OpenVINO, so from the outside it works the same.

Test formula detection: Pick a page with a known formula region. Run old vs new detection to ensure the bounding boxes align. Check that the coordinates match (OpenVINO might produce slightly different float rounding, but should be negligible). Also test confidence values – should be similar.

Multi-field classification (if any): If there’s a YOLOv8MFCModel or some classification step (like heading classifier), handle it. Possibly MinerU re-uses the YOLOv8 for classification by treating it as detection with class outputs or uses a separate simple classifier model. We’ll search the code for any classifier model usage. If found and it’s small, we might convert it to ONNX and use OpenVINO (or just leave on CPU if it’s fast enough). Most likely, heading classification might be rule-based, so no code change needed beyond config toggling. We’ll confirm.

4.4 OCR / PaddleOCR integration steps

Prepare ONNX models: Convert the PPOCR detection and recognition models. Since MinerU 2.x uses PP-OCRv5 for multilingual text, we will get:

PP-OCRv5_det ONNX (text detector).

PP-OCRv5_rec ONNX (text recognizer). Possibly the “server” model which supports English+Chinese+digits by default. For broader language support (37 languages), PaddleOCR provides specific models (like for Latin, Cyrillic, Indic, etc.). However, they also have a general multilingual model. We might start with the default (which likely covers English/Latin; if MinerU specifically mentioned 37 languages, they might use a multi-language recognition model from PaddleOCR).

If needed, the classification model PP-LCNet for script direction – but this is optional; PaddleOCR uses it to detect if text is rotated 90°, etc. MinerU might or might not include it. If they did (the installation snippet mentions PP-LCNet_x1_0_doc_cls model
allpcb.com
), we should also convert that to ONNX. It’s very lightweight (a simple image classifier) and can be run on CPU easily, so even if left on CPU it’s fine. We could also incorporate it via OpenVINO EP if desired.

We can utilize existing converted models if available to save time.

Implement OpenVINO OCR class: Create a new class or modify MultiLanguageOCR:

On init, load the ONNXRuntime sessions for det and rec (and cls if needed) with OpenVINO providers as shown above.

On recognize(page_image):

Preprocess the image as PaddleOCR expects (usually resizing longest side to a certain length (e.g. 960 or 1280) while maintaining aspect, then making a copy because PaddleOCR uses an opposite pixel mean sometimes – but PP-OCRv5 uses simple normalization).

Feed into det_session to get a detection map output. The PaddleOCR detector outputs a segmentation map of text regions. Paddle’s postprocess (DBPostProcessor) will be needed to get boxes from this map. We may reuse PaddleOCR’s post-processing code (we can import paddleocr just for the utility if available, or implement a basic DB post-processing: essentially binarize output by threshold, find contours, filter by size/shape). This is a bit of work but manageable. Alternatively, since we might still have PaddleOCR installed, we could hack by feeding the output of our ONNX det back into their postprocessing. But to avoid complexity, it might be simpler to still instantiate PaddleOCR in a special mode: PaddleOCR allows specifying det_model_dir and rec_model_dir. Perhaps we can trick it by giving it our ONNX model through onnxruntime? That might be too hacky. Let’s do manual postproc.

After obtaining text box coordinates, sort them (top to bottom, left to right).

For each box, crop the image region and resize to the input size expected by the recognizer (usually 32px height, variable width, keep aspect, pad if needed).

Run rec_session on each crop to get character probabilities, then decode to string. If using a CTC-based model, decoding is straightforward; if it’s an attention-based model, the ONNX likely outputs the sequence of characters directly. (We might need the character dictionary to map indices to chars – PaddleOCR provides a dict for each model, e.g. en_dict.txt or a combined dict for multilingual. MinerU likely includes these or they rely on PaddleOCR’s internal dictionaries.)

Collect the text results with their positions and confidences.

This is a substantial chunk of code. However, note that MinerU might have already integrated RapidOCR or similar, which is an ONNXRuntime-based OCR solution
repos.ecosyste.ms
pypi.org
. If so, their code might already have onnxruntime inference and we could simply ensure the OpenVINO EP is used. For example, the PyPI rapid-doc mention suggests that except OCR and PP-DocLayout, they tried OpenVINO EP
pypi.org
. We should check if MinerU has any mention of onnxruntime usage for OCR. If they currently use PaddleOCR in pure python, then we implement as above.

Optionally, if timeline is tight, we could initially run PaddleOCR’s detector on CPU (since that might be okay) and just accelerate recognizer with OpenVINO (since recognizer runs many times for each text line, that might actually be a bigger bottleneck). But ideally, do both.

Integrate into pipeline: The pipeline likely calls something like:

if ocr_required:
    texts = ocr_engine.recognize(images, lang=config_lang)


We ensure our ocr_engine is initialized accordingly. Possibly, we instantiate it with a given language list or model path. We might allow a config like ocr_backend: "openvino" to trigger our code. If env var is set, we replace ocr_engine with our OpenVINO-backed implementation.

We will also add a fallback: if something fails or if a particular script isn’t supported, perhaps default back to PaddleOCR CPU. For example, certain languages’ models may not be converted.

Test OCR: Use a sample image with mixed text (English and maybe another language if possible) to verify that detection boxes and recognized text are correct. Compare with PaddleOCR’s output for the same image. They should be nearly identical (OpenVINO inference shouldn’t change accuracy). Adjust thresholds if needed (the DB detector requires choosing a binarization threshold and box polygon smoothing – Paddle default threshold is 0.3). We might use PaddleOCR’s published hyperparameters.

Performance check: Running the new OCR on a page and measure speed vs old. If the text detection is still a bit slow, we could try limiting max image size or using a smaller OCR model (like the “mobile” version for lower latency). But since we target Arc GPU, we can likely handle the “server” models.

4.5 MFR / formula recognition steps

Keep on CPU: In UnimernetModel (or wherever formulas are recognized):

Ensure that when loading the model, we do not attempt to put it on GPU (remove .to(device) if device is XPU or CUDA). Instead, always use CPU or at most try device="CPU".

If MinerU allows specifying device in config for this model, override it to CPU in an Intel deployment.

Document this choice clearly so users know formula recognition will be CPU-bound.

(Optional) Investigate ONNX: If we have time, attempt a quick ONNX export of the UniMERNet model for curiosity (using torch.onnx.export). However, likely it involves dynamic decoding which is not easily exportable. So this is more of a future task.

Test formula output: Ensure that with the rest of pipeline on OpenVINO, the UniMERNet can still run on CPU and produce correct LaTeX. There should be no change to formula accuracy or output, just possibly slower if it was on a GPU before.

4.6 Configuration and environment

Device string management: We standardize how devices are specified:

If using OpenVINO for a model, we use "intel:gpu" or "intel:cpu" accordingly. We might want to allow an environment variable to specify this. e.g. OPENVINO_DEVICE=GPU or =CPU. Then in code, do:

ov_device = os.getenv("OPENVINO_DEVICE", "GPU")
device_str = f"intel:{ov_device.lower()}"


Then use device_str in Ultralytics calls or ORT provider config.

For PyTorch models that remain, if device=="cuda" is in config but no CUDA present (on an Intel machine), we should intercept that and use CPU or XPU. Possibly MinerU has logic to default to CPU if CUDA not found. We’ll confirm. We might explicitly set device_cpu for those in our documentation to avoid any confusion.

Environment variables or config toggles: Introduce:

MINERU_USE_OPENVINO=1 to enable OpenVINO for YOLO and OCR.

MINERU_OPENVINO_DEVICE=GPU (or CPU) as discussed.

Possibly MINERU_USE_ONNXRUNTIME_OPENVINO=1 if we go the ORT route, but we can tie that under the generic use_openvino flag.

We will implement these checks in model init code for layout, mfd, ocr.

Dependencies: Ensure requirements.txt or pyproject.toml include necessary packages:

openvino==2025.0 (for OpenVINO runtime if we use it directly).

onnxruntime-openvino==1.16.0 (if we use the ORT EP, although installing onnxruntime with openvino EP might just be onnxruntime==1.16.0 and it includes OpenVINO EP as an option if OpenVINO is installed – we should verify if a separate package is needed or if the OpenVINO EP is built-in. There is an onnxruntime-openvino package which might be specific.)

Alternatively, onnxruntime-gpu plus OpenVINO enabled? Actually onnxruntime usually has openvino as an EP plugin. We might need pip install onnxruntime-openvino. We’ll confirm the best practice from Microsoft’s docs or simply use OpenVINO runtime directly.

ultralytics>=x.y (to ensure we have YOLOv10 support and OpenVINO integration, probably need a fairly recent version given YOLOv10 is mentioned. We might pin a version that is tested, e.g. ultralytics 8.0.XX from 2024).

paddleocr can potentially be removed if we fully replace it, but we might keep it for now in case of fallback or data structures.

Logging and error handling: Add clear logs when loading models:

Print “Loaded DocLayout-YOLO OpenVINO model on GPU” or similar, so we can verify from logs that OpenVINO path was taken.

If OpenVINO model files are not found, fall back to PyTorch with a warning. (E.g., if user forgot to export, we either auto-export if possible – though doing heavy export at runtime isn’t ideal – or at least warn and use original model on CPU).

Similarly for OCR, if ONNX files missing, either instruct user to run a script to generate them or catch and fallback to PaddleOCR.

Testing after integration: We will run the entire pipeline on a few sample PDFs through the CLI or API to ensure everything works end-to-end:

A digital PDF (to test layout + content merging without OCR).

A scanned PDF (to test OCR path).

A math-heavy PDF (to test formula detect+recognize).

A table-heavy PDF (to see how table parsing behaves on CPU).
Check the outputs (Markdown/JSON) for completeness and correctness.

By following this checklist, our team can methodically implement the migration. We’ll use context7 to open each mentioned file to verify function names and adjust the pseudocode into actual code diffs.

Testing & benchmarking strategy

With the new OpenVINO-enabled pipeline, testing involves both functional validation and performance benchmarking:

Functional Testing

We will prepare a small suite of PDFs that cover all the key components:

Complex Layout PDF – e.g. an academic paper with multi-column layout, various headings, lists, figures, tables. Goal: Verify layout detection (DocLayout-YOLO) correctly identifies all regions and that reading order assembly is correct. Also test heading classification if enabled. Check: Number of detected blocks and their types should match expected (compare against original MinerU output as ground truth). Any shift in coordinates or missing elements would indicate an issue in detection or parsing. We expect identical results because the model hasn’t changed – if differences appear, inspect the thresholding or NMS (maybe OpenVINO vs PyTorch minor discrepancies). We can measure IoU of each detected region vs baseline to ensure high overlap.

Table-heavy PDF – a document with complex tables (spanning multiple pages, with/without borders). Goal: Ensure table detector and recognizer still work. Check: The output HTML/LaTeX of tables should be logically the same as before. Minor differences might occur if any nondeterminism in generation, but structure should match. Functionally, verify that our pipeline didn’t break cross-page table merging. Since table model remains on CPU, this also tests that mixing CPU (table) and OpenVINO (layout) doesn’t cause any thread safety issues (OpenVINO and PyTorch can run in parallel threads). Also measure the time spent on table processing separately – likely still a bottleneck.

Math-heavy PDF – e.g. a scientific article dense with equations (display and inline if possible). Goal: Test formula detection and recognition. Check: All formula regions are found and the LaTeX output from UniMERNet is correct. We will compare the LaTeX with ground truth if available or at least ensure it’s not gibberish. Since formula detection is now OpenVINO, ensure that none are missed. Also test that formulas inside dense text (if they exist as inline) are handled appropriately (though DocLayout might treat them as separate blocks or not at all – need to know how inline formulas are dealt with; possibly they appear within paragraphs and MinerU relies on OCR or something for them – unclear, but at least block equations should be handled).

OCR-challenging PDF – a scan with mixed languages (for example, an English document that has some French or Spanish text, or a multilingual dataset page). Goal: Validate OCR pipeline. Check: The recognized text content should match the actual text. Because we replaced the OCR backend, we must carefully verify accuracy. We’ll use known text for comparison or at least manually check a few lines. We should also test a case with vertical text or rotated text if possible (to see if our pipeline handles orientation – PaddleOCR’s classifier or our strategy to rotate images if needed). If we didn’t implement the orientation classifier, heavily rotated text might be misread; we will note that as a limitation if so. Also, test non-Latin script (if available, e.g. Chinese or Cyrillic text) to ensure the model and decoding covers it. If not, consider loading the appropriate model/dictionary.

For each of the above, we will run MinerU’s pipeline end-to-end and produce the markdown/JSON output. We will then:

Diff the structured output against a baseline (perhaps run the original MinerU pipeline on the same input if possible). They should largely match. Differences in ordering or minor content would be inspected.

Specifically log any errors or warnings that appear (e.g. any fallback messages we added).

Validate that no component crashes or returns None when it shouldn’t.

If any regressions are found, debug those components (e.g., if text detection finds too few boxes, adjust threshold; if layout missed something, maybe we need to raise conf_thres or ensure dynamic shape was handled, etc.).

Performance Benchmarking

Our goal is to quantify speed improvements on Intel hardware (Arc GPU and CPU). We will measure:

Per-page model timings: instrument the pipeline to log time spent in each model:

Layout detection time per page.

Formula detection time per page (for pages with formulas).

OCR time per page (for pages where OCR is applied).

Table processing time per table.

Overall pipeline time per page (from PDF page image to structured output).

We’ll run the test PDFs on:

Intel Arc GPU (with OpenVINO): This is the target scenario. We expect big gains in layout and formula detection. For example, if layout took 100ms on CPU per page, on Arc it might drop to ~15-20ms
docs.ultralytics.com
 for a large model. OCR detection might drop from, say, 500ms CPU to maybe 100-200ms on GPU (depending on image size).

CPU (OpenVINO vs original): Also test on CPU only. OpenVINO-optimized models on CPU can be compared to the original PyTorch/Paddle models on CPU. We expect:

YOLO models: ~2-3× faster on CPU with OpenVINO
ultralytics.com
docs.ultralytics.com
. Ultralytics data shows YOLOv8n from 16ms -> ~7ms on CPU with OpenVINO FP32
docs.ultralytics.com
, and int8 gave ~6-7ms with slight mAP drop. For larger models, speedup is even more (YOLOx from 212ms -> 18ms FP32)
docs.ultralytics.com
, meaning heavy models get ~10× speedup on CPU.

OCR detection: According to a Medium report, converting PP-OCR detector to OpenVINO gave up to 15× speedup on CPU
medium.com
. We should measure ourselves; possibly from ~1s to ~200ms for text detection on a page.

OCR recognition: Each line is small, speed may not change drastically per line, but overall throughput might improve by ~2-3× by using vectorized operations on CPU or GPU.

Table model: still PyTorch on CPU, so no change. This will dominate CPU time if many tables; we’ll note that.

We will compile results in a small table (for internal evaluation) like:

Layout: CPU original vs CPU OpenVINO vs Arc OpenVINO.

OCR: CPU vs Arc.

etc.

The expectation is that on Arc GPU, the pipeline throughput (pages per second) will improve significantly especially for visually-heavy pages. For example, if originally a page with OCR took ~3 seconds on CPU, with GPU it might take <1 second. We’ll particularly highlight:

Latency per page for different doc types (text-only vs scanned) before and after.

If possible, do a throughput test on multi-page PDF (like 50 pages) to see if GPU is fully utilized (maybe batch processing helps achieve higher throughput by parallelizing across pages or within page for OCR lines, etc.). OpenVINO’s “AUTO” device might even execute OCR detector on GPU and recognition on CPU concurrently – something to consider if it can overlap, but that may be too granular.

We should also monitor system metrics during tests:

GPU utilization (to ensure we are indeed using the Arc GPU effectively).

CPU utilization (should drop for tasks moved to GPU, freeing CPU for other tasks or for parallel processing of multiple pages in multi-thread scenarios).

Memory usage (OpenVINO models tend to use less memory than PyTorch; but we load more models now possibly – verify we don’t exceed typical memory).

Accuracy/regression testing:

Though we don’t expect changes in accuracy (same weights, just different runtime), we should do a quick sanity:

Use the official evaluation scripts from PDF-Extract-Kit if available (they have evaluation for layout, OCR, etc. 
pdf-extract-kit.readthedocs.io
). We could run their evaluation on a small dataset to ensure mAP for layout and formula det is unchanged (should be within 0.001).

Check that the number of OCR errors (character error rate) on a sample is same or better. OpenVINO shouldn’t change it, but if our custom postprocessing had a bug, it might. So this is a good check.

Stability and edge cases:

We will test:

Running the pipeline in Docker (since user uses containers) to ensure any GPU access issues are resolved (Arc GPU needs --device /dev/dri etc. and proper drivers on host).

Test with multiple instances or threads if MinerU supports parallel processing, to see if OpenVINO can handle concurrent calls. (OpenVINO by default should be thread-safe; onnxruntime too, but may need session per thread or enable intra_op threads.)

By performing these tests, we’ll gain confidence that:

The pipeline outputs the same quality of results.

The speed benefits are realized and measured, which we can report (e.g., “layout analysis X times faster, OCR Y times faster on Arc GPU vs CPU”).

We catch any new issues (like an OpenVINO plugin error or onnxruntime not finding the device) before deploying to production.

Intel-specific notes and pitfalls (Arc/XPU + OpenVINO)

Deploying on Intel Arc GPUs with OpenVINO and oneAPI requires some special considerations. Our combined experience yields the following do’s and don’ts:

Driver and Runtime Setup: Make sure the Intel oneAPI GPU drivers are installed on the host. For Arc on Linux, this means installing Intel’s Level-Zero/OpenCL drivers (e.g. the Intel GPU driver package). OpenVINO will use Level-Zero (oneAPI) to access the Arc GPU. On first OpenVINO GPU inference, it may take a couple of seconds to load caches – that’s normal. Ensure the Docker container has access to /dev/dri for GPU. It’s recommended to use the latest OpenVINO (2025.0 or newer) which has improved Arc support, and pair it with an updated driver (Arc support was significantly improved in 2024 drivers). Using an older driver can cause suboptimal performance or even offloading to CPU silently.

Avoiding PyTorch XPU conflicts: We found PyTorch XPU (Intel’s GPU plugin for PyTorch) is still experimental (e.g. missing certain torchvision ops). Running PyTorch XPU and OpenVINO in the same process could also lead to competition for the GPU. Recommendation: Offload all heavy vision inference to OpenVINO, and run any remaining PyTorch models on CPU. If one does attempt to use PyTorch XPU for something, be aware that memory is limited (Arc A770 has 16GB, shared with system RAM possibly) – and OpenVINO will also allocate memory on the GPU. It’s possible to run both, but you might need to set environment variables to limit OpenVINO’s memory (OpenVINO uses GPU memory caching). In general, it’s cleaner to not use PyTorch XPU concurrently with OpenVINO on the same GPU.

OpenVINO Device Strings: When specifying devices for OpenVINO, use the correct identifiers. "GPU" directs OpenVINO to use any available GPU (which includes Arc or integrated Iris Xe). "CPU" uses CPU (with multi-threading and vectorization). "AUTO" can be used in OpenVINO (not via Ultralytics directly, but via core.compile_model("AUTO")) to let OpenVINO choose the best device (it will pick GPU if free, else CPU). We could use AUTO for simplicity so it runs on GPU when present. In Ultralytics, "intel:gpu" is effectively mapping to OpenVINO’s GPU device
docs.ultralytics.com
. There is also "HETERO:GPU,CPU" mode in OpenVINO if we wanted to split parts of a model, but not needed here.

Arc GPU Performance Quirks: Arc GPUs perform best with FP16/BF16 precision. OpenVINO will automatically use FP16 precision kernels on Arc by default. We don’t usually need to do anything special, but it’s good to know. Also, Arc’s pure int8 support is limited – it can run int8 on the Vector Engines but often FP16 is just as good. So focus on FP16 optimization rather than int8 for Arc. On CPU, int8 (with AVX512 VNNI) is very beneficial, but that’s separate.

Threading and CPU usage: OpenVINO’s CPU inference and GPU inference both use the CPU for some scheduling overhead. By default, OpenVINO might use a number of threads equal to your cores. Since we’ll mainly use GPU, we can consider setting OMP_NUM_THREADS to a lower number to avoid stealing too much CPU time (the GPU tasks don’t need many helper threads). Alternatively, use OpenVINO Core to set intra_op_parallelism_threads for CPU if needed. For Arc GPU plugin, also ensure hyper-threading doesn’t confuse measurements – not a big issue but keep in mind.

Memory and GPU-sharing: If the Arc GPU is also used for display (in a desktop scenario), heavy compute might throttle if the GPU gets too hot or if VRAM is constrained. For a headless server with Arc, no issue. But if running a GUI on the same GPU, monitor utilization. It might be worth setting power management to maximum performance mode for consistent results. Also, avoid running other GPU-heavy tasks (like video encoding) simultaneously as that could affect throughput.

oneAPI and OpenCL ICD: The container/host must have the Intel GPU ICD (Installable Client Driver) for oneAPI. If OpenVINO cannot find a GPU device, it usually means the ICD is missing or not mounted in container. The fix is to install intel-level-zero-gpu and intel-opencl-icd on the host and ensure /etc/OpenCL/vendors/intel.icd is present, and similarly for Level Zero (it should just work if driver is there). In Docker, we may need to include --device /dev/dri and perhaps use the intel/intel-oneapi-runtime:latest base or similar.

OnnxRuntime OpenVINO EP issues: If using onnxruntime with OpenVINO EP, be aware of a known issue: sometimes on multi-thread, ORT+OpenVINO can throw pthread_setaffinity_np warnings or similar (there’s a GitHub issue about this)
github.com
. It’s usually benign, but if it occurs, setting an env var OMP_WAIT_POLICY=PASSIVE or pinning threads might help. We’ll watch for that in testing. Alternatively, using OpenVINO runtime directly avoids this.

Don’t mix incompatible drivers: If the system has an older version of oneAPI or an older OpenVINO in PATH, it could conflict. We should use a consistent version. We might containerize the whole stack with Intel’s official OpenVINO runtime image as a base to ensure all dependencies align.

Monitoring: Use tools like intel_gpu_top (on Linux) to monitor GPU usage. This can confirm that inference is actually happening on GPU (you’ll see EU array active) and how much GPU memory is used. If it shows 0% GPU usage and only CPU is high, something is misconfigured.

Fallbacks: Always have a CPU fallback path. If for some reason the OpenVINO execution fails (maybe unsupported operation – though for YOLO and PPOCR it should be fine), catch the exception and log a clear error. Then either try default onnxruntime CPU or PyTorch CPU to not crash the pipeline. It’s better the pipeline runs (slower) than fails entirely.

Pitfall – GPU memory allocation: By default, OpenVINO will try to allocate a chunk of GPU memory for graphs and streams. If we load multiple large models (layout YOLO, OCR det, OCR rec all on GPU), we might push the limits of a smaller GPU. Arc A770 has 16GB, which is plenty for our models, but if using an integrated GPU with 2GB, it could be an issue. Monitor memory. If needed, we can load some models on CPU or use OpenVINO’s AUTO device such that if GPU memory is low, it will offload to CPU. Also, releasing models (by destroying the compiled model or session) when not needed can free memory (e.g. if OCR is only needed for scanned PDFs, maybe don’t load it for digital PDFs; but dynamic loading/unloading might complicate things).

PyTorch CPU optimizations: Since we still use PyTorch on CPU for some parts, ensure BLAS libraries are utilized. PyTorch should use MKL by default. No special action needed, just note that having MKL in the environment yields good performance for the remaining CPU tasks (like UniMERNet).

In summary, do use the latest OpenVINO and Intel drivers, do leverage Intel GPU with OpenVINO’s device flags, and do monitor resource usage. Don’t rely on PyTorch XPU for unsupported ops, and don’t assume everything will magically use the GPU (explicitly set the device for OpenVINO). By adhering to these guidelines, we can avoid common pitfalls and ensure a stable, fast deployment on Arc.

Open questions / things to confirm with code (for context7)

Finally, to proceed with implementation, we have some specific details to verify in the MinerU codebase. We will use context7 (our code inspection tool) to open these files and confirm our assumptions:

DocLayoutYOLOModel class signature and parsing – In mineru/model/layout/doclayoutyolo.py:

Confirm how __init__ is defined (parameters like weight or weight_dir, and device). Check if it already has any logic for different devices or CPU/GPU.

Inspect _parse_prediction or similar method to see how it handles Ultralytics results (does it access prediction[0].boxes.xyxy or .cpu() or such). We need to ensure compatibility with OpenVINO results (likely fine, but we’ll verify if any Torch-specific calls need change).

See if predict_images or predict_pdfs methods exist and how they utilize predict. If they batch images, ensure our device handling works in that context.

YOLOv8MFDModel in mineru/model/mfd/yolo_v8.py:

Open this file to see the class YOLOv8MFDModel (or similar name).

Check how it loads the model (likely using Ultralytics YOLO as well). Does it have a weight or weight_dir param? We saw in a blog that weight is used
blog.csdn.net
.

Confirm if there is also a classification model (perhaps a YOLOv8MFCModel or if MFD covers detection and classification tasks together). If there's anything named “MFC” or usage of classification heads.

Determine what the output of this model’s predict is used for. Possibly it returns a list of formula bounding boxes. See if anywhere in pipeline they do something like if formulas: for box in formulas: crop and send to recognizer.

OCR integration – Find where PaddleOCR is used:

Search for “PaddleOCR” or “ppocr” in the repo. Possibly in mineru/model/ocr/ or directly in pipeline code.

Look for a MultiLanguageOCR or OCRRecognizer class. The blog snippet shows usage in HybridOCRStrategy
blog.csdn.net
 of self.ocr_engine = MultiLanguageOCR(). That suggests MultiLanguageOCR might be in mineru/model/ocr/multi_language_ocr.py or similar.

Once found, check how it loads models. Does it use paddleocr.PaddleOCR class internally? Likely yes, e.g. self.ocr = PaddleOCR(lang=..., det_model_dir=..., rec_model_dir=...). See what arguments it uses and how it chooses models for different languages (lang param might be passed through).

This will tell us what model files are expected (e.g. it might download en_PP-OCRv5_rec etc.). Also check if onnxruntime is mentioned (maybe not, if they fully rely on Paddle).

We will plan to intercept/replace this class’s implementation with our ONNXRuntime inference, but to do so cleanly, understanding its API: does it have a method like get_textlines(image) or does it directly return structured output? We suspect they use ocr.ocr(image) from Paddle which returns [ [box, text, conf], ... ].

Table model implementation:

Open mineru/model/table/... – possibly there’s wired.py or table_recognizer.py.

Identify WiredTableDetector.detect() and TableStructureRecognizer.recognize(). See what models or libraries they call. For instance, WiredTableDetector might use OpenCV or a small ML model (maybe a U-Net for line detection). If it’s a small ML model in ONNX, maybe onnxruntime is used (maybe in the code, we’ll search for onnx).

Check TableStructureRecognizer: Does it load the HF Model from local path? Perhaps using from transformers import AutoModel with model=StructEqTable. Or maybe they integrated ModelScope’s API (the HF commit readme suggests using ModelScope to download models
huggingface.co
huggingface.co
, they might then use ModelScope’s pipeline to run it). If they use ModelScope, it might be running on GPU with PyTorch by default.

Confirm that it indeed only tries to use CUDA. If it has an option for device, maybe we can set it to CPU. We likely see something like device = torch.device("cuda" if torch.cuda.is_available() else "cpu") in that code. On an Arc machine without CUDA, it would default to CPU automatically. If they hard-require CUDA, that’s an issue – hopefully not.

Heading classification or other small models:

Search for “heading” or “classifier” in the code. See if any model is used for heading classification (maybe a simple fasttext or huggingface model). If found, note how it’s loaded. It may already be an ONNX or PyTorch on CPU. We might not change it, but just confirm we’re not missing migrating something obvious.

Likewise, search for any usage of “onnxruntime” to see if any part of the code already uses onnx (if yes, perhaps for AMD or fallback for Paddle?).

Device handling in pipeline:

In pipeline_analyze.py (or wherever the pipeline_process_flow is implemented as per the pseudocode
blog.csdn.net
), see how it manages devices. Possibly it doesn’t explicitly pass devices to model methods, relying on each model object being initialized on the correct device. But maybe they do something like:

layout_results = layout_model.predict(page_image)  # model internally on GPU
formulas = formula_model.predict(page_image)  # etc.


If so, we’re fine. If they were moving image tensors to GPU, we’ll strip that out because Ultralytics expects numpy or PIL image on CPU and handles transfer internally.

Ensure no part of the pipeline assumes a CUDA context. For example, if they use torch for merging outputs or something, it should all be on CPU. That should be fine but worth a check.

Model weight path management:

Confirm how MinerU knows where weight files are. Possibly they auto-download to ~/.cache via huggingface. The weight loading mechanism might try HF hub if local not provided. The CSDN article
blog.csdn.net
blog.csdn.net
 mentioned older approach was HF cache. They added support for custom local paths in 2.0.6. So likely in config or code, there’s logic: if weight_dir provided, load from there; else download from HF.

We will use this: we’ll point weight_dir to our local weights (and IR). For instance, we might have an env var MINERU_MODEL_DIR where we’ve placed all required model files (OpenVINO IRs and any others). We’ll set weight_dir for layout to that, so it doesn’t attempt to download PyTorch weights.

We also might need to trick it: if we keep the same naming, e.g., put yolov10l_ft_openvino_model folder under the expected weight_dir, our code will load it. We should ensure the code doesn’t try to validate the model file extension, etc.

Also ensure that for OCR we either use local ONNX or if in a container we bake them in.

Summarize: after code changes, we’ll likely run MinerU with something like mineru.run(config="mineru_config.json", use_openvino=1, model_dir="/models"). We might need to add these options.

These open questions will guide our code inspection. We’ll resolve them with context7, then implement accordingly. Each item corresponds to verifying or retrieving exact code to plug into our solution.