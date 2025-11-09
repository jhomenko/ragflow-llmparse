# direct_vision_parser.py
import base64, io, json, os, re, logging, requests
from io import BytesIO
import pdfplumber
from PIL import Image

DEFAULT_SYS = "You are a meticulous PDF-to-Markdown transcriber. Output ONLY valid Markdown."
DEFAULT_TEMP = 0.0
DEFAULT_MAX_TOK = 4096

def _to_jpeg_bytes(pil_img, max_side=2000, quality=90):
    img = pil_img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / float(max(w, h))
        img = img.resize((int(w*s), int(h*s)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return buf.getvalue()

def _clean_md(s: str) -> str:
    # Strip ```markdown fences if the model adds them
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:markdown)?\s*", "", s)
        s = re.sub(r"\s*```$", "", s)
    return s.strip()

class DirectVisionParser:
    """
    Minimal, robust PDF->image->llama.cpp vision path using OpenAI-compatible /v1/chat/completions.
    Env:
      LLAMA_SERVER_URL : e.g. http://127.0.0.1:8080
      LLAMA_MODEL      : e.g. Qwen2.5VL-3B
    """
    def __init__(self, server_url=None, model=None, system_prompt=None, temp=None, max_tokens=None):
        self.server_url = server_url or os.environ.get("LLAMA_SERVER_URL", "http://127.0.0.1:8080")
        self.model = model or os.environ.get("LLAMA_MODEL", "Qwen2.5VL-3B")
        self.system_prompt = system_prompt or os.environ.get("VLM_SYSTEM_PROMPT", DEFAULT_SYS)
        self.temp = DEFAULT_TEMP if temp is None else temp
        self.max_tokens = DEFAULT_MAX_TOK if max_tokens is None else max_tokens

    def _post_one(self, jpg_bytes: bytes, user_prompt: str) -> str:
        b64 = base64.b64encode(jpg_bytes).decode("ascii")
        payload = {
            "model": self.model,
            "temperature": self.temp,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
                ]}
            ]
        }
        r = requests.post(f"{self.server_url}/v1/chat/completions",
                          headers={"Content-Type": "application/json"},
                          data=json.dumps(payload), timeout=120)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return _clean_md(content)

    def __call__(self, filename_or_bytes, from_page=0, to_page=100000, prompt_fn=None, zoomin=3, callback=None):
        """
        Returns: ([(text, meta_str), ...], [])
        meta_str keeps the same @@<page>\t<x0>\t<x1>\t<top>\t<bottom>## shape used upstream.
        """
        cb = callback or (lambda *_: None)
        # open PDF
        with pdfplumber.open(filename_or_bytes if isinstance(filename_or_bytes, str)
                             else BytesIO(filename_or_bytes)) as pdf:
            total = len(pdf.pages)
            start = max(0, from_page)
            end = min(to_page, total)
            out = []
            for idx, pg in enumerate(pdf.pages[start:end], start=start):
                # Rasterize page
                pil_img = pg.to_image(resolution=72 * zoomin).original
                jpg = _to_jpeg_bytes(pil_img)
                # Prompt: inject page number if your template expects it
                user_prompt = prompt_fn(idx + 1) if prompt_fn else "Transcribe this PDF page to clean Markdown."
                txt = self._post_one(jpg, user_prompt) or ""
                w, h = pil_img.size
                out.append((txt, f"@@{(idx+1)}\t0.0\t{w/zoomin:.1f}\t0.0\t{h/zoomin:.1f}##"))
                cb((idx-start+1)/(end-start or 1), f"Processed: {idx+1}/{end}")
        return out, []
