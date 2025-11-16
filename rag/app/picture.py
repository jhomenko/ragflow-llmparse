#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

import io
import re
import logging
import os
from typing import Optional

import numpy as np
from PIL import Image

from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from deepdoc.vision import OCR
from rag.nlp import tokenize
from rag.utils import clean_markdown_block
from rag.nlp import rag_tokenizer
from rag.llm.working_vlm_module import describe_image_working


class VisionLLMCallError(Exception):
    """Raised when the working VLM cannot be reached or returns transport errors."""


class VisionLLMResponseError(Exception):
    """Raised when the working VLM returns unusable text (e.g., repetition loops)."""


def extract_base_model_name(full_model_name):
    """
    Extract base model name from RAGFlow's composite model format.

    RAGFlow stores models as: {model_name}___{provider}@{api_type}
    Example: "Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible"

    This function returns just the base model name: "Qwen2.5VL-3B"

    Args:
        full_model_name: Full model name from LLMBundle

    Returns:
        Base model name (everything before '___')
    """
    if not full_model_name:
        logging.info("extract_base_model_name: received empty or None model name, returning 'unknown'")
        return "unknown"

    # Split on '___' and take first part
    if "___" in full_model_name:
        base_name = full_model_name.split("___")[0]
        logging.info(f"Extracted base model name: '{base_name}' from '{full_model_name}'")
        return base_name

    # No separator found, return as-is
    return full_model_name

ocr = OCR()

_REPEAT_HINT = (
    "Please avoid repeating the same sentence or table multiple times. "
    "If content is duplicated in the source, summarize it once and continue."
)


def _normalize_working_result(result):
    text = None
    token_count = None
    if isinstance(result, tuple):
        if len(result) >= 1:
            text = result[0]
        if len(result) >= 2:
            token_count = result[1]
    else:
        text = result

    if text is None:
        text = ""
    if not isinstance(text, str):
        try:
            text = str(text)
        except Exception:
            text = ""

    text = clean_markdown_block(text).strip()

    if token_count is None or not isinstance(token_count, (int, float)):
        token_count = 0
    return text, int(token_count)


def _has_repetitive_pattern(text: str) -> bool:
    normalized = (text or "").strip()
    if len(normalized) < 300:
        return False

    for pattern_len in (50, 100, 200):
        if len(normalized) < pattern_len * 3:
            continue
        pattern = normalized[:pattern_len]
        # str.count works well enough for obvious loops (copy/paste)
        repeats = normalized.count(pattern)
        if repeats >= 3:
            return True

    # Fallback: check first few non-empty lines; if mostly identical and there are many lines, treat as repetition
    lines = [ln.strip() for ln in normalized.splitlines() if ln.strip()]
    if len(lines) >= 15:
        sample = lines[:5]
        if len(set(sample)) <= 1:
            return True
    return False


def _validate_vlm_text(text: str) -> None:
    if not text or not text.strip():
        logging.warning("vision_llm_chunk: VLM returned empty text")
        return

    if _has_repetitive_pattern(text):
        raise VisionLLMResponseError("Detected repeated pattern suggesting the model looped")


def _call_working_vlm_with_retry(image_bytes: bytes, prompt: str, model_name: str, callback):
    max_attempts = max(1, int(os.getenv("VLM_PAGE_MAX_ATTEMPTS", "2")))
    prompt = prompt or ""
    last_error: Optional[Exception] = None

    for attempt in range(max_attempts):
        temperature = min(0.1 + attempt * 0.1, 0.4)
        attempt_prompt = prompt
        if attempt > 0:
            attempt_prompt = (prompt.rstrip() + "\n\n" + _REPEAT_HINT).strip()
            logging.info(
                "vision_llm_chunk: retrying with repetition hint, attempt %s/%s, temperature=%.2f",
                attempt + 1,
                max_attempts,
                temperature,
            )
        else:
            logging.info("vision_llm_chunk: attempt %s/%s at temperature %.2f", attempt + 1, max_attempts, temperature)

        try:
            result = describe_image_working(
                image_bytes=image_bytes,
                prompt=attempt_prompt,
                model_name=model_name,
                temperature=temperature,
            )
        except Exception as exc:
            last_error = VisionLLMCallError(f"Working VLM call failed: {exc}")
            logging.exception("vision_llm_chunk: working VLM module failed on attempt %s: %s", attempt + 1, exc)
            continue

        text, token_count = _normalize_working_result(result)
        try:
            _validate_vlm_text(text)
            logging.info("Working VLM response: %s chars, %s tokens", len(text), token_count)
            logging.debug("Working VLM preview: %s...", text[:200])
            try:
                callback(0.95, f"Working VLM tokens: {token_count}, preview: {text[:128]}")
            except Exception:
                pass
            return text
        except VisionLLMResponseError as resp_err:
            last_error = resp_err
            logging.warning("vision_llm_chunk: unusable response on attempt %s: %s", attempt + 1, resp_err)
            continue

    raise last_error or VisionLLMCallError("Working VLM exhausted all attempts without success")


def chunk(filename, binary, tenant_id, lang, callback=None, **kwargs):
    img = Image.open(io.BytesIO(binary)).convert('RGB')
    doc = {
        "docnm_kwd": filename,
        "title_tks": rag_tokenizer.tokenize(re.sub(r"\.[a-zA-Z]+$", "", filename)),
        "image": img,
        "doc_type_kwd": "image"
    }
    bxs = ocr(np.array(img))
    txt = "\n".join([t[0] for _, t in bxs if t[0]])
    eng = lang.lower() == "english"
    callback(0.4, "Finish OCR: (%s ...)" % txt[:12])
    if (eng and len(txt.split()) > 32) or len(txt) > 32:
        tokenize(doc, txt, eng)
        callback(0.8, "OCR results is too long to use CV LLM.")
        return [doc]

    try:
        callback(0.4, "Use CV LLM to describe the picture.")
        cv_mdl = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, lang=lang)
        img_binary = io.BytesIO()
        img.save(img_binary, format='JPEG')
        img_binary.seek(0)
        ans = cv_mdl.describe(img_binary.read())
        callback(0.8, "CV LLM respond: %s ..." % ans[:32])
        txt += "\n" + ans
        tokenize(doc, txt, eng)
        return [doc]
    except Exception as e:
        callback(prog=-1, msg=str(e))

    return []


def vision_llm_chunk(binary, vision_model, prompt=None, callback=None):
    """
    Process image bytes to markdown text via VLM.

    Args:
        binary: JPEG/PNG bytes (NOT PIL Image)
        vision_model: LLMBundle instance configured for IMAGE2TEXT
        prompt: Custom prompt string (optional)
        callback: Progress callback (optional)

    Returns:
        Markdown text string
    """
    callback = callback or (lambda prog, msg: None)

    # Validate input type
    if not isinstance(binary, (bytes, bytearray)):
        err = "vision_llm_chunk expected 'bytes' for binary parameter, got %s" % type(binary).__name__
        try:
            callback(-1, err)
        except Exception:
            pass
        logging.error(err)
        return ""

    # Validate binary is not empty
    if len(binary) == 0:
        err = "vision_llm_chunk: empty binary data"
        try:
            callback(-1, err)
        except Exception:
            pass
        logging.error(err)
        return ""

    # Warn if very small (suspicious)
    if len(binary) < 100:
        logging.warning(f"vision_llm_chunk: suspiciously small image ({len(binary)} bytes)")

    # Validate vision_model presence and required method(s)
    if vision_model is None or not (hasattr(vision_model, "describe_with_prompt") or hasattr(vision_model, "describe")):
        err = "vision_llm_chunk: vision_model is not configured or missing describe methods"
        try:
            callback(-1, err)
        except Exception:
            pass
        logging.error(err)
        return ""

    # Validate/normalize prompt
    if prompt is not None and not isinstance(prompt, str):
        logging.warning(f"Invalid prompt type: {type(prompt)}, converting to string")
        try:
            prompt = str(prompt)
        except Exception:
            prompt = ""

    prompt = prompt or ""
    logging.debug(f"vision_llm_chunk: binary size={len(binary)} bytes, prompt length={len(prompt or '')}")

    use_working_module = os.getenv("USE_WORKING_VLM", "true").lower() == "true"
    if not use_working_module:
        err = "vision_llm_chunk: working VLM module disabled via USE_WORKING_VLM"
        logging.error(err)
        try:
            callback(-1, err)
        except Exception:
            pass
        raise VisionLLMCallError(err)

    full_model_name = getattr(vision_model, "llm_name", None) or getattr(vision_model, "name", None) or "unknown"
    model_name = extract_base_model_name(full_model_name)
    logging.info(f"vision_llm_chunk: Using working VLM module for model={model_name}")

    try:
        text = _call_working_vlm_with_retry(binary, prompt or "", model_name, callback)
        return text
    except Exception as exc:
        logging.exception("vision_llm_chunk failed after retries: %s", exc)
        try:
            callback(-1, f"vision_llm_chunk error: {exc}")
        except Exception:
            pass
        raise
