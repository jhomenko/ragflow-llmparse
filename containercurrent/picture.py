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

import numpy as np
from PIL import Image

from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from deepdoc.vision import OCR
from rag.nlp import tokenize
from rag.utils import clean_markdown_block
from rag.nlp import rag_tokenizer


ocr = OCR()


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

    try:
        # Log model being used (best-effort)
        model_name = getattr(vision_model, "llm_name", None) or getattr(vision_model, "name", None) or "unknown"
        logging.debug(f"vision_llm_chunk: Calling VLM model: {model_name}")

        # Call the vision model. Prefer describe_with_prompt if available.
        if hasattr(vision_model, "describe_with_prompt"):
            result = vision_model.describe_with_prompt(binary, prompt)
        elif hasattr(vision_model, "describe"):
            result = vision_model.describe(binary)
        else:
            err = "vision_llm_chunk: vision_model has no usable describe method"
            logging.error(err)
            try:
                callback(-1, err)
            except Exception:
                pass
            return ""

        # Normalize result to text and optional token count
        txt = None
        token_count = None
        if isinstance(result, tuple):
            if len(result) >= 1:
                txt = result[0]
            if len(result) >= 2:
                token_count = result[1]
        else:
            txt = result

        # Handle None response
        if txt is None:
            logging.warning("vision_llm_chunk: VLM returned None, treating as empty")
            try:
                callback(0.5, "VLM returned no text")
            except Exception:
                pass
            txt = ""

        # Handle non-string response
        if not isinstance(txt, str):
            logging.warning(f"vision_llm_chunk: VLM returned non-string: {type(txt)}, converting to str")
            try:
                txt = str(txt)
            except Exception:
                txt = ""

        # Clean up possible markdown fences
        txt = clean_markdown_block(txt).strip()

        # Check if response looks like an error message
        error_patterns = ["error", "failed", "cannot", "unable", "invalid", "exception"]
        if txt and any(pattern in txt.lower()[:200] for pattern in error_patterns):
            logging.warning(f"vision_llm_chunk: VLM response may be an error message: {txt[:200]}")

        # Warn on very long responses (might indicate model confusion)
        if txt and len(txt) > 50000:  # ~50KB of text
            logging.warning(f"vision_llm_chunk: Very long VLM response: {len(txt)} chars, may need truncation")
            try:
                callback(0.7, f"VLM very long response ({len(txt)} chars)")
            except Exception:
                pass

        # Handle missing or invalid token count
        if token_count is None:
            token_count = 0
        elif not isinstance(token_count, (int, float)):
            logging.debug(f"vision_llm_chunk: Invalid token_count type: {type(token_count)}, setting to 0")
            token_count = 0

        # Final cleanup and validation
        txt = txt.strip() if isinstance(txt, str) else ""
        if not txt:
            logging.warning("vision_llm_chunk: VLM returned empty string after cleanup")
            try:
                callback(0.9, "VLM returned empty text")
            except Exception:
                pass
            return ""

        # Log token usage and preview
        logging.info(f"vision_llm_chunk: VLM response tokens={token_count}, chars={len(txt)}")
        logging.debug(f"vision_llm_chunk: VLM preview: {txt[:200]}...")
        try:
            callback(0.95, f"VLM tokens: {token_count}, preview: {txt[:128]}")
        except Exception:
            pass

        # If extremely long, return as-is but warn again
        if len(txt) > 200000:
            logging.warning(f"vision_llm_chunk: Returning extremely long VLM output ({len(txt)} chars)")

        return txt

    except Exception as e:
        # Log error and return empty string (do not crash)
        logging.exception("vision_llm_chunk failed")
        try:
            callback(-1, f"vision_llm_chunk error: {e}")
        except Exception:
            pass
        return ""
