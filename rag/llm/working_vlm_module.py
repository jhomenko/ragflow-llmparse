#!/usr/bin/env python3
"""
Working VLM Module - Uses proven OpenAI client code
Ported from test_vlm_pdf_complete.py
"""

import logging
import base64
import os
from typing import Optional, Tuple

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

logger = logging.getLogger(__name__)


class WorkingVLMClient:
    """
    VLM client using the proven working code path.
    Direct port of test_vlm_pdf_complete.py's OpenAI usage.
    """

    def __init__(self, base_url: Optional[str] = None, api_key: str = "not-needed"):
        self.base_url = base_url or os.getenv(
            "VLM_BASE_URL", "http://192.168.68.186:8080/v1"
        )

        if OpenAI is None:
            logger.error("openai.OpenAI client is not available")
            raise RuntimeError("openai.OpenAI client not available")

        # Use the exact working OpenAI client setup (no extra_body, no extras)
        self.client = OpenAI(api_key=api_key, base_url=self.base_url)

        logger.info("WorkingVLMClient initialized: base_url=%s", self.base_url)

    def describe_image(self, image_bytes: bytes, prompt: str, model_name: str = "Qwen2.5VL-3B") -> Tuple[str, int]:
        """
        Describe an image using the working VLM code path.

        Args:
            image_bytes: JPEG/PNG bytes
            prompt: Prompt text to send to the model
            model_name: Model name to use on the VLM server

        Returns:
            (text, token_count) tuple

        Raises:
            Exception if the VLM call fails
        """
        try:
            # Convert bytes to base64 (same as the working test)
            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            # ADD: Log image encoding details
            logger.info(f"Image encoded: {len(image_bytes)} bytes → {len(image_b64)} base64 chars")
            logger.info(f"Base64 preview (first 100 chars): {image_b64[:100]}")

            # Construct messages exactly like the working test
            system_message = {
                "role": "system",
                "content": (
                    "You are a meticulous PDF-to-Markdown transcriber. "
                    "Your task is to convert PDF pages into clean, well-structured Markdown. "
                    "Preserve text, tables, headings, and formatting. "
                    "Output ONLY the Markdown content, no explanations."
                ),
            }

            user_message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }

            logger.info(f"Calling VLM: model={model_name} prompt_len={len(prompt)}")

            # ADD: Log exact parameters being sent
            logger.info("VLM call parameters: max_tokens=8192, temperature=0.1, stop=[]")
            try:
                logger.info(f"System message length: {len(system_message['content'])}")
            except Exception:
                logger.debug("System message length: UNKNOWN")
            logger.info(f"User prompt (first 200 chars): {prompt[:200]}")

            # Call VLM exactly like the working test:
            # model, messages, max_tokens=4096, temperature=0.1
            # NO extra_body, NO stop, NO other extras
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[system_message, user_message],
                max_tokens=8192,
                temperature=0.1,
                stop=[],  # Explicitly disable default stop tokens
            )

            # ADD: Log raw response details
            logger.info(f"VLM raw response type: {type(response)}")
            logger.info(f"VLM response object: {response}")

            # ADD: Log choices structure
            try:
                choices_len = len(response.choices)
            except Exception:
                try:
                    choices_len = int(getattr(response, "choices", 0) and len(response.choices))
                except Exception:
                    choices_len = 0
            logger.info(f"Response has {choices_len} choice(s)")
            if choices_len:
                try:
                    logger.info(f"Choice[0] message type: {type(response.choices[0].message)}")
                    logger.info(f"Choice[0] message: {response.choices[0].message}")
                    logger.info(f"Choice[0] finish_reason: {getattr(response.choices[0], 'finish_reason', None)}")
                except Exception as e:
                    logger.debug(f"Failed to log choice[0] details: {e}")

            # Extract text
            try:
                text = response.choices[0].message.content
                logger.info(f"Extracted content type: {type(text)}")
                logger.info(f"Extracted content length: {len(text) if text else 0}")
                logger.info(f"Extracted content preview (first 500 chars): {text[:500] if text else 'NONE'}")
            except Exception as e:
                logger.error(f"Failed to extract content: {e}")
                text = getattr(getattr(response.choices[0], "message", {}), "content", "") if getattr(response, "choices", None) else ""
                logger.info(f"Fallback content: {text}")

            # Extract token usage safely
            token_count = 0
            try:
                # response.usage may be an object or dict
                token_count = getattr(response.usage, "total_tokens", None) or (response.usage.get("total_tokens") if isinstance(response.usage, dict) else None) or 0
                logger.info(f"Token usage: {getattr(response, 'usage', None)}")
            except Exception as e:
                logger.warning(f"Failed to get token count: {e}")
                try:
                    token_count = int(getattr(response.usage, "total_tokens", 0))
                except Exception:
                    token_count = 0

            logger.info(f"VLM response received: chars={len(text)} tokens={token_count}")

            # ADD: Check for suspicious patterns
            if len(text) < 100:
                logger.warning(f"⚠️ SUSPICIOUSLY SHORT RESPONSE: Only {len(text)} chars!")
                logger.warning(f"Full response text: '{text}'")
    
            if token_count and token_count > 1000 and len(text) < 100:
                logger.error(f"❌ TOKEN/CHAR MISMATCH: {token_count} tokens but only {len(text)} chars!")
                logger.error(f"This suggests response truncation or extraction failure")
    
            # Repetition detection: identify repeating blocks that indicate truncation mid-table
            if isinstance(text, str) and len(text) > 200:
                # Check for repeating patterns (50-char and 100-char blocks repeated 3+ times)
                for pattern_len in [50, 100]:
                    if len(text) >= pattern_len * 3:
                        pattern = text[:pattern_len]
                        # Use count on the full text to detect repetition frequency
                        try:
                            repeats = text.count(pattern)
                        except Exception:
                            repeats = 0
                        if repeats >= 3:
                            logger.warning(f"⚠️ REPETITION DETECTED: Pattern (len={pattern_len}) repeats {repeats} times")
                            logger.warning("This indicates max_tokens hit during generation. Consider increasing further or retrying.")
                            break
    
            return text, token_count

        except Exception as e:
            logger.exception(f"VLM call failed: {e}")
            raise


# Singleton instance (created on first use)
_client_instance: Optional[WorkingVLMClient] = None


def get_working_vlm_client(base_url: Optional[str] = None) -> WorkingVLMClient:
    """
    Get singleton WorkingVLMClient instance.

    Args:
        base_url: Optional override for VLM server URL

    Returns:
        WorkingVLMClient
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = WorkingVLMClient(base_url=base_url)
    return _client_instance


def describe_image_working(image_bytes: bytes, prompt: str, model_name: str = "Qwen2.5VL-3B") -> Tuple[str, int]:
    """
    Convenience wrapper to describe image using the working client.

    Returns:
        (text, token_count)
    """
    client = get_working_vlm_client()
    return client.describe_image(image_bytes, prompt, model_name)