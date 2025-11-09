# VLM Message Format Fix - CRITICAL BUG FOUND

## Root Cause Identified

The VLM is **echoing the user prompt** instead of generating content because we're sending messages in the wrong format for vision models.

## Evidence

**Server Log:**
```
Token usage: completion_tokens=257, prompt_tokens=1373, total_tokens=1630
VLM response received: chars=1192 tokens=1630
```

**Model Output:**
The exact prompt text verbatim: `"## INSTRUCTION Transcribe the content..."`

**Analysis:**
- Prompt tokens: 1373 (system ~170 + user ~1192 + image)
- Response length: 1192 chars = **Exact user prompt length!**
- The VLM is returning the user prompt as if it's the response

## The Problem

### Current Code (rag/llm/working_vlm_module.py:64-80)
```python
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

response = self.client.chat.completions.create(
    model=model_name,
    messages=[system_message, user_message],  # ← PROBLEM: Separate system + user
    max_tokens=4096,
    temperature=0.1,
    stop=[],
)
```

### Why This Fails

Vision models (especially Qwen2.5VL) **don't properly support separate system messages** when images are involved. The API is:
1. Either ignoring the system message and echoing the user prompt
2. Or concatenating them incorrectly, causing the model to return the prompt text

### Working Test Script Format

The test script that worked used:
```python
prompt="Transcribe this PDF page to clean Markdown."  # Simple, short prompt
```

But RAGFlow now sends a **long instructional prompt** (~1192 chars) from the template, which the VLM is echoing back!

## The Fix

### Solution: Combine System + User into Single User Message

**File:** `rag/llm/working_vlm_module.py`  
**Lines:** 63-80

**Change from:**
```python
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
logger.info("VLM call parameters: max_tokens=4096, temperature=0.1, stop=[]")
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
    max_tokens=4096,
    temperature=0.1,
    stop=[],  # Explicitly disable default stop tokens
)
```

**Change to:**
```python
# CRITICAL FIX: VLMs don't support separate system messages properly
# Combine everything into a single user message with the image
# This matches the working test script behavior

# No separate system message - combine it into the user prompt
combined_prompt = prompt  # The prompt already contains full instructions from template

logger.info(f"Calling VLM: model={model_name} prompt_len={len(combined_prompt)}")
logger.info("VLM call parameters: max_tokens=4096, temperature=0.1, stop=[]")
logger.info(f"Combined prompt (first 200 chars): {combined_prompt[:200]}")

# Single user message with text + image (standard VLM format)
user_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": combined_prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ],
}

# Call VLM with ONLY user message (no system message)
response = self.client.chat.completions.create(
    model=model_name,
    messages=[user_message],  # ← FIX: Single user message only
    max_tokens=4096,
    temperature=0.1,
    stop=[],  # Explicitly disable default stop tokens
)
```

## Why This Fix Works

1. **Matches Working Test**: Your test script sent a simple prompt in the user message with the image
2. **Standard VLM Format**: Vision models expect: `[{"role": "user", "content": [text, image]}]`
3. **No System Message Confusion**: Eliminates the system message that VLMs don't handle properly
4. **Prompt Contains Instructions**: The template already has all instructions, no need for separate system message

## Verification

After applying this fix:

1. **Expected Behavior:**
   - VLM receives: Single user message with prompt + image
   - VLM generates: Actual transcription (not prompt echo)
   - Response length: 2000-4000 chars (actual content)

2. **Test Command:**
   ```bash
   # Restart container (file change only, no rebuild needed)
   docker-compose restart ragflow
   
   # Upload test PDF and check response length
   ```

3. **Success Criteria:**
   - Response is NOT the prompt text
   - Response contains actual PDF content
   - Response length matches test script (~3547 chars)

## Additional Debugging

If issue persists, add logging to see what the VLM actually received:

```python
logger.info(f"Message structure being sent: {[user_message]}")
logger.info(f"Full response object: {response}")
logger.info(f"Response message content: {response.choices[0].message.content[:500]}")
```

## Implementation Priority

**CRITICAL - Implement Immediately**

This is blocking all VLM PDF parsing. The fix is simple but must be tested carefully.

## Related Files

- **Primary:** `rag/llm/working_vlm_module.py` (lines 63-101)
- **Calls from:** `rag/app/picture.py` (line 168)
- **Used by:** `deepdoc/parser/pdf_parser.py` (line 1495)