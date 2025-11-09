# VLM Empty Response Root Cause Analysis & Fix Plan

## Executive Summary

After extensive code analysis, I've identified **THREE CRITICAL BUGS** causing the VLM to return nearly empty responses (only 6 tokens instead of full markdown):

### 🔴 Critical Bug #1: Return Value Mismatch (MOST LIKELY CAUSE)
**File**: [`api/db/services/llm_service.py:153-165`](api/db/services/llm_service.py:153)

```python
def describe_with_prompt(self, image, prompt):
    # ...
    txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)  # Gets tuple
    # ...
    return txt  # ❌ RETURNS ONLY STRING, NOT TUPLE!
```

**Impact**: [`rag/app/picture.py:136,151-156`](rag/app/picture.py:136) expects a tuple `(text, token_count)` but gets only a string. This causes the result unpacking to fail silently or behave unexpectedly.

### 🔴 Critical Bug #2: Missing System Message
**File**: [`rag/llm/cv_model.py:158-164`](rag/llm/cv_model.py:158)

The working curl test includes:
```json
{
  "messages": [
    {"role": "system", "content": "You are a meticulous PDF-to-Markdown transcriber..."},
    {"role": "user", "content": [...]}
  ]
}
```

But RAGFlow's `vision_llm_prompt()` only creates:
```python
def vision_llm_prompt(self, b64, prompt=None):
    return [
        {
            "role": "user",  # ❌ NO SYSTEM MESSAGE!
            "content": self._image_prompt(prompt if prompt else vision_llm_describe_prompt(), b64)
        }
    ]
```

**Impact**: The VLM may not understand its role as a transcriber, leading to minimal/confused output.

### 🟡 Potential Bug #3: Image Format in Messages
**File**: [`rag/llm/cv_model.py:60-75`](rag/llm/cv_model.py:60)

The working test sends:
```json
{
  "type": "image_url",
  "image_url": {"url": "data:image/jpeg;base64,<base64>"}
}
```

RAGFlow's `_image_prompt()` on line 72 does:
```python
"url": img if isinstance(img, str) and img.startswith("data:") else f"data:image/png;base64,{img}"
```

**Issue**: It assumes PNG when not a data URL, but we're sending JPEG bytes. However, `image2base64()` (line 114-143) should detect JPEG magic numbers correctly.

---

## Detailed Call Chain Analysis

### Current Flow (BROKEN):

```
1. parser.py:375
   └─> VisionParser(vision_model=vision_model)(blob, prompt_text=prompt_text)

2. pdf_parser.py:1436
   └─> picture_vision_llm_chunk(binary=jpg_bytes, vision_model=self.vision_model, prompt=final_prompt)

3. picture.py:136
   └─> result = vision_model.describe_with_prompt(binary, prompt)
       ├─> Expects: (text, token_count) tuple
       └─> Gets: STRING ONLY ❌

4. llm_service.py:153-165
   └─> txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)
       └─> return txt  ❌ MISSING TOKEN COUNT!

5. cv_model.py:187-194 (GptV4.describe_with_prompt)
   └─> messages=self.vision_llm_prompt(b64, prompt)  ❌ NO SYSTEM MESSAGE
       └─> res = self.client.chat.completions.create(...)
           └─> return (res.choices[0].message.content.strip(), total_token_count_from_response(res))
```

### Working Curl Test Flow:

```json
{
  "model": "MiMo7b",
  "messages": [
    {"role": "system", "content": "You are a meticulous PDF-to-Markdown transcriber. Your task is to convert PDF pages into clean, well-structured Markdown..."},
    {"role": "user", "content": [
      {"type": "text", "text": "<FULL PROMPT FROM vision_llm_describe_prompt.md>"},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<BASE64>"}}
    ]}
  ],
  "max_tokens": 4096,
  "temperature": 0.1
}
```

---

## Implementation Plan

### Phase 5: Fix LLMBundle Return Value (CRITICAL - DO FIRST)

**Priority**: 🔴 CRITICAL
**Estimated Time**: 5 minutes
**Files**: [`api/db/services/llm_service.py`](api/db/services/llm_service.py:153)

#### Task 5.1: Fix describe_with_prompt to Return Tuple

**Change Required**:
```python
# Line 153-165: BEFORE
def describe_with_prompt(self, image, prompt):
    if self.langfuse:
        generation = self.langfuse.start_generation(trace_context=self.trace_context, name="describe_with_prompt", metadata={"model": self.llm_name, "prompt": prompt})

    txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)
    if not TenantLLMService.increase_usage(self.tenant_id, self.llm_type, used_tokens):
        logging.error("LLMBundle.describe can't update token usage for {}/IMAGE2TEXT used_tokens: {}".format(self.tenant_id, used_tokens))

    if self.langfuse:
        generation.update(output={"output": txt}, usage_details={"total_tokens": used_tokens})
        generation.end()

    return txt  # ❌ WRONG!
```

**Fixed Version**:
```python
# Line 153-165: AFTER
def describe_with_prompt(self, image, prompt):
    if self.langfuse:
        generation = self.langfuse.start_generation(trace_context=self.trace_context, name="describe_with_prompt", metadata={"model": self.llm_name, "prompt": prompt})

    txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)
    if not TenantLLMService.increase_usage(self.tenant_id, self.llm_type, used_tokens):
        logging.error("LLMBundle.describe can't update token usage for {}/IMAGE2TEXT used_tokens: {}".format(self.tenant_id, used_tokens))

    if self.langfuse:
        generation.update(output={"output": txt}, usage_details={"total_tokens": used_tokens})
        generation.end()

    return txt, used_tokens  # ✅ RETURN TUPLE!
```

**Why This Fixes It**:
- [`rag/app/picture.py:136,151-156`](rag/app/picture.py:136) expects `(text, token_count)` tuple
- Without this, the result handling fails silently or returns wrong data
- This is likely causing the "empty" response

---

### Phase 6: Add System Message Support (HIGH PRIORITY)

**Priority**: 🔴 HIGH
**Estimated Time**: 15 minutes
**Files**: [`rag/llm/cv_model.py`](rag/llm/cv_model.py:158)

#### Task 6.1: Update vision_llm_prompt to Include System Message

**Change Required**:
```python
# Line 158-164: BEFORE
def vision_llm_prompt(self, b64, prompt=None):
    return [
        {
            "role": "user",
            "content": self._image_prompt(prompt if prompt else vision_llm_describe_prompt(), b64)
        }
    ]
```

**Fixed Version**:
```python
# Line 158-167: AFTER
def vision_llm_prompt(self, b64, prompt=None):
    """
    Create vision LLM prompt with system message for better context.
    System message tells the model its role as a PDF transcriber.
    """
    return [
        {
            "role": "system",
            "content": (
                "You are a meticulous PDF-to-Markdown transcriber. "
                "Your task is to convert PDF pages into clean, well-structured Markdown. "
                "Preserve all text, tables, headings, and formatting. "
                "Output ONLY the Markdown content, no explanations."
            )
        },
        {
            "role": "user",
            "content": self._image_prompt(prompt if prompt else vision_llm_describe_prompt(), b64)
        }
    ]
```

**Why This Matters**:
- The working curl test includes a system message
- System messages provide crucial context for the model's role
- Without it, the model may not understand it should transcribe comprehensively

#### Task 6.2: Verify System Message in Chat API Call

**File**: [`rag/llm/cv_model.py:187-194`](rag/llm/cv_model.py:187)

Verify that `describe_with_prompt` correctly passes messages to the API:
```python
def describe_with_prompt(self, image, prompt=None):
    b64 = self.image2base64(image)
    messages = self.vision_llm_prompt(b64, prompt)  # Now includes system message
    
    # Add debug logging
    logging.debug(f"VLM API call - model: {self.model_name}, messages: {len(messages)} messages")
    for i, msg in enumerate(messages):
        logging.debug(f"  Message {i}: role={msg['role']}, content_length={len(str(msg['content']))}")
    
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages,  # ✅ System + User messages
        extra_body=self.extra_body,
    )
    return res.choices[0].message.content.strip(), total_token_count_from_response(res)
```

---

### Phase 7: Verify API Call Parameters Match Working Test

**Priority**: 🟡 MEDIUM
**Estimated Time**: 20 minutes

#### Task 7.1: Add max_tokens and temperature Parameters

**File**: [`rag/llm/cv_model.py:187-194`](rag/llm/cv_model.py:187)

The working curl test uses:
- `max_tokens: 4096`
- `temperature: 0.1`

Update the API call:
```python
def describe_with_prompt(self, image, prompt=None):
    b64 = self.image2base64(image)
    messages = self.vision_llm_prompt(b64, prompt)
    
    # Match working curl test parameters
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages,
        max_tokens=4096,  # ✅ Add explicit token limit
        temperature=0.1,  # ✅ Low temperature for consistent transcription
        extra_body=self.extra_body,
    )
    return res.choices[0].message.content.strip(), total_token_count_from_response(res)
```

#### Task 7.2: Make Parameters Configurable

Add configuration support in [`rag/flow/parser/parser.py`](rag/flow/parser/parser.py:214):

```python
# In Parser._pdf() around line 340-375
vision_model = LLMBundle(
    self._canvas._tenant_id,
    LLMType.IMAGE2TEXT,
    llm_name=layout_recognize,
    lang=conf.get("lang", "Chinese"),
)

# Add VLM generation config
vlm_gen_conf = {
    "max_tokens": conf.get("vlm_max_tokens", 4096),
    "temperature": conf.get("vlm_temperature", 0.1),
}

# Pass to VisionParser (will need to update VisionParser signature)
lines, _ = VisionParser(vision_model=vision_model)(
    blob,
    callback=self.callback,
    zoomin=conf.get("zoomin", 3),
    prompt_text=prompt_text,
    gen_conf=vlm_gen_conf,  # NEW
)
```

---

### Phase 8: Add Comprehensive VLM Debugging Tools

**Priority**: 🟢 LOW (but helpful)
**Estimated Time**: 30 minutes

#### Task 8.1: Create VLM Request/Response Logger

**New File**: `rag/llm/vlm_debug.py`

```python
"""
VLM Debugging Utilities

Provides tools to log, inspect, and troubleshoot VLM API calls.
"""
import json
import logging
from datetime import datetime
from pathlib import Path


class VLMDebugger:
    """Log VLM requests and responses for debugging."""
    
    def __init__(self, log_dir="logs/vlm_debug"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = True
    
    def log_request(self, model_name, messages, image_size=None, **kwargs):
        """Log VLM request details."""
        if not self.enabled:
            return
        
        timestamp = datetime.now().isoformat()
        request_data = {
            "timestamp": timestamp,
            "model": model_name,
            "image_size_bytes": image_size,
            "num_messages": len(messages),
            "messages_structure": [
                {
                    "role": msg["role"],
                    "content_type": type(msg["content"]).__name__,
                    "content_items": len(msg["content"]) if isinstance(msg["content"], list) else 1
                }
                for msg in messages
            ],
            "kwargs": {k: str(v)[:100] for k, v in kwargs.items()}
        }
        
        # Extract text prompts (but not images)
        for i, msg in enumerate(messages):
            if msg["role"] == "user" and isinstance(msg["content"], list):
                for item in msg["content"]:
                    if item.get("type") == "text":
                        request_data[f"message_{i}_text_preview"] = item["text"][:500]
        
        log_file = self.log_dir / f"request_{timestamp.replace(':', '-')}.json"
        with open(log_file, 'w') as f:
            json.dump(request_data, f, indent=2)
        
        logging.info(f"VLM request logged to {log_file}")
    
    def log_response(self, model_name, response_text, token_count, duration_ms=None):
        """Log VLM response details."""
        if not self.enabled:
            return
        
        timestamp = datetime.now().isoformat()
        response_data = {
            "timestamp": timestamp,
            "model": model_name,
            "response_length_chars": len(response_text),
            "response_length_tokens": token_count,
            "duration_ms": duration_ms,
            "response_preview": response_text[:1000],
            "response_stats": {
                "total_lines": len(response_text.splitlines()),
                "has_markdown_headers": "##" in response_text or "# " in response_text,
                "has_tables": "|" in response_text,
                "has_code_blocks": "```" in response_text,
                "word_count": len(response_text.split()),
            }
        }
        
        log_file = self.log_dir / f"response_{timestamp.replace(':', '-')}.json"
        with open(log_file, 'w') as f:
            json.dump(response_data, f, indent=2)
        
        logging.info(f"VLM response logged to {log_file}")
    
    def compare_with_working_test(self, messages, working_test_path="working_vlm_test.json"):
        """Compare current request with known working test."""
        if not Path(working_test_path).exists():
            logging.warning(f"Working test file not found: {working_test_path}")
            return
        
        with open(working_test_path) as f:
            working_test = json.load(f)
        
        comparison = {
            "structure_match": len(messages) == len(working_test.get("messages", [])),
            "has_system_message": any(m["role"] == "system" for m in messages),
            "working_has_system": any(m["role"] == "system" for m in working_test.get("messages", [])),
            "user_message_structure": [
                {
                    "has_text": any(isinstance(item, dict) and item.get("type") == "text" for item in (m.get("content") if isinstance(m.get("content"), list) else [])),
                    "has_image": any(isinstance(item, dict) and item.get("type") == "image_url" for item in (m.get("content") if isinstance(m.get("content"), list) else []))
                }
                for m in messages if m["role"] == "user"
            ]
        }
        
        logging.info(f"VLM request comparison: {json.dumps(comparison, indent=2)}")
        return comparison


# Global debugger instance
vlm_debugger = VLMDebugger()
```

#### Task 8.2: Integrate Debugger into cv_model.py

```python
# At top of rag/llm/cv_model.py
from rag.llm.vlm_debug import vlm_debugger

# In GptV4.describe_with_prompt (line 187-194)
def describe_with_prompt(self, image, prompt=None):
    import time
    start_time = time.time()
    
    b64 = self.image2base64(image)
    messages = self.vision_llm_prompt(b64, prompt)
    
    # Debug logging
    vlm_debugger.log_request(
        model_name=self.model_name,
        messages=messages,
        image_size=len(image) if isinstance(image, bytes) else None,
        max_tokens=4096,
        temperature=0.1
    )
    
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages,
        max_tokens=4096,
        temperature=0.1,
        extra_body=self.extra_body,
    )
    
    duration_ms = (time.time() - start_time) * 1000
    response_text = res.choices[0].message.content.strip()
    token_count = total_token_count_from_response(res)
    
    vlm_debugger.log_response(
        model_name=self.model_name,
        response_text=response_text,
        token_count=token_count,
        duration_ms=duration_ms
    )
    
    return response_text, token_count
```

---

### Phase 9: Testing & Validation

**Priority**: 🔴 CRITICAL
**Estimated Time**: 45 minutes

#### Task 9.1: Create Minimal Test Script

**New File**: `test_vlm_fix.py`

```python
#!/usr/bin/env python3
"""
Minimal test script to verify VLM fixes.
Usage: python test_vlm_fix.py <pdf_file>
"""
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from deepdoc.parser.pdf_parser import VisionParser


def test_vlm_parsing(pdf_path, tenant_id="test_tenant"):
    """Test VLM PDF parsing with fixes."""
    
    print(f"\n{'='*80}")
    print(f"Testing VLM PDF Parsing: {pdf_path}")
    print(f"{'='*80}\n")
    
    # 1. Create vision model
    print("1. Creating VLM Bundle...")
    vision_model = LLMBundle(
        tenant_id=tenant_id,
        llm_type=LLMType.IMAGE2TEXT,
        llm_name="Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible",
        lang="English"
    )
    print(f"   ✓ Model: {vision_model.llm_name}")
    
    # 2. Load prompt
    print("\n2. Loading prompt...")
    prompt_path = Path("rag/prompts/vision_llm_describe_prompt.md")
    if prompt_path.exists():
        prompt_text = prompt_path.read_text(encoding="utf-8")
        print(f"   ✓ Loaded {len(prompt_text)} chars from {prompt_path}")
    else:
        prompt_text = "Transcribe this PDF page to clean Markdown."
        print(f"   ⚠ Using default prompt: {prompt_text}")
    
    # 3. Parse PDF
    print("\n3. Parsing PDF with VisionParser...")
    parser = VisionParser(vision_model=vision_model)
    
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    
    def progress_callback(prog, msg):
        print(f"   [{prog*100:5.1f}%] {msg}")
    
    lines, _ = parser(
        pdf_bytes,
        from_page=0,
        to_page=1,  # Just first page
        callback=progress_callback,
        zoomin=3,
        prompt_text=prompt_text
    )
    
    # 4. Check results
    print(f"\n4. Results:")
    print(f"   Pages processed: {len(lines)}")
    
    for idx, (text, metadata) in enumerate(lines):
        print(f"\n   Page {idx+1}:")
        print(f"     - Text length: {len(text)} chars")
        print(f"     - Metadata: {metadata}")
        print(f"     - Preview: {text[:200]}...")
        
        if len(text) < 50:
            print(f"     ⚠ WARNING: Suspiciously short response!")
        else:
            print(f"     ✓ Response looks good")
    
    print(f"\n{'='*80}")
    print("Test complete!")
    print(f"{'='*80}\n")
    
    return lines


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_vlm_fix.py <pdf_file>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    test_vlm_parsing(pdf_path)
```

#### Task 9.2: Test Sequence

1. **Before Fixes**: Run test to capture baseline (should show empty responses)
2. **After Fix #1**: Apply LLMBundle return tuple fix, run test
3. **After Fix #2**: Add system message, run test
4. **After Fix #3**: Add max_tokens/temperature, run test
5. **Compare**: Compare with working curl test output

#### Task 9.3: Validation Checklist

- [ ] VLM returns >1000 characters per page (not just 14)
- [ ] Token count is >500 (not just 6)
- [ ] Response contains markdown headers (##, ###)
- [ ] Response contains actual content (not just "--- Page 1 ---")
- [ ] VLM server logs show large token output
- [ ] RAGFlow logs show proper text preview
- [ ] Chunks are created with actual content

---

### Phase 10: Documentation

**Priority**: 🟢 LOW
**Estimated Time**: 30 minutes

#### Task 10.1: Update VLM_IMPLEMENTATION.md

Add "Troubleshooting" section documenting:
- The three critical bugs found
- How to diagnose VLM issues
- How to use the debug logger
- Common failure modes

#### Task 10.2: Create VLM Configuration Guide

**New File**: `docs/VLM_CONFIGURATION.md`

Document:
- How to configure VLM models
- Prompt engineering best practices
- Parameter tuning (max_tokens, temperature)
- System message customization

---

## Summary of Root Causes

| Bug | Severity | File | Line | Impact |
|-----|----------|------|------|--------|
| Return value not tuple | 🔴 CRITICAL | llm_service.py | 165 | Result unpacking fails, returns wrong data |
| Missing system message | 🔴 HIGH | cv_model.py | 158-164 | VLM lacks context, produces minimal output |
| Missing API parameters | 🟡 MEDIUM | cv_model.py | 187-194 | May hit token limits or inconsistent output |

## Implementation Order

1. **IMMEDIATE** (5 min): Fix Task 5.1 - Return tuple from LLMBundle
2. **HIGH** (15 min): Fix Task 6.1 - Add system message
3. **MEDIUM** (20 min): Fix Task 7.1 - Add max_tokens/temperature
4. **TEST** (45 min): Run Task 9.1-9.3 - Validate fixes
5. **POLISH** (optional): Tasks 8.1-8.2, 10.1-10.2 - Debug tools and docs

## Expected Outcome

After applying fixes:
- VLM should return 1000-5000 characters per page (vs current 14)
- Token count should be 500-2000+ (vs current 6)
- Response should contain full markdown transcription
- Chunks should have actual content for RAG retrieval

## Next Steps

Please confirm you want to proceed with implementation. I recommend:

1. **Start with Phase 5** (fix return value) - this is likely the primary issue
2. **Then Phase 6** (add system message) - this provides crucial context
3. **Test after each phase** to isolate which fix resolves the issue
4. **Add debugging tools** (Phase 8) for future troubleshooting

Would you like me to proceed with implementation, or do you have questions about the plan?