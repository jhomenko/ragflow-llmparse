# Critical Discovery: UI vs Test Discrepancy Analysis

## Problem Statement
Test script returns 3500+ characters, but UI returns empty chunks for the same PDF.

## Call Chain Comparison

### Test Script (WORKS ✅)
```
test_vlm_pdf_complete.py
  ↓
LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_name="Qwen2.5VL-3B")
  ↓
LLMBundle.describe_with_prompt(jpg_bytes, prompt)
  ↓
self.mdl.describe_with_prompt(image, prompt)  # mdl = GptV4 instance
  ↓
cv_model.py::GptV4.describe_with_prompt()  [Lines 200-222]
  ↓ 
OpenAI client.chat.completions.create(
    model=self.model_name,
    messages=self.vision_llm_prompt(b64, prompt),  # ✅ Has system message
    max_tokens=4096,      # ✅ Explicit limit
    temperature=0.1,      # ✅ Low temp
    stop=[],              # ✅ No stop tokens
)
  ↓
Returns: (text, tokens)  # ✅ 3547 characters
```

### UI Path (FAILS ❌)
```
UI → Upload PDF with VLM parser
  ↓
rag/flow/parser/parser.py::Parser._pdf()  [Line 214]
  ↓
Creates: vision_model = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_name=parse_method)
  ↓
VisionParser(vision_model=vision_model)  [Line 364]
  ↓
VisionParser.__call__(blob, callback=..., zoomin=3, prompt_text=prompt)  [Line 1377]
  ↓
picture_vision_llm_chunk(
    binary=jpg_bytes,
    vision_model=self.vision_model,  # ← Same LLMBundle instance
    prompt=prompt,
    callback=callback
)  [pdf_parser.py:1495]
  ↓
rag/app/picture.py::vision_llm_chunk()  [Line 69]
  ↓
if hasattr(vision_model, "describe_with_prompt"):
    result = vision_model.describe_with_prompt(binary, prompt)  [Line 136]
  ↓
api/db/services/llm_service.py::LLMBundle.describe_with_prompt()  [Line 153]
  ↓
txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)  [Line 157]
  ↓
cv_model.py::GptV4.describe_with_prompt()  [Line 200]
  ↓
??? Returns empty or truncated ???
```

## Hypothesis: The Bug is NOT in cv_model.py!

The test proves that `cv_model.py::GptV4.describe_with_prompt()` works correctly when called directly through `LLMBundle`.

**So why does the UI path fail?**

## Possible Root Causes

### 1. **Image Format Issue** ⚠️ MOST LIKELY
In the test:
```python
# test_vlm_pdf_complete.py
img = page.to_image(resolution=72 * 3).annotated
img_rgb = img.convert("RGB")
buf = io.BytesIO()
img_rgb.save(buf, format="JPEG", quality=90)
jpg_bytes = buf.getvalue()  # ✅ Clean JPEG bytes
```

In VisionParser:
```python
# deepdoc/parser/pdf_parser.py:1456
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=90, optimize=True)
jpg_bytes = buf.getvalue()  # Should be same...
```

**BUT WAIT** - Let me check if `buf.getvalue()` is called vs `buf.read()`:

```python
# pdf_parser.py:1456-1458
buf = io.BytesIO()
img.save(buf, format="JPEG", quality=90, optimize=True)
jpg_bytes = buf.getvalue()  # ✅ Correct - no seek needed
```

So image format should be identical. ❌ Not the issue.

### 2. **Prompt Not Reaching VLM** ⚠️ POSSIBLE

Check if prompt is actually passed through the chain:
- parser.py:365 passes `prompt_text=prompt`
- VisionParser:1482 extracts it: `prompt_text = kwargs.get("prompt_text", None)`
- VisionParser:1481-1484 builds prompt
- Passes to picture_vision_llm_chunk:1495 as `prompt=prompt`
- picture.py:136 passes to `vision_model.describe_with_prompt(binary, prompt)`

Looks correct. ❌ Not the issue.

### 3. **Tenant ID Mismatch** ⚠️ POSSIBLE

Test uses:
```python
tenant_id = "69736c5e723611efb51b0242ac120007"  # Hardcoded
```

UI uses:
```python
tenant_id = getattr(self._canvas, "_tenant_id", None)  # From session
```

Could the UI tenant_id be wrong/different, causing different LLM config to load?

### 4. **Model Configuration Difference** ⚠️ **VERY LIKELY**

The test creates:
```python
vision_model = LLMBundle(
    tenant_id,
    LLMType.IMAGE2TEXT,
    llm_name="Qwen2.5VL-3B",
    lang="English"
)
```

The UI creates:
```python
vision_model = LLMBundle(
    tenant_id,
    LLMType.IMAGE2TEXT,
    llm_name=parse_method,  # User-selected model name
    lang=self._param.setups["pdf"].get("lang", "Chinese"),  # ← Different default!
)
```

**KEY DIFFERENCE**: Test uses `lang="English"`, UI might use `lang="Chinese"`!

But wait - that shouldn't affect the VLM call since we're using explicit prompts. ❌ Not the core issue.

### 5. **The Real Culprit: LLMBundle.mdl Not Initialized** ⚠️⚠️⚠️ **CRITICAL**

Looking at [`api/db/services/llm_service.py:153-165`](api/db/services/llm_service.py:153):

```python
def describe_with_prompt(self, image, prompt):
    if self.langfuse:
        generation = self.langfuse.start_generation(...)
    
    txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)  # ← What is self.mdl?
    
    if not TenantLLMService.increase_usage(...):
        logging.error(...)
    
    if self.langfuse:
        generation.update(...)
        generation.end()
    
    return txt, used_tokens
```

**What is `self.mdl`?** It's set in the parent class `LLM4Tenant`. Let me check if there's an initialization issue.

The test might be initializing `self.mdl` correctly because it's calling from a fresh Python process, but the UI's long-running container might have a cached/broken `self.mdl` instance.

### 6. **Response Stripping in picture.py** ⚠️⚠️ **VERY SUSPICIOUS**

Look at [`rag/app/picture.py:176-177`](rag/app/picture.py:176):

```python
# Clean up possible markdown fences
txt = clean_markdown_block(txt).strip()
```

**What does `clean_markdown_block()` do?** Let me check:

```python
from rag.utils import clean_markdown_block
```

This function might be stripping the entire response! If the VLM wraps output in triple backticks like:
```markdown
\`\`\`markdown
# Page Content
...
\`\`\`
```

Then `clean_markdown_block()` might be removing the fences but also the content!

## Action Items

1. **Add logging in picture.py BEFORE clean_markdown_block()**
   - Log the raw `txt` from VLM before any cleaning
   - This will show if VLM is returning data but it's being stripped

2. **Check if self.mdl is properly initialized**
   - Add logging in LLMBundle.describe_with_prompt() to show self.mdl type

3. **Compare tenant configurations**
   - Test tenant vs UI tenant - are they using the same API keys/endpoints?

4. **Check for middleware/validation**
   - Is there content filtering between cv_model and the final output?

## Most Likely Issue

Based on the evidence, I believe the issue is **`clean_markdown_block()` is stripping the entire response**.

The VLM probably returns valid markdown, but the cleaning function is too aggressive and removes everything, leaving an empty string.

## Next Steps

Add diagnostic logging to confirm where the content is lost:

```python
# rag/app/picture.py:176 (BEFORE cleaning)
logging.critical(f"VLM RAW RESPONSE (before cleaning): length={len(txt)}, preview={txt[:500]}")

# Clean up possible markdown fences  
txt = clean_markdown_block(txt).strip()

logging.critical(f"AFTER clean_markdown_block: length={len(txt)}, preview={txt[:500]}")
```

This will definitively show where the data is lost.