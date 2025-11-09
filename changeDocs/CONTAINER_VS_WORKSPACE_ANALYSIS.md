# Container vs Workspace Code Analysis

## Critical Discovery

After analyzing the actual container files, I've found the **REAL ROOT CAUSE** of why the test works but the UI fails.

## The Container Code is ALREADY FIXED!

Looking at the container files you provided:

### 1. `containercurrent/cv_model.py` (Lines 199-208)
```python
def describe_with_prompt(self, image, prompt=None):
    b64 = self.image2base64(image)
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=self.vision_llm_prompt(b64, prompt),
        max_tokens=4096,      # ✅ ALREADY PRESENT
        temperature=0.1,      # ✅ ALREADY PRESENT
        extra_body=self.extra_body,
    )
    return res.choices[0].message.content.strip(), total_token_count_from_response(res)
```

### 2. `containercurrent/llm_service.py` (Lines 153-165)
```python
def describe_with_prompt(self, image, prompt):
    if self.langfuse:
        generation = self.langfuse.start_generation(...)
    
    txt, used_tokens = self.mdl.describe_with_prompt(image, prompt)  # ✅ RETURNS TUPLE
    # ... token tracking ...
    
    return txt, used_tokens  # ✅ CORRECT RETURN FORMAT
```

### 3. `containercurrent/pdf_parser.py` (Lines 1434-1569) - VisionParser
The VisionParser in the container:
- ✅ Converts PIL Images to JPEG bytes (lines 1444-1458)
- ✅ Passes bytes to `picture_vision_llm_chunk()` (line 1495-1500)
- ✅ Has comprehensive logging and validation
- ✅ Has all the fixes we implemented!

### 4. `containercurrent/picture.py` (Lines 69-230) - vision_llm_chunk
The picture.py in the container:
- ✅ Expects bytes input (validated at line 85-92)
- ✅ Calls `vision_model.describe_with_prompt(binary, prompt)` (line 136)
- ✅ Handles tuple return value (lines 151-157)
- ✅ Has all validation and error handling

## The REAL Problem

**The container ALREADY HAS the fixes**. So why does the UI produce empty chunks while the test works?

## Root Cause Theory: Tenant Configuration Mismatch

Since both paths use the SAME code, the issue must be in **HOW the LLMBundle is initialized**:

### Test Script Configuration
```python
vision_model = LLMBundle(
    tenant_id="your_tenant_id",  # Hardcoded tenant
    llm_type=LLMType.IMAGE2TEXT,
    llm_name="Qwen2.5VL-3B",     # Explicit model name
    lang="English"
)
```

### UI Path Configuration (from parser.py:302-307)
```python
vision_model = LLMBundle(
    tenant_id,                    # From canvas._tenant_id
    LLMType.IMAGE2TEXT,
    llm_name=parse_method,        # From UI selection
    lang=self._param.setups["pdf"].get("lang", "Chinese"),  # ⚠️ "Chinese" default
)
```

## Key Differences

1. **Language Parameter**:
   - Test: `"English"`
   - UI: `"Chinese"` (default)

2. **Tenant ID**:
   - Test: Explicit hardcoded value
   - UI: Dynamic from session

3. **Model Name Resolution**:
   - Test: Direct `"Qwen2.5VL-3B"`
   - UI: Whatever string is in `parse_method` config

## The Critical Bug: Language Parameter in VLM Prompt

Look at `containercurrent/cv_model.py` line 163-176:

```python
def vision_llm_prompt(self, b64, prompt=None):
    """Create vision LLM prompt with system message"""
    system_msg = {
        "role": "system",
        "content": (
            "You are a meticulous PDF-to-Markdown transcriber. "
            "Your task is to convert PDF pages into clean, well-structured Markdown. "
            "Preserve all text, tables, headings, and formatting. "
            "Output ONLY the Markdown content, no explanations."
        )
    }
    user_msg = {
        "role": "user",
        "content": self._image_prompt(prompt if prompt else vision_llm_describe_prompt(), b64)
    }
    return [system_msg, user_msg]
```

BUT look at the base `prompt()` method at lines 145-156 - it checks `self.lang`:

```python
def prompt(self, b64):
    return [
        {
            "role": "user",
            "content": self._image_prompt(
                "请用中文详细描述一下图中的内容，..."
                if self.lang.lower() == "chinese"  # ⚠️ USES self.lang
                else "Please describe the content of this picture..."
```

## Hypothesis: Lang Parameter Affects Model Initialization

The `lang` parameter might affect:
1. **Prompt template selection** in `vision_llm_describe_prompt()`
2. **Model configuration** in the database
3. **API endpoint** or **model variant** selection

## Next Steps to Diagnose

You need to check the **ACTUAL parameters** being used in the UI:

1. Add this diagnostic logging to `parser.py` line 315:

```python
logging.critical(f"=== VLM CONFIG DEBUG ===")
logging.critical(f"tenant_id: {tenant_id}")
logging.critical(f"parse_method: {parse_method}")
logging.critical(f"lang: {conf.get('lang', 'Chinese')}")
logging.critical(f"prompt_path: {prompt_path}")
logging.critical(f"vision_model created: {vision_model is not None}")
if vision_model:
    logging.critical(f"vision_model.llm_name: {getattr(vision_model, 'llm_name', 'N/A')}")
    logging.critical(f"vision_model.lang: {getattr(vision_model, 'lang', 'N/A')}")
```

2. Check what's in your VLM prompt file:
```bash
docker exec <container> cat /ragflow/rag/prompts/vision_llm_describe_prompt.md
```

3. Compare tenant configurations:
```bash
# Get test tenant config
docker exec <container> python3 -c "
from api.db.services.tenant_llm_service import TenantLLMService
tenant = TenantLLMService.query(tenant_id='test_tenant')
print(tenant)
"

# Get UI tenant config  
docker exec <container> python3 -c "
from api.db.services.tenant_llm_service import TenantLLMService
tenant = TenantLLMService.query(tenant_id='<your_ui_tenant>')
print(tenant)
"
```

## Smoking Gun Candidates

1. **Wrong tenant configuration**: UI tenant has different model settings
2. **Language mismatch**: Chinese vs English affects prompt/model behavior
3. **Model name mismatch**: `parse_method` doesn't resolve to same model as test
4. **Prompt template issue**: `vision_llm_describe_prompt()` returns empty/wrong prompt for Chinese lang

## The Most Likely Culprit

Based on the server logs you showed earlier (6-token responses), I suspect:

**The UI is passing `lang="Chinese"` which causes the VLM to use a Chinese prompt template, and the model is being configured differently for Chinese language, resulting in truncated output.**

Try this quick test:

1. In your UI, when selecting VLM parser, explicitly set:
   - Language: `"English"` (not Chinese)
   
2. Check if that produces 3547-character responses like your test

If YES → The problem is the **lang parameter affecting model behavior**
If NO → The problem is **tenant-specific model configuration**