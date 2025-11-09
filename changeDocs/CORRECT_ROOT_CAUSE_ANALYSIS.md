# Correct Root Cause Analysis

## The Real Situation

**FACT**: Both test script AND UI run in the SAME Docker container with the SAME (old) code.
**FACT**: Test script returns 3547 characters ✅
**FACT**: UI returns empty chunks ❌

**CONCLUSION**: The difference is NOT in the code version, but in HOW the code is being called.

## Critical Question

If both use the same LLMBundle.describe_with_prompt() method, why does one work and the other fail?

## Hypothesis: Different Execution Paths

### Test Script Path
```python
# test_vlm_pdf_complete.py (in container)
vision_model = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_name="Qwen2.5VL-3B", lang="English")
txt, tokens = vision_model.describe_with_prompt(jpg_bytes, prompt)
# Result: 3547 chars ✅
```

### UI Path
```python
# UI upload → parser.py → VisionParser → picture_vision_llm_chunk → LLMBundle.describe_with_prompt
vision_model = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_name=parse_method, lang="Chinese")
# ... (complex chain)
# Result: Empty ❌
```

## Key Differences to Investigate

### 1. Language Parameter
- Test: `lang="English"`
- UI: `lang="Chinese"` (default in parser.py:306)

Could this affect the VLM behavior? Probably not directly, but let's check.

### 2. Tenant ID
- Test: `tenant_id = "69736c5e723611efb51b0242ac120007"` (hardcoded)
- UI: `tenant_id = self._canvas._tenant_id` (from session)

Could different tenants have different LLM configurations?

### 3. Model Name
- Test: `llm_name="Qwen2.5VL-3B"` (explicit)
- UI: `llm_name=parse_method` (from UI config)

Are these actually the same model? Or does the UI use a different model name that maps to different settings?

### 4. The LLMBundle Initialization

The most likely issue: **LLMBundle might load different configurations based on tenant_id**.

Let me trace this:

```python
# api/db/services/llm_service.py:86
class LLMBundle(LLM4Tenant):
    def __init__(self, tenant_id, llm_type, llm_name=None, lang="Chinese", **kwargs):
        super().__init__(tenant_id, llm_type, llm_name, lang, **kwargs)
```

This inherits from `LLM4Tenant`. The parent class must be loading model configuration from the database based on tenant_id!

## Root Cause: Tenant-Specific LLM Configuration

**The test works because it uses a specific tenant_id that has the model configured correctly.**

**The UI fails because the logged-in user's tenant has the model configured differently (or not at all).**

### What LLM4Tenant Does

1. Takes tenant_id + llm_name
2. Queries database for that tenant's LLM configuration
3. Loads API keys, base_url, and other parameters
4. Creates the actual VLM client (cv_model.py instance)

**If the tenant doesn't have the model configured, or has wrong settings, the VLM call fails.**

## Evidence Needed

Please copy these files from the running container so I can verify:

```bash
# Copy from container
docker cp <container_id>:/ragflow/api/db/services/tenant_llm_service.py ./tenant_llm_service.py.container
docker cp <container_id>:/ragflow/rag/llm/cv_model.py ./cv_model.py.container

# Also get the database schema to understand tenant configuration
docker exec <container_id> cat /ragflow/docker/init.sql > init.sql.container
```

## Alternative Hypothesis: picture_vision_llm_chunk() Breaks It

Let's look at what picture_vision_llm_chunk does that your test doesn't:

### Test (direct call):
```python
txt, tokens = vision_model.describe_with_prompt(jpg_bytes, prompt)
```

### UI (through picture_vision_llm_chunk):
```python
# rag/app/picture.py:136
result = vision_model.describe_with_prompt(binary, prompt)

# Lines 148-157: Normalizes result
if isinstance(result, tuple):
    txt = result[0]
    token_count = result[1]
else:
    txt = result

# Line 177: Cleans markdown
txt = clean_markdown_block(txt).strip()
```

**But we already verified clean_markdown_block() is safe.**

## Most Likely Issue: Return Value Handling

Wait - let's check if the OLD version of LLMBundle.describe_with_prompt() in the container returns a tuple or just a string!

If the old code returns just `txt` instead of `(txt, tokens)`, then:

```python
# picture.py:136
result = vision_model.describe_with_prompt(binary, prompt)  # Returns just "string"

# picture.py:151-153
if isinstance(result, tuple):  # False!
    txt = result[0]
    token_count = result[1]
else:
    txt = result  # txt = "string" ✅ Should work

# But wait... line 221 returns txt
return txt
```

This should still work even if it returns a string.

UNLESS... the old code is CRASHING and returning an empty string from the exception handler!

## Request: Container File Inspection

Please run these commands IN the container and share the output:

```bash
# 1. Check the actual LLMBundle.describe_with_prompt implementation
docker exec <container_id> grep -A 20 "def describe_with_prompt" /ragflow/api/db/services/llm_service.py

# 2. Check cv_model.py implementation  
docker exec <container_id> grep -A 30 "def describe_with_prompt" /ragflow/rag/llm/cv_model.py | head -50

# 3. Check if there's error logging we're missing
docker exec <container_id> grep -r "describe_with_prompt" /ragflow/logs/ 2>/dev/null || echo "No logs found"

# 4. Most importantly: Add debug logging to the RUNNING container
docker exec <container_id> python3 << 'EOF'
import sys
sys.path.insert(0, '/ragflow')
from api.db.services.llm_service import LLMBundle
from api.db import LLMType

# Check if describe_with_prompt exists and what it returns
llm = LLMBundle.__dict__.get('describe_with_prompt')
if llm:
    import inspect
    print("=== LLMBundle.describe_with_prompt signature ===")
    print(inspect.getsource(llm))
EOF
```

This will show us EXACTLY what code is running in the container.