# Model Name Extraction Fix - Implementation Plan

## Problem Analysis

RAGFlow's LLMBundle stores model names in a composite format:
```
Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible
```

But your VLM server expects just:
```
Qwen2.5VL-3B
```

**Current Behavior**:
```python
# rag/app/picture.py:138
model_name = getattr(vision_model, "llm_name") or getattr(vision_model, "name") or "unknown"
# Returns: "Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible"

# working_vlm_module.py:83
response = self.client.chat.completions.create(
    model=model_name,  # Sends full composite name to VLM
    ...
)
```

**Error**:
```
Error code: 400 - {'error': 'could not find real modelID for Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible'}
```

## Solution

Extract the base model name (before `___`) in `picture.py` before passing to working module.

### Format Pattern

RAGFlow uses this naming convention:
```
{model_name}___{provider}@{api_type}
```

Examples:
- `Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible`
- `Qwen2.5VL-7B___OpenAI-API@OpenAI-API-Compatible`
- `gpt-4___OpenAI@OpenAI`

**We need**: Everything before `___`

## Implementation Plan

### Step 1: Add Model Name Extraction Function

**File**: `rag/app/picture.py`

**Location**: Add near top of file (after imports, before functions)

**Code**:
```python
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
        return "unknown"
    
    # Split on '___' and take first part
    if "___" in full_model_name:
        base_name = full_model_name.split("___")[0]
        logging.info(f"Extracted base model name: '{base_name}' from '{full_model_name}'")
        return base_name
    
    # No separator found, return as-is
    return full_model_name
```

**Why This Works**:
- Simple string split on `___` delimiter
- Takes first part (base model name)
- Handles cases where no separator exists (returns original)
- Logs extraction for debugging
- Safe with None/empty strings

### Step 2: Use Extraction in vision_llm_chunk()

**File**: `rag/app/picture.py`

**Location**: Line ~138 in `vision_llm_chunk()` function

**Current Code**:
```python
model_name = getattr(vision_model, "llm_name") or getattr(vision_model, "name") or "unknown"
```

**Change To**:
```python
# Get full model name from vision_model
full_model_name = getattr(vision_model, "llm_name") or getattr(vision_model, "name") or "unknown"

# Extract base model name (e.g., "Qwen2.5VL-3B" from "Qwen2.5VL-3B___OpenAI-API@...")
model_name = extract_base_model_name(full_model_name)
```

**Why This Works**:
- Gets composite name from LLMBundle
- Extracts just the base model name
- Passes clean name to working_vlm_module
- VLM server receives correct model name

### Step 3: Update Test Scripts

**File**: `test_vision_parser_integration.py`

**Location**: Where LLMBundle is created (around line where llm_name is set)

**Current Code**:
```python
vision_model = LLMBundle(
    tenant_id="test_tenant",
    llm_type=LLMType.IMAGE2TEXT,
    llm_name="Qwen2.5VL-3B",  # Simple name in test
    lang="English"
)
```

**Add Test Case**:
```python
# Test with composite model name (as RAGFlow uses in production)
vision_model = LLMBundle(
    tenant_id="test_tenant",
    llm_type=LLMType.IMAGE2TEXT,
    llm_name="Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible",  # Full composite name
    lang="English"
)
```

**Why**:
- Tests the extraction logic
- Ensures compatibility with production model names
- Validates logging

## Implementation Steps

### Implementation Order

1. **Add extraction function** (`extract_base_model_name()`) to `picture.py`
2. **Modify vision_llm_chunk()** to use extraction
3. **Test with existing test script** - should work immediately
4. **Add composite name test** to validate extraction
5. **Update documentation** with model name format notes

### Files to Modify

1. **`rag/app/picture.py`**
   - Add `extract_base_model_name()` function
   - Modify `vision_llm_chunk()` to extract base name
   - Lines affected: ~25 (new function), ~138 (function call)

2. **`test_vision_parser_integration.py`** (optional)
   - Add test case with composite model name
   - Validate extraction works

3. **`WORKING_VLM_MODULE_GUIDE.md`** (documentation)
   - Add note about model name format
   - Explain extraction behavior

## Expected Behavior After Fix

### Before Fix (Current):
```
2025-11-08 00:56:59,902 INFO Using working VLM module: model=Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible
2025-11-08 00:56:59,958 INFO Calling VLM: model=Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible prompt_len=1178
2025-11-08 00:56:59,978 ERROR VLM call failed: Error code: 400 - {'error': 'could not find real modelID for Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible'}
```

### After Fix (Expected):
```
2025-11-08 01:00:00,000 INFO Extracted base model name: 'Qwen2.5VL-3B' from 'Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible'
2025-11-08 01:00:00,001 INFO Using working VLM module: model=Qwen2.5VL-3B
2025-11-08 01:00:00,010 INFO Calling VLM: model=Qwen2.5VL-3B prompt_len=1178
2025-11-08 01:00:02,500 INFO Working VLM response: 3547 chars, 1399 tokens
```

## Testing Plan

### Test 1: Unit Test Extraction Function
```python
# Quick test in Python REPL
from rag.app.picture import extract_base_model_name

# Test normal case
assert extract_base_model_name("Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible") == "Qwen2.5VL-3B"

# Test simple case (no separator)
assert extract_base_model_name("Qwen2.5VL-3B") == "Qwen2.5VL-3B"

# Test empty
assert extract_base_model_name("") == "unknown"
assert extract_base_model_name(None) == "unknown"

print("✓ All extraction tests passed")
```

### Test 2: Integration Test
```bash
# Run existing integration test
docker exec ragflow-server python3 test_vision_parser_integration.py

# Should see in logs:
# - "Extracted base model name: 'Qwen2.5VL-3B' from '...'"
# - "Calling VLM: model=Qwen2.5VL-3B"
# - "Working VLM response: 3547 chars, 1399 tokens"
```

### Test 3: UI Test
1. Upload PDF via RAGFlow UI
2. Select model (any VLM model)
3. Check logs for extraction message
4. Verify chunks created with full content

## Rollback Plan

If issues arise:

1. **Revert picture.py**:
   ```python
   # Remove extraction, use original:
   model_name = getattr(vision_model, "llm_name") or getattr(vision_model, "name") or "unknown"
   ```

2. **Restart container**:
   ```bash
   docker-compose restart ragflow
   ```

## Edge Cases Handled

1. **No separator in name**: Returns name as-is
2. **Empty/None name**: Returns "unknown"
3. **Multiple separators**: Only splits on first `___`
4. **Already base name**: Returns unchanged (no harm)

## Benefits

1. ✅ **Simple**: One-line string split
2. ✅ **Safe**: Handles all edge cases
3. ✅ **Fast**: No regex, just string operation
4. ✅ **Backward compatible**: Works with simple names too
5. ✅ **Debuggable**: Logs extraction
6. ✅ **Non-breaking**: Doesn't affect other code paths

## Summary

**Problem**: RAGFlow sends `Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible` to VLM server

**Solution**: Extract base name (`Qwen2.5VL-3B`) before sending

**Implementation**: 
- Add `extract_base_model_name()` function
- Use it in `vision_llm_chunk()` before calling working module

**Files Modified**: 1 file (`rag/app/picture.py`)

**Lines Changed**: ~10 lines (new function + 2-line change in existing function)

**Testing**: Existing tests work immediately, just need to verify logs

**Risk**: Very low - simple string operation with fallbacks

This is indeed a simple fix that should resolve the model name issue completely!