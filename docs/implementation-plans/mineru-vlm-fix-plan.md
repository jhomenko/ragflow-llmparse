# MinerU VLM Implementation Fix Plan

## Current Error

```
TypeError: MinerUParser.parse_pdf() got an unexpected keyword argument 'parse_method'
```

**Location:** `rag/app/naive.py` line 78 → calls `pdf_parser.parse_pdf(..., parse_method=parse_method)`

**Root Cause:** Parameter name mismatch between caller and method signature.

---

## Analysis

### The Mismatch

| File | Location | Parameter Name |
|------|----------|----------------|
| `rag/app/naive.py` | Line 86 | `parse_method=parse_method` (caller) |
| `deepdoc/parser/mineru_parser.py` | Line 543 | `method: str = "auto"` (signature) |

The `by_mineru()` function is calling `parse_pdf()` with a keyword argument `parse_method`, but the actual method signature uses `method`.

### Code in `naive.py` (Lines 78-87):
```python
sections, tables = pdf_parser.parse_pdf(
    filepath=filename,
    binary=binary,
    callback=callback,
    output_dir=os.environ.get("MINERU_OUTPUT_DIR", ""),
    backend=backend,
    server_url=os.environ.get("MINERU_SERVER_URL", ""),
    delete_output=bool(int(os.environ.get("MINERU_DELETE_OUTPUT", 1))),
    parse_method=parse_method  # <-- WRONG: should be "method=parse_method"
)
```

### Code in `mineru_parser.py` (Lines 534-546):
```python
def parse_pdf(
    self,
    filepath: str | PathLike[str],
    binary: BytesIO | bytes,
    callback: Optional[Callable] = None,
    *,
    output_dir: Optional[str] = None,
    backend: str = "pipeline",
    lang: Optional[str] = None,
    method: str = "auto",           # <-- The actual parameter name
    server_url: Optional[str] = None,
    delete_output: bool = True,
) -> tuple:
```

---

## Fix Required

### Option A: Fix the Caller (Recommended - Minimal Change)

**File:** `rag/app/naive.py`  
**Line:** 86  
**Change:** `parse_method=parse_method` → `method=parse_method`

```python
# Before (line 86):
parse_method=parse_method

# After:
method=parse_method
```

**Full corrected block (lines 78-87):**
```python
sections, tables = pdf_parser.parse_pdf(
    filepath=filename,
    binary=binary,
    callback=callback,
    output_dir=os.environ.get("MINERU_OUTPUT_DIR", ""),
    backend=backend,
    server_url=os.environ.get("MINERU_SERVER_URL", ""),
    delete_output=bool(int(os.environ.get("MINERU_DELETE_OUTPUT", 1))),
    method=parse_method  # <-- FIXED: use correct parameter name
)
```

### Option B: Fix the Method Signature (Alternative)

**File:** `deepdoc/parser/mineru_parser.py`  
**Line:** 543  
**Change:** `method: str = "auto"` → `parse_method: str = "auto"`

This would require also updating:
- Line 590: `self._run_mineru(..., method=method, ...)` → `method=parse_method`
- Line 591: `self._read_output(..., method=method, ...)` → `method=parse_method`

**Not recommended** because it changes the API signature which might break other callers.

---

## Additional Issues to Verify

While fixing this, also verify these potential issues don't exist:

### 1. Health Check Endpoint
**File:** `deepdoc/parser/mineru_parser.py` (lines 133-154)  
**Issue:** Uses `/openapi.json` but user's server uses `/health`  
**Fix:** Add fallback to try `/health` first

### 2. Container File Sync
The error suggests the container might not have the latest files. After fixing, ensure:
```bash
# Restart container with fresh mounts
docker-compose down
docker-compose up -d

# Or if using volume mounts, clear cache
docker exec -it <container> find /ragflow -name "*.pyc" -delete
```

---

## Implementation Checklist

| # | Task | File | Line(s) | Status |
|---|------|------|---------|--------|
| 1 | Change `parse_method=` to `method=` | rag/app/naive.py | 86 | Pending |
| 2 | (Optional) Add `/health` fallback | mineru_parser.py | 133-154 | Pending |
| 3 | Restart container | - | - | Pending |
| 4 | Test PDF parsing | - | - | Pending |

---

## Verification Commands

After applying fixes:

```bash
# 1. Check the fix is in place
grep -n "method=parse_method" /ragflow/rag/app/naive.py

# 2. Clear Python cache
find /ragflow -name "*.pyc" -delete
find /ragflow -name "__pycache__" -type d -exec rm -rf {} +

# 3. Restart the task executor
# (depends on your setup - may need to restart container)

# 4. Re-run the failed task via RAGFlow UI
```

---

## Summary

**Primary Fix:** Change line 86 in `rag/app/naive.py` from:
```python
parse_method=parse_method
```
to:
```python
method=parse_method
```

This is a one-line fix that aligns the caller with the actual method signature.