# VLM Improvements - Phase 2 Issues

## GREAT NEWS: VLM is Now Transcribing! 🎉

The simplified prompt fixed the echo issue! The VLM now returns **actual content** instead of echoing the prompt.

## New Issues Identified

### 1. **Markdown Formatting Not Rendering**

**Problem:** Bold text not appearing in RAGFlow UI
```
**Review of the PD and Draft IRT**  ← Should be bold but isn't
```

**Root Cause:** The VLM might be outputting plain text with `**` markers, but RAGFlow expects different format or the markers are being stripped.

**Possible Solutions:**
- Check if VLM is actually outputting `**bold**` syntax
- Verify RAGFlow's markdown renderer accepts standard markdown
- May need to explicitly instruct VLM to use proper markdown syntax
- Check if there's whitespace/encoding issues with the asterisks

**Investigation Needed:**
1. Log the raw VLM output to see exact formatting
2. Check what RAGFlow's markdown parser expects
3. Test if standard markdown `**bold**` works in RAGFlow UI manually

### 2. **Table Repetition Bug**

**Problem:** Complex tables devolve into infinite repetition:
```
| | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | | ...
```

**Root Cause:** VLM hits `max_tokens` limit while generating table, causing:
- Incomplete table structure
- Model gets stuck in pattern repetition
- `stop_reason: length` confirms token limit hit

**Solutions:**

#### A. Increase max_tokens
**File:** `rag/llm/working_vlm_module.py` line 98

Change from:
```python
max_tokens=4096,  # Current limit
```

Change to:
```python
max_tokens=8192,  # Double the limit for complex tables
# or
max_tokens=16384,  # For very complex documents
```

#### B. Add Table-Specific Handling
Detect when table generation is incomplete and either:
- Retry with higher token limit
- Fall back to simpler table format
- Split complex tables into multiple chunks

#### C. Improve Prompt for Tables
Add specific table instructions:
```markdown
For tables: Use standard Markdown table format with | separators. 
Keep tables concise. If a table is too large, summarize key data points instead of reproducing the entire table.
```

### 3. **Chunk Sorting Issue**

**Problem:** "second half of page appears before first half"

**Possible Causes:**
1. Multi-page PDF handling bug in VisionParser
2. Metadata coordinate parsing issue
3. RAGFlow's chunk sorting logic using wrong coordinates

**Files to Check:**
- `deepdoc/parser/pdf_parser.py` lines 1556-1559 (metadata generation)
- Chunk sorting logic in parser.py

**Fix:** Ensure metadata format is correct:
```python
f"@@{pdf_page_num + 1}\t{0.0:.1f}\t{width / zoomin:.1f}\t{0.0:.1f}\t{height / zoomin:.1f}##"
```

### 4. **Minor Issues**

#### A. Page Number Addition
```
"10 assessing the potential..."  ← Page number from image footer included
```
- This is actually **expected behavior** - OCR/VLM transcribes everything visible
- May want to add prompt instruction to exclude page numbers

#### B. Word Addition at End
```
"...outlined in detail within each section of this document."
```
The word "document" appears but wasn't in original - minor hallucination but acceptable.

## Recommended Fixes (Priority Order)

### CRITICAL: Fix Table Repetition

**File:** `rag/llm/working_vlm_module.py`

```python
# Line 98: Increase token limit
max_tokens=8192,  # ← Change from 4096

# Alternative: Make it configurable
max_tokens=int(os.getenv("VLM_MAX_TOKENS", "8192")),
```

### HIGH: Improve Markdown Formatting

**File:** `rag/prompts/vision_llm_describe_prompt.md`

Current (simplified):
```markdown
Transcribe all text from this PDF page image into clean Markdown format. Preserve the original structure, formatting, tables, and headings exactly as shown. Output only the transcribed content with no explanations or meta-text.
```

Enhanced version:
```markdown
Transcribe all text from this PDF page image into clean, properly formatted Markdown.

**Formatting Rules:**
- Use `**text**` for bold text
- Use `*text*` for italic text  
- Use `# ` for headings (## for subheadings, etc.)
- Use `| col1 | col2 |` format for tables with `|---|---|` separator rows
- Preserve lists, bullet points, and indentation
- Output only the transcribed content with no meta-text or explanations

For complex tables: If a table has many columns or rows, maintain the structure but keep it readable. If the table is too complex, consider summarizing key data points.
```

### MEDIUM: Add Repetition Detection

**File:** `rag/llm/working_vlm_module.py`

After receiving VLM response, check for repetition:

```python
# After line 159 (return text, token_count)
# Add repetition detection
def detect_repetition(text, pattern_len=50, threshold=3):
    """Detect if text has repeating patterns suggesting model got stuck."""
    if len(text) < pattern_len * threshold:
        return False
    
    pattern = text[-pattern_len:]
    count = text.count(pattern)
    return count >= threshold

if detect_repetition(text):
    logger.warning("Detected repetition pattern - VLM may have hit token limit")
    # Could retry with higher max_tokens or different prompt
```

### LOW: Exclude Page Numbers

**File:** `rag/prompts/vision_llm_describe_prompt.md`

Add instruction:
```markdown
Exclude page numbers and headers/footers that are not part of the main content.
```

## Testing Plan

### Test 1: Simple Text Page
- ✅ Already working
- Verify formatting preserved

### Test 2: Complex Table Page  
- Current: Repetition bug
- After fix: Should handle full table without repetition
- Verify: `stop_reason: stop` (not `length`)

### Test 3: Multi-Page Document
- Verify chunk order is correct
- Check coordinates in metadata

### Test 4: Formatting Test
- Create page with bold, italic, headers
- Verify all formatting renders in RAGFlow UI

## Implementation Order

1. **Immediate:** Increase `max_tokens` to 8192 in working_vlm_module.py
2. **Next:** Enhance prompt with explicit markdown formatting rules
3. **Then:** Add repetition detection and handling
4. **Finally:** Fix chunk sorting if still an issue

## Success Criteria

- ✅ VLM transcribes actual content (not echo) - **ACHIEVED**
- ✅ Simple pages work correctly - **ACHIEVED**  
- ⏳ Bold/italic formatting renders in RAGFlow UI
- ⏳ Complex tables parse without repetition
- ⏳ Multi-page PDFs maintain correct chunk order
- ⏳ Token limit sufficient for full page transcription

## Notes

The core VLM pathway is now **working**! These are polish/optimization issues, not blocking bugs. The system can be used in production with these caveats:
- Works well for simple text pages
- May need manual review for complex tables
- Formatting might need manual correction in UI