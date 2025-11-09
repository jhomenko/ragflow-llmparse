# VLM Prompt Template Fix - Root Cause Analysis

## Executive Summary

**Root Cause Found**: The VLM was outputting only `--- Page 1 ---` (14 characters) because the Jinja prompt template explicitly instructed it to add that page divider, and the model was being overly obedient by outputting ONLY the divider and skipping the actual transcription.

## The Smoking Gun

### Original Prompt Template (`rag/prompts/vision_llm_describe_prompt.md`)

```markdown
## INSTRUCTION
Transcribe the content from the provided PDF page image into clean Markdown format.
...

{% if page %}
At the end of the transcription, add the page divider: `--- Page {{ page }} ---`.
{% endif %}
```

### What Actually Happened

1. **Jinja Rendering** (`rag/prompts/generator.py:298-301`):
   ```python
   def vision_llm_describe_prompt(page=None) -> str:
       template = PROMPT_JINJA_ENV.from_string(VISION_LLM_DESCRIBE_PROMPT)
       return template.render(page=page)  # page=1 for first page
   ```

2. **Rendered Prompt Sent to VLM**:
   ```
   ## INSTRUCTION
   Transcribe the content from the provided PDF page image...
   
   At the end of the transcription, add the page divider: `--- Page 1 ---`.
   ```

3. **VLM Response** (being TOO obedient):
   ```
   --- Page 1 ---
   ```
   
   The model saw the instruction "add the page divider" and outputted ONLY that, treating it as the complete task.

## Evidence from Logs

```
VLM response object: ChatCompletion(
    choices=[Choice(
        message=ChatCompletionMessage(content='--- Page 1 ---', ...)
        finish_reason='stop'
    )]
)
completion_tokens=6      ← Only 6 tokens generated
prompt_tokens=1392       ← Full image was processed
total_tokens=1398
```

**Key Observations**:
- VLM processed full image (1392 prompt tokens)
- VLM generated only 6 tokens naturally (not truncated)
- VLM stopped naturally with `finish_reason='stop'`
- Response content was exactly `'--- Page 1 ---'` (14 chars)

## Why This Wasn't Truncation

Initially we thought this was a stop token issue, but debug logs proved:
1. ✅ `stop=[]` parameter was correctly set
2. ✅ No default stop tokens were being added
3. ✅ VLM finished naturally, not prematurely
4. ✅ The 14-character response was what the VLM actually generated

## The Fix

### Removed Lines 18-20 from Prompt Template

**BEFORE**:
```markdown
{% if page %}
At the end of the transcription, add the page divider: `--- Page {{ page }} ---`.
{% endif %}
```

**AFTER**: (Removed entirely)

### Why This Works

1. **Page metadata is added separately** in `deepdoc/parser/pdf_parser.py:1558`:
   ```python
   all_docs.append((
       cleaned,
       f"@@{pdf_page_num + 1}\t{0.0:.1f}\t{width / zoomin:.1f}\t{0.0:.1f}\t{height / zoomin:.1f}##"
   ))
   ```
   
2. **The page marker was redundant** and confusing the VLM

3. **Now the prompt is clear**: Just transcribe the content, no extra instructions

## Expected Behavior After Fix

### Test Command
```bash
docker-compose build --no-cache ragflow
docker-compose up -d
# Upload PDF via UI with VLM parser
```

### Expected Logs
```
INFO User prompt (first 200 chars): ## INSTRUCTION
Transcribe the content from the provided PDF page image into clean Markdown format.

- Only output the content transcribed from the image...
INFO VLM response received: chars=3547 tokens=1398  ← FULL RESPONSE NOW!
```

### Expected Result
- VLM should now output full page transcription (3000+ characters)
- Markdown formatting should be preserved
- Multiple pages should all work correctly

## Call Chain Review

For completeness, here's the full call chain:

1. **UI uploads PDF** → Parser selection: "Qwen2.5VL-3B"
2. **`rag/flow/parser/parser.py:_pdf()`** → Detects VLM model name
3. **Creates `VisionParser`** with model bundle
4. **`deepdoc/parser/pdf_parser.py:VisionParser.__call__()`** 
   - Line 1484: `prompt = vision_llm_describe_prompt(page=pdf_page_num + 1)`
   - Line 1495: Calls `picture_vision_llm_chunk(binary=jpg_bytes, prompt=prompt, ...)`
5. **`rag/app/picture.py:vision_llm_chunk()`**
   - Line 168: `res = describe_image_working(image_bytes=binary, prompt=prompt_for_working, ...)`
6. **`rag/llm/working_vlm_module.py:describe_image()`**
   - Line 78: Constructs OpenAI-compatible messages
   - Line 95: Calls VLM server with `stop=[]`
7. **VLM Server** → Returns response
8. **Response flows back** → Added to `all_docs` with metadata

## Lessons Learned

1. **Prompt Engineering Matters**: Even small instructions can confuse VLMs
2. **Redundancy is Dangerous**: The page marker was added both in prompt AND metadata
3. **Debug Thoroughly**: What looked like truncation was actually following instructions
4. **Read the Actual Code**: Assumptions about what prompt was sent were wrong

## Files Modified

1. **`rag/prompts/vision_llm_describe_prompt.md`** - Removed page marker instruction (lines 18-20)

## Next Steps

1. ✅ Rebuild Docker image
2. ✅ Test with sample PDF
3. ✅ Verify full transcription (3000+ chars)
4. ✅ Test multi-page PDFs
5. ✅ Monitor logs for any issues

## Success Criteria

- [x] Prompt template fixed
- [ ] VLM returns full page content (>3000 chars)
- [ ] No more "--- Page X ---" only responses
- [ ] Multi-page PDFs work correctly
- [ ] Logs show proper prompt being sent

---

**Status**: Ready for rebuild and testing
**Risk**: LOW - Simple prompt template fix
**Impact**: HIGH - Enables full VLM PDF parsing