# Debug Logging Rebuild Guide

## What We Just Added

Comprehensive debug logging to [`rag/llm/working_vlm_module.py`](rag/llm/working_vlm_module.py:1) to track exactly what's happening with the VLM calls.

## Rebuild Steps

```bash
# 1. Rebuild with new debug logging
docker-compose down
docker-compose build --no-cache ragflow
docker-compose up -d

# 2. Wait for services to start
sleep 30

# 3. Upload PDF via UI
# Select model: Qwen2.5VL-3B

# 4. Watch detailed logs
docker logs -f ragflow-server 2>&1 | grep -A 50 "Calling VLM"
```

## What to Look For in Logs

### Expected Success Pattern

```
INFO Image encoded: 245678 bytes → 327570 base64 chars
INFO Calling VLM: model=Qwen2.5VL-3B prompt_len=1178
INFO VLM call parameters: max_tokens=4096, temperature=0.1, stop=[]
INFO VLM raw response type: <class 'openai.types.chat.chat_completion.ChatCompletion'>
INFO Response has 1 choice(s)
INFO Choice[0] finish_reason: stop
INFO Extracted content type: <class 'str'>
INFO Extracted content length: 3547
INFO Extracted content preview (first 500 chars): # Assessing the Potential...
INFO Token usage: CompletionUsage(completion_tokens=..., prompt_tokens=..., total_tokens=1399)
INFO VLM response received: chars=3547 tokens=1399
```

### Current Failure Pattern (What We're Seeing)

```
INFO Image encoded: ??? bytes → ??? base64 chars
INFO Calling VLM: model=Qwen2.5VL-3B prompt_len=1178
INFO VLM call parameters: max_tokens=4096, temperature=0.1, stop=[]
INFO VLM raw response type: ???
INFO Response has 1 choice(s)
INFO Choice[0] message: ???
INFO Choice[0] finish_reason: ???
INFO Extracted content length: 14
WARNING ⚠️ SUSPICIOUSLY SHORT RESPONSE: Only 14 chars!
WARNING Full response text: '--- Page 1 ---'
ERROR ❌ TOKEN/CHAR MISMATCH: 1398 tokens but only 14 chars!
```

## Key Diagnostic Questions

The new logs will answer:

### 1. Image Encoding
**Question**: Is the image being encoded to base64 correctly?
**Look for**: `INFO Image encoded: X bytes → Y base64 chars`
**Expected**: ~250KB bytes → ~330KB base64 chars

### 2. Parameters
**Question**: Is `stop=[]` actually being sent?
**Look for**: `INFO VLM call parameters: max_tokens=4096, temperature=0.1, stop=[]`
**Expected**: Should show `stop=[]` explicitly

### 3. Raw Response
**Question**: What does the VLM server actually return?
**Look for**: `INFO VLM response object: ...`
**Expected**: Full ChatCompletion object with content

### 4. Content Extraction
**Question**: What do we extract from `response.choices[0].message.content`?
**Look for**: `INFO Extracted content preview (first 500 chars): ...`
**Expected**: Should show actual markdown content, not "--- Page 1 ---"

### 5. Finish Reason
**Question**: Why did the VLM stop generating?
**Look for**: `INFO Choice[0] finish_reason: ...`
**Expected**: 
- `stop` = natural completion ✅
- `length` = hit max_tokens ⚠️
- Other = problem ❌

## Scenarios and Interpretations

### Scenario A: Image Not Encoded
```
ERROR Failed to encode image
```
**Diagnosis**: Image conversion failing before VLM call
**Fix**: Check PIL Image to JPEG conversion in VisionParser

### Scenario B: VLM Returns Short Content
```
INFO Extracted content length: 14
INFO Extracted content preview: '--- Page 1 ---'
INFO Choice[0] finish_reason: stop
```
**Diagnosis**: VLM server is returning short response
**Possible causes**:
1. `stop=[]` not actually being sent (check parameters log)
2. VLM server has server-side stop tokens configured
3. Image quality too low for VLM to process

### Scenario C: Content Lost in Extraction
```
INFO VLM response object: ChatCompletion(choices=[Choice(message=ChatCompletionMessage(content='...long content...', role='assistant'))])
INFO Extracted content length: 14
```
**Diagnosis**: Response contains full content but extraction fails
**Fix**: Review content extraction logic

### Scenario D: Token/Char Severe Mismatch
```
INFO Token usage: total_tokens=1398
INFO Extracted content length: 14
ERROR ❌ TOKEN/CHAR MISMATCH
```
**Diagnosis**: VLM processed full image (1398 tokens) but returned minimal text
**Possible causes**:
1. Server-side stop tokens (even though we sent `stop=[]`)
2. VLM model configuration issue
3. Image contains minimal extractable text

## Comparison with Working Test

Your `test_vlm_pdf_complete.py` produces:

```python
# Test with stop=[]
INFO VLM response: 3547 chars, 1399 tokens ✅

# Test with stop=["<|im_end|>", "\n\n"]
INFO VLM response: ~100 chars (truncated after first heading) ❌
```

The working module should match the first case exactly.

## Next Steps Based on Logs

### If Image Encoding Fails
→ Fix in [`deepdoc/parser/pdf_parser.py`](deepdoc/parser/pdf_parser.py:1366) VisionParser

### If stop=[] Not Sent
→ Verify Docker rebuild completed successfully
→ Check file was actually updated in container: 
```bash
docker exec ragflow-server grep "stop=\[\]" /ragflow/rag/llm/working_vlm_module.py
```

### If VLM Returns Short Content
→ Test VLM server directly with curl (your working test)
→ Compare exact request payloads
→ Check VLM server logs for any server-side truncation

### If Content Lost in Extraction
→ Review response object structure in logs
→ May need to adjust extraction logic

## Docker Rebuild Verification

Before testing, verify the changes are in the container:

```bash
# Check stop=[] is in the file
docker exec ragflow-server grep -n "stop=\[\]" /ragflow/rag/llm/working_vlm_module.py

# Should show:
# 89:            stop=[],  # Explicitly disable default stop tokens

# Check debug logging is present
docker exec ragflow-server grep -n "Image encoded:" /ragflow/rag/llm/working_vlm_module.py

# Should show multiple lines with logging statements
```

## Full Log Capture

To capture complete debug output:

```bash
# Start fresh log capture
docker logs ragflow-server 2>&1 | tail -n 0 -f > vlm_debug_full.log &
LOG_PID=$!

# Upload PDF via UI
# Wait for processing to complete

# Stop log capture
kill $LOG_PID

# Search for our debug section
grep -A 100 "Calling VLM" vlm_debug_full.log > vlm_debug_section.log

# Analyze the output
cat vlm_debug_section.log
```

## Critical Success Indicators

✅ **Image encoded**: Shows reasonable byte → base64 conversion
✅ **Parameters log**: Shows `stop=[]` explicitly
✅ **Response length**: > 1000 characters
✅ **No mismatch error**: Token count matches character count
✅ **Finish reason**: `stop` (natural completion)

## Critical Failure Indicators

❌ **Image encoding fails**: No base64 output
❌ **No stop parameter log**: Rebuild didn't work
❌ **Mismatch error**: High tokens, low chars
❌ **Finish reason**: `length` or other unexpected value
❌ **Short response warning**: < 100 chars

## What This Will Tell Us

The debug logs will definitively answer:

1. **Is the fix applied?** (stop=[] in logs)
2. **Is the image correct?** (encoding details)
3. **What does VLM return?** (raw response object)
4. **Where is content lost?** (extraction logs)
5. **Why is it failing?** (mismatch detection)

After you rebuild and test, share the logs section starting from "Calling VLM" through "Working VLM response" and we'll know exactly what's happening.

---

*Created: 2025-11-08*
*Purpose: Debug the persistent 14-character response issue with comprehensive logging*