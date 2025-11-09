# Stop Token Analysis - Why RAGFlow Gets Only 6 Tokens

## Problem Statement
RAGFlow's VLM calls return only 6 tokens, while your curl test returns full transcriptions using the **exact same VLM server**. The llama.cpp logs show `truncated = 0`, proving this is a **STOP TOKEN** issue, not a token limit.

## Evidence

### Working Curl Test
```bash
HOSTPORT="http://192.168.68.186:8080"
python3 test_vision_direct.py | curl -sS "${HOSTPORT}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @- | jq '.choices[0].message.content'
```

**Returns:** Full page transcription (1000-5000 characters)

### RAGFlow VLM Call
**Returns:** Only 14 characters (6 tokens)

### Server Logs Comparison
```
# Your curl test:
eval time = 2847.34 ms / 1234 tokens  ✅ Full generation

# RAGFlow:
eval time = 84.50 ms / 6 tokens       ❌ Early stop
slot release: stop processing: truncated = 0  ⚠️ STOP TOKEN HIT
```

## Root Cause Analysis

### Theory 1: OpenAI Client Default Stop Tokens
The OpenAI Python client **may be adding default stop tokens** when none are explicitly provided:

```python
# RAGFlow's current code (cv_model.py:199-208)
res = self.client.chat.completions.create(
    model=self.model_name,
    messages=self.vision_llm_prompt(b64, prompt),
    max_tokens=4096,
    temperature=0.1,
    extra_body=self.extra_body,  # None for most providers
)
# ❌ NO EXPLICIT stop=[] parameter!
```

**Default OpenAI stop tokens** that might be problematic for Qwen2.5VL:
- `<|im_end|>` (ChatML format end token)
- `<|endoftext|>` (GPT tokenizer end token)
- `\n\n---\n\n` (Section separator)

### Theory 2: Prompt Format Issue
The system message might be causing the model to think it should stop early:

```python
# Current system message (cv_model.py:163-171)
system_msg = {
    "role": "system",
    "content": (
        "You are a meticulous PDF-to-Markdown transcriber. "
        "Your task is to convert PDF pages into clean, well-structured Markdown. "
        "Preserve all text, tables, headings, and formatting. "
        "Output ONLY the Markdown content, no explanations."
        #        ^^^^ This might trigger early stopping!
    )
}
```

The phrase "Output ONLY" might cause the model to generate a stop token immediately after producing minimal output.

### Theory 3: Model-Specific Stop Tokens
Qwen2.5VL might have specific stop tokens configured in llama.cpp that don't match what OpenAI clients expect:

```
# Qwen2.5VL expected stop tokens (from model config):
- "<|im_end|>"
- "<|endoftext|>"
- "<|im_start|>"

# But if OpenAI client sends:
- "\n\n"  (markdown section break)
- "---"   (horizontal rule)
```

## Solution: Explicitly Disable Stop Tokens

### Fix #4: Add `stop=[]` Parameter

**File:** `rag/llm/cv_model.py`
**Line:** 199-208

```python
def describe_with_prompt(self, image, prompt=None):
    b64 = self.image2base64(image)
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=self.vision_llm_prompt(b64, prompt),
        max_tokens=4096,
        temperature=0.1,
        stop=[],  # ✅ FIX #4: Explicitly disable default stop tokens
        extra_body=self.extra_body,
    )
    return res.choices[0].message.content.strip(), total_token_count_from_response(res)
```

### Why This Should Fix It

1. **Removes OpenAI Client Defaults:** The OpenAI Python client won't inject its default stop tokens
2. **Lets Model Generate Naturally:** The llama.cpp server will only stop when the model itself generates its configured stop token
3. **Matches Your Curl Test:** Your curl test likely doesn't send any stop tokens either

### Alternative: Use Model-Specific Stop Tokens

If `stop=[]` doesn't work, explicitly set Qwen2.5VL's stop tokens:

```python
# For Qwen2.5VL specifically
stop=["<|im_end|>", "<|endoftext|>"]
```

## Testing Strategy

### Step 1: Add Debug Logging
```python
def describe_with_prompt(self, image, prompt=None):
    b64 = self.image2base64(image)
    
    # Debug: Log the exact parameters being sent
    params = {
        "model": self.model_name,
        "messages": self.vision_llm_prompt(b64, prompt),
        "max_tokens": 4096,
        "temperature": 0.1,
        "stop": [],  # ✅ Explicitly empty
    }
    logging.info(f"VLM API call params: {params}")
    
    res = self.client.chat.completions.create(**params, extra_body=self.extra_body)
    
    # Debug: Log response details
    logging.info(f"VLM response tokens: {res.usage.total_tokens}")
    logging.info(f"VLM finish_reason: {res.choices[0].finish_reason}")
    logging.info(f"VLM content length: {len(res.choices[0].message.content)}")
    
    return res.choices[0].message.content.strip(), total_token_count_from_response(res)
```

### Step 2: Compare with Curl Test
Create a test that mimics your exact curl command but through RAGFlow:

```python
# test_stop_tokens.py
import base64
import json
from openai import OpenAI

# Load your test image
with open("test_page.jpg", "rb") as f:
    img_bytes = f.read()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

client = OpenAI(
    api_key="not-needed",
    base_url="http://192.168.68.186:8080/v1"
)

# Test 1: With stop=[]
print("=" * 80)
print("TEST 1: With stop=[]")
print("=" * 80)
res1 = client.chat.completions.create(
    model="Qwen2.5VL-3B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe this PDF page to Markdown."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
    }],
    max_tokens=4096,
    temperature=0.1,
    stop=[]  # ✅ Explicitly empty
)
print(f"Tokens: {res1.usage.total_tokens}")
print(f"Finish reason: {res1.choices[0].finish_reason}")
print(f"Content ({len(res1.choices[0].message.content)} chars):")
print(res1.choices[0].message.content[:500])

# Test 2: Without stop parameter (let client use defaults)
print("\n" + "=" * 80)
print("TEST 2: Without stop parameter")
print("=" * 80)
res2 = client.chat.completions.create(
    model="Qwen2.5VL-3B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe this PDF page to Markdown."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
    }],
    max_tokens=4096,
    temperature=0.1,
    # NO stop parameter
)
print(f"Tokens: {res2.usage.total_tokens}")
print(f"Finish reason: {res2.choices[0].finish_reason}")
print(f"Content ({len(res2.choices[0].message.content)} chars):")
print(res2.choices[0].message.content[:500])

# Test 3: With common stop tokens
print("\n" + "=" * 80)
print("TEST 3: With OpenAI-style stop tokens")
print("=" * 80)
res3 = client.chat.completions.create(
    model="Qwen2.5VL-3B",
    messages=[{
        "role": "user",
        "content": [
            {"type": "text", "text": "Transcribe this PDF page to Markdown."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
        ]
    }],
    max_tokens=4096,
    temperature=0.1,
    stop=["<|im_end|>", "\n\n"]  # Common OpenAI stop tokens
)
print(f"Tokens: {res3.usage.total_tokens}")
print(f"Finish reason: {res3.choices[0].finish_reason}")
print(f"Content ({len(res3.choices[0].message.content)} chars):")
print(res3.choices[0].message.content[:500])
```

## Expected Results

**If stop=[] fixes it:**
- TEST 1: ~1000-5000 characters ✅
- TEST 2: ~14 characters ❌ (reproduces bug)
- TEST 3: ~14 characters ❌ (confirms stop tokens are the issue)

**This would prove:** The OpenAI client adds default stop tokens that cause premature termination.

## Implementation Plan

1. ✅ Add `stop=[]` to `describe_with_prompt()` in cv_model.py:199-208
2. ✅ Add debug logging to track finish_reason and token counts
3. ✅ Rebuild Docker container with `--no-cache`
4. ✅ Run test_stop_tokens.py to confirm hypothesis
5. ✅ Test through RAGFlow UI with a real PDF
6. ✅ Compare with your working curl test results

## Additional Considerations

### OpenAI Client Version
Check which version of the openai package is installed:
```bash
pip show openai
```

Different versions might handle stop tokens differently.

### llama.cpp Server Configuration
Your llama.cpp server might have default stop tokens configured. Check the server startup command:
```bash
# Look for --stop or similar flags
ps aux | grep llama
```

### Model's Built-in Stop Tokens
Qwen2.5VL has these tokens in its tokenizer:
```
<|im_start|>  # Start of message
<|im_end|>    # End of message
<|endoftext|> # End of text
```

If the OpenAI client is sending `<|im_end|>` as a stop token, the model will terminate as soon as it closes the first message tag.

## Next Steps

1. **Implement Fix #4** (add `stop=[]`)
2. **Run test_stop_tokens.py** to validate hypothesis
3. **Check server logs** for differences in API requests
4. **Compare exact JSON payloads** between curl test and RAGFlow

If `stop=[]` doesn't solve it, we'll need to investigate:
- System message phrasing causing early stops
- Message format differences (system vs user role)
- Image encoding differences (MIME type, size, etc.)