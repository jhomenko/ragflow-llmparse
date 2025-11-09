# VLM Still Returning 6 Tokens - Debug Analysis

## Current Situation

Your VLM server log shows:
```
prompt eval time = 3436.85 ms / 1393 tokens (image + prompt processed)
eval time = 83.65 ms / 6 tokens (ONLY 6 TOKENS OUTPUT ❌)
```

## Two Possible Issues

### Issue A: Docker Cache (CONFIRMED)
- Test file not in container = cached build
- **Solution**: Force clean rebuild (see REBUILD_INSTRUCTIONS.md)
- **Verify fixes are applied** after rebuild

### Issue B: API Call Format May Need Adjustment

Looking at the VLM logs, I notice the image processing happens **after** the response is sent. This suggests a potential timing or API format issue.

## Additional Debugging Steps

### Step 1: Verify Code Changes Are Applied

After clean rebuild, check inside container:

```bash
docker exec ragflow-server python3 << 'EOF'
import sys
sys.path.insert(0, '/ragflow')

# Check Bug #1 fix
from api.db.services.llm_service import LLMBundle
import inspect
src = inspect.getsource(LLMBundle.describe_with_prompt)
print("=== Bug #1 Check: Return Statement ===")
if "return txt, used_tokens" in src:
    print("✓ FIXED: Returns tuple")
else:
    print("✗ NOT FIXED: Still returns only txt")
print()

# Check Bug #2 fix
from rag.llm.cv_model import GptV4
src2 = inspect.getsource(GptV4.vision_llm_prompt)
print("=== Bug #2 Check: System Message ===")
if '"role": "system"' in src2 or "'role': 'system'" in src2:
    print("✓ FIXED: Has system message")
else:
    print("✗ NOT FIXED: No system message")
print()

# Check Bug #3 fix  
src3 = inspect.getsource(GptV4.describe_with_prompt)
print("=== Bug #3 Check: API Parameters ===")
if "max_tokens" in src3:
    print("✓ FIXED: Has max_tokens parameter")
else:
    print("✗ NOT FIXED: No max_tokens parameter")
if "temperature" in src3:
    print("✓ FIXED: Has temperature parameter")
else:
    print("✗ NOT FIXED: No temperature parameter")
EOF
```

### Step 2: Check VLM Server Compatibility

Your VLM server uses `llama.cpp` with OpenAI-compatible API. Verify parameters are supported:

```bash
# Test VLM server directly with our parameters
curl -X POST http://192.168.68.186:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-VL-3B-Instruct",
    "messages": [
      {
        "role": "system",
        "content": "You are a PDF transcriber. Convert images to markdown."
      },
      {
        "role": "user",
        "content": [
          {"type": "text", "text": "Describe this image in detail."},
          {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
        ]
      }
    ],
    "max_tokens": 4096,
    "temperature": 0.1
  }' | jq '.choices[0].message.content' | wc -c
```

Expected: >1000 characters  
If still 6 tokens: VLM server may not respect max_tokens parameter

### Step 3: Check llama.cpp Server Configuration

The VLM server may have hard-coded limits. Check startup parameters:

```bash
# Look for -n or --n-predict in server startup
docker logs vlm-container-name 2>&1 | grep -E "n-predict|n_predict|max.*tokens"
```

Common issues with llama.cpp server:
- Default `n_predict` may be very low (e.g., 128)
- Server may need `--ctx-size` parameter
- Model may need specific prompt format

### Step 4: Test with Different max_tokens Values

If server ignores `max_tokens`, try these alternatives in `rag/llm/cv_model.py`:

**Option 1**: Add to extra_body
```python
res = self.client.chat.completions.create(
    model=self.model_name,
    messages=self.vision_llm_prompt(b64, prompt),
    max_tokens=4096,
    temperature=0.1,
    extra_body={
        "n_predict": 4096,  # llama.cpp specific
        "stop": ["<|im_end|>"],  # Qwen2 stop token
        **(self.extra_body or {})
    },
)
```

**Option 2**: Check if server uses different parameter name
```python
res = self.client.chat.completions.create(
    model=self.model_name,
    messages=self.vision_llm_prompt(b64, prompt),
    max_completion_tokens=4096,  # Alternative parameter name
    temperature=0.1,
    extra_body=self.extra_body,
)
```

### Step 5: Enable Detailed API Logging

Add logging to see exact request/response:

```python
# In rag/llm/cv_model.py, line 189
def describe_with_prompt(self, image, prompt=None):
    import json
    b64 = self.image2base64(image)
    messages = self.vision_llm_prompt(b64, prompt)
    
    # Log request details
    logging.info(f"VLM API Request:")
    logging.info(f"  Model: {self.model_name}")
    logging.info(f"  Messages: {len(messages)} messages")
    for i, msg in enumerate(messages):
        role = msg.get('role')
        content = msg.get('content')
        if isinstance(content, str):
            logging.info(f"    [{i}] {role}: {len(content)} chars")
        elif isinstance(content, list):
            logging.info(f"    [{i}] {role}: {len(content)} items")
    
    request_params = {
        "model": self.model_name,
        "max_tokens": 4096,
        "temperature": 0.1,
    }
    logging.info(f"  Parameters: {json.dumps(request_params, indent=2)}")
    
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=messages,
        max_tokens=4096,
        temperature=0.1,
        extra_body=self.extra_body,
    )
    
    response_text = res.choices[0].message.content.strip()
    token_count = total_token_count_from_response(res)
    
    logging.info(f"VLM API Response:")
    logging.info(f"  Tokens: {token_count}")
    logging.info(f"  Length: {len(response_text)} chars")
    logging.info(f"  Preview: {response_text[:200]}...")
    
    return response_text, token_count
```

### Step 6: Check VLM Server Settings

Your llama.cpp server might need these startup parameters:

```bash
# Correct startup for llama.cpp server with vision model
./llama-server \
  -m /models/Qwen2.5VL/Qwen2.5-VL-3B-Instruct-Q4_1.gguf \
  --mmproj /models/Qwen2.5VL/mmproj-BF16.gguf \
  --host 0.0.0.0 \
  --port 8080 \
  -c 32000 \               # Context size
  -n 4096 \                # Max tokens to predict (IMPORTANT!)
  --n-gpu-layers 99 \      # Offload to GPU
  --parallel 1 \           # Parallel requests
  --log-disable            # Disable verbose logging
```

The `-n 4096` parameter is critical - it sets the maximum prediction length.

## Recommended Action Plan

1. **First**: Clean rebuild with `--no-cache` (REBUILD_INSTRUCTIONS.md)
2. **Verify**: Run Step 1 to confirm all fixes are applied
3. **Test**: Try the VLM again through RAGFlow UI
4. **If still 6 tokens**: Run Steps 2-6 to diagnose VLM server configuration
5. **Check VLM Server**: May need to restart with correct `-n` parameter

## Quick Test Command

```bash
# After rebuild, test directly in container
docker exec ragflow-server bash -c '
cd /ragflow && 
python3 -c "
from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from pathlib import Path
import sys

# Quick inline test
try:
    vm = LLMBundle(\"test\", LLMType.IMAGE2TEXT, \"Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible\", \"English\")
    print(\"✓ LLMBundle created successfully\")
    
    # Check if describe_with_prompt exists and returns tuple
    import inspect
    sig = inspect.signature(vm.describe_with_prompt)
    print(f\"✓ describe_with_prompt signature: {sig}\")
    
except Exception as e:
    print(f\"✗ Error: {e}\")
    sys.exit(1)
"
'
```

## Most Likely Root Cause

Based on the logs, I suspect:
1. Docker cache preventing fixes from being applied (CONFIRMED)
2. VLM server may have low default `-n` parameter (needs verification)

The combination of both issues would explain why you're still seeing 6 tokens.