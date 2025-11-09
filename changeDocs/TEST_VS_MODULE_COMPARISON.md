# Critical Comparison: Test Script vs RAGFlow Modules

## Your Test Script (Works - 3547 chars)

```python
# test_vlm_pdf_complete.py line 241
client = OpenAI(api_key="not-needed", base_url=VLM_SERVER)

# Lines 251-254 - Uses stop=[]
results["stop_empty"] = test_vlm_call(
    client, messages, [],
    "stop=[] (RAGFlow Fix #4 - Disable defaults)"
)

# Line 143 - The actual call
res = client.chat.completions.create(
    model=MODEL_NAME,
    messages=messages,
    max_tokens=4096,
    temperature=0.1,
    stop=[]  # ← EXPLICIT stop=[]
)
```

## RAGFlow Modules (Fails - ~6 tokens)

Looking at `containercurrent/cv_model.py` lines 199-208:

```python
def describe_with_prompt(self, image, prompt=None):
    b64 = self.image2base64(image)
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=self.vision_llm_prompt(b64, prompt),
        max_tokens=4096,
        temperature=0.1,
        extra_body=self.extra_body,
    )
    return res.choices[0].message.content.strip(), total_token_count_from_response(res)
```

## THE SMOKING GUN: Missing `stop=[]`!

**Your test explicitly sets `stop=[]`**

**RAGFlow's code does NOT set `stop`!**

When `stop` is not specified, the OpenAI client (or the server) may use DEFAULT stop tokens that cause early termination!

## The Bug

In your earlier stop token test, you found:
- `stop=[]`: 2864 characters
- No stop parameter: 3547 characters  

So actually `stop=[]` produces LESS content than no stop parameter for your setup!

But the UI is getting ~6 tokens, which suggests it's hitting some OTHER stop condition.

## Hypothesis: `extra_body` is the Culprit

Look at line 206 in `cv_model.py`:
```python
extra_body=self.extra_body,
```

This `extra_body` parameter might contain configuration that adds stop tokens or other restrictions!

Check what's in `extra_body`:

```bash
docker exec <container> python3 << 'EOF'
from api.db import LLMType
from api.db.services.llm_service import LLMBundle

# Create the same LLMBundle the UI uses
vision_model = LLMBundle(
    tenant_id="YOUR_TENANT_ID",  # Use your actual tenant ID
    llm_type=LLMType.IMAGE2TEXT,
    llm_name="Qwen2.5VL-3B",
    lang="Chinese"  # Or whatever default is used
)

# Check what extra_body contains
print("extra_body:", getattr(vision_model.mdl, "extra_body", None))
print("model_name:", getattr(vision_model.mdl, "model_name", None))
print("client base_url:", getattr(vision_model.mdl.client, "base_url", None))

# Try to see all attributes
import pprint
print("\nAll mdl attributes:")
pprint.pprint({k: v for k, v in vars(vision_model.mdl).items() if not k.startswith('_')})
EOF
```

## Other Potential Differences

1. **Client initialization**:
   - Test: `OpenAI(api_key="not-needed", base_url=VLM_SERVER)`
   - Module: Client created somewhere in LLMBundle initialization

2. **Messages format**:
   - Test: Uses explicit system message + user message with image
   - Module: Uses `self.vision_llm_prompt(b64, prompt)` which ALSO creates system message

3. **Base URL**:
   - Test: `"http://192.168.68.186:8080/v1"`
   - Module: Might be different or have additional path components

## The Fix

Add `stop=[]` to `cv_model.py` line 201-208:

```python
def describe_with_prompt(self, image, prompt=None):
    b64 = self.image2base64(image)
    res = self.client.chat.completions.create(
        model=self.model_name,
        messages=self.vision_llm_prompt(b64, prompt),
        max_tokens=4096,
        temperature=0.1,
        stop=[],  # ← ADD THIS LINE
        extra_body=self.extra_body,
    )
    return res.choices[0].message.content.strip(), total_token_count_from_response(res)
```

BUT WAIT - you said earlier that `stop=[]` gives you LESS content (2864 vs 3547). 

So maybe the fix is to explicitly set `stop=None` or investigate what's in `extra_body`.

## Root Cause Summary

The difference between your test and the modules:

1. **Your test**: Explicitly controls ALL parameters including stop tokens
2. **Modules**: Let some parameters be implicit, which may trigger default behavior
3. **`extra_body`**: Might contain additional configuration that affects output

The answer is in one of:
- `extra_body` content
- Client initialization differences  
- Different API endpoint behavior

Run that diagnostic above to see what `extra_body` contains!