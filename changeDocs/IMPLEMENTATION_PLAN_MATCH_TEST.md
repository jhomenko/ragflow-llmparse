# Implementation Plan: Make RAGFlow Match Working Test Script

## The Problem is Clear

Your log shows: `vision_llm_chunk: VLM response tokens=1399, chars=14`

**The VLM itself is returning only 14 characters!** This confirms it's not post-processing - the API call is wrong.

## What Your Test Does (Works - 3547 chars)

```python
# test_vlm_pdf_complete.py
client = OpenAI(api_key="not-needed", base_url="http://192.168.68.186:8080/v1")

res = client.chat.completions.create(
    model="Qwen2.5VL-3B",
    messages=[
        {
            "role": "system",
            "content": "You are a meticulous PDF-to-Markdown transcriber..."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }
    ],
    max_tokens=4096,
    temperature=0.1,
    # NO stop parameter (works best for you)
)
```

## What RAGFlow Does (Fails - 14 chars)

Looking at `containercurrent/cv_model.py`:

```python
res = self.client.chat.completions.create(
    model=self.model_name,
    messages=self.vision_llm_prompt(b64, prompt),
    max_tokens=4096,
    temperature=0.1,
    extra_body=self.extra_body,  # ← This might be the problem!
)
```

## Key Differences

1. **`extra_body` parameter** - Your test doesn't have this
2. **Possible model name mismatch** - Need to verify `self.model_name` is exactly "Qwen2.5VL-3B"
3. **Client initialization** - Different base_url or API key might affect behavior

## The Fix: Make RAGFlow Exactly Match Your Test

### Step 1: Remove/Disable `extra_body`

```bash
docker exec <container_id> python3 << 'EOF'
# Modify cv_model.py to NOT use extra_body for VLM calls
with open("/ragflow/rag/llm/cv_model.py", "r") as f:
    content = f.read()

# Find the describe_with_prompt method and remove extra_body
old_code = """res = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.vision_llm_prompt(b64, prompt),
            max_tokens=4096,
            temperature=0.1,
            extra_body=self.extra_body,
        )"""

new_code = """res = self.client.chat.completions.create(
            model=self.model_name,
            messages=self.vision_llm_prompt(b64, prompt),
            max_tokens=4096,
            temperature=0.1,
            # extra_body=self.extra_body,  # DISABLED - causes issues with VLM
        )"""

content = content.replace(old_code, new_code)

with open("/ragflow/rag/llm/cv_model.py", "w") as f:
    f.write(content)

print("✅ Removed extra_body from describe_with_prompt")
EOF
```

### Step 2: Verify Model Name

```bash
docker exec <container_id> python3 << 'EOF'
# Check what model name is being used
import re
with open("/ragflow/rag/llm/cv_model.py", "r") as f:
    content = f.read()

# Find where model_name is set
model_name_match = re.search(r'self\.model_name = (.+)', content)
if model_name_match:
    print(f"Model name assignment: {model_name_match.group(0)}")
else:
    print("Could not find model_name assignment")
EOF
```

### Step 3: Test Again

Upload the same PDF and check if it now returns ~3500 chars like your test script.

## If That Doesn't Work

Then we need to check the **Client initialization**. The issue might be in how the OpenAI client is created in LLMBundle/TenantLLMService.

### Verify Client Setup

```bash
docker exec <container_id> grep -A 10 "OpenAI(api_key" /ragflow/rag/llm/cv_model.py
```

Compare the base_url, api_key, and any other parameters to your test script.

## Expected Outcome

After removing `extra_body`, the VLM should return ~3500 chars just like your test script, because all other parameters will match.

##Implementation

Run Step 1 above, then test. If it works, we're done. If not, run Steps 2-3 for further diagnosis.