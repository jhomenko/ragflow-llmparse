# Simple Direct Fix - No More Diagnostics Needed

## The Problem

Your test script explicitly sets `stop=[]` and works. RAGFlow's code doesn't set `stop` at all.

## The Solution

Since you can't easily test with LLMBundle (no tenant setup), let's just **add comprehensive logging to the actual code path** and see what the VLM returns in real time.

## Quick Fix: Add Debug Logging

Add these 3 log statements to track the VLM response through the pipeline:

### 1. Log in cv_model.py (what VLM returns)

```bash
docker exec <container_id> python3 << 'EOF'
import re

# Read file
with open('/ragflow/rag/llm/cv_model.py', 'r') as f:
    content = f.read()

# Find the describe_with_prompt method and add logging after the API call
# Insert after line 207 (after res = self.client.chat.completions.create)
pattern = r'(res = self\.client\.chat\.completions\.create\([^)]+\))'
replacement = r'''\1
        
        # DEBUG: Log VLM response
        import logging
        response_text = res.choices[0].message.content
        logging.critical(f"==== cv_model.describe_with_prompt VLM RESPONSE ====")
        logging.critical(f"Response length: {len(response_text)} chars")
        logging.critical(f"Response preview: {response_text[:300]}")
        logging.critical(f"Finish reason: {res.choices[0].finish_reason}")
        logging.critical(f"Total tokens: {getattr(res.usage, 'total_tokens', 'N/A')}")'''

content = re.sub(pattern, replacement, content, count=1, flags=re.DOTALL)

# Write back
with open('/ragflow/rag/llm/cv_model.py', 'w') as f:
    f.write(content)

print("✅ Added logging to cv_model.py")
EOF
```

### 2. Log in picture.py (what gets returned to VisionParser)

```bash
docker exec <container_id> python3 << 'EOF'
# Read file
with open('/ragflow/rag/app/picture.py', 'r') as f:
    lines = f.readlines()

# Find line 221 (return txt) and add logging before it
for i, line in enumerate(lines):
    if 'return txt' in line and i > 200:  # Around line 221
        # Insert before the return
        lines.insert(i, '        logging.critical(f"==== picture.vision_llm_chunk RETURNING ====")\n')
        lines.insert(i+1, '        logging.critical(f"Returning text length: {len(txt)} chars")\n')
        lines.insert(i+2, '        logging.critical(f"Returning text preview: {txt[:200]}")\n')
        lines.insert(i+3, '\n')
        break

# Write back
with open('/ragflow/rag/app/picture.py', 'w') as f:
    f.writelines(lines)

print("✅ Added logging to picture.py")
EOF
```

### 3. Log in pdf_parser.py (what VisionParser receives)

```bash
docker exec <container_id> python3 << 'EOF'
# Read file
with open('/ragflow/deepdoc/parser/pdf_parser.py', 'r') as f:
    lines = f.readlines()

# Find line ~1500 where text = picture_vision_llm_chunk returns
for i, line in enumerate(lines):
    if 'picture_vision_llm_chunk(' in line and i > 1490:
        # Insert after this line
        lines.insert(i+1, '                logging.critical(f"==== VisionParser received from picture_vision_llm_chunk ====")\n')
        lines.insert(i+2, '                logging.critical(f"Received text type: {type(text)}, length: {len(str(text)) if text else 0}")\n')
        lines.insert(i+3, '\n')
        break

# Write back
with open('/ragflow/deepdoc/parser/pdf_parser.py', 'w') as f:
    f.writelines(lines)

print("✅ Added logging to pdf_parser.py")
EOF
```

### 4. Restart Container

```bash
docker restart <container_id>
# Wait for it to come back up
sleep 15
```

### 5. Upload PDF and Check Logs

```bash
# In one terminal, tail the logs
docker logs -f <container_id> 2>&1 | grep "====" 

# In another terminal/browser, upload your test PDF via UI
```

## What This Will Show

You'll see EXACTLY:
1. What the VLM returns (length, preview, finish reason)
2. What picture.py returns
3. What VisionParser receives

This will pinpoint where the content disappears or if the VLM itself is returning short responses.

## If VLM Returns Short Response

Then the issue IS with how the API is being called. The fix would be to match your test's parameters more closely.

## If VLM Returns Long Response But It Gets Lost

Then we know it's in the post-processing (cleaning, parsing, etc.) and we can fix that specific step.

No more guessing - just add the logging, upload a PDF, and we'll see exactly what's happening!