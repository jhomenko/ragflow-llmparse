# The REAL Bug - Message Construction Difference

## Key Observation

Both the working test and non-working module use system + user messages, BUT there's a subtle difference:

### Working Test (test_vlm_pdf_complete.py:103-120)
```python
def create_messages(img_b64, prompt="Transcribe this PDF page to clean Markdown."):
    return [
        {
            "role": "system",
            "content": (
                "You are a meticulous PDF-to-Markdown transcriber. "
                "Your task is to convert PDF pages into clean, well-structured Markdown. "
                "Preserve all text, tables, headings, and formatting. "
                "Output ONLY the Markdown content, no explanations."
            )
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},  # ← SHORT prompt: "Transcribe..."
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }
    ]
```

**User prompt in test:** `"Transcribe this PDF page to clean Markdown."` (~45 chars)

### Non-Working Module (working_vlm_module.py:64-80)
```python
system_message = {
    "role": "system",
    "content": (
        "You are a meticulous PDF-to-Markdown transcriber. "
        "Your task is to convert PDF pages into clean, well-structured Markdown. "
        "Preserve text, tables, headings, and formatting. "
        "Output ONLY the Markdown content, no explanations."
    ),
}

user_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": prompt},  # ← LONG prompt from template (~1192 chars!)
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ],
}
```

**User prompt in RAGFlow:** The FULL template from `vision_llm_describe_prompt.md` (~1192 chars)

## The REAL Bug

The issue is NOT the message format - both use system + user correctly!

The issue is **PROMPT LENGTH**:
- ✅ Working test: Short prompt (45 chars) → VLM generates content
- ❌ RAGFlow: Long template prompt (1192 chars) → VLM **echoes the prompt back**

## Why Long Prompts Cause Echoing

When the user text portion is very long and contains detailed instructions in a specific format (like markdown headers with "## INSTRUCTION"), the VLM might:

1. **Interpret the structured prompt AS the expected output format**
2. **Return it as an example/template** rather than generating new content
3. **Get confused** between "instructions" and "output format"

The prompt from the template starts with:
```markdown
## INSTRUCTION

Transcribe the content from the provided PDF page image into clean Markdown format.

- Only output the content transcribed from the image.
- Do NOT output this instruction or any other explanation.
...
## RULES
1. Do NOT generate examples...
```

This structured format may be **confusing the model** into thinking it should output the instructions themselves!

## The Actual Fix Needed

### Option 1: Simplify the Prompt Template

**File:** `rag/prompts/vision_llm_describe_prompt.md`

**Change from:**
```markdown
## INSTRUCTION

Transcribe the content from the provided PDF page image into clean Markdown format.

- Only output the content transcribed from the image.
- Do NOT output this instruction or any other explanation.
- If the content is missing or you do not understand the input, return an empty string.

## RULES

1. Do NOT generate examples, demonstrations, or templates.
2. Do NOT output any extra text such as 'Example', 'Example Output', or similar.
...
(many more lines)
```

**Change to:**
```markdown
Transcribe all text from this PDF page image into clean Markdown format. Preserve the original structure, formatting, tables, and headings exactly as shown. Output only the transcribed content with no explanations or meta-text.
```

### Option 2: Move Long Instructions to System Message

Keep the working_vlm_module the same, but restructure how we use the prompt:

**File:** `rag/llm/working_vlm_module.py`

```python
# Put detailed instructions in system message
system_message = {
    "role": "system",
    "content": prompt  # ← Use the full template here as instructions
}

# Simple user message with just the image
user_message = {
    "role": "user",
    "content": [
        {"type": "text", "text": "Transcribe this page to Markdown:"},  # ← Simple task
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
    ],
}
```

## Recommendation

**Option 1 is better** because:
1. Simpler prompt = less confusion for VLM
2. Your test proves short prompts work
3. The long detailed rules might be counterproductive
4. Matches your working test approach

## Implementation

1. Simplify `rag/prompts/vision_llm_describe_prompt.md` to ~2-3 sentences
2. Keep the message structure (system + user) the same
3. Test with the simplified prompt

The VLM should then generate content instead of echoing the prompt!