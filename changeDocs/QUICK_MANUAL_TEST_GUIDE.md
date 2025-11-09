# Quick Manual Testing Without Docker Rebuild

## How to Test Stop Token Changes Manually

If you want to quickly test different stop token configurations **without rebuilding Docker**, you can directly edit the running container's Python files.

### Method 1: Edit File in Running Container

```bash
# 1. Get your container ID
docker ps

# 2. Enter the container
docker exec -it <container_id> bash

# 3. Edit the cv_model.py file
vi /ragflow/rag/llm/cv_model.py
# or
nano /ragflow/rag/llm/cv_model.py

# 4. Find line ~206 in GptV4.describe_with_prompt()
# Current code (with stop=[]):
res = self.client.chat.completions.create(
    model=self.model_name,
    messages=self.vision_llm_prompt(b64, prompt),
    max_tokens=4096,
    temperature=0.1,
    stop=[],  # ← LINE TO CHANGE
    extra_body=self.extra_body,
)

# 5. Test different configurations:

# Option A: Remove stop parameter entirely (test "no stop")
res = self.client.chat.completions.create(
    model=self.model_name,
    messages=self.vision_llm_prompt(b64, prompt),
    max_tokens=4096,
    temperature=0.1,
    # stop=[],  # ← COMMENTED OUT
    extra_body=self.extra_body,
)

# Option B: Use None (same as removing it)
res = self.client.chat.completions.create(
    model=self.model_name,
    messages=self.vision_llm_prompt(b64, prompt),
    max_tokens=4096,
    temperature=0.1,
    stop=None,  # ← TEST THIS
    extra_body=self.extra_body,
)

# Option C: Keep stop=[] (current implementation)
# (no changes needed)

# 6. Save and exit (vi: :wq, nano: Ctrl+X, Y, Enter)

# 7. Restart the RAGFlow service to pick up changes
# From OUTSIDE the container:
docker-compose restart ragflow
# OR from inside:
supervisorctl restart ragflow_server
```

### Method 2: Copy Modified File into Container

```bash
# 1. Edit rag/llm/cv_model.py on your host machine
vi rag/llm/cv_model.py

# 2. Copy it into the running container
docker cp rag/llm/cv_model.py <container_id>:/ragflow/rag/llm/cv_model.py

# 3. Restart service
docker-compose restart ragflow
```

### Test Configuration Matrix

Test these configurations to understand the differences:

| Config | Code | Expected Result |
|--------|------|-----------------|
| **stop=[]** | `stop=[]` | Should produce ~2864 chars (your test) |
| **No stop param** | `# stop=[]` (commented) | Should produce ~3547 chars (your test) |
| **stop=None** | `stop=None` | Likely same as no stop param |

### Quick Test After Each Change

```bash
# Inside container, run the enhanced test script:
cd /ragflow
python3 test_vlm_pdf_complete.py jair_page.pdf 0

# It will create output files:
# - vlm_output_stop=[]_(RAGFlow_Fix_#4_-_Disable_defaults).md
# - vlm_output_No_stop_parameter_(Client_defaults).md
# - vlm_output_OpenAI_stop_tokens_(Should_fail).md

# Compare the files:
wc -l vlm_output_*.md
diff vlm_output_stop=[]*.md vlm_output_No_stop*.md
```

## Understanding the Difference

### Why No Stop (3547 chars) > stop=[] (2864 chars)?

There are several possible reasons:

**Theory 1: Model Sampling Variability**
- LLMs use temperature-based sampling (even at 0.1)
- Each run can produce slightly different output lengths
- The model might naturally stop at different points

**Theory 2: Stop Token Interpretation**
- `stop=[]` might be interpreted differently by llama.cpp
- Some implementations treat `stop=[]` as "use model defaults"
- No stop parameter might mean "no stop constraints at all"

**Theory 3: Model Internal Stop Logic**
- Qwen2.5VL has built-in stop tokens: `<|im_end|>`, `<|endoftext|>`
- `stop=[]` might still respect these
- No stop parameter might override even these

### Recommended Test Sequence

1. **First**, run [`test_vlm_pdf_complete.py`](test_vlm_pdf_complete.py:1) with current code
2. **Review** the 3 output files it creates
3. **Compare** `stop=[]` vs `no stop` outputs:
   - Are they both complete transcriptions?
   - Does one cut off mid-sentence?
   - Is one more accurate?
4. **Manually test** by commenting out `stop=[]`
5. **Upload same PDF** via RAGFlow UI
6. **Compare** chunk content with saved files

### Decision Criteria

**Use `stop=[]` if:**
- ✅ Both outputs are equally complete
- ✅ You want explicit, predictable behavior
- ✅ You plan to deploy to production (safer)

**Use no stop parameter if:**
- ✅ No stop produces significantly better results
- ✅ The longer output is more complete/accurate
- ✅ You're OK with implicit OpenAI client behavior

**Test more if:**
- ⚠️ One cuts off mid-sentence
- ⚠️ Quality differs significantly
- ⚠️ Results are inconsistent across runs

## Current File Locations in Container

```
/ragflow/rag/llm/cv_model.py              ← Line ~206 (stop token config)
/ragflow/api/db/services/llm_service.py   ← Bug #1 fix
/ragflow/deepdoc/parser/pdf_parser.py     ← VisionParser
/ragflow/rag/flow/parser/parser.py        ← Parser._pdf()
```

## Verification After Changes

```bash
# Check if file was modified
stat /ragflow/rag/llm/cv_model.py

# Search for the exact line
grep -n "stop=" /ragflow/rag/llm/cv_model.py

# Should show something like:
# 206:            stop=[],
# or
# 206:            # stop=[],

# View the exact section
sed -n '199,210p' /ragflow/rag/llm/cv_model.py
```

## Rollback if Needed

```bash
# If something breaks, restore from your local copy:
docker cp rag/llm/cv_model.py <container_id>:/ragflow/rag/llm/cv_model.py
docker-compose restart ragflow
```

## Important Notes

⚠️ **Temporary Changes**: Manual edits will be lost if you:
- Rebuild the Docker image
- Delete and recreate the container
- Update the container

✅ **Permanent Changes**: To make permanent:
1. Test manually first (as above)
2. Once you confirm the best config, update source code
3. Rebuild Docker image with `--no-cache`
4. Deploy new image

## Next Steps

1. Run updated [`test_vlm_pdf_complete.py`](test_vlm_pdf_complete.py:1) to get output files
2. Review the saved markdown files
3. Decide based on actual content quality
4. Manually test the winner configuration
5. Make permanent in source code if satisfied