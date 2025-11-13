# Hybrid VLM Table Parsing Implementation Plan

## Executive Summary

This document outlines the implementation plan for integrating Vision Language Model (VLM) based table parsing into RAGFlow's existing deepdoc pipeline. The hybrid approach leverages deepdoc's accurate table detection with VLM's superior semantic understanding of complex table structures.

**Key Benefits:**
- **Accuracy**: VLMs excel at understanding merged cells, nested headers, and complex table layouts
- **Speed**: Deepdoc handles fast layout detection; VLM only processes table regions
- **Compatibility**: Drop-in replacement maintaining existing schema and cross-page support
- **Flexibility**: Opt-in via environment variables; fallback to existing TableStructureRecognizer

---

## Architecture Overview

### Current Flow
```
PDF Pages → RAGFlowPdfParser.__images__() → OCR + Layout Detection
         → _table_transformer_job() → Crop table regions
         → TableStructureRecognizer.__call__() → HTML/Text output
         → _extract_table_figure() → Insert into document flow
```

### Proposed Hybrid Flow
```
PDF Pages → RAGFlowPdfParser.__images__() → OCR + Layout Detection
         → _table_transformer_job() → Crop table regions
         → [ENV: USE_VLM_TABLE_PARSING?]
            ├─ YES → _vlm_table_parser() → HTML output (VLM)
            └─ NO  → TableStructureRecognizer.__call__() → HTML output (existing)
         → _extract_table_figure() → Insert into document flow
```

**Key Integration Point**: Between table cropping and table parsing, routing based on environment configuration.

---

## Implementation Details

### 1. Environment Variables

**File**: Add to `docker-compose.yml` or `.env`

```bash
# Enable VLM-based table parsing (default: false for backward compatibility)
USE_VLM_TABLE_PARSING=false

# VLM model to use for table parsing (optional, defaults to document VLM model)
VLM_TABLE_MODEL=Qwen2.5VL-3B

# Table-specific VLM timeout (seconds)
VLM_TABLE_TIMEOUT_SEC=

# Fallback to TableStructureRecognizer on VLM failure (default: true)
VLM_TABLE_FALLBACK_ENABLED=true

# Output format: "html" or "markdown" (default: html to match existing output)
VLM_TABLE_OUTPUT_FORMAT=html
```

**Rationale**: 
- Backward compatible (disabled by default)
- Allows separate model selection for table parsing
- Timeout control for table-specific processing
- Safety net with fallback enabled

---

### 2. Smart Resize Integration

**File**: `deepdoc/parser/pdf_parser.py`

**Critical**: Use existing `smart_resize()` function (lines 76-108) for table image preprocessing.

```python
def _vlm_table_parser(self, table_images, table_positions, vision_model=None):
    """
    Parse tables using VLM with smart resizing for Qwen3-VL compatibility.
    
    Args:
        table_images: List of PIL Image objects (cropped table regions)
        table_positions: List of (page_num, left, right, top, bottom) tuples
        vision_model: LLMBundle instance for IMAGE2TEXT (optional)
    
    Returns:
        List of HTML strings (same format as TableStructureRecognizer.construct_table)
    """
    # Get resize factor from environment (default: 32 for Qwen3-VL)
    resize_factor = int(os.getenv("VLM_RESIZE_FACTOR", "32"))
    timeout = int(os.getenv("VLM_TABLE_TIMEOUT_SEC", "30"))
    output_format = os.getenv("VLM_TABLE_OUTPUT_FORMAT", "html").lower()
    
    results = []
    
    for idx, (img, pos) in enumerate(zip(table_images, table_positions)):
        try:
            # Convert to RGB
            img = img.convert("RGB")
            
            # Apply smart_resize to ensure dimensions are multiples of factor
            width, height = img.size
            target_height, target_width = smart_resize(
                height, width,
                factor=resize_factor,
                target_max_dimension=1024  # Balance quality vs speed
            )
            
            # Resize with high-quality resampling
            img = img.resize((target_width, target_height), 
                           resample=Image.Resampling.LANCZOS)
            
            logging.debug(f"Table {idx}: Resized from {width}x{height} to "
                        f"{target_width}x{target_height} (factor={resize_factor})")
            
            # Convert to JPEG bytes
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90, optimize=True)
            jpg_bytes = buf.getvalue()
            
            # Construct table-specific prompt
            prompt = self._get_table_prompt(output_format)
            
            # Call VLM with timeout
            from rag.app.picture import vision_llm_chunk
            
            table_output = vision_llm_chunk(
                binary=jpg_bytes,
                vision_model=vision_model,
                prompt=prompt,
                callback=lambda p, m: logging.debug(f"Table {idx}: {m}")
            )
            
            # Validate and clean output
            if output_format == "html":
                table_output = self._validate_html_table(table_output)
            else:  # markdown
                table_output = self._validate_markdown_table(table_output)
            
            results.append(table_output)
            
        except Exception as e:
            logging.error(f"VLM table parsing failed for table {idx}: {e}")
            
            # Check if fallback is enabled
            if os.getenv("VLM_TABLE_FALLBACK_ENABLED", "true").lower() == "true":
                logging.info(f"Falling back to TableStructureRecognizer for table {idx}")
                # Return None to signal fallback needed
                results.append(None)
            else:
                # Return empty table
                results.append("<table><tr><td>Table parsing failed</td></tr></table>")
    
    return results
```

---

### 3. Table-Specific Prompts

**File**: Create `rag/prompts/table_vlm_prompt.md`

```markdown
# HTML Format Prompt
You are a table extraction specialist. Analyze the provided table image and convert it to clean HTML format.

**Requirements:**
1. Use standard HTML table tags: <table>, <tr>, <th>, <td>
2. Preserve all headers, rows, columns, and cell content exactly
3. For merged cells, use colspan and rowspan attributes
4. Maintain data types (numbers, dates, text)
5. Include table caption if visible: <caption>...</caption>
6. Do NOT add any CSS, styles, or classes
7. Do NOT add markdown formatting
8. Output ONLY the HTML table, no explanations

**Example Output:**
```html
<table>
<caption>Sales Report 2024</caption>
<tr>
  <th>Product</th>
  <th colspan="2">Q1 Sales</th>
</tr>
<tr>
  <td>Widget A</td>
  <td>1,250</td>
  <td>$45,000</td>
</tr>
</table>
```

Begin table extraction now.
```

**File**: Add prompt getter method to `RAGFlowPdfParser`

```python
def _get_table_prompt(self, output_format="html"):
    """Load table-specific VLM prompt."""
    from pathlib import Path
    
    prompt_file = os.getenv("VLM_TABLE_PROMPT_PATH")
    if not prompt_file:
        base = Path(__file__).resolve().parent.parent.parent
        prompt_file = base / "rag" / "prompts" / "table_vlm_prompt.md"
    else:
        prompt_file = Path(prompt_file)
    
    if prompt_file.exists():
        prompt = prompt_file.read_text(encoding="utf-8")
    else:
        # Fallback inline prompt
        if output_format == "html":
            prompt = """Extract this table as HTML. Use <table>, <tr>, <th>, <td> tags. 
            Use colspan/rowspan for merged cells. Output only the HTML table, no markdown."""
        else:
            prompt = """Extract this table as markdown. Use | for columns, proper alignment. 
            Preserve all headers and data exactly."""
    
    return prompt
```

---

### 4. Output Validation

**File**: `deepdoc/parser/pdf_parser.py` - Add validation methods

```python
def _validate_html_table(self, html_output):
    """
    Validate and clean HTML table output from VLM.
    Ensures compatibility with downstream processing.
    """
    import re
    
    # Remove markdown code fences if present
    html_output = re.sub(r'^```html\n?', '', html_output, flags=re.MULTILINE)
    html_output = re.sub(r'\n?```$', '', html_output, flags=re.MULTILINE)
    html_output = html_output.strip()
    
    # Verify it contains table tags
    if '<table' not in html_output.lower():
        logging.warning("VLM output missing <table> tag, wrapping content")
        html_output = f"<table>\n{html_output}\n</table>"
    
    # Basic HTML validation (optional, can be expanded)
    if '<tr' not in html_output.lower():
        logging.warning("VLM output missing <tr> tags")
    
    return html_output

def _validate_markdown_table(self, md_output):
    """Validate markdown table output."""
    md_output = md_output.strip()
    
    # Check for markdown table format (| separator)
    if '|' not in md_output:
        logging.warning("VLM output doesn't appear to be markdown table format")
    
    return md_output
```

---

### 5. Modify Table Transformer Job

**File**: `deepdoc/parser/pdf_parser.py:255-297`

**Changes to `_table_transformer_job()` method:**

```python
def _table_transformer_job(self, ZM):
    """
    Table processing with optional VLM routing.
    Modified to support hybrid VLM/TableStructureRecognizer approach.
    """
    logging.debug("Table processing...")
    imgs, pos = [], []
    tbcnt = [0]
    MARGIN = 10
    self.tb_cpns = []
    
    assert len(self.page_layout) == len(self.page_images)
    
    # Step 1: Crop table regions (UNCHANGED)
    for p, tbls in enumerate(self.page_layout):
        tbls = [f for f in tbls if f["type"] == "table"]
        tbcnt.append(len(tbls))
        if not tbls:
            continue
        for tb in tbls:
            left, top, right, bott = tb["x0"] - MARGIN, tb["top"] - MARGIN, \
                                     tb["x1"] + MARGIN, tb["bottom"] + MARGIN
            left *= ZM
            top *= ZM
            right *= ZM
            bott *= ZM
            pos.append((left, top))
            imgs.append(self.page_images[p].crop((left, top, right, bott)))
    
    assert len(self.page_images) == len(tbcnt) - 1
    if not imgs:
        return
    
    # Step 2: Route to VLM or TableStructureRecognizer
    use_vlm_tables = os.getenv("USE_VLM_TABLE_PARSING", "false").lower() == "true"
    
    if use_vlm_tables and hasattr(self, 'vision_model'):
        logging.info(f"Using VLM for table parsing ({len(imgs)} tables)")
        
        # Get VLM model (use document model or specific table model)
        table_model_name = os.getenv("VLM_TABLE_MODEL")
        if table_model_name and table_model_name != getattr(self.vision_model, 'llm_name', ''):
            # Create separate model instance for tables
            try:
                from api.db.services.llm_service import LLMBundle
                from api.db import LLMType
                table_vision_model = LLMBundle(
                    self.vision_model.tenant_id,
                    LLMType.IMAGE2TEXT,
                    llm_name=table_model_name
                )
            except Exception as e:
                logging.warning(f"Failed to create table-specific model, using document model: {e}")
                table_vision_model = self.vision_model
        else:
            table_vision_model = self.vision_model
        
        # Call VLM table parser
        vlm_results = self._vlm_table_parser(imgs, pos, table_vision_model)
        
        # Handle results with fallback support
        recos = []
        fallback_imgs = []
        fallback_indices = []
        
        for idx, result in enumerate(vlm_results):
            if result is None:  # Fallback needed
                fallback_imgs.append(imgs[idx])
                fallback_indices.append(idx)
                recos.append(None)  # Placeholder
            else:
                # Convert HTML to expected format
                recos.append([{"html": result}])  # Wrap in expected structure
        
        # Process fallback tables with TableStructureRecognizer
        if fallback_imgs:
            logging.info(f"Processing {len(fallback_imgs)} tables with TableStructureRecognizer fallback")
            fallback_recos = self.tbl_det(fallback_imgs)
            for i, idx in enumerate(fallback_indices):
                recos[idx] = fallback_recos[i]
    else:
        # Use existing TableStructureRecognizer (UNCHANGED)
        logging.debug("Using TableStructureRecognizer for table parsing")
        recos = self.tbl_det(imgs)
    
    # Step 3: Process results and assign to page components (MODIFIED)
    tbcnt = np.cumsum(tbcnt)
    for i in range(len(tbcnt) - 1):
        pg = []
        for j, tb_items in enumerate(recos[tbcnt[i]:tbcnt[i + 1]]):
            poss = pos[tbcnt[i]:tbcnt[i + 1]]
            
            # Handle VLM HTML output format
            if isinstance(tb_items, list) and len(tb_items) > 0 and \
               isinstance(tb_items[0], dict) and 'html' in tb_items[0]:
                # VLM HTML result - store directly
                # Note: This will be handled differently in _extract_table_figure
                for it in tb_items:
                    it["x0"] = poss[j][0] / ZM
                    it["x1"] = (poss[j][0] + imgs[tbcnt[i] + j].size[0]) / ZM
                    it["top"] = (poss[j][1] / ZM) + self.page_cum_height[i]
                    it["bottom"] = ((poss[j][1] + imgs[tbcnt[i] + j].size[1]) / ZM) + \
                                   self.page_cum_height[i]
                    it["pn"] = i
                    it["layoutno"] = j
                    it["source"] = "vlm"  # Mark source for later processing
                    pg.append(it)
            else:
                # Standard TableStructureRecognizer output (UNCHANGED)
                for it in tb_items:
                    it["x0"] = it["x0"] + poss[j][0]
                    it["x1"] = it["x1"] + poss[j][0]
                    it["top"] = it["top"] + poss[j][1]
                    it["bottom"] = it["bottom"] + poss[j][1]
                    for n in ["x0", "x1", "top", "bottom"]:
                        it[n] /= ZM
                    it["top"] += self.page_cum_height[i]
                    it["bottom"] += self.page_cum_height[i]
                    it["pn"] = i
                    it["layoutno"] = j
                    pg.append(it)
        self.tb_cpns.extend(pg)
    
    # Step 4: Continue with existing row/column/spanning logic (UNCHANGED for non-VLM)
    # VLM tables bypass this as they're already structured HTML
    non_vlm_cpns = [c for c in self.tb_cpns if c.get("source") != "vlm"]
    if non_vlm_cpns:
        # Existing gather() logic for headers, rows, spans, columns...
        # (Lines 299-340 remain unchanged)
        def gather(kwd, fzy=10, ption=0.6):
            eles = Recognizer.sort_Y_firstly([r for r in non_vlm_cpns if re.match(kwd, r["label"])], fzy)
            eles = Recognizer.layouts_cleanup(self.boxes, eles, 5, ption)
            return Recognizer.sort_Y_firstly(eles, 0)
        
        headers = gather(r".*header$")
        rows = gather(r".* (row|header)")
        spans = gather(r".*spanning")
        clmns = sorted([r for r in non_vlm_cpns if re.match(r"table column$", r["label"])], 
                      key=lambda x: (x["pn"], x["layoutno"], x["x0"]))
        clmns = Recognizer.layouts_cleanup(self.boxes, clmns, 5, 0.5)
        
        # Apply R, H, C, SP tags only to non-VLM boxes
        for b in self.boxes:
            if b.get("layout_type", "") != "table":
                continue
            # ... existing logic ...
```

---

### 6. Modify Extract Table Figure

**File**: `deepdoc/parser/pdf_parser.py:798-971`

**Changes to `_extract_table_figure()` method:**

```python
# Around line 950-958 in the table processing loop:
for k, bxs in tables.items():
    if not bxs:
        continue
    
    # Check if this is VLM-sourced table
    is_vlm_table = any(b.get("source") == "vlm" for b in bxs)
    
    if is_vlm_table:
        # VLM tables: extract HTML directly
        html_content = ""
        for b in bxs:
            if "html" in b:
                html_content += b["html"]
        
        # Get caption if present
        caption = "\n".join([b["text"] for b in bxs if TableStructureRecognizer.is_caption(b)])
        if caption:
            # Insert caption into HTML if not already present
            if "<caption>" not in html_content.lower():
                html_content = html_content.replace("<table>", f"<table>\n<caption>{caption}</caption>", 1)
        
        poss = []
        res.append((cropout(bxs, "table", poss), [html_content]))
        positions.append(poss)
    else:
        # Standard deepdoc path (UNCHANGED)
        bxs = Recognizer.sort_Y_firstly(bxs, np.mean([(b["bottom"] - b["top"]) / 2 for b in bxs]))
        poss = []
        res.append((cropout(bxs, "table", poss), 
                   self.tbl_det.construct_table(bxs, html=return_html, is_english=self.is_english)))
        positions.append(poss)
```

---

### 7. Pass Vision Model to RAGFlowPdfParser

**File**: `rag/flow/parser/parser.py` (around line 219-280)

**Modify `_pdf()` method to pass vision_model to RAGFlowPdfParser:**

```python
def _pdf(self, name, blob):
    # ... existing code ...
    
    if conf.get("parse_method").lower() == "deepdoc":
        # Create parser instance
        parser = RAGFlowPdfParser()
        
        # Check if VLM table parsing is enabled
        if os.getenv("USE_VLM_TABLE_PARSING", "false").lower() == "true":
            # Get vision model for table parsing
            table_model_name = os.getenv("VLM_TABLE_MODEL", conf.get("parse_method"))
            try:
                vision_model = LLMBundle(
                    self._canvas._tenant_id,
                    LLMType.IMAGE2TEXT,
                    llm_name=table_model_name,
                    lang=conf.get("lang", "English")
                )
                # Attach to parser instance
                parser.vision_model = vision_model
                logging.info(f"VLM table parsing enabled with model: {table_model_name}")
            except Exception as e:
                logging.warning(f"Failed to initialize VLM for table parsing: {e}")
        
        # Call parser
        bboxes = parser.parse_into_bboxes(blob, callback=self.callback)
    elif conf.get("parse_method").lower() == "plain_text":
        # ... rest unchanged ...
```

---

## Testing Strategy

### Phase 1: Unit Testing

**Test File**: Create `test_vlm_table_parsing.py`

```python
import os
import io
from PIL import Image
import pytest

def test_smart_resize_factor_alignment():
    """Verify images are resized to multiples of VLM_RESIZE_FACTOR"""
    from deepdoc.parser.pdf_parser import smart_resize
    
    # Test various inputs
    test_cases = [
        (800, 600, 32, 1024),  # Normal case
        (2000, 1500, 32, 1024),  # Downscale needed
        (400, 300, 32, 1024),  # Upscale case
    ]
    
    for h, w, factor, max_dim in test_cases:
        new_h, new_w = smart_resize(h, w, factor, max_dim)
        assert new_h % factor == 0, f"Height {new_h} not multiple of {factor}"
        assert new_w % factor == 0, f"Width {new_w} not multiple of {factor}"
        assert max(new_h, new_w) <= max_dim, f"Dimension exceeds {max_dim}"

def test_vlm_table_prompt_loading():
    """Verify table prompt loads correctly"""
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
    
    parser = RAGFlowPdfParser()
    prompt = parser._get_table_prompt("html")
    
    assert "table" in prompt.lower()
    assert "html" in prompt.lower() or "<table>" in prompt.lower()

def test_html_validation():
    """Test HTML output validation"""
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
    
    parser = RAGFlowPdfParser()
    
    # Test with markdown fence
    html_with_fence = "```html\n<table><tr><td>Test</td></tr></table>\n```"
    cleaned = parser._validate_html_table(html_with_fence)
    assert "```" not in cleaned
    assert "<table>" in cleaned
    
    # Test missing table tag
    html_incomplete = "<tr><td>Test</td></tr>"
    cleaned = parser._validate_html_table(html_incomplete)
    assert "<table>" in cleaned
```



---

## Deployment Plan

### Phase 1: Development
- [ ] Implement `_vlm_table_parser()` method
- [ ] Add smart resize integration
- [ ] Create table prompt template
- [ ] Add validation methods
- [ ] Unit tests

### Phase 2: Integration
- [ ] Modify `_table_transformer_job()`
- [ ] Update `_extract_table_figure()`
- [ ] Pass vision_model to parser
- [ ] Handle VLM HTML format in downstream code

### Phase 3: Testing
- [ ] Test with single-page tables
- [ ] Test with multi-page tables
- [ ] Test fallback scenarios
- [ ] Performance benchmarking
- [ ] Edge case handling

### Phase 4: Documentation & Deployment
- [ ] Update configuration documentation
- [ ] Create usage examples
- [ ] Deployment guide
- [ ] Rollback procedures
- [ ] Production deployment

---

## Risk Mitigation

### Risk 1: VLM Output Format Inconsistency
**Mitigation:**
- Strict prompt engineering with examples
- Robust validation and cleaning
- Fallback to TableStructureRecognizer

### Risk 2: Performance Degradation
**Mitigation:**
- VLM only processes table regions (not full pages)
- Configurable timeout per table
- Parallel processing possible in future

### Risk 3: Cross-page Table Handling
**Mitigation:**
- Leverage existing concatenation logic (lines 838-853)
- VLM processes concatenated image
- Test extensively with multi-page tables

### Risk 4: HTML Schema Compatibility
**Mitigation:**
- Match TableStructureRecognizer output format exactly
- Validation ensures required tags present
- Integration tests verify downstream compatibility

---

## Success Criteria

### Functional Requirements
- ✅ VLM tables produce valid HTML output
- ✅ Cross-page tables concatenate correctly
- ✅ Fallback to deepdoc works reliably
- ✅ Environment variable control functional
- ✅ Smart resize with factor alignment (multiples of 32)

### Performance Requirements
- ✅ No memory leaks or resource exhaustion
- ✅ Graceful degradation on VLM failure

### Quality Requirements
- ✅ VLM accuracy ≥ deepdoc on complex tables (merged cells, nested headers)
- ✅ HTML output validates and renders correctly
- ✅ Caption extraction preserved
- ✅ Cell content accuracy ≥ 95%

---

## Future Enhancements

### Phase 2 Features (Post-MVP)
1. **Parallel table processing**: Process multiple tables concurrently
2. **VLM bbox extraction**: Add Option 1 (bbox coordinates) as secondary feature
3. **Adaptive routing**: Use deepdoc for simple tables, VLM for complex ones
4. **Caching**: Cache VLM results to reduce API costs
5. **Quality scoring**: Automatic quality assessment to choose best parser

### Integration Opportunities
1. **Figure parsing**: Extend VLM to figures (already has infrastructure)
2. **Mixed content**: Tables with embedded images/charts
3. **Multi-modal RAG**: Leverage table structure in retrieval

---

## Rollback Plan

If issues arise in production:

### Immediate Rollback
```bash
# Disable VLM table parsing
docker exec ragflow-server bash -c "export USE_VLM_TABLE_PARSING=false"
docker restart ragflow-server
```

### Code Rollback
```bash
# Revert to previous commit
git revert <commit-hash>
docker-compose build --no-cache
docker-compose up -d
```

### Data Integrity
- VLM table parsing doesn't modify existing data
- New documents can be re-processed with deepdoc if needed
- No database schema changes required

---

## Appendix

### A. File Change Summary

| File | Change Type | Lines Changed | Description |
|------|-------------|---------------|-------------|
| `deepdoc/parser/pdf_parser.py` | Modify | ~200 | Add VLM table parser, modify table transformer |
| `rag/prompts/table_vlm_prompt.md` | Create | ~50 | Table-specific VLM prompt |
| `rag/flow/parser/parser.py` | Modify | ~20 | Pass vision_model to parser |
| `docker-compose.yml` | Modify | ~10 | Add environment variables |
| `test_vlm_table_parsing.py` | Create | ~150 | Unit and integration tests |

### B. Configuration Examples

**Simple enable:**
```yaml
# docker-compose.yml
environment:
  - USE_VLM_TABLE_PARSING=true
```

**Advanced configuration:**
```yaml
environment:
  - USE_VLM_TABLE_PARSING=true
  - VLM_TABLE_MODEL=Qwen2.5VL-7B
  - VLM_TABLE_TIMEOUT_SEC=60
  - VLM_TABLE_FALLBACK_ENABLED=true
  - VLM_TABLE_OUTPUT_FORMAT=html
  - VLM_RESIZE_FACTOR=32
```

### C. Monitoring & Metrics

**Key metrics to track:**
- Table parsing success rate (VLM vs fallback)
- VLM API token usage
- Error rate by table complexity
- User satisfaction with table extraction quality

**Logging recommendations:**
```python
logging.info(f"VLM table parsing: {success_count}/{total_count} successful, "
            f"{fallback_count} fallbacks")
```

---

## Conclusion

This implementation plan provides a comprehensive, production-ready approach to integrating VLM-based table parsing into RAGFlow. The hybrid architecture leverages the strengths of both deepdoc (speed, accuracy for layout) and VLMs (semantic understanding, complex structures) while maintaining backward compatibility and providing multiple safety mechanisms.

**Key Advantages:**
- ✅ Non-breaking changes (opt-in via environment variables)
- ✅ Maintains existing schema and data flow
- ✅ Smart resize integration for Qwen3-VL compatibility
- ✅ Robust fallback mechanisms
- ✅ Comprehensive testing strategy
- ✅ Clear deployment and rollback procedures
