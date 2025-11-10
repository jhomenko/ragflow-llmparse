# VisionParser Image Resizing Implementation Plan

## Executive Summary

This document outlines the implementation plan for modifying the `VisionParser` class in `deepdoc/parser/pdf_parser.py` to ensure images are properly sized for VLM (Vision Language Model) processing according to specifications.

## Objectives

1. **High-Quality Input**: Extract PDF images at 600 DPI for better VLM processing
2. **VLM Compliance**: Ensure image dimensions are multiples of configurable factor (default: 32)
3. **Balanced Resolution**: Target 1024px max dimension for speed/accuracy balance
4. **Aspect Ratio Preservation**: Maintain original aspect ratios during resizing
5. **Configurability**: Allow factor adjustment via environment variable

## Current Implementation Analysis

### File: `deepdoc/parser/pdf_parser.py`

**Imports (lines 17-54):**
- ✅ `math` - Already imported (line 18)
- ✅ `os` - Already imported (line 19)
- ✅ `Image` from PIL - Already imported (line 35)
- ✅ `logging` - Already imported (line 17)
- ✅ `io` - Already imported (line 27)

**VisionParser Class (lines 1361-1569):**

**Current Issues:**
- Line 1370: Uses `72 * zoomin` resolution (typically 216 DPI) - **NEEDS UPDATE**
- Lines 1447-1453: Resizes to max 2000px without factor alignment - **NEEDS REPLACEMENT**

## Implementation Strategy

### Phase 1: Add Utility Functions (Before line 61)

Insert four utility functions immediately after the lock definition (after line 59) and before the `RAGFlowPdfParser` class definition:

```python
def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer >= 'number' divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer <= 'number' divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(height: int, width: int, factor: int = 32, target_max_dimension: int = 1024) -> tuple[int, int]:
    """
    Rescales image dimensions to meet these conditions:
    1. Both dimensions divisible by 'factor'
    2. Longest dimension <= 'target_max_dimension'
    3. Aspect ratio maintained
    
    Args:
        height: Original height in pixels
        width: Original width in pixels
        factor: Dimension divisibility factor (default: 32)
        target_max_dimension: Maximum allowed dimension (default: 1024)
    
    Returns:
        tuple[int, int]: (adjusted_height, adjusted_width)
    """
    max_dimension = max(height, width)
    scale = target_max_dimension / max_dimension
    new_height = int(height * scale)
    new_width = int(width * scale)
    
    adjusted_height = round_by_factor(new_height, factor)
    adjusted_width = round_by_factor(new_width, factor)
    
    if max(adjusted_height, adjusted_width) > target_max_dimension:
        final_scale = (target_max_dimension - factor) / max(new_height, new_width)
        adjusted_height = round_by_factor(int(height * final_scale), factor)
        adjusted_width = round_by_factor(int(width * final_scale), factor)
    
    adjusted_height = max(adjusted_height, factor)
    adjusted_width = max(adjusted_width, factor)
    
    return adjusted_height, adjusted_width
```

### Phase 2: Modify VisionParser.__images__ Method

**Location:** Line 1370

**Current Code:**
```python
self.page_images = [p.to_image(resolution=72 * zoomin).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]
```

**New Code:**
```python
# Extract images at high DPI (600) to ensure high quality for VLM processing
high_dpi = 600
self.page_images = [p.to_image(resolution=high_dpi).annotated for i, p in enumerate(self.pdf.pages[page_from:page_to])]
```

**Rationale:**
- Fixed 600 DPI provides consistent high-quality input
- Independent of `zoomin` parameter for VLM processing
- Ensures sufficient detail for text and image recognition

### Phase 3: Update VisionParser.__call__ Image Processing

**Location:** Lines 1443-1453

**Current Code:**
```python
# Convert to RGB
img = img_pil.convert("RGB")

# Resize if longest side > 2000px (maintain aspect ratio)
max_side = max(img.size)
if max_side > 2000:
    scale = 2000.0 / max_side
    new_size = (int(img.size[0] * scale), int(img.size[1] * scale))
    img = img.resize(new_size, resample=Image.Resampling.LANCZOS)
    logging.debug(f"VisionParser: Page {idx+1}: Resized to {img.size[0]}x{img.size[1]}")
```

**New Code:**
```python
# Convert to RGB
img = img_pil.convert("RGB")

# Get resize factor from environment variable or default to 32
resize_factor = int(os.getenv("VLM_RESIZE_FACTOR", "32"))

# Apply smart_resize to ensure dimensions are multiples of the factor
# and maintain proper aspect ratio with max 1024 on the long dimension
width, height = img.size
target_height, target_width = smart_resize(
    height, width, 
    factor=resize_factor,
    target_max_dimension=1024  # Balanced resolution for general use
)

# Resize the image to the calculated dimensions
img = img.resize((target_width, target_height), resample=Image.Resampling.LANCZOS)
logging.debug(f"VisionParser: Page {idx+1}: Resized from {width}x{height} to {target_width}x{target_height} (factor={resize_factor})")
```

## Implementation Subtasks

### Subtask 1: Add Utility Functions
- **File:** `deepdoc/parser/pdf_parser.py`
- **Insert Location:** After line 59 (after LOCK_KEY definition)
- **Lines to Add:** ~60 lines (4 functions with docstrings)
- **Dependencies:** Uses `math` module (already imported)

### Subtask 2: Modify High DPI Extraction
- **File:** `deepdoc/parser/pdf_parser.py`
- **Modification Location:** Line 1370
- **Change Type:** Replace single line + add comment
- **Dependencies:** None (uses existing pdfplumber API)

### Subtask 3: Update Resize Logic
- **File:** `deepdoc/parser/pdf_parser.py`
- **Modification Location:** Lines 1447-1453 (replace 7 lines with ~17 lines)
- **Change Type:** Replace conditional resize logic
- **Dependencies:** `smart_resize()` function, `os` module

### Subtask 4: Environment Variable Support
- **Variable Name:** `VLM_RESIZE_FACTOR`
- **Default Value:** `"32"` (string, converted to int)
- **Valid Values:** Any positive integer (typically 16, 32, 64)
- **Usage:** `int(os.getenv("VLM_RESIZE_FACTOR", "32"))`

## Testing Strategy

### Unit Tests
1. **Utility Function Tests:**
   - Test `round_by_factor()` with various inputs
   - Test `ceil_by_factor()` edge cases
   - Test `floor_by_factor()` edge cases
   - Test `smart_resize()` with different aspect ratios

2. **Integration Tests:**
   - Test with various PDF files (different page sizes)
   - Verify dimensions are multiples of configured factor
   - Confirm max dimension doesn't exceed 1024px
   - Validate aspect ratio preservation

### Validation Checklist
- [ ] Images extracted at 600 DPI
- [ ] All output dimensions are multiples of 32 (default)
- [ ] Longest dimension ≤ 1024 pixels
- [ ] Aspect ratios preserved within rounding tolerance
- [ ] Environment variable `VLM_RESIZE_FACTOR` works correctly
- [ ] VLM processing still works with new image sizes
- [ ] Logging messages show correct before/after dimensions
- [ ] Performance is acceptable (not significantly slower)

## Benefits

1. **VLM Compliance**: Qwen2-VL and similar models require dimensions divisible by specific factors
2. **High Quality**: 600 DPI extraction ensures detailed input
3. **Balanced Performance**: 1024px max keeps processing fast while maintaining accuracy
4. **Flexibility**: Environment variable allows easy adjustment per deployment
5. **Aspect Ratio**: Proper aspect ratio maintenance prevents distortion
6. **Minimal Impact**: Changes are surgical and localized to VisionParser

## Rollback Plan

If issues arise:
1. Revert line 1370 to original: `resolution=72 * zoomin`
2. Revert lines 1447-1453 to original resize logic
3. Remove utility functions (lines 61-120 approximately)

## Configuration Examples

```bash
# Default (Qwen2-VL compatible)
# No environment variable needed, defaults to 32

# For models requiring 16-pixel alignment
export VLM_RESIZE_FACTOR=16

# For models requiring 64-pixel alignment
export VLM_RESIZE_FACTOR=64

# For high-precision models
export VLM_RESIZE_FACTOR=8
```

## Performance Considerations

**Expected Impact:**
- **DPI Change (72*3=216 → 600)**: ~8x more pixels initially, but resized down
- **Final Resolution (2000 → 1024)**: ~4x fewer pixels sent to VLM
- **Net Effect**: Slightly faster VLM processing, similar extraction time
- **Memory**: Temporarily higher during extraction, lower during VLM call

## Compatibility Notes

- **Backward Compatible**: Existing code using RAGFlowPdfParser unaffected
- **API Stable**: No changes to VisionParser public interface
- **Dependencies**: No new package dependencies required
- **Python Version**: Works with Python 3.7+ (type hints compatible)

## Code Review Checklist

- [ ] All imports present and correct
- [ ] Type hints are accurate
- [ ] Docstrings are clear and complete
- [ ] Logging statements are informative
- [ ] Error handling is appropriate
- [ ] Code follows existing style conventions
- [ ] No hardcoded values (except reasonable defaults)
- [ ] Environment variable handling is safe

## Deployment Notes

**Pre-deployment:**
1. Review and test on sample PDFs
2. Verify VLM model compatibility with new dimensions
3. Check memory usage with large PDFs

**Deployment:**
1. Deploy code changes
2. Set `VLM_RESIZE_FACTOR` if non-default needed
3. Monitor logs for resize operations
4. Validate VLM output quality

**Post-deployment:**
1. Monitor performance metrics
2. Check for any dimension-related errors
3. Validate VLM output remains high quality

## Timeline Estimate

- **Subtask 1** (Utility Functions): 15 minutes
- **Subtask 2** (High DPI): 5 minutes  
- **Subtask 3** (Resize Logic): 10 minutes
- **Subtask 4** (Testing): 30 minutes
- **Total**: ~60 minutes implementation + testing

## Success Criteria

✅ All images have dimensions divisible by configured factor  
✅ Longest dimension ≤ 1024 pixels  
✅ Aspect ratios preserved  
✅ VLM processing produces quality output  
✅ No performance degradation  
✅ Environment variable works correctly  
✅ All existing tests pass  
✅ New validation tests pass  

## Next Steps

1. Review this plan with stakeholders
2. Obtain approval to proceed
3. Switch to Code mode for implementation
4. Execute subtasks sequentially
5. Run validation tests
6. Document changes in changelog