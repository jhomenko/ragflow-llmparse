import os
import io
from PIL import Image
import pytest

# Test 1: smart_resize factor alignment
def test_smart_resize_factor_alignment():
    """Verify images are resized to multiples of VLM_RESIZE_FACTOR"""
    from deepdoc.parser.pdf_parser import smart_resize

    test_cases = [
        (800, 600, 32, 1024),   # Normal case
        (2000, 1500, 32, 1024), # Downscale needed
        (400, 300, 32, 1024),   # Upscale case
    ]

    for h, w, factor, max_dim in test_cases:
        new_h, new_w = smart_resize(h, w, factor, max_dim)
        assert new_h % factor == 0, f"Height {new_h} not multiple of {factor}"
        assert new_w % factor == 0, f"Width {new_w} not multiple of {factor}"
        assert max(new_h, new_w) <= max_dim, f"Dimension exceeds {max_dim}"


# Test 2: prompt loading
def test_vlm_table_prompt_loading():
    """Verify table prompt loads correctly"""
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser

    parser = RAGFlowPdfParser()
    prompt = parser._get_table_prompt("html")

    assert "table" in prompt.lower()
    assert "html" in prompt.lower() or "<table>" in prompt.lower()


# Test 3: HTML validation
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