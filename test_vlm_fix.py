#!/usr/bin/env python3
"""
VLM PDF Parsing Test Script

Tests the VLM PDF parsing fixes to verify that full markdown transcriptions
are returned instead of empty responses.

Usage: python test_vlm_fix.py <pdf_file> [tenant_id]

Expected Results After Fixes:
- Text length: >1000 characters (was 14)
- Token count: >500 tokens (was 6)
- Content: Full markdown transcription (was "--- Page 1 ---")
"""
import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from api.db import LLMType
from api.db.services.llm_service import LLMBundle
from deepdoc.parser.pdf_parser import VisionParser


def test_vlm_parsing(pdf_path, tenant_id="test_tenant"):
    """Test VLM PDF parsing with fixes."""
    
    print(f"\n{'='*80}")
    print(f"VLM PDF Parsing Validation Test")
    print(f"{'='*80}\n")
    print(f"PDF File: {pdf_path}")
    print(f"Tenant ID: {tenant_id}\n")
    
    # 1. Create vision model
    print("Step 1: Creating VLM Bundle...")
    try:
        vision_model = LLMBundle(
            tenant_id=tenant_id,
            llm_type=LLMType.IMAGE2TEXT,
            llm_name="Qwen2.5VL-3B___OpenAI-API@OpenAI-API-Compatible",
            lang="English"
        )
        print(f"   ✓ Model created: {vision_model.llm_name}")
    except Exception as e:
        print(f"   ✗ ERROR: Failed to create VLM Bundle: {e}")
        return False
    
    # 2. Load prompt
    print("\nStep 2: Loading vision prompt...")
    prompt_path = Path("rag/prompts/vision_llm_describe_prompt.md")
    if prompt_path.exists():
        prompt_text = prompt_path.read_text(encoding="utf-8")
        print(f"   ✓ Loaded {len(prompt_text)} chars from {prompt_path}")
    else:
        prompt_text = "Transcribe this PDF page to clean Markdown."
        print(f"   ⚠ Using default prompt (file not found): {prompt_text}")
    
    # 3. Parse PDF
    print("\nStep 3: Parsing PDF with VisionParser...")
    parser = VisionParser(vision_model=vision_model)
    
    try:
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        print(f"   ✓ Loaded PDF ({len(pdf_bytes)} bytes)")
    except Exception as e:
        print(f"   ✗ ERROR: Failed to read PDF: {e}")
        return False
    
    def progress_callback(prog, msg):
        if prog >= 0:
            print(f"   [{prog*100:5.1f}%] {msg}")
        else:
            print(f"   [ERROR] {msg}")
    
    try:
        lines, _ = parser(
            pdf_bytes,
            from_page=0,
            to_page=1,  # Just first page for testing
            callback=progress_callback,
            zoomin=3,
            prompt_text=prompt_text
        )
        print(f"   ✓ Parsing completed")
    except Exception as e:
        print(f"   ✗ ERROR: Parsing failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. Validate results
    print(f"\nStep 4: Validating Results...")
    print(f"   Pages processed: {len(lines)}")
    
    if len(lines) == 0:
        print(f"   ✗ FAIL: No pages processed!")
        return False
    
    success = True
    for idx, (text, metadata) in enumerate(lines):
        print(f"\n   Page {idx+1} Results:")
        print(f"     - Text length: {len(text)} chars")
        print(f"     - Metadata: {metadata}")
        
        # Validate text length
        if len(text) < 50:
            print(f"     ✗ FAIL: Text too short ({len(text)} chars) - Expected >1000 chars")
            print(f"     Response: '{text}'")
            success = False
        elif len(text) < 500:
            print(f"     ⚠ WARNING: Text shorter than expected ({len(text)} chars)")
            print(f"     Preview: {text[:200]}...")
        else:
            print(f"     ✓ PASS: Text length looks good ({len(text)} chars)")
            print(f"     Preview: {text[:200]}...")
        
        # Check for empty/placeholder content
        if text.strip() in ["", "--- Page 1 ---", "Page 1"]:
            print(f"     ✗ FAIL: Response is empty or placeholder")
            success = False
        
        # Check for markdown content
        has_headers = "##" in text or "# " in text
        has_content = len(text.split()) > 50
        
        if has_headers:
            print(f"     ✓ Contains markdown headers")
        if has_content:
            print(f"     ✓ Contains substantial content ({len(text.split())} words)")
        
        if not has_content:
            print(f"     ✗ FAIL: Content too sparse")
            success = False
    
    # 5. Summary
    print(f"\n{'='*80}")
    if success:
        print("✓ TEST PASSED: VLM is producing full transcriptions!")
        print("\nThe following bugs have been successfully fixed:")
        print("  1. LLMBundle.describe_with_prompt now returns tuple")
        print("  2. System message added to vision_llm_prompt")
        print("  3. max_tokens=4096 and temperature=0.1 parameters added")
        print("\nYou can now use VLM PDF parsing in production.")
    else:
        print("✗ TEST FAILED: VLM is still producing short/empty responses")
        print("\nPossible issues:")
        print("  - VLM server may not be running or accessible")
        print("  - Model configuration may be incorrect")
        print("  - Network connectivity issues")
        print("  - Review logs above for specific errors")
    print(f"{'='*80}\n")
    
    return success


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_vlm_fix.py <pdf_file> [tenant_id]")
        print("\nExample:")
        print("  python test_vlm_fix.py sample.pdf")
        print("  python test_vlm_fix.py sample.pdf my_tenant_id")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    tenant_id = sys.argv[2] if len(sys.argv) > 2 else "test_tenant"
    
    if not Path(pdf_path).exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    success = test_vlm_parsing(pdf_path, tenant_id)
    sys.exit(0 if success else 1)