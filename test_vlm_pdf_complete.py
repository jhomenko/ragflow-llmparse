#!/usr/bin/env python3
"""
Complete VLM PDF test using RAGFlow's actual conversion pipeline.
Tests both PDF conversion and stop token configurations.
"""
import sys
import io
import base64
from pathlib import Path
from PIL import Image
from typing import TYPE_CHECKING, Any
import pdfplumber

if TYPE_CHECKING:
    # Help type-checkers without importing at runtime
    import pdfplumber as pdfplumber_t  # type: ignore
    from openai import OpenAI as OpenAI_t  # type: ignore

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# Configuration
VLM_SERVER = "http://192.168.68.186:8080/v1"
MODEL_NAME = "Qwen2.5VL-3B"
TEST_PDF = "test.pdf"  # User specifies PDF
ZOOMIN = 3  # Same as RAGFlow default

def pdf_to_image_ragflow_style(pdf_path, page_num=0, zoomin=3):
    """
    Convert PDF page to image using RAGFlow's production method (pdfplumber).
    This mirrors deepdoc/parser/pdf_parser.py::__images__ behavior.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            if page_num >= total_pages:
                print(f"❌ Error: PDF only has {total_pages} pages, requested page {page_num}")
                sys.exit(1)

            page = pdf.pages[page_num]
            # Use the same rasterization used in RAGFlow: resolution = 72 * zoomin
            # Use annotated to preserve any annotations/alpha similar to production pipeline
            img_obj = page.to_image(resolution=72 * zoomin, antialias=True).annotated

            # pdfplumber returns a PIL Image for .annotated/.original
            if isinstance(img_obj, Image.Image):
                img = img_obj
            else:
                # fallback: try to get .original or convert bytes
                try:
                    img = getattr(img_obj, "original", None) or Image.frombytes("RGB", (img_obj.width, img_obj.height), img_obj.samples)
                except Exception:
                    img = Image.frombytes("RGB", (img_obj.width, img_obj.height), img_obj.samples)

            print(f"✅ Converted PDF page {page_num} to image (pdfplumber)")
            try:
                # Attempt to log original vector page size if available
                rect = page.rect
                print(f"   Original size: {rect.width:.1f}x{rect.height:.1f}")
            except Exception:
                pass
            print(f"   Zoomed size: {img.size[0]}x{img.size[1]} (zoom={zoomin})")

            return img

    except Exception as e:
        print(f"❌ Error converting PDF with pdfplumber: {e}")
        sys.exit(1)

def pil_image_to_jpeg_bytes(pil_img):
    """
    Convert PIL Image to JPEG bytes.
    This matches VisionParser's conversion in Bug Fix #1.
    """
    img_rgb = pil_img.convert("RGB")

    # Optional: Resize if too large (prevent token overflow)
    max_side = 2000
    w, h = img_rgb.size
    if max(w, h) > max_side:
        scale = max_side / float(max(w, h))
        new_size = (int(w * scale), int(h * scale))
        img_rgb = img_rgb.resize(new_size, Image.Resampling.LANCZOS)
        print(f"   Resized to {new_size[0]}x{new_size[1]} (max_side={max_side})")

    # Convert to JPEG bytes
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=90, optimize=True)
    jpg_bytes = buf.getvalue()

    print(f"   JPEG size: {len(jpg_bytes):,} bytes")

    return jpg_bytes

def jpeg_bytes_to_base64(jpg_bytes):
    """Convert JPEG bytes to base64 for API transmission."""
    return base64.b64encode(jpg_bytes).decode("utf-8")

def create_messages(img_b64, prompt="Transcribe this PDF page to clean Markdown."):
    """Create OpenAI API messages with system message (matches Bug Fix #2)."""
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
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }
    ]

def test_vlm_call(client, messages, stop_config, test_name):
    """Run VLM test with specific stop token configuration."""
    print("\n" + "=" * 80)
    print(f"TEST: {test_name}")
    print("=" * 80)

    params = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": 4096,      # Bug Fix #3
        "temperature": 0.1,      # Bug Fix #3
    }

    # Apply stop token config
    if stop_config is not None:
        params["stop"] = stop_config
        print(f"Stop tokens: {stop_config}")
    else:
        print("Stop tokens: (not specified)")

    try:
        res = client.chat.completions.create(**params)

        content = res.choices[0].message.content
        tokens = getattr(res.usage, "total_tokens", None) or res.usage.get("total_tokens", 0)
        finish_reason = res.choices[0].finish_reason

        print(f"\n✅ SUCCESS")
        print(f"Tokens: {tokens}")
        print(f"Finish reason: {finish_reason}")
        print(f"Content length: {len(content)} characters")
        print(f"\nFirst 300 characters:")
        print("-" * 80)
        print(content[:300])
        print("-" * 80)

        # Verdict
        is_good = (tokens is not None and tokens > 500) and len(content) > 1000
        status = "✅ GOOD" if is_good else "⚠️  SHORT"
        print(f"\n{status}: {tokens} tokens, {len(content)} characters")
 
        # Save output to file for later comparison
        try:
            output_filename = f"vlm_output_{test_name.replace(' ', '_').replace('/', '_')}.md"
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(f"# VLM Test Output\n\n")
                f.write(f"**Test:** {test_name}\n")
                f.write(f"**Stop Config:** {stop_config}\n")
                f.write(f"**Tokens:** {tokens}\n")
                f.write(f"**Finish Reason:** {finish_reason}\n")
                f.write(f"**Length:** {len(content)} characters\n\n")
                f.write(f"---\n\n")
                f.write(content)
            print(f"✅ Saved output to: {output_filename}")
        except Exception as _e:
            print(f"⚠️  Could not save output file: {_e}")
 
        return {
            "success": True,
            "tokens": tokens,
            "finish_reason": finish_reason,
            "length": len(content),
            "content": content,
            "is_good": is_good
        }

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}

def main():
    """Main test execution."""
    print("\n" + "=" * 80)
    print("COMPLETE VLM PDF PARSING TEST")
    print("Using RAGFlow's Actual Conversion Pipeline")
    print("=" * 80)

    # Parse arguments
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else TEST_PDF
    page_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    # Check PDF exists (validate the actual selected file)
    if not Path(pdf_path).exists():
        print(f"\n❌ Error: Test PDF not found: {pdf_path}")
        print("Please provide a test PDF file.")
        print(f"Usage: python3 {sys.argv[0]} [pdf_file] [page_num]")
        sys.exit(1)

    print(f"\nTest PDF: {pdf_path}")
    print(f"Test page: {page_num}")
    print(f"VLM Server: {VLM_SERVER}")
    print(f"Model: {MODEL_NAME}")

    # Step 1: Convert PDF to Image (RAGFlow way)
    print("\n" + "=" * 80)
    print("STEP 1: PDF → Image Conversion (RAGFlow Pipeline)")
    print("=" * 80)
    pil_img = pdf_to_image_ragflow_style(pdf_path, page_num, ZOOMIN)

    # Step 2: Convert Image to JPEG bytes (RAGFlow way)
    print("\n" + "=" * 80)
    print("STEP 2: Image → JPEG Bytes Conversion")
    print("=" * 80)
    jpg_bytes = pil_image_to_jpeg_bytes(pil_img)

    # Step 3: Encode to base64
    print("\n" + "=" * 80)
    print("STEP 3: JPEG Bytes → Base64 Encoding")
    print("=" * 80)
    img_b64 = jpeg_bytes_to_base64(jpg_bytes)
    print(f"✅ Base64 length: {len(img_b64):,} characters")

    # Step 4: Create messages with system prompt
    messages = create_messages(img_b64)
    print(f"✅ Messages created (system + user with image)")

    # Step 5: Create OpenAI client
    client = OpenAI(api_key="not-needed", base_url=VLM_SERVER)

    # Step 6: Run tests
    print("\n" + "=" * 80)
    print("STEP 4: VLM API Tests with Different Stop Token Configs")
    print("=" * 80)

    results = {}

    # Test 1: stop=[] (RAGFlow Fix #4)
    results["stop_empty"] = test_vlm_call(
        client, messages, [],
        "stop=[] (RAGFlow Fix #4 - Disable defaults)"
    )

    # Test 2: No stop parameter
    results["stop_none"] = test_vlm_call(
        client, messages, None,
        "No stop parameter (Client defaults)"
    )

    # Test 3: OpenAI common stop tokens
    results["stop_openai"] = test_vlm_call(
        client, messages, ["<|im_end|>", "\n\n"],
        "OpenAI stop tokens (Should fail)"
    )

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    for name, result in results.items():
        if result.get("success"):
            status = "✅ GOOD" if result.get("is_good") else "⚠️  SHORT"
            tokens = result["tokens"]
            length = result["length"]
            print(f"{status} | {name:15s} | {tokens:5d} tokens | {length:6d} chars")
        else:
            print(f"❌ FAIL | {name:15s} | {result.get('error', 'Unknown error')}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    stop_empty = results.get("stop_empty", {})
    stop_none = results.get("stop_none", {})
    stop_openai = results.get("stop_openai", {})

    # Check if PDF conversion is working
    if all(r.get("success") for r in results.values()):
        print("✅ PDF → Image → JPEG → Base64 pipeline: WORKING")
        print("   All conversions completed successfully")

    # Check stop token impact
    empty_good = stop_empty.get("is_good", False)
    none_good = stop_none.get("is_good", False)
    openai_good = stop_openai.get("is_good", False)

    if empty_good and none_good and not openai_good:
        print("\n✅ CONFIRMED: OpenAI stop tokens cause truncation")
        print(f"   - stop=[]: {stop_empty.get('length', 0)} chars")
        print(f"   - no stop: {stop_none.get('length', 0)} chars")
        print(f"   - OpenAI stops: {stop_openai.get('length', 0)} chars")
        print("\n   → RAGFlow SHOULD use stop=[] (already implemented in Bug Fix #4)")

    elif empty_good and not none_good:
        print("\n⚠️  WARNING: Client adds default stop tokens")
        print("   → stop=[] is REQUIRED for full responses")

    elif not empty_good:
        print("\n❌ PROBLEM: Even stop=[] produces short responses")
        print("   Possible issues:")
        print("   1. PDF conversion losing content")
        print("   2. Image quality too low")
        print("   3. Prompt not clear enough")
        print("   4. Model configuration issue")

    # DETAILED COMPARISON
    print("\n" + "=" * 80)
    print("DETAILED COMPARISON")
    print("=" * 80)
 
    stop_empty_len = stop_empty.get("length", 0)
    stop_none_len = stop_none.get("length", 0)
 
    if stop_empty_len != stop_none_len:
        diff = abs(stop_none_len - stop_empty_len)
        longer = "no stop" if stop_none_len > stop_empty_len else "stop=[]"
        print(f"\n⚠️  OUTPUT DIFFERS: {diff} character difference")
        print(f"   - stop=[]: {stop_empty_len} chars")
        print(f"   - no stop: {stop_none_len} chars")
        print(f"   - Longer output: {longer}")
        print(f"\n   This suggests the model naturally stopped at different points.")
        print(f"   Check saved files to see if one is more complete.")
    else:
        print(f"\n✅ IDENTICAL OUTPUT: Both produce {stop_empty_len} chars")
        print(f"   Either option is safe to use.")
 
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)
 
    if stop_none_len > stop_empty_len and stop_none.get("is_good"):
        print("✅ USE: No stop parameter (produces longer output)")
        print("   Your OpenAI client doesn't add problematic defaults.")
        print("   The model generates more content naturally.")
    elif stop_empty_len >= stop_none_len * 0.95:  # Within 5%
        print("✅ USE: stop=[] (safer for production)")
        print("   Output length similar, but explicit behavior.")
        print("   Protects against future client updates.")
    else:
        print("⚠️  INVESTIGATE: Significant difference in outputs")
        print("   Review saved files to determine which is more complete.")
 
    # Comparison with direct curl test
    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("1. Compare these results with your working curl test")
    print("2. If results match curl → PDF pipeline is good ✅")
    print("3. If results differ → Check image conversion quality")
    print("4. Rebuild Docker with --no-cache to apply all fixes")
    print("5. Test through RAGFlow UI with same PDF")

if __name__ == "__main__":
    main()