#!/usr/bin/env python3
import sys
import os
import traceback

# allow imports as if running from repo root
sys.path.insert(0, '/ragflow')

try:
    import fitz  # PyMuPDF
except Exception as e:
    print(f"✗ Missing dependency: PyMuPDF (fitz) - {e}")
    sys.exit(1)

try:
    # Try to import the direct function from the working VLM module.
    from rag.llm.working_vlm_module import describe_image_working
except Exception as e:
    print("✗ Failed to import describe_image_working from rag.llm.working_vlm_module")
    traceback.print_exc()
    sys.exit(1)


TEST_PDF_PATH = "test_data/sample.pdf"
MODEL_NAME = "Qwen2.5VL-3B"
PROMPT = "Please describe the image in detail, focusing on visible objects, layout, and notable textual content."

def ensure_test_pdf(path: str):
    """Create a minimal single-page PDF if none exists using PyMuPDF."""
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        doc = fitz.open()
        doc.new_page(width=595, height=842)  # A4-like single page
        doc.save(path)
        doc.close()
        print(f"✓ Created minimal test PDF at {path}")
    except Exception:
        print("✗ Failed to create minimal test PDF")
        traceback.print_exc()
        sys.exit(1)


def pdf_first_page_to_png_bytes(pdf_path: str) -> bytes:
    """Load the first page of a PDF and return PNG bytes."""
    try:
        doc = fitz.open(pdf_path)
        if doc.page_count < 1:
            raise RuntimeError("PDF has no pages")
        page = doc.load_page(0)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # increase resolution
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes
    except Exception:
        print("✗ Error converting PDF first page to image bytes")
        traceback.print_exc()
        raise


def main():
    try:
        ensure_test_pdf(TEST_PDF_PATH)
        img_bytes = pdf_first_page_to_png_bytes(TEST_PDF_PATH)
    except Exception:
        print("✗ Pre-test setup failed")
        sys.exit(1)

    try:
        # Call the working VLM module function directly
        response = describe_image_working(img_bytes, PROMPT, MODEL_NAME)
    except Exception:
        print("✗ describe_image_working raised an exception")
        traceback.print_exc()
        sys.exit(1)

    # Basic validation of the response
    try:
        if not isinstance(response, str):
            print("✗ Response is not a string")
            sys.exit(1)

        resp_len = len(response.strip())
        token_count = len(response.split())

        print("\n--- Response preview ---")
        preview = response.strip()[:400].replace("\n", " ")
        print(preview)
        print("--- End preview ---\n")

        print(f"Token count (approx): {token_count}")
        print(f"Character count: {resp_len}")

        assert resp_len > 100, "Response is too short; expected >100 characters"

        print("✓ test_working_vlm_module: PASS")
        sys.exit(0)
    except AssertionError as ae:
        print(f"✗ Assertion failed: {ae}")
        sys.exit(1)
    except Exception:
        print("✗ Unexpected error while validating response")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()