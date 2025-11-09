#!/usr/bin/env python3
import sys
import os
import traceback
from pathlib import Path

# allow imports as if running from repo root
sys.path.insert(0, '/ragflow')

# Set environment variables required for integration test
os.environ["USE_WORKING_VLM"] = "true"
os.environ.setdefault("VLM_BASE_URL", "http://192.168.68.186:8080/v1")

try:
    import pdfplumber
except Exception as e:
    print(f"✗ Missing dependency: pdfplumber - {e}")
    sys.exit(1)

try:
    from api.db import LLMType
    from api.db.services.llm_service import LLMBundle
    from deepdoc.parser.pdf_parser import VisionParser
except Exception:
    print("✗ Failed to import VisionParser or LLMBundle")
    traceback.print_exc()
    sys.exit(1)

TEST_PDF_PATH = "test.pdf"
MODEL_NAME = "Qwen2.5VL-3B"
PROMPT = "Please describe the page in detail, focusing on layout and visible objects and notable text."
TENANT_ID = "test_tenant"


# ensure_test_pdf removed; tests must provide a real 'test.pdf' file in the repository.
# PDF creation via PyMuPDF was removed in favor of using RAGFlow's actual test PDF.


def read_pdf_bytes(path: str) -> bytes:
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        print(f"✗ Failed to read PDF file: {path}")
        traceback.print_exc()
        sys.exit(1)


def main():
    try:
        # Require a real test PDF to be present (matches test_vlm_pdf_complete.py behavior).
        if not Path(TEST_PDF_PATH).exists():
            print(f"✗ Error: Test PDF not found: {TEST_PDF_PATH}")
            sys.exit(1)
        pdf_bytes = read_pdf_bytes(TEST_PDF_PATH)
    except Exception:
        print("✗ Pre-test setup failed")
        sys.exit(1)

    try:
        vision_model = LLMBundle(TENANT_ID, LLMType.IMAGE2TEXT, llm_name=MODEL_NAME, lang="English")
        print(f"✓ Created LLMBundle: {getattr(vision_model, 'llm_name', MODEL_NAME)}")
    except Exception:
        print("✗ Failed to create LLMBundle")
        traceback.print_exc()
        sys.exit(1)

    try:
        parser = VisionParser(vision_model=vision_model)
    except Exception:
        print("✗ Failed to instantiate VisionParser")
        traceback.print_exc()
        sys.exit(1)

    try:
        # parse first 2 pages (0..2)
        pages, _ = parser(pdf_bytes, from_page=0, to_page=2, prompt_text=PROMPT)
        print(f"✓ VisionParser returned {len(pages)} page(s)")
    except Exception:
        print("✗ VisionParser raised an exception")
        traceback.print_exc()
        sys.exit(1)

    if not pages:
        print("✗ No pages returned by VisionParser")
        sys.exit(1)

    success = True
    for idx, (text, metadata) in enumerate(pages):
        try:
            txt = (text or "").strip()
            char_count = len(txt)
            token_count = len(txt.split())
            print(f"\nPage {idx+1}: chars={char_count}, tokens≈{token_count}, metadata={metadata}")
            preview = txt[:400].replace("\n", " ")
            print(f"Preview: {preview}\n")
            if char_count <= 100:
                print(f"✗ FAIL: Page {idx+1} response too short ({char_count} chars) - expected >100")
                success = False
            else:
                print(f"✓ Page {idx+1} PASS")
        except Exception:
            print(f"✗ Error validating page {idx+1}")
            traceback.print_exc()
            success = False

    if success:
        print("\n✓ test_vision_parser_integration: PASS")
        sys.exit(0)
    else:
        print("\n✗ test_vision_parser_integration: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()