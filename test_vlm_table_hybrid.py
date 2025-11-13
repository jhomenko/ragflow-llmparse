#!/usr/bin/env python3
"""
Diagnostic script: test_vlm_table_hybrid.py

- Runs a PDF through deepdoc/RAGFlow pipeline with explicit VLM table parsing enabled.
- Verifies key environment variables.
- Enables verbose logging to terminal and test_output_table_hybrid.txt.
- Confirms whether hybrid VLM table parsing was activated and reports table outputs.

Usage:
    python3 test_vlm_table_hybrid.py /path/to/test.pdf

The script will print progress to stdout and also write a detailed log + summary
to ./test_output_table_hybrid.txt
"""
import os
import sys
import logging
from pathlib import Path
import traceback
import time

# Configure verbose logging to both console and file
LOGFILE = "test_output_table_hybrid.txt"
logger = logging.getLogger("test_vlm_table_hybrid")
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(fmt)
logger.addHandler(ch)

fh = logging.FileHandler(LOGFILE, mode="w", encoding="utf-8")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)

logger.info("Starting VLM table hybrid diagnostic")

# Print and record important environment variables
env_vars = [
    "USE_VLM_TABLE_PARSING",
    "VLM_TABLE_MODEL",
    "VLM_TABLE_OUTPUT_FORMAT",
    "VLM_TABLE_FALLBACK_ENABLED",
    "VLM_RESIZE_FACTOR",
    "USE_WORKING_VLM",
]
logger.info("Environment variables relevant to VLM table parsing:")
for k in env_vars:
    logger.info("  %s = %s", k, os.getenv(k, "<unset>"))

# Basic arg parsing
if len(sys.argv) < 2:
    logger.error("Usage: python3 test_vlm_table_hybrid.py /path/to/test.pdf [page_from] [page_to]")
    sys.exit(2)

pdf_path = Path(sys.argv[1])
if not pdf_path.exists():
    logger.error("PDF file not found: %s", pdf_path)
    sys.exit(3)

page_from = int(sys.argv[2]) if len(sys.argv) > 2 else 0
page_to = int(sys.argv[3]) if len(sys.argv) > 3 else 100000

# Import parser and create a minimal vision_model placeholder
try:
    from deepdoc.parser.pdf_parser import RAGFlowPdfParser
    from api.db import LLMType
    from api.db.services.llm_service import LLMBundle
except Exception:
    logger.exception("Failed to import RAGFlowPdfParser or LLMBundle")
    sys.exit(4)

# Create a real LLMBundle instance using tenant_id and VLM_TABLE_MODEL
tenant_id = os.getenv("TENANT_ID", "test_tenant")  # Use test tenant if not provided
vlm_model_name = os.getenv("VLM_TABLE_MODEL", "Qwen3VL-4B")

try:
    vision_model = LLMBundle(tenant_id, LLMType.IMAGE2TEXT, llm_name=vlm_model_name)
    logger.info("Created LLMBundle with tenant_id: %s, model: %s", tenant_id, vlm_model_name)
    
    # Validate that the created vision model has required methods
    if not hasattr(vision_model, 'describe') or not hasattr(vision_model, 'describe_with_prompt'):
        logger.error("LLMBundle does not have required methods (describe, describe_with_prompt)")
        sys.exit(6)
    else:
        logger.info("LLMBundle has required methods (describe, describe_with_prompt)")
        
except Exception as e:
    logger.error("Failed to create LLMBundle: %s", str(e))
    sys.exit(5)

logger.info("LLMBundle created successfully: %s", vision_model)

# Instantiate parser with vision_model attached via constructor (preferred)
try:
    parser = RAGFlowPdfParser(vision_model=vision_model)
    logger.info("RAGFlowPdfParser instantiated with vision_model attribute: %s", hasattr(parser, "vision_model") and getattr(parser, "vision_model"))
except TypeError:
    # For older constructors that don't accept vision_model, create then attach
    logger.warning("Constructor did not accept vision_model param, creating and attaching attribute manually")
    parser = RAGFlowPdfParser()
    parser.vision_model = vision_model
    logger.info("Attached vision_model to parser instance")

# Progress callback used by parser
def progress_cb(prog=None, msg=""):
    try:
        if prog is None:
            logger.info("progress: %s", msg)
        else:
            logger.info("progress: %.3f - %s", prog if isinstance(prog, float) else float(prog), msg)
    except Exception:
        logger.debug("progress_cb: unable to log progress")

# Run full parsing pathway (will exercise _table_transformer_job)
start_ts = time.time()
try:
    logger.info("Calling parser on %s (pages %s..%s)", pdf_path, page_from, page_to)
    # return_html True so tables (if VLM-provided) come back as HTML strings
    try:
        # Use VisionParser.__call__ method which supports from_page and to_page
        bboxes, tables = parser(str(pdf_path), from_page=page_from, to_page=page_to, callback=progress_cb)
    except TypeError:
        # If the call fails, try the RAGFlowPdfParser.__call__ method
        try:
            bboxes, tables = parser(str(pdf_path), need_image=True, zoomin=3, return_html=True)
        except TypeError:
            # Some __call__ signatures accept (fnm, need_image, zoomin, return_html)
            bboxes, tables = parser(str(pdf_path), True, 3, True)
    duration = time.time() - start_ts
    logger.info("Parsing complete (%.2fs). Boxes: %d, Tables extracted: %d", duration, len(bboxes) if bboxes else 0, len(tables) if tables else 0)
except Exception:
    logger.exception("Parser run failed")
    sys.exit(5)

# Inspect tables for VLM indicators
logger.info("Inspecting extracted tables for VLM/HTML content and fallback cases")
vlm_like = 0
fallback_like = 0
malformed_like = 0

# tables expected as list of tuples: ( (image, html_or_text), positions ) OR (image, html) depending on code path
for idx, tbl in enumerate(tables or []):
    try:
        # Normalized shape handling
        candidate = tbl
        html_or_text = None
        try:
            # Some callers wrap as ((img, html), meta)
            if isinstance(candidate, tuple) and len(candidate) >= 1 and isinstance(candidate[0], tuple) and len(candidate[0]) >= 2:
                html_or_text = candidate[0][1]
            elif isinstance(candidate, tuple) and len(candidate) >= 2 and (isinstance(candidate[1], str) or isinstance(candidate[1], str)):
                # (img, html) simple
                html_or_text = candidate[1]
            elif isinstance(candidate, tuple) and len(candidate) == 1:
                html_or_text = candidate[0]
            else:
                html_or_text = candidate
        except Exception:
            html_or_text = str(candidate)

        html_str = html_or_text if isinstance(html_or_text, str) else repr(html_or_text)
        logger.info("Table[%d] preview: %.200s", idx, html_str[:200])

        # Heuristics to detect VLM-produced HTML/markdown vs fallback structure
        if "<table" in html_str.lower() or "|" in html_str and "-" in html_str:
            vlm_like += 1
            logger.info("Table[%d] appears VLM-like (contains table/markdown markers)", idx)
        elif html_str.strip() == "" or html_str is None:
            malformed_like += 1
            logger.warning("Table[%d] empty or None", idx)
        else:
            # Likely constructed by TableStructureRecognizer (html constructed from components)
            fallback_like += 1
            logger.info("Table[%d] appears fallback/structured output", idx)

    except Exception:
        logger.exception("Error inspecting table %d", idx)

logger.info("Summary: total tables=%d, vlm_like=%d, fallback_like=%d, malformed=%d", len(tables or []), vlm_like, fallback_like, malformed_like)

# Confirm whether VLM routing was expected vs observed
use_vlm_env = str(os.getenv("USE_VLM_TABLE_PARSING", "false")).lower() in ("1", "true", "yes", "on")
if use_vlm_env and vision_model is not None:
    if vlm_like > 0:
        logger.info("CONFIRMATION: VLM table parsing activated and produced %d table(s) that look like HTML/markdown", vlm_like)
    else:
        logger.warning("EXPECTED VLM activation (env true + vision_model present) but no VLM-like tables were found. Check logs for _vlm_table_parser failures.")
else:
    logger.info("VLM table parsing not expected by env or vision_model; observe fallback outputs.")

# Save a short machine-readable summary for quick checks
summary = {
    "pdf": str(pdf_path),
    "use_vlm_env": use_vlm_env,
    "vision_model_attached": getattr(parser, "vision_model", None) is not None,
    "tables_total": len(tables or []),
    "tables_vlm_like": vlm_like,
    "tables_fallback_like": fallback_like,
    "tables_malformed": malformed_like,
    "duration_s": duration,
}

logger.info("WROTE summary to console and log file. Also appending JSON-style summary to %s", LOGFILE)
logger.info("SUMMARY: %s", summary)

# Also append human-friendly snippet of first VLM-like table (if present)
if vlm_like > 0:
    for idx, tbl in enumerate(tables or []):
        try:
            html_or_text = None
            if isinstance(tbl, tuple) and isinstance(tbl[0], tuple) and len(tbl[0]) >= 2:
                html_or_text = tbl[0][1]
            elif isinstance(tbl, tuple) and len(tbl) >= 2:
                html_or_text = tbl[1]
            else:
                html_or_text = tbl
            if not isinstance(html_or_text, str):
                html_or_text = str(html_or_text)
            if "<table" in html_or_text.lower() or "|" in html_or_text:
                logger.info("FIRST VLM TABLE (first 2000 chars):\n%s", html_or_text[:2000])
                break
        except Exception:
            continue

logger.info("Diagnostic complete.")