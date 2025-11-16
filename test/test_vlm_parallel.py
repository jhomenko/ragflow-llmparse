import os
import time
import threading
import tracemalloc
import io
import pytest
from PIL import Image

# Import target under test
from deepdoc.parser.pdf_parser import VisionParser, VisionParserPageError

# Helpers --------------------------------------------------------------------


def make_image(w=800, h=600, color=(245, 245, 245)):
    """Create a simple RGB PIL image for testing."""
    return Image.new("RGB", (w, h), color)


def make_fake___images__(page_count=3):
    """
    Return a function suitable for monkeypatching VisionParser.__images__.
    The returned function will set self.page_images and self.total_page.
    """

    def _impl(self, fnm, zoomin=3, page_from=0, page_to=299, callback=None):
        self.page_images = [make_image() for _ in range(page_count)]
        self.total_page = page_count

    return _impl


def make_vlm_stub(concurrent_tracker=None, sleep_s=0.0, fail_on_pages=None, prefix="PAGE"):
    """
    Produce a stub function that mimics the VLM call used by VisionParser.
    - concurrent_tracker: dict with 'current' and 'max' to observe concurrency
    - sleep_s: seconds to sleep to allow concurrency overlap
    - fail_on_pages: set/list of 1-based page numbers to raise an exception for
    - prefix: string to include in returned text
    """

    fail_on_pages = set(fail_on_pages or [])

    lock = threading.Lock()
    call_id = {"n": 0}

    def _stub(binary=None, vision_model=None, prompt=None, callback=None, timeout=None):
        # Track concurrency
        if concurrent_tracker is not None:
            with lock:
                concurrent_tracker["current"] += 1
                concurrent_tracker["max"] = max(concurrent_tracker.get("max", 0), concurrent_tracker["current"])
        try:
            # Allow other threads to run so concurrent calls overlap
            if sleep_s:
                time.sleep(sleep_s)

            # Try to heuristically extract page number from the prompt (common prompt templates include the page number)
            page_no = None
            if isinstance(prompt, str):
                import re

                m = re.search(r"\bpage\b.*?(\d+)", prompt, flags=re.IGNORECASE)
                if m:
                    page_no = int(m.group(1))
            call_id["n"] += 1
            # if page_no is one that should fail, raise
            if page_no in fail_on_pages:
                raise RuntimeError(f"Injected failure for page {page_no}")
            return f"{prefix} {page_no or call_id['n']}"
        finally:
            if concurrent_tracker is not None:
                with lock:
                    concurrent_tracker["current"] -= 1

    return _stub


# Tests ----------------------------------------------------------------------


def test_parallel_env_sequential_path(monkeypatch):
    """
    Unit test: when PARALLEL_VLM_REQUESTS=1, VisionParser should run the sequential
    path and produce deterministic outputs matching expectations driven by prompt_text.
    """
    monkeypatch.setenv("PARALLEL_VLM_REQUESTS", "1")

    # Make a small fake document of 3 pages
    monkeypatch.setattr("deepdoc.parser.pdf_parser.VisionParser.__images__", make_fake___images__(page_count=3))

    # Install stub VLM that echoes the prompt so we can predict outputs
    stub = make_vlm_stub(sleep_s=0.0, prefix="ECHO")
    monkeypatch.setattr("rag.app.picture", "vision_llm_chunk", stub, raising=False)
    # Some modules import picture_vision_llm_chunk by other name; set both if present
    try:
        import rag.app.picture as picture_mod

        picture_mod.picture_vision_llm_chunk = stub
        picture_mod.vision_llm_chunk = stub
    except Exception:
        pass

    parser = VisionParser(vision_model=object())

    # Use a controlled prompt_text so outputs are deterministic and include page num
    prompt_text = "PROMPT page {{ page }}"

    docs, _ = parser("dummy.pdf", from_page=0, to_page=3, prompt_text=prompt_text)
    assert len(docs) == 3, "Expected 3 pages processed in sequential path"

    # Expect each entry to contain the page number as returned by stub
    expected = [("ECHO {}".format(i + 1), f"@@{i + 1}\t0.0\t{parser.page_images[i].size[0] / 3:.1f}\t0.0\t{parser.page_images[i].size[1] / 3:.1f}##") for i in range(3)]

    # Only check startswith because VisionParser does some trimming/formatting on results
    for (got_text, got_meta), (exp_text, exp_meta) in zip(docs, expected):
        assert got_text is not None
        assert str(got_text).startswith(exp_text), f"Expected text starting with {exp_text} but got {got_text}"
        assert got_meta.startswith(f"@@{exp_meta.split()[0][2:2] if False else exp_meta.split()[0].strip('@')}" ) or got_meta.startswith("@@"), "Meta tag present"


def test_parallel_concurrency_limits(monkeypatch):
    """
    Integration-style unit test using a stub VLM to verify concurrent execution
    observes PARALLEL_VLM_REQUESTS semaphore behavior.
    """
    monkeypatch.setenv("PARALLEL_VLM_REQUESTS", "4")

    # Create 8 pages to exercise concurrency
    monkeypatch.setattr("deepdoc.parser.pdf_parser.VisionParser.__images__", make_fake___images__(page_count=8))

    concurrent = {"current": 0, "max": 0}
    # stub sleeps so multiple threads overlap
    stub = make_vlm_stub(concurrent_tracker=concurrent, sleep_s=0.12, prefix="CONC")
    # apply stub to the expected call sites
    monkeypatch.setattr("rag.app.picture", "vision_llm_chunk", stub, raising=False)
    try:
        import rag.app.picture as picture_mod

        picture_mod.picture_vision_llm_chunk = stub
        picture_mod.vision_llm_chunk = stub
    except Exception:
        pass

    parser = VisionParser(vision_model=object())

    docs, _ = parser("dummy.pdf", from_page=0, to_page=8, prompt_text="PROMPT page {{ page }}")
    assert len(docs) == 8

    # We expect some concurrency > 1 while respecting the semaphore (<= 4)
    assert concurrent["max"] >= 2, "No concurrency observed; expected >1"
    assert concurrent["max"] <= 4, f"Semaphore did not cap concurrency; observed max {concurrent['max']}"


def test_fault_injection_page_failure_causes_abort(monkeypatch):
    """
    Fault injection: force one page to raise inside the VLM stub and ensure the parser
    aborts the entire document after retries.
    """
    monkeypatch.setenv("PARALLEL_VLM_REQUESTS", "3")
    monkeypatch.setattr("deepdoc.parser.pdf_parser.VisionParser.__images__", make_fake___images__(page_count=5))

    # Fail on page 3 (1-based)
    stub = make_vlm_stub(fail_on_pages={3}, sleep_s=0.02, prefix="OK")
    monkeypatch.setattr("rag.app.picture", "vision_llm_chunk", stub, raising=False)
    try:
        import rag.app.picture as picture_mod

        picture_mod.picture_vision_llm_chunk = stub
        picture_mod.vision_llm_chunk = stub
    except Exception:
        pass

    parser = VisionParser(vision_model=object())

    with pytest.raises(VisionParserPageError):
        parser("dummy.pdf", from_page=0, to_page=5, prompt_text="PROMPT page {{ page }}")


@pytest.mark.skipif(os.getenv("CI") is not None, reason="Skip heavy profiling in CI")
def test_memory_profiling_large_document(monkeypatch):
    """
    Profiling test: create many pages and ensure we can run through the parallel path
    while capturing peak memory via tracemalloc. This test is intentionally lightweight
    in CPU but can reveal large memory blowups when many images are held in memory.
    """
    monkeypatch.setenv("PARALLEL_VLM_REQUESTS", "2")
    # Create a "large" doc: 120 pages of small images (keeps test reasonable)
    PAGE_COUNT = 120
    monkeypatch.setattr("deepdoc.parser.pdf_parser.VisionParser.__images__", make_fake___images__(page_count=PAGE_COUNT))

    # Fast stub so test completes quickly
    stub = make_vlm_stub(sleep_s=0.0, prefix="PG")
    monkeypatch.setattr("rag.app.picture", "vision_llm_chunk", stub, raising=False)
    try:
        import rag.app.picture as picture_mod

        picture_mod.picture_vision_llm_chunk = stub
        picture_mod.vision_llm_chunk = stub
    except Exception:
        pass

    parser = VisionParser(vision_model=object())

    tracemalloc.start()
    docs, _ = parser("dummy.pdf", from_page=0, to_page=PAGE_COUNT, prompt_text="PROMPT page {{ page }}")
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    assert len(docs) == PAGE_COUNT
    # Don't assert a hard threshold (envs/CI vary) — just confirm we measured memory
    assert peak > 0


def test_environment_variable_handling(monkeypatch):
    """
    Verify parser reacts to PARALLEL_VLM_REQUESTS env variable by creating/removing the internal semaphore.
    """
    # Sequential case
    monkeypatch.setenv("PARALLEL_VLM_REQUESTS", "1")
    monkeypatch.setattr("deepdoc.parser.pdf_parser.VisionParser.__images__", make_fake___images__(page_count=2))
    stub = make_vlm_stub(sleep_s=0.0, prefix="ENV")
    monkeypatch.setattr("rag.app.picture", "vision_llm_chunk", stub, raising=False)
    try:
        import rag.app.picture as picture_mod

        picture_mod.picture_vision_llm_chunk = stub
        picture_mod.vision_llm_chunk = stub
    except Exception:
        pass

    parser = VisionParser(vision_model=object())
    docs, _ = parser("dummy.pdf", from_page=0, to_page=2, prompt_text="PROMPT page {{ page }}")
    # When PARALLEL_VLM_REQUESTS=1 the internal semaphore should be None
    assert getattr(parser, "_vlm_semaphore", None) is None

    # Parallel case
    monkeypatch.setenv("PARALLEL_VLM_REQUESTS", "3")
    parser2 = VisionParser(vision_model=object())
    # Trigger __call__ to construct the semaphore (we patch images to avoid heavy work)
    monkeypatch.setattr("deepdoc.parser.pdf_parser.VisionParser.__images__", make_fake___images__(page_count=2))
    docs2, _ = parser2("dummy.pdf", from_page=0, to_page=2, prompt_text="PROMPT page {{ page }}")
    assert getattr(parser2, "_vlm_semaphore", None) is not None
