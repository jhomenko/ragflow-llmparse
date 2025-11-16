import pytest

from rag.app.picture import (
    VisionLLMCallError,
    VisionLLMResponseError,
    _REPEAT_HINT,
    vision_llm_chunk,
)


class DummyVisionModel:
    llm_name = "Qwen2.5VL-3B___OpenAI-API@OpenAI-Compatible"

    def describe_with_prompt(self, *args, **kwargs):
        return ""


def _make_repetitive_text():
    block = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    return block * 4


def test_vision_llm_chunk_retries_on_repetition(monkeypatch):
    calls = {"count": 0, "temps": [], "prompts": []}

    def fake_describe(image_bytes, prompt, model_name, temperature=0.1):
        calls["count"] += 1
        calls["temps"].append(round(temperature, 2))
        calls["prompts"].append(prompt)
        if calls["count"] == 1:
            return _make_repetitive_text(), 512
        return ("# Heading\nValid content", 1024)

    monkeypatch.setattr("rag.app.picture.describe_image_working", fake_describe)
    monkeypatch.setenv("VLM_PAGE_MAX_ATTEMPTS", "3")

    result = vision_llm_chunk(
        binary=b"\xff" * 1024,
        vision_model=DummyVisionModel(),
        prompt="PROMPT",
    )

    assert result.startswith("# Heading")
    assert calls["count"] == 2, "Expected a retry after detecting repetition"
    assert calls["temps"][0] == 0.1
    assert calls["temps"][1] > calls["temps"][0]
    assert calls["prompts"][1].endswith(_REPEAT_HINT.strip())


def test_vision_llm_chunk_raises_after_exhausting_retries(monkeypatch):
    def fake_describe(*args, **kwargs):
        return _make_repetitive_text(), 256

    monkeypatch.setattr("rag.app.picture.describe_image_working", fake_describe)
    monkeypatch.setenv("VLM_PAGE_MAX_ATTEMPTS", "2")

    with pytest.raises(VisionLLMResponseError):
        vision_llm_chunk(
            binary=b"\x00" * 2048,
            vision_model=DummyVisionModel(),
            prompt="PROMPT",
        )


def test_vision_llm_chunk_raises_on_transport_error(monkeypatch):
    def fake_describe(*args, **kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("rag.app.picture.describe_image_working", fake_describe)

    with pytest.raises(VisionLLMCallError):
        vision_llm_chunk(
            binary=b"\x00" * 2048,
            vision_model=DummyVisionModel(),
            prompt="PROMPT",
        )
