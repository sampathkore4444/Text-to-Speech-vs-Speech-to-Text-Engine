"""STT post-processing tests: cleaning and redaction hooks."""

from __future__ import annotations

from speechai.redaction.pii import RedactionPolicy, Redactor
from speechai.stt.postprocess import TextPostProcessor, clean_text


def test_clean_text() -> None:
    assert clean_text("  hello   world ,  sir.  ") == "Hello world, sir."


def test_fix_spacing() -> None:
    assert clean_text("hello,world") == "Hello, world"
    assert clean_text("hello , world") == "Hello, world"


def test_postprocessor_redacts() -> None:
    processor = TextPostProcessor(Redactor(RedactionPolicy(mode="mask")))
    result = processor.process("My card is 4242 4242 4242 4242.")
    assert result.redacted is True
    assert "4242 4242 4242 4242" not in result.text
    assert "XXXX" in result.text


def test_postprocessor_no_redact() -> None:
    processor = TextPostProcessor(Redactor(RedactionPolicy(mode="mask")))
    result = processor.process("My card is 4242 4242 4242 4242.", redact=False)
    assert result.redacted is False
    assert "4242 4242 4242 4242" in result.text
