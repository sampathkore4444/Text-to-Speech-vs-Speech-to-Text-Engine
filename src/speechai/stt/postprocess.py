"""Deterministic post-processing applied to ASR output.

Whisper already restores punctuation and capitalization; this module adds
deterministic normalization (whitespace, spacing around punctuation, sentence
casing) and hooks the PII redactor so transcripts leaving the platform are
clean of sensitive identifiers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from speechai.redaction.pii import Finding, Redactor


@dataclass
class PostProcessed:
    text: str
    redacted: bool = False
    findings: list[Finding] = field(default_factory=list)


class TextPostProcessor:
    """Cleans ASR text and applies PII redaction."""

    def __init__(self, redactor: Redactor | None = None) -> None:
        self.redactor = redactor

    def process(self, text: str, *, redact: bool = True) -> PostProcessed:
        cleaned = clean_text(text)
        findings: list[Finding] = []
        if redact and self.redactor is not None:
            cleaned, findings = self.redactor.redact(cleaned)
        return PostProcessed(text=cleaned, redacted=bool(findings), findings=findings)


def clean_text(text: str) -> str:
    """Whitespace normalization + spacing + sentence casing."""
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    text = fix_spacing(text)
    text = cap_sentences(text)
    return text


def fix_spacing(text: str) -> str:
    # "hello , world" -> "hello, world"
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    # "hello,world" -> "hello, world"
    text = re.sub(r"([,.;:!?])(?=[A-Za-z0-9])", r"\1 ", text)
    return text


def cap_sentences(text: str) -> str:
    return re.sub(r"(^|[.!?]\s+)([a-z])", lambda m: m.group(1) + m.group(2).upper(), text)
