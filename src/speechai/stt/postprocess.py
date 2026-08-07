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
from speechai.stt.base import Segment


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


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences on ``.``/``!``/``?`` followed by whitespace."""
    if not text:
        return []
    parts = [part.strip() for part in _SENTENCE_SPLIT.split(text)]
    return [part for part in parts if part]


def refine_segments(segments: list[Segment]) -> list[Segment]:
    """Expand engine segments so each row is a single sentence.

    faster-whisper often returns one segment spanning several sentences when
    the pauses between them are short (a single file transcribes as one
    "line"). Splitting on sentence boundaries - with timings interpolated
    proportionally to the character offsets within the segment - gives every
    consumer clean per-line, timed rows. Segments that are already a single
    sentence are returned unchanged.
    """
    refined: list[Segment] = []
    for seg in segments:
        sentences = split_sentences(seg.text)
        if len(sentences) <= 1:
            refined.append(seg)
            continue
        span = max(seg.end - seg.start, 1e-3)
        text_len = max(len(seg.text), 1)
        cursor = 0
        for sentence in sentences:
            start = seg.start + span * (cursor / text_len)
            cursor += len(sentence)
            end = seg.start + span * (cursor / text_len)
            refined.append(Segment(text=sentence, start=start, end=end, confidence=seg.confidence))
            cursor += 1  # the whitespace separator
        refined[-1].end = seg.end  # clamp the final boundary to the segment end
    return refined
