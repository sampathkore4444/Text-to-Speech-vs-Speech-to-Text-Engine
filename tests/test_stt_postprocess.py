"""STT post-processing tests: cleaning and redaction hooks."""

from __future__ import annotations

from speechai.redaction.pii import RedactionPolicy, Redactor
from speechai.stt.base import Segment
from speechai.stt.postprocess import TextPostProcessor, clean_text, refine_segments, split_sentences


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


def test_split_sentences() -> None:
    assert split_sentences("One. Two! Three?") == ["One.", "Two!", "Three?"]
    assert split_sentences("") == []
    assert split_sentences("No punctuation here") == ["No punctuation here"]


def test_refine_segments_splits_multi_sentence_segments() -> None:
    seg = Segment(text="First sentence. Second sentence!", start=1.0, end=3.0, confidence=0.9)
    refined = refine_segments([seg])
    assert len(refined) == 2
    assert refined[0].text == "First sentence."
    assert refined[1].text == "Second sentence!"
    assert refined[0].confidence == 0.9
    # timings stay inside the original span, in order, clamped to seg.end
    assert refined[0].start >= seg.start
    assert refined[1].end == seg.end
    assert refined[0].end <= refined[1].start + 1e-6


def test_refine_segments_keeps_single_sentence() -> None:
    seg = Segment(text="Just one line.", start=0.0, end=1.0, confidence=0.5)
    assert refine_segments([seg]) == [seg]


class _FakeWord:
    """Minimal stand-in for faster-whisper's Word dataclass."""

    def __init__(self, word: str, start: float, end: float, probability: float = 0.9) -> None:
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


def test_group_word_segments_splits_on_gaps() -> None:
    from speechai.stt.whisper_engine import _group_word_segments

    words = [
        _FakeWord(" We", 0.0, 0.3),
        _FakeWord(" detected", 0.3, 0.7),
        _FakeWord(" activity", 0.7, 1.2),
        _FakeWord(" account,", 1.2, 1.6),
        _FakeWord(" press", 1.9, 2.3),  # gap 0.3s -> new row
        _FakeWord(" one.", 2.3, 2.9),
        _FakeWord(" Thanks", 3.1, 3.6),  # gap 0.2s + previous ends '.' -> new row
        _FakeWord(" bye.", 3.6, 4.2),
    ]
    rows = _group_word_segments(words)
    assert len(rows) == 3
    assert rows[0].text == "We detected activity account,"
    assert rows[0].start == 0.0
    assert rows[0].end == 1.6
    assert rows[1].text == "press one."
    assert rows[1].start == 1.9
    assert rows[1].end == 2.9
    assert rows[2].text == "Thanks bye."
    assert rows[2].end == 4.2
    assert rows[0].confidence == 0.9


def test_group_word_segments_single_row() -> None:
    from speechai.stt.whisper_engine import _group_word_segments

    words = [_FakeWord(" Hello", 0.0, 0.4), _FakeWord(" world", 0.4, 0.9)]
    rows = _group_word_segments(words)
    assert len(rows) == 1
    assert rows[0].text == "Hello world"
