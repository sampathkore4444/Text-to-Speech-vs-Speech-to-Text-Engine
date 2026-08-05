"""Bank text normalization tests (numbers, currency, dates, redaction)."""

from __future__ import annotations

from speechai.redaction.pii import RedactionPolicy, Redactor
from speechai.tts.textnorm import NormalizedText, TextNormalizer, split_sentences


def _normalize(text: str) -> str:
    normalizer = TextNormalizer(Redactor(RedactionPolicy(mode="mask")))
    return normalizer.normalize(text).text


def test_currency() -> None:
    result = _normalize("Your balance is $1,250.50")
    assert result.lower() == (
        "your balance is one thousand two hundred and fifty dollars and fifty cents"
    )


def test_integer_currency() -> None:
    result = _normalize("Pay $100")
    assert "one hundred dollars" in result


def test_percent() -> None:
    assert "three point five percent" in _normalize("The rate is 3.5%").lower()


def test_date_dmy() -> None:
    assert "the fifteenth of august twenty twenty six" in _normalize("Due on 15/08/2026").lower()


def test_time() -> None:
    assert "nine thirty am" in _normalize("At 9:30 am sharp").lower()


def test_long_reference_redacted_as_account() -> None:
    # 12+ digit runs are treated as account numbers: redacted, never spoken.
    result = _normalize("Reference 123456789012")
    assert result == "Reference redacted"


def test_plain_integer() -> None:
    assert "one thousand five hundred" in _normalize("Amount 1500").lower()


def test_decimal_spoken() -> None:
    assert "three point two five" in _normalize("Rate 3.25").lower()


def test_card_number_never_spoken() -> None:
    result = _normalize("My card is 4242 4242 4242 4242")
    assert "redacted" in result
    assert "4242" not in result


def test_no_redactor_leaves_numbers() -> None:
    normalizer = TextNormalizer()  # no redactor -> numbers expanded normally
    result = normalizer.normalize("Card 4242 4242 4242 4242").text
    assert "four thousand two hundred and forty two" in result
    assert "4242" not in result  # spoken as words instead


def test_split_sentences() -> None:
    sentences = split_sentences("Hello world. This is a bank. Last one!")
    assert sentences == ["Hello world.", "This is a bank.", "Last one!"]


def test_normalized_text_dataclass() -> None:
    normalizer = TextNormalizer(Redactor(RedactionPolicy(mode="mask")))
    out: NormalizedText = normalizer.normalize("Call 98765 43210 now")
    assert out.redacted is True
    assert out.findings and out.findings[0].pii_type == "phone"
