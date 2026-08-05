"""Bank-specific text normalization for TTS input.

Converts written numbers, currency amounts, dates, times, percentages and
reference numbers into words that a TTS engine speaks naturally, and applies
PII redaction so sensitive identifiers are *never* spoken.

Examples::

    "$1,250.50"              -> "one thousand two hundred fifty dollars and fifty cents"
    "15/08/2026"             -> "the fifteenth of August twenty twenty six"
    "3.5%"                   -> "three point five percent"
    "4242 4242 4242 4242"    -> "redacted"           (after redaction)
    "9:30 am"                -> "nine thirty am"
    "REF 123456789012"       -> "ref one two three four five six seven eight nine zero one two"
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from num2words import num2words

from speechai.redaction.pii import Finding, Redactor

_MONTHS = [
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
]

_CURRENCY_SYMBOLS = {"$": "dollars", "€": "euros", "£": "pounds", "₹": "rupees"}
_CURRENCY_CODES = {"usd": "dollars", "eur": "euros", "gbp": "pounds", "inr": "rupees"}

# Masked PII runs produced by the redactor, e.g. "XXXX XXXX 3456".
_MASKED_RUN = re.compile(r"\bX[\dX\s-]*\b")


@dataclass
class NormalizedText:
    text: str
    redacted: bool = False
    findings: list[Finding] = field(default_factory=list)


class TextNormalizer:
    """Redact + expand a string of text for speech synthesis."""

    def __init__(self, redactor: Redactor | None = None, *, speak_redacted_as: str = "redacted") -> None:
        self.redactor = redactor
        self.speak_redacted_as = speak_redacted_as

    def normalize(self, text: str) -> NormalizedText:
        work = text.strip()
        findings: list[Finding] = []
        if self.redactor is not None:
            work, findings = self.redactor.redact(work)
            if self.speak_redacted_as:
                work = _MASKED_RUN.sub(self.speak_redacted_as, work)
        work = _expand_all(work)
        return NormalizedText(text=work, redacted=bool(findings), findings=findings)


# ---------------------------------------------------------------------------
# Expansion pipeline (order matters: specific patterns before generic numbers)
# ---------------------------------------------------------------------------
def _expand_all(text: str) -> str:
    text = _expand_dates(text)
    text = _expand_times(text)
    text = _expand_percent(text)
    text = _expand_currency(text)
    text = _expand_digit_runs(text)
    text = _expand_plain_numbers(text)
    return _clean_ws(text)


def _expand_dates(text: str) -> str:
    # dd/mm/yyyy or dd-mm-yyyy or dd.mm.yyyy
    def _repl(m: re.Match[str]) -> str:
        day, month, year = int(m.group(1)), int(m.group(2)), m.group(3)
        if month < 1 or month > 12:
            return m.group(0)
        return f"the {_ordinal_word(day)} of {_MONTHS[month - 1]} {_year_to_words(year)}"

    text = re.sub(r"\b(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{2,4})\b", _repl, text)
    # ISO yyyy-mm-dd
    text = re.sub(
        r"\b(\d{4})[-/](\d{1,2})[-/](\d{1,2})\b",
        lambda m: f"the {_ordinal_word(int(m.group(3)))} of {_MONTHS[int(m.group(2)) - 1]} {_year_to_words(m.group(1))}"
        if 1 <= int(m.group(2)) <= 12
        else m.group(0),
        text,
    )
    return text


def _expand_times(text: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        hour, minute, meridiem = int(m.group(1)), int(m.group(2)), (m.group(3) or "").lower()
        if hour > 23 or minute > 59:
            return m.group(0)
        hour12 = hour % 12 or 12
        if minute == 0:
            return f"{_int_to_words(hour12)} {meridiem}".strip()
        return f"{_int_to_words(hour12)} {_two_digit_words(minute)} {meridiem}".strip()

    return re.sub(r"\b(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?\b", _repl, text, flags=re.IGNORECASE)


def _expand_percent(text: str) -> str:
    return re.sub(
        r"(\d+(?:\.\d+)?)\s*%",
        lambda m: f"{_number_to_words(m.group(1))} percent",
        text,
    )


def _expand_currency(text: str) -> str:
    def _repl(m: re.Match[str]) -> str:
        symbol = m.group(1)
        word = _CURRENCY_SYMBOLS.get(symbol, "dollars")
        return _amount_to_words(m.group(2), word)

    text = re.sub(r"([$€£₹])\s*([\d,]+(?:\.\d{1,2})?)", _repl, text)
    # Explicit currency codes: "USD 50" / "50 USD"
    text = re.sub(
        r"\b(usd|eur|gbp|inr)\s+([\d,]+(?:\.\d{1,2})?)",
        lambda m: _amount_to_words(m.group(2), _CURRENCY_CODES[m.group(1).lower()]),
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"\b([\d,]+(?:\.\d{1,2})?)\s*(usd|eur|gbp|inr)\b",
        lambda m: _amount_to_words(m.group(1), _CURRENCY_CODES[m.group(2).lower()]),
        text,
        flags=re.IGNORECASE,
    )
    return text


def _expand_digit_runs(text: str) -> str:
    """Long reference numbers are spoken digit-by-digit."""
    return re.sub(r"\b\d{9,}\b", lambda m: " ".join(_digit_words(d) for d in m.group(0)), text)


def _expand_plain_numbers(text: str) -> str:
    # Thousands-grouped numbers: 1,250.50
    text = re.sub(
        r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b",
        lambda m: _number_to_words(m.group(0).replace(",", "")),
        text,
    )
    # Remaining plain integers/decimals
    text = re.sub(r"\b\d+(?:\.\d+)?\b", lambda m: _number_to_words(m.group(0)), text)
    return text


# ---------------------------------------------------------------------------
def _number_to_words(value: str) -> str:
    if "." in value:
        whole, frac = value.split(".", 1)
        whole_words = _int_to_words(int(whole)) if whole.lstrip("0") else "zero"
        frac_digits = " ".join(_digit_words(d) for d in frac)
        return f"{whole_words} point {frac_digits}"
    return _int_to_words(int(value))


def _amount_to_words(value: str, unit: str) -> str:
    value = value.replace(",", "")
    if "." in value:
        whole, cents = value.split(".", 1)
        cents = cents.ljust(2, "0")
        whole_words = _int_to_words(int(whole)) if whole.lstrip("0") else "zero"
        if cents == "00":
            return f"{whole_words} {unit}"
        if unit == "rupees":
            return f"{whole_words} {unit} and {_int_to_words(int(cents))} paise"
        return f"{whole_words} {unit} and {_int_to_words(int(cents))} cents"
    return f"{_int_to_words(int(value))} {unit}"


def _int_to_words(value: int) -> str:
    if value == 0:
        return "zero"
    # num2words emits commas and hyphens ("twenty-six"); normalize to plain words.
    return re.sub(r"[\s,-]+", " ", num2words(value, lang="en")).strip()


def _ordinal_word(value: int) -> str:
    return num2words(value, lang="en", to="ordinal")


def _two_digit_words(value: int) -> str:
    if value < 10:
        return _digit_words(str(value))
    return _int_to_words(value)


def _year_to_words(year: str) -> str:
    if len(year) == 4 and year[0] in "12" and year[2:] != "00":
        return f"{_int_to_words(int(year[:2]))} {_int_to_words(int(year[2:]))}"
    return _int_to_words(int(year))


def _digit_words(digit: str) -> str:
    return {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
            "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}[digit]


def _clean_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences for chunked synthesis."""
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text)]
    return [p for p in parts if p]
