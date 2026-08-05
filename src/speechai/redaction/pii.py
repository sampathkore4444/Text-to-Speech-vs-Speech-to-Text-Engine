"""Banking PII redaction for STT output and TTS input.

Detects and masks/redacts sensitive identifiers commonly handled by banks:
credit/debit card numbers (with Luhn validation), account numbers, IFSC
codes, Aadhaar numbers, PAN, SSN, phone numbers and email addresses.

Security model (configurable via ``redaction.mode``):

- ``mask``  - keep the last N digits visible (default 4), e.g.
  ``4242 4242 4242 4242`` -> ``XXXX XXXX XXXX 4242``. Low enough
  exposure for agent disambiguation, safe for most workflows.
- ``redact`` - replace the whole value with ``[REDACTED]``.
- ``none``  - leave untouched. Never use in production banking.

The redactor errs on the side of redaction: a false positive masks a few
digits, a false negative can leak a PAN/card number.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_ORDER = ("card", "aadhaar", "ssn", "pan", "ifsc", "account", "phone", "email")

_PATTERNS: dict[str, re.Pattern[str]] = {
    "card": re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)"),
    "account": re.compile(r"(?<!\d)\d{9,18}(?!\d)"),
    "ifsc": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "aadhaar": re.compile(r"\b[2-9]\d{3}[ ]\d{4}[ ]\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Permissive candidate: a run of 7-15 digits with common phone separators.
    # `_is_phone` validates digit count and rejects decimal/currency false positives.
    "phone": re.compile(
        r"(?<![0-9A-Za-z])(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,5}\)[\s.-]?)?"
        r"\d[\d\s().-]{5,17}(?![0-9A-Za-z])"
    ),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
}

# Currency tokens that must not be treated as phone numbers.
_CURRENCY_WORDS = ("rs", "inr", "usd", "eur", "gbp", "cad", "aud", "aed", "sar")

DEFAULT_PATTERNS_ENABLED = {name: True for name in _ORDER}


@dataclass
class Finding:
    """A redaction event: what was found, where, and how it was masked."""

    pii_type: str
    start: int
    end: int
    masked: str


@dataclass
class RedactionPolicy:
    mode: str = "mask"
    mask_keep_last: int = 4
    patterns: dict[str, bool] = field(default_factory=lambda: dict(DEFAULT_PATTERNS_ENABLED))

    def is_enabled(self, pii_type: str) -> bool:
        return self.patterns.get(pii_type, True)

    @classmethod
    def from_settings(cls, redaction_cfg: Any) -> RedactionPolicy:
        if not redaction_cfg.enabled:
            return cls(mode="none")
        return cls(
            mode=redaction_cfg.mode,
            mask_keep_last=redaction_cfg.mask_keep_last,
            patterns=dict(redaction_cfg.patterns),
        )


class Redactor:
    """Applies the configured policy to a string of text."""

    def __init__(self, policy: RedactionPolicy | None = None) -> None:
        self.policy = policy or RedactionPolicy()

    # ------------------------------------------------------------------
    def redact(self, text: str) -> tuple[str, list[Finding]]:
        """Return (redacted_text, findings). No-op when policy mode is 'none'."""
        if self.policy.mode == "none":
            return text, []
        work = text
        findings: list[Finding] = []
        for pii_type in _ORDER:
            if not self.policy.is_enabled(pii_type):
                continue
            pattern = _PATTERNS[pii_type]

            def _replace(m: re.Match[str], kind: str = pii_type, current: str = work) -> str:
                value = m.group(0)
                if kind == "card" and not luhn_valid(re.sub(r"\D", "", value)):
                    return value  # numeric but not a valid card number
                if kind == "phone" and not _is_phone(value, current, m.start()):
                    return value
                replacement = self._mask(value, kind)
                findings.append(Finding(kind, m.start(), m.end(), replacement))
                return replacement

            work = pattern.sub(_replace, work)
        return work, findings

    def has_sensitive(self, text: str) -> bool:
        _, findings = self.redact(text)
        return bool(findings)

    # ------------------------------------------------------------------
    def _mask(self, value: str, pii_type: str) -> str:
        if self.policy.mode == "redact" or pii_type == "email":
            return "[REDACTED]"
        digits = re.findall(r"\d", value)
        if not digits:
            return value
        keep = self.policy.mask_keep_last
        total = len(digits)
        out: list[str] = []
        digit_idx = 0
        for ch in value:
            if ch.isdigit():
                # If there are no more digits than we'd reveal, hide all of them.
                visible = digit_idx >= total - keep and total > keep
                out.append(ch if visible else "X")
                digit_idx += 1
            else:
                out.append(ch)
        return "".join(out)


def luhn_valid(digits: str) -> bool:
    """Validate a card number with the Luhn checksum."""
    if not digits or not digits.isdigit():
        return False
    total = 0
    parity = len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_phone(candidate: str, text: str, start: int) -> bool:
    candidate = candidate.rstrip("().- ")  # drop trailing punctuation
    digits = re.sub(r"\D", "", candidate)
    if not 7 <= len(digits) <= 15:
        return False
    if re.search(r"\.\d", candidate):  # looks like a decimal amount, e.g. 1,250.50
        return False
    prefix = text[max(0, start - 6) : start].lower()
    if any(prefix.endswith(token) for token in _CURRENCY_WORDS):
        return False
    return True
