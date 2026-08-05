"""Bank PII redaction tests."""

from __future__ import annotations

from speechai.redaction.pii import RedactionPolicy, Redactor, luhn_valid


def test_luhn() -> None:
    assert luhn_valid("4242424242424242") is True
    assert luhn_valid("4242424242424241") is False
    assert luhn_valid("abc") is False


def test_mask_card_number() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask", mask_keep_last=4))
    text, findings = redactor.redact("My card is 4242 4242 4242 4242.")
    assert text == "My card is XXXX XXXX XXXX 4242."
    assert len(findings) == 1
    assert findings[0].pii_type == "card"


def test_non_card_digits_untouched() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask"))
    text, findings = redactor.redact("The order number is 12345.")
    assert text == "The order number is 12345."
    assert findings == []


def test_redact_mode_removes_value() -> None:
    redactor = Redactor(RedactionPolicy(mode="redact"))
    text, findings = redactor.redact("My card is 4242 4242 4242 4242")
    assert "[REDACTED]" in text
    assert "4242" not in text


def test_none_mode_is_noop() -> None:
    redactor = Redactor(RedactionPolicy(mode="none"))
    text, findings = redactor.redact("My card is 4242 4242 4242 4242")
    assert text == "My card is 4242 4242 4242 4242"
    assert findings == []


def test_account_number_masked() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask"))
    text, findings = redactor.redact("Account number 123456789012 is active.")
    assert "123456789012" not in text
    assert "9012" in text
    assert findings[0].pii_type == "account"


def test_ifsc_masked() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask"))
    text, findings = redactor.redact("IFSC: HDFC0001234")
    assert findings and findings[0].pii_type == "ifsc"
    assert "HDFC0001234" not in text


def test_aadhaar_masked() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask"))
    text, findings = redactor.redact("Aadhaar 2345 6789 0123 verified")
    assert findings and findings[0].pii_type == "aadhaar"
    assert "2345 6789 0123" not in text


def test_pan_masked() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask"))
    text, findings = redactor.redact("PAN ABCDE1234F on file")
    assert findings and findings[0].pii_type == "pan"
    assert "ABCDE1234F" not in text
    assert "XXXX" in text  # digits hidden even though they are few


def test_ssn_masked() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask"))
    text, findings = redactor.redact("SSN 123-45-6789")
    assert findings and findings[0].pii_type == "ssn"


def test_phone_masked() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask"))
    text, findings = redactor.redact("Call me at 98765 43210")
    assert findings and findings[0].pii_type == "phone"
    assert "98765" not in text


def test_email_redacted() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask"))
    text, findings = redactor.redact("Contact support@example.com")
    assert findings and findings[0].pii_type == "email"
    assert "[REDACTED]" in text


def test_amount_not_treated_as_phone() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask"))
    text, findings = redactor.redact("The total is $1,250.50")
    assert text == "The total is $1,250.50"
    assert findings == []


def test_has_sensitive() -> None:
    redactor = Redactor(RedactionPolicy(mode="mask"))
    assert redactor.has_sensitive("Card 4242 4242 4242 4242") is True
    assert redactor.has_sensitive("Have a nice day") is False


def test_patterns_can_be_disabled() -> None:
    policy = RedactionPolicy(mode="mask")
    policy.patterns["card"] = False
    policy.patterns["aadhaar"] = False  # aadhaar matches 4-4-4 digit groups too
    redactor = Redactor(policy)
    text, findings = redactor.redact("Card 4242 4242 4242 4242")
    assert findings == []
    assert "4242 4242 4242 4242" in text
