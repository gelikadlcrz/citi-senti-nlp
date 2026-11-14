"""Fixed label space from the CITI-Senti aspect taxonomy v1.1.

The category codes and order here MUST be identical for every baseline model.
"""
from __future__ import annotations

import re

ASPECT_NAMES: dict[str, str] = {
    "A01": "Policy Benefits and Impact",
    "A02": "Implementation and Enforcement",
    "A03": "Cost, Fees and Penalties",
    "A04": "Accessibility and Convenience",
    "A05": "Fairness and Inclusivity",
    "A06": "Rules and Policy Provisions",
    "A07": "Government Administration and Responsiveness",
    "A08": "Public Safety and Security",
    "A09": "Environmental Impact",
    "A10": "Traffic and Transportation Impact",
    "A11": "Other",
}
ASPECT_CODES: tuple[str, ...] = tuple(ASPECT_NAMES)
ASPECT_TO_INDEX: dict[str, int] = {code: i for i, code in enumerate(ASPECT_CODES)}

SENTIMENT_LABELS: tuple[str, ...] = ("Negative", "Neutral", "Positive")
SENTIMENT_TO_INDEX: dict[str, int] = {
    label: i for i, label in enumerate(SENTIMENT_LABELS)
}


def _normalize_words(value: str) -> str:
    # Punctuation and spacing variants are ignored for matching known category names.
    return " ".join(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


_CATEGORY_NAMES_TO_CODES = {
    _normalize_words(name): code for code, name in ASPECT_NAMES.items()
}


def normalize_aspect(value: str) -> str:
    """Convert 'A01', 'A01 — Name', or the official name to a stable code."""
    raw = str(value).strip()
    match = re.match(r"^(A(?:0[1-9]|1[01]))(?=$|\W)", raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    code = _CATEGORY_NAMES_TO_CODES.get(_normalize_words(raw))
    if code is not None:
        return code
    raise ValueError(
        f"Unrecognized aspect_category {value!r}. Use the official A01–A11 "
        "code, 'A01 — Name', or its official taxonomy name."
    )


def normalize_sentiment(value: str) -> str:
    raw = str(value).strip().casefold()
    for label in SENTIMENT_LABELS:
        if raw == label.casefold():
            return label
    raise ValueError(
        f"Invalid sentiment {value!r}; expected Positive, Neutral, or Negative."
    )
