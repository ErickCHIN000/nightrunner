"""Schema / format ids. Files written before the rename carry the old `beastpack.` prefix and stay readable."""

from __future__ import annotations

LEGACY_PREFIX = "beastpack."
PREFIX = "nightrunner."


def normalize(value) -> str:
    s = str(value or "")
    return PREFIX + s[len(LEGACY_PREFIX):] if s.startswith(LEGACY_PREFIX) else s


def matches(value, schema: str) -> bool:
    """True when *value* is *schema* or its pre-rename spelling."""
    return normalize(value) == schema


def startswith(value, prefix: str) -> bool:
    return normalize(value).startswith(prefix)
