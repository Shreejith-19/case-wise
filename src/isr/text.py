"""Shared lexical analysis: one tokenizer for indexing and querying.

NFKC + lowercase, alphanumeric tokens (single characters kept: section/clause numbers
like "2" or "a" matter in law), English stopwords removed, Snowball stemming.
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import Stemmer
from bm25s.tokenization import STOPWORDS_EN

_TOKEN = re.compile(r"[0-9a-z]+")
_STOP = frozenset(STOPWORDS_EN) - {"no", "not", "nor"}   # negation matters in statutes
_STEMMER = Stemmer.Stemmer("english")


@lru_cache(maxsize=500_000)
def _stem(word: str) -> str:
    return _STEMMER.stemWord(word)


def raw_tokens(text: str) -> list[str]:
    return _TOKEN.findall(unicodedata.normalize("NFKC", text).lower())


def tokenize(text: str) -> list[str]:
    """Stemmed content tokens, in order (duplicates kept)."""
    return [_stem(t) for t in raw_tokens(text) if t not in _STOP]
