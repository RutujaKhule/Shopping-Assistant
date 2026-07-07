"""
utils/__init__.py
=========================================================
Shared Helpers
=========================================================
Small, dependency-free helpers shared across the app:

1. build_manual_search_query() - builds a concise, human-crafted search
   query from Brand + Product Name + Category (per project requirement:
   NEVER search using Gemini's own free-text `search_query` field).

2. Accessory filtering - a single source of truth for the "ignore
   accessories, only return the real product" rule. Used by:
   - services/search.py   (filter raw Tavily results)
   - services/embeddings.py (skip accessory content before chunking/embedding)
   - rag/retrieval.py      (skip accessory chunks at answer-context time)

Keeping this logic in one place means the accessory blocklist only has
to be updated once and every layer of the app (search, RAG storage, RAG
retrieval) stays consistent.
"""

import re
from typing import Any, Iterable, List, Optional

# -------------------------------------------------------
# Accessory blocklist (Feature: "Never allow accessory results")
# -------------------------------------------------------

ACCESSORY_KEYWORDS: List[str] = [
    "phone cover",
    "back cover",
    "cover",
    "case",
    "tempered glass",
    "screen guard",
    "screen protector",
    "cable",
    "adapter",
    "charger",
    "skin",
    "pouch",
    "keyboard cover",
    "watch strap",
    "strap",
    "accessory",
    "accessories",
]

# Pre-compile word-boundary patterns once so filtering stays cheap even
# when called for every search result / chunk.
_ACCESSORY_PATTERNS = [
    re.compile(r"\b" + re.escape(keyword) + r"\b", re.IGNORECASE)
    for keyword in ACCESSORY_KEYWORDS
]


def contains_accessory_keyword(text: Optional[str]) -> bool:
    """
    Return True if `text` looks like it refers to an accessory (case,
    cover, charger, cable, etc.) rather than the real, standalone
    product.
    """
    if not text:
        return False
    return any(pattern.search(text) for pattern in _ACCESSORY_PATTERNS)


def is_accessory_result(result: Any) -> bool:
    """
    Duck-typed check for SearchResult-like objects (or anything with
    `.title` / `.url` / `.content` attributes). A result is treated as
    an accessory listing if its title or URL clearly names an
    accessory. We deliberately do NOT check `.content` here, since
    legitimate product pages often *mention* accessories ("comes with a
    charger") without being an accessory listing themselves.
    """
    title = getattr(result, "title", "") or ""
    url = getattr(result, "url", "") or ""
    return contains_accessory_keyword(title) or contains_accessory_keyword(url)


def filter_accessory_results(results: Iterable[Any]) -> List[Any]:
    """Drop accessory listings from a list of SearchResult-like objects."""
    return [r for r in results if not is_accessory_result(r)]


# -------------------------------------------------------
# Manual search-query builder
# (Feature: "Never search using Gemini's generated search_query")
# -------------------------------------------------------

_UNKNOWN_VALUES = {"", "unknown", "n/a", "na", "none", "null"}


def _clean_token(value: Optional[str]) -> str:
    if not value:
        return ""
    value = value.strip()
    if value.lower() in _UNKNOWN_VALUES:
        return ""
    return value


def build_manual_search_query(
    brand: Optional[str],
    product_name: Optional[str],
    category: Optional[str] = None,
) -> str:
    """
    Build a concise search query manually from Brand + Product Name +
    Category, exactly per the project spec:

        Brand:    Apple
        Product:  iPhone 15
        Category: Smartphone
        ->        "Apple iPhone 15 Smartphone"

    Rules:
    - Never include colors/variants unless they're already part of the
      product name.
    - Never duplicate a brand or category word that's already present
      in the product name (e.g. product_name="Apple iPhone 15" +
      brand="Apple" should NOT become "Apple Apple iPhone 15").
    - Falls back gracefully to whatever fields are actually available.
    """
    brand_c = _clean_token(brand)
    product_c = _clean_token(product_name)
    category_c = _clean_token(category)

    product_lower = product_c.lower()

    parts: List[str] = []
    if brand_c and brand_c.lower() not in product_lower:
        parts.append(brand_c)
    if product_c:
        parts.append(product_c)
    if category_c and category_c.lower() not in product_lower:
        parts.append(category_c)

    query = " ".join(parts).strip()
    query = re.sub(r"\s+", " ", query)

    return query or product_c or "product"