"""
services/search.py
=========================================================
Real-Time Search Service (Tavily)
=========================================================
Modified with lazy initialization to prevent crashes during import on Streamlit Cloud.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import requests
import streamlit as st

from utils import build_manual_search_query, filter_accessory_results

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "8"))

RETAILER_DOMAINS: Dict[str, List[str]] = {
    "Amazon India": ["amazon.in"],
    "Flipkart": ["flipkart.com"],
    "Croma": ["croma.com"],
    "Reliance Digital": ["reliancedigital.in"],
    "Vijay Sales": ["vijaysales.com"],
}

MAX_PARALLEL_WORKERS = 5
MAX_CACHE_ENTRIES = 256


def get_tavily_api_key() -> Optional[str]:
    """Dynamically fetches TAVILY_API_KEY from Streamlit Secrets or environment variables."""
    return st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY")


class SearchServiceError(Exception):
    """Raised when a search call fails or returns no usable results."""
    pass


@dataclass
class SearchResult:
    """A single normalized web search result."""
    title: str
    url: str
    content: str
    score: float = 0.0
    source: str = "web"

    def to_text(self) -> str:
        """Render this result as a labeled text block for embedding/RAG."""
        return (
            f"Source: {self.source}\n"
            f"Title: {self.title}\n"
            f"URL: {self.url}\n"
            f"Content: {self.content}\n"
        )


@dataclass
class ProductSearchBundle:
    """All real-time data gathered for one product in one place."""
    product_name: str
    general_info: List[SearchResult] = field(default_factory=list)
    prices_by_retailer: Dict[str, List[SearchResult]] = field(default_factory=dict)
    reviews: List[SearchResult] = field(default_factory=list)
    similar_products: List[SearchResult] = field(default_factory=list)

    def to_corpus_text(self) -> str:
        """Flatten every gathered result into one text corpus for FAISS embedding."""
        blocks: List[str] = []
        for result in self.general_info:
            blocks.append(result.to_text())
        for retailer, results in self.prices_by_retailer.items():
            for result in results:
                result.source = retailer
                blocks.append(result.to_text())
        for result in self.reviews:
            blocks.append(result.to_text())
        for result in self.similar_products:
            blocks.append(result.to_text())
        return "\n---\n".join(blocks)


class SearchService:
    """Wrapper around the Tavily Search API for all real-time product research."""

    def __init__(self, max_results: int = MAX_SEARCH_RESULTS) -> None:
        # Removed _ensure_api_key from here to prevent import-time exceptions
        self.max_results = max_results
        self._cache: Dict[Tuple[str, Tuple[str, ...], int], List[SearchResult]] = {}

    def _ensure_api_key(self) -> None:
        """Fail fast only when a search is actively performed."""
        api_key = get_tavily_api_key()
        if not api_key or api_key == "your_tavily_api_key_here":
            raise SearchServiceError(
                "TAVILY_API_KEY is missing or unset. Add a valid key to "
                "your Streamlit Secrets or .env file before using SearchService."
            )

    def _cache_key(
        self, query: str, max_results: int, include_domains: Optional[List[str]]
    ) -> Tuple[str, Tuple[str, ...], int]:
        domains_key = tuple(sorted(include_domains)) if include_domains else tuple()
        return (query.strip().lower(), domains_key, max_results)

    def _search(
        self,
        query: str,
        max_results: Optional[int] = None,
        include_domains: Optional[List[str]] = None,
        topic: str = "general",
    ) -> List[SearchResult]:
        self._ensure_api_key()
        api_key = get_tavily_api_key()
        
        effective_max = max_results or self.max_results
        cache_key = self._cache_key(query, effective_max, include_domains)

        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit for query: %s", query)
            return cached

        payload = {
            "api_key": api_key,
            "query": query,
            "topic": topic,
            "search_depth": "basic",
            "max_results": effective_max,
        }

        if include_domains:
            payload["include_domains"] = include_domains

        try:
            response = requests.post(
                "[https://api.tavily.com/search](https://api.tavily.com/search)",
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        except requests.exceptions.Timeout:
            raise SearchServiceError("Tavily timeout. Please try again.")
        except Exception as e:
            raise SearchServiceError(str(e))

        results = []
        for item in data.get("results", []):
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", ""),
                    score=item.get("score", 0),
                )
            )

        results = filter_accessory_results(results)

        if len(self._cache) >= MAX_CACHE_ENTRIES:
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = results

        return results

    def search_prices_by_retailer(self, product_query: str) -> Dict[str, List[SearchResult]]:
        prices: Dict[str, List[SearchResult]] = {}

        def _search_retailer(retailer: str, domains: List[str]) -> Tuple[str, List[SearchResult]]:
            try:
                results = self._search(product_query, max_results=3, include_domains=domains)
                for r in results:
                    r.source = retailer
                return retailer, results
            except SearchServiceError as exc:
                logger.warning("Price search failed for %s: %s", retailer, exc)
                return retailer, []

        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(RETAILER_DOMAINS))) as executor:
            futures = [executor.submit(_search_retailer, retailer, domains) for retailer, domains in RETAILER_DOMAINS.items()]
            for future in as_completed(futures):
                retailer, results = future.result()
                prices[retailer] = results

        return prices

    def search_reviews(self, product_query: str) -> List[SearchResult]:
        queries = [
            f"{product_query} customer reviews",
            f"{product_query} pros cons",
            f"{product_query} expert review",
        ]

        all_results: List[SearchResult] = []
        seen_urls = set()

        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(queries))) as executor:
            futures = [executor.submit(self._search, q, 4) for q in queries]
            for future in as_completed(futures):
                try:
                    results = future.result()
                except SearchServiceError as exc:
                    logger.warning("Review search failed: %s", exc)
                    continue
                for r in results:
                    if r.url and r.url in seen_urls:
                        continue
                    if r.url:
                        seen_urls.add(r.url)
                    r.source = "reviews"
                    all_results.append(r)

        return all_results

    def search_similar_products(self, product_query: str, category: Optional[str] = None) -> List[SearchResult]:
        category_hint = f" {category}" if category else ""
        query = f"best alternatives to {product_query}{category_hint} 2026"
        results = self._search(query)
        for r in results:
            r.source = "similar_products"
        return results

    def search_similar_products_by_suggestions(self, suggestions: List[Dict[str, str]], category: Optional[str] = None) -> List[SearchResult]:
        if not suggestions:
            return []

        def _search_one(item: Dict[str, str]) -> Optional[SearchResult]:
            name = (item.get("name") or "").strip()
            if not name:
                return None
            brand = (item.get("brand") or "").strip()
            query = build_manual_search_query(brand, name, category)

            try:
                results = self._search(query, max_results=3)
            except SearchServiceError as exc:
                logger.warning("Similar product search failed for %s: %s", name, exc)
                return None

            if not results:
                return None

            top = results[0]
            top.source = "similar_products"
            if not top.title:
                top.title = name
            return top

        found: List[SearchResult] = []
        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(suggestions))) as executor:
            futures = [executor.submit(_search_one, item) for item in suggestions]
            for future in as_completed(futures):
                result = future.result()
                if result:
                    found.append(result)

        return found

    def search_budget_alternatives(self, category: str, max_price_inr: int) -> List[SearchResult]:
        query = f"best {category} under {max_price_inr} rupees 2026"
        results = self._search(query)
        for r in results:
            r.source = "budget_alternatives"
        return results

    def search_product_details(self, product_query: str) -> List[SearchResult]:
        query = f"{product_query} specifications features price"
        results = self._search(query=query, max_results=5)
        for r in results:
            r.source = "general"
        return results

    def build_product_bundle(self, product_query: str, category: Optional[str] = None) -> ProductSearchBundle:
        logger.info("Building product search bundle for: %s", product_query)
        bundle = ProductSearchBundle(product_name=product_query)

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_general = executor.submit(self.search_product_details, product_query)
            future_prices = executor.submit(self.search_prices_by_retailer, product_query)
            future_reviews = executor.submit(self.search_reviews, product_query)

            bundle.general_info = future_general.result()
            bundle.prices_by_retailer = future_prices.result()
            bundle.reviews = future_reviews.result()

        bundle.similar_products = self.search_similar_products(product_query, category)
        return bundle
