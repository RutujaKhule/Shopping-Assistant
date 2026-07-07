"""
services/search.py
=========================================================
Real-Time Search Service (Tavily)
=========================================================
This module is the app's only source of "ground truth" product data.
Per project requirements, NOTHING here is static or hardcoded — every
spec, price, and review snippet is fetched live from the web via the
Tavily Search API.

Responsibilities:
1. General product research      -> specs, features, availability
2. Retailer-specific price search -> Amazon India, Flipkart, Croma,
                                      Reliance Digital, Vijay Sales
3. Review discovery               -> real customer opinions from the web
4. Similar / budget alternative discovery

Quality & performance improvements:
- Every result returned by `_search()` is filtered through
  `utils.filter_accessory_results()`, so accessory listings (covers,
  cases, chargers, cables, etc.) never leak into the app, no matter
  which higher-level method is calling it.
- Retailer price search stays domain-restricted via `include_domains`
  (more reliable than a bare "site:" query string) and now uses a
  short, product-first query instead of noisy filler words.
- Independent Tavily calls (general info / retailer prices / reviews,
  and each retailer within the price search) are run in parallel with
  a thread pool, since they don't depend on each other.
- Identical queries are cached in-memory for the lifetime of this
  service instance, so repeated chat questions don't re-hit Tavily for
  data we already fetched.

All results are returned as `SearchResult` objects and can be flattened
into plain text via `to_corpus_text()` for embedding in
services/embeddings.py + rag/vector_store.py.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv
import requests

from utils import build_manual_search_query, filter_accessory_results

# -------------------------------------------------------
# Setup
# -------------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "8"))

# Retailers we check for real-time price comparison (Feature 2).
# Mapped to the domains Tavily should restrict its search to, so results
# are guaranteed to come from the retailer itself rather than aggregators.
# We rely on `include_domains` (rather than a "site:" string in the query
# text) because Tavily's own domain filter is the reliable way to pin
# results to a specific retailer - "site:" operators are a search-engine
# convention that Tavily's query text does not consistently honor.
RETAILER_DOMAINS: Dict[str, List[str]] = {
    "Amazon India": ["amazon.in"],
    "Flipkart": ["flipkart.com"],
    "Croma": ["croma.com"],
    "Reliance Digital": ["reliancedigital.in"],
    "Vijay Sales": ["vijaysales.com"],
}

# Cap on parallel worker threads for any batch of Tavily calls.
MAX_PARALLEL_WORKERS = 5

# Cap on how many (query -> results) pairs we keep in the in-memory cache.
MAX_CACHE_ENTRIES = 256


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
    source: str = "web"  # e.g. "Flipkart", "reviews", "general"

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
    """
    All real-time data gathered for one product in one place. This is
    what gets handed off to embeddings/FAISS (Feature 8) and to the
    comparison/reviews services.
    """
    product_name: str
    general_info: List[SearchResult] = field(default_factory=list)
    prices_by_retailer: Dict[str, List[SearchResult]] = field(default_factory=dict)
    reviews: List[SearchResult] = field(default_factory=list)
    similar_products: List[SearchResult] = field(default_factory=list)

    def to_corpus_text(self) -> str:
        """
        Flatten every gathered result into one text corpus, ready to be
        chunked and embedded for FAISS retrieval.
        """
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
    """
    Wrapper around the Tavily Search API for all real-time product
    research needs of the app.

    Usage:
        search = SearchService()
        bundle = search.build_product_bundle("Apple iPhone 15 Smartphone")
    """

    def __init__(self, max_results: int = MAX_SEARCH_RESULTS) -> None:
        if not TAVILY_API_KEY or TAVILY_API_KEY == "your_tavily_api_key_here":
            raise SearchServiceError(
                "TAVILY_API_KEY is missing or unset. Add a valid key to "
                "your .env file (TAVILY_API_KEY=...) before using SearchService."
            )
        self.max_results = max_results

        # Simple in-memory cache: (query, sorted include_domains, max_results) -> results.
        # Keeps the app snappy when the same product/question is searched
        # more than once in a session, without needing an external cache.
        self._cache: Dict[Tuple[str, Tuple[str, ...], int], List[SearchResult]] = {}

    # -----------------------------------------------------
    # Low-level search primitive
    # -----------------------------------------------------

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
        effective_max = max_results or self.max_results
        cache_key = self._cache_key(query, effective_max, include_domains)

        cached = self._cache.get(cache_key)
        if cached is not None:
            logger.info("Cache hit for query: %s", query)
            return cached

        payload = {
            "api_key": TAVILY_API_KEY,
            "query": query,
            "topic": topic,
            "search_depth": "basic",
            "max_results": effective_max,
        }

        if include_domains:
            payload["include_domains"] = include_domains

        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=15,
            )

            response.raise_for_status()

            data = response.json()

        except requests.exceptions.Timeout:
            raise SearchServiceError(
                "Tavily timeout. Please try again."
            )

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

        # Global accessory filter: no matter which higher-level method
        # called us, accessory listings (covers, cases, chargers, cables,
        # etc.) never make it into the app's data.
        results = filter_accessory_results(results)

        if len(self._cache) >= MAX_CACHE_ENTRIES:
            # Cheap eviction: drop an arbitrary (oldest-inserted, in
            # practice) entry rather than pulling in a full LRU dependency.
            self._cache.pop(next(iter(self._cache)))
        self._cache[cache_key] = results

        return results

    # -----------------------------------------------------
    # FEATURE 2: Real-time price comparison across retailers
    # -----------------------------------------------------

    def search_prices_by_retailer(
        self, product_query: str
    ) -> Dict[str, List[SearchResult]]:
        """
        Search each configured retailer individually so price/availability
        data is traceable back to a specific website (needed for the price
        comparison table and "best deal" highlighting).

        Retailers are queried in parallel (they're independent HTTP calls),
        and the query itself is the concise product query as-is - no
        "price buy online" filler - since `include_domains` already pins
        each search to that retailer's own listing pages.

        Returns:
            Dict mapping retailer name -> list of SearchResult (usually 1-3
            per retailer, since we only need the product listing page(s)).
        """
        prices: Dict[str, List[SearchResult]] = {}

        def _search_retailer(retailer: str, domains: List[str]) -> Tuple[str, List[SearchResult]]:
            try:
                results = self._search(
                    product_query, max_results=3, include_domains=domains
                )
                for r in results:
                    r.source = retailer
                return retailer, results
            except SearchServiceError as exc:
                # Don't let one retailer's failure break the whole comparison;
                # log it and continue with an empty list for that retailer.
                logger.warning("Price search failed for %s: %s", retailer, exc)
                return retailer, []

        with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_WORKERS, len(RETAILER_DOMAINS))) as executor:
            futures = [
                executor.submit(_search_retailer, retailer, domains)
                for retailer, domains in RETAILER_DOMAINS.items()
            ]
            for future in as_completed(futures):
                retailer, results = future.result()
                prices[retailer] = results

        return prices

    # -----------------------------------------------------
    # FEATURE 7: Review discovery
    # -----------------------------------------------------

    def search_reviews(self, product_query: str) -> List[SearchResult]:
        """
        Search for real customer reviews and expert opinions across the
        web (not limited to retailer sites, since review/tech blogs often
        have richer commentary).

        Runs three complementary, concise queries in parallel and merges
        the de-duplicated results, instead of relying on a single broad
        query:
            "<product> customer reviews"
            "<product> pros cons"
            "<product> expert review"
        """
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

    # -----------------------------------------------------
    # FEATURE 4 & 5: Similar products / budget alternatives
    # -----------------------------------------------------

    def search_similar_products(
        self, product_query: str, category: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Legacy similar-products search, kept as a safety-net fallback for
        callers that don't have Gemini-generated suggestions on hand
        (e.g. if suggest_similar_products() fails). Prefer
        search_similar_products_by_suggestions() when possible - it
        produces much more relevant, less noisy results.
        """
        category_hint = f" {category}" if category else ""
        query = f"best alternatives to {product_query}{category_hint} 2026"
        results = self._search(query)
        for r in results:
            r.source = "similar_products"
        return results

    def search_similar_products_by_suggestions(
        self,
        suggestions: List[Dict[str, str]],
        category: Optional[str] = None,
    ) -> List[SearchResult]:
        """
        FEATURE 5 (improved): search the live web for each Gemini-suggested
        similar product individually, rather than relying on a single
        vague "best alternatives to X" query.

        Each suggestion is turned into a concise, manually-built query
        (Brand + Product Name + Category) and searched independently, in
        parallel. Accessory results are already filtered out by
        `_search()`. The single best (first, highest-scored) result per
        suggested product is kept, so the "Similar Products" row shows one
        card per real alternative product rather than a pile of
        unrelated pages.

        Args:
            suggestions: Output of GeminiService.suggest_similar_products(),
                i.e. a list of {"name": ..., "brand": ...} dicts.
            category: The original product's category, reused for each
                suggestion's query for relevance.

        Returns:
            A list of SearchResult, one per suggested product that
            returned a usable, non-accessory listing. May be shorter than
            `suggestions` if some searches failed or returned nothing.
        """
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

    def search_budget_alternatives(
        self, category: str, max_price_inr: int
    ) -> List[SearchResult]:
        """
        Search for budget-friendly alternatives under a given price cap.

        Example: search_budget_alternatives("laptops", 60000)
                 -> "best laptops under 60000 rupees 2026"
        """
        query = f"best {category} under {max_price_inr} rupees 2026"
        results = self._search(query)
        for r in results:
            r.source = "budget_alternatives"
        return results

    # -----------------------------------------------------
    # High-level orchestrator: build everything at once
    # -----------------------------------------------------

    # FEATURE 1: General Product Search

    def search_product_details(self, product_query: str) -> List[SearchResult]:
        """
        Search product specifications, features, price and overview
        using a concise, product-first query (no filler adjectives like
        "latest original review" - see project requirement on search
        quality).
        """

        query = f"{product_query} specifications features price"

        results = self._search(
            query=query,
            max_results=5,
        )

        for r in results:
            r.source = "general"

        return results

    def build_product_bundle(
        self, product_query: str, category: Optional[str] = None
    ) -> ProductSearchBundle:
        """
        Run all searches needed to fully populate the app for a given
        identified product: general specs, retailer prices, reviews, and
        a legacy similar-products fallback. This is the single entry
        point app.py should call right after Gemini Vision identifies
        the product and app.py builds the manual search query.

        General info, retailer prices, and reviews are independent
        Tavily calls, so they run in parallel via a thread pool rather
        than sequentially - this is the main latency win for the
        "Optimize Performance" requirement.

        NOTE: `similar_products` here is populated with the legacy,
        lower-quality search as a safe default. app.py immediately
        overwrites it with the higher-quality, Gemini-suggestion-driven
        results from search_similar_products_by_suggestions() when that
        succeeds - this method's own similar-products call is only a
        fallback if that improved flow fails.

        Args:
            product_query: The manually-built search query (Brand +
                Product Name + Category - see utils.build_manual_search_query()),
                NOT Gemini's raw `search_query` field.
            category: Optional product category, improves similar-product
                search relevance.

        Returns:
            A fully populated ProductSearchBundle.
        """
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