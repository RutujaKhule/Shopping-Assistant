"""
services/comparison.py
=========================================================
Comparison & Recommendation Service
=========================================================
Implements:

  FEATURE 2 - Real-time price comparison across retailers, with
              cheapest / best-rated / fastest-delivery / best-value
              highlighting and savings calculation.

  FEATURE 3 - AI buying recommendation: Gemini analyzes price, rating,
              specs, and reviews across retailers to produce a single
              "best buy" verdict.

  FEATURE 6 - Product-vs-product comparison (specs, pros/cons, winner).

Design note: raw Tavily search results are unstructured text snippets,
not clean structured data (there is no product database here — see
project requirements). So this module uses Gemini itself as an
extraction step: it reads each retailer's search snippet and pulls out
structured fields (price, rating, delivery, availability) as JSON.
This keeps the "no static/hardcoded data" requirement intact while
still producing a clean comparison table.
"""

import logging
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Any

from services.gemini import GeminiService, GeminiServiceError
from services.search import SearchResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ComparisonServiceError(Exception):
    """Raised when price extraction or comparison generation fails."""
    pass


@dataclass
class PriceInfo:
    """Structured, per-retailer pricing data extracted from search results."""
    retailer: str
    available: bool
    price: Optional[float] = None
    currency: str = "INR"
    discount_percent: Optional[float] = None
    delivery_charge: Optional[float] = None
    delivery_days_estimate: Optional[float] = None
    delivery_time_text: Optional[str] = None
    rating: Optional[float] = None
    product_title: Optional[str] = None
    url: Optional[str] = None
    notes: Optional[str] = None

    # Computed later relative to the cheapest option
    price_difference_from_cheapest: Optional[float] = None
    savings_vs_most_expensive: Optional[float] = None

    # Badges filled in by highlight_best_options()
    is_cheapest: bool = False
    is_best_rated: bool = False
    is_fastest_delivery: bool = False
    is_best_value: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ComparisonService:
    """
    Usage:
        comparator = ComparisonService(gemini_service)
        price_table = comparator.build_price_comparison_table(bundle.prices_by_retailer)
        price_table = comparator.highlight_best_options(price_table)
        recommendation = comparator.generate_buying_recommendation(product_name, price_table)
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self.gemini = gemini_service

    # -----------------------------------------------------
    # FEATURE 2: Extract structured price info per retailer
    # -----------------------------------------------------

    def extract_price_info(self, retailer: str, results: List[SearchResult]) -> PriceInfo:
        """
        Use Gemini to read a retailer's raw search snippet(s) and extract
        structured pricing/availability/rating data.

        Args:
            retailer: Retailer display name (e.g. "Flipkart").
            results: Search results for that retailer (usually 1-3).

        Returns:
            A PriceInfo object. If no results were found or extraction
            fails, `available` is set to False rather than raising, so a
            single retailer's failure never breaks the whole comparison.
        """
        if not results:
            return PriceInfo(retailer=retailer, available=False, notes="No listing found.")

        combined_snippets = "\n\n".join(
            f"Title: {r.title}\nURL: {r.url}\nContent: {r.content}" for r in results
        )

        prompt = f"""
You are extracting structured product listing data from raw web search
snippets from {retailer}. Read the snippets below and extract what you
can. Respond with ONLY a valid JSON object (no markdown, no commentary)
using exactly this schema:

{{
  "available": true or false (false only if the snippets clearly show the product is out of stock or not sold here),
  "price": number or null (numeric price in INR, no currency symbols or commas),
  "discount_percent": number or null,
  "delivery_charge": number or null (0 if free delivery is mentioned),
  "delivery_days_estimate": number or null (estimated days as a plain number, e.g. 2),
  "delivery_time_text": "string or null, the raw delivery time phrase if mentioned",
  "rating": number or null (out of 5),
  "product_title": "string or null, the exact listed product title",
  "url": "string or null, the most relevant product page URL from the snippets",
  "notes": "string or null, anything else notable (e.g. limited stock, exchange offer)"
}}

If a field cannot be determined from the snippets, use null. Do not
guess prices — only extract values that are actually present in the
text.

SNIPPETS FROM {retailer}:
---
{combined_snippets}
---
"""
        try:
            raw = self.gemini.generate_text(prompt, temperature=0.0)
            data = GeminiService.parse_json_response(raw)  # reuse tolerant JSON parsing
        except (GeminiServiceError, Exception) as exc:  # noqa: BLE001
            logger.warning("Price extraction failed for %s: %s", retailer, exc)
            return PriceInfo(
                retailer=retailer,
                available=False,
                notes="Could not extract structured pricing data.",
            )

        return PriceInfo(
            retailer=retailer,
            available=bool(data.get("available", True)),
            price=data.get("price"),
            discount_percent=data.get("discount_percent"),
            delivery_charge=data.get("delivery_charge"),
            delivery_days_estimate=data.get("delivery_days_estimate"),
            delivery_time_text=data.get("delivery_time_text"),
            rating=data.get("rating"),
            product_title=data.get("product_title"),
            url=data.get("url") or (results[0].url if results else None),
            notes=data.get("notes"),
        )

    def build_price_comparison_table(
        self, prices_by_retailer: Dict[str, List[SearchResult]]
    ) -> List[PriceInfo]:
        """
        Build the full price comparison table (Feature 2) from every
        retailer's raw search results.
        """
        table: List[PriceInfo] = []
        for retailer, results in prices_by_retailer.items():
            table.append(self.extract_price_info(retailer, results))
        return table

    # -----------------------------------------------------
    # FEATURE 2: Highlighting + savings math
    # -----------------------------------------------------

    def highlight_best_options(self, price_table: List[PriceInfo]) -> List[PriceInfo]:
        """
        Mark cheapest / best-rated / fastest-delivery / best-value entries,
        and compute price differences and savings relative to the
        cheapest and most expensive available listings.

        Mutates and returns the same list for convenience.
        """
        available = [p for p in price_table if p.available and p.price is not None]
        if not available:
            logger.warning("No available priced listings to highlight.")
            return price_table

        cheapest = min(available, key=lambda p: p.price)
        most_expensive = max(available, key=lambda p: p.price)
        cheapest.is_cheapest = True

        rated = [p for p in available if p.rating is not None]
        if rated:
            best_rated = max(rated, key=lambda p: p.rating)
            best_rated.is_best_rated = True

        with_delivery = [p for p in available if p.delivery_days_estimate is not None]
        if with_delivery:
            fastest = min(with_delivery, key=lambda p: p.delivery_days_estimate)
            fastest.is_fastest_delivery = True

        # "Best value" heuristic: normalize price (lower is better) and
        # rating (higher is better) into one score. This is a simple,
        # transparent heuristic — the AI recommendation (Feature 3) gives
        # a richer, reasoned verdict on top of this.
        price_range = max(p.price for p in available) - min(p.price for p in available)
        for p in available:
            price_score = (
                1.0
                if price_range == 0
                else 1.0 - ((p.price - cheapest.price) / price_range)
            )
            rating_score = (p.rating / 5.0) if p.rating is not None else 0.5
            p_value_score = (0.5 * price_score) + (0.5 * rating_score)
            p.notes = (p.notes or "") + f" [value_score={p_value_score:.2f}]"

        best_value = max(
            available,
            key=lambda p: float((p.notes or "0").split("value_score=")[-1].rstrip("]")),
        )
        best_value.is_best_value = True

        # Price difference / savings relative to the cheapest option.
        for p in price_table:
            if p.available and p.price is not None:
                p.price_difference_from_cheapest = round(p.price - cheapest.price, 2)
                p.savings_vs_most_expensive = round(most_expensive.price - p.price, 2)

        return price_table

    # -----------------------------------------------------
    # FEATURE 3: AI Buying Recommendation
    # -----------------------------------------------------

    def generate_buying_recommendation(
        self,
        product_name: str,
        price_table: List[PriceInfo],
        review_summary_text: Optional[str] = None,
    ) -> str:
        """
        Ask Gemini to analyze price, rating, delivery, and (optionally)
        review sentiment across retailers and produce a single clear
        "best buy" recommendation.

        Args:
            product_name: The identified product's name.
            price_table: Output of build_price_comparison_table() +
                highlight_best_options().
            review_summary_text: Optional pre-generated review summary
                (from services/reviews.py) to factor into the verdict.

        Returns:
            A short, human-readable recommendation string.
        """
        table_lines = []
        for p in price_table:
            if not p.available:
                table_lines.append(f"- {p.retailer}: Not available.")
                continue
            table_lines.append(
                f"- {p.retailer}: price=₹{p.price}, rating={p.rating}, "
                f"discount={p.discount_percent}%, delivery={p.delivery_time_text or p.delivery_days_estimate}, "
                f"cheapest={p.is_cheapest}, best_rated={p.is_best_rated}, "
                f"fastest_delivery={p.is_fastest_delivery}, best_value={p.is_best_value}"
            )
        table_text = "\n".join(table_lines)

        review_block = f"\nREVIEW SUMMARY:\n{review_summary_text}\n" if review_summary_text else ""

        prompt = f"""
You are an AI shopping assistant. Analyze the retailer data below for
"{product_name}" and give a short, decisive buying recommendation (3-5
sentences). Mention which retailer offers the best overall value and
why, and note any meaningful tradeoffs (e.g. faster delivery elsewhere,
better warranty, higher rating). Be specific and concrete, not generic.

RETAILER DATA:
{table_text}
{review_block}
"""
        try:
            return self.gemini.generate_text(prompt, temperature=0.3)
        except GeminiServiceError as exc:
            logger.warning("Buying recommendation generation failed: %s", exc)
            return (
                "Unable to generate an AI recommendation right now. "
                "Please refer to the price comparison table above."
            )

    # -----------------------------------------------------
    # FEATURE 6: Product-vs-product comparison
    # -----------------------------------------------------

    def compare_products(
        self,
        product_a_name: str,
        product_a_context: str,
        product_b_name: str,
        product_b_context: str,
    ) -> str:
        """
        Generate a structured comparison between two products, covering
        specs, performance, battery, display, camera, storage, RAM,
        processor, pros/cons, and a final winner verdict.

        Args:
            product_a_name / product_b_name: Display names of the two products.
            product_a_context / product_b_context: Retrieved text context
                for each product (e.g. from RAGPipeline.retrieve() +
                build_context(), or a fresh SearchService bundle for the
                second product).

        Returns:
            A markdown-formatted comparison as a string, ready to render
            directly in the Streamlit UI.
        """
        prompt = f"""
Compare these two products in detail using ONLY the information given
below. Use this markdown structure exactly:

## {product_a_name} vs {product_b_name}

| Aspect | {product_a_name} | {product_b_name} |
|---|---|---|
| Price | ... | ... |
| Display | ... | ... |
| Processor | ... | ... |
| RAM | ... | ... |
| Storage | ... | ... |
| Battery | ... | ... |
| Camera | ... | ... |
| Rating | ... | ... |

**Pros of {product_a_name}:** ...
**Cons of {product_a_name}:** ...

**Pros of {product_b_name}:** ...
**Cons of {product_b_name}:** ...

**Winner:** State which product wins overall and one sentence why. If
it depends on use case, say so explicitly (e.g. "best for gaming" vs
"best for battery life").

If a field is not available in the context for either product, write
"Not available" for that cell rather than guessing.

CONTEXT FOR {product_a_name}:
---
{product_a_context}
---

CONTEXT FOR {product_b_name}:
---
{product_b_context}
---
"""
        try:
            return self.gemini.generate_text(prompt, temperature=0.3)
        except GeminiServiceError as exc:
            logger.warning("Product comparison generation failed: %s", exc)
            raise ComparisonServiceError(f"Comparison generation failed: {exc}") from exc
