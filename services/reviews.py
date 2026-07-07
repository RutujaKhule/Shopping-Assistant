"""
services/reviews.py
=========================================================
Review Summary Service
=========================================================
Implements FEATURE 7: AI Review Summary.

Takes the raw review snippets gathered by SearchService.search_reviews()
(real customer/expert opinions scraped live from the web — no static
data) and asks Gemini to distill them into a structured summary:
overall rating, pros, cons, common complaints, common positive
feedback, and a final verdict.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from services.gemini import GeminiService, GeminiServiceError
from services.search import SearchResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class ReviewsServiceError(Exception):
    """Raised when review summarization fails."""
    pass


@dataclass
class ReviewSummary:
    """Structured summary of a product's aggregated online reviews."""
    overall_rating: Optional[float] = None
    pros: List[str] = field(default_factory=list)
    cons: List[str] = field(default_factory=list)
    common_complaints: List[str] = field(default_factory=list)
    common_positive_feedback: List[str] = field(default_factory=list)
    recommended_for: Optional[str] = None
    final_verdict: Optional[str] = None
    source_count: int = 0
    has_data: bool = True

    def to_markdown(self) -> str:
        """
        Render the summary as clean markdown for direct display in the
        Streamlit UI (see ui/components.py).
        """
        if not self.has_data:
            return "_No customer reviews were found for this product yet._"

        lines: List[str] = []

        if self.overall_rating is not None:
            lines.append(f"**Overall Rating:** {self.overall_rating:.1f} / 5")

        if self.pros:
            lines.append("\n**Pros:**")
            lines.extend(f"- {item}" for item in self.pros)

        if self.cons:
            lines.append("\n**Cons:**")
            lines.extend(f"- {item}" for item in self.cons)

        if self.common_positive_feedback:
            lines.append("\n**Common Positive Feedback:**")
            lines.extend(f"- {item}" for item in self.common_positive_feedback)

        if self.common_complaints:
            lines.append("\n**Common Complaints:**")
            lines.extend(f"- {item}" for item in self.common_complaints)

        if self.recommended_for:
            lines.append(f"\n**Recommended for:** {self.recommended_for}")

        if self.final_verdict:
            lines.append(f"\n**Final Verdict:** {self.final_verdict}")

        lines.append(f"\n_Based on {self.source_count} online source(s)._")

        return "\n".join(lines)


class ReviewsService:
    """
    Usage:
        reviews_service = ReviewsService(gemini_service)
        summary = reviews_service.summarize_reviews(product_name, bundle.reviews)
        st.markdown(summary.to_markdown())
    """

    def __init__(self, gemini_service: GeminiService) -> None:
        self.gemini = gemini_service

    def summarize_reviews(
        self, product_name: str, review_results: List[SearchResult]
    ) -> ReviewSummary:
        """
        Summarize a list of raw review search results into a structured
        ReviewSummary using Gemini.

        Args:
            product_name: The identified product's name (for prompt context).
            review_results: Raw review snippets from SearchService.search_reviews().

        Returns:
            A ReviewSummary. If no review results were found, returns a
            summary with has_data=False rather than calling Gemini.
        """
        if not review_results:
            logger.warning("No review results to summarize for %s", product_name)
            return ReviewSummary(has_data=False, source_count=0)

        combined_snippets = "\n\n".join(
            f"Source: {r.url}\nTitle: {r.title}\nContent: {r.content}"
            for r in review_results
        )

        prompt = f"""
You are analyzing real customer and expert reviews for "{product_name}"
gathered from the web. Read the review snippets below and produce a
concise, honest summary. Respond with ONLY a valid JSON object (no
markdown, no commentary) using exactly this schema:

{{
  "overall_rating": number or null (out of 5, your best estimate based on sentiment and any stated ratings),
  "pros": ["short phrase", "short phrase", ...] (3-5 items max),
  "cons": ["short phrase", "short phrase", ...] (3-5 items max),
  "common_complaints": ["short phrase", ...] (specific recurring issues mentioned by multiple reviewers, 2-4 items),
  "common_positive_feedback": ["short phrase", ...] (specific recurring praise, 2-4 items),
  "recommended_for": "string, a short phrase on who this product suits best (e.g. 'students and office users'), or null if unclear",
  "final_verdict": "string, one or two sentence honest overall verdict"
}}

Base every item strictly on the snippets provided. Do not invent
specific claims that aren't supported by the text. If the snippets are
too sparse to determine a field confidently, use null for that field
(or an empty list for list fields) rather than guessing.

REVIEW SNIPPETS:
---
{combined_snippets}
---
"""
        try:
            raw = self.gemini.generate_text(prompt, temperature=0.2)
            data = GeminiService.parse_json_response(raw)
        except GeminiServiceError as exc:
            logger.warning("Review summarization failed for %s: %s", product_name, exc)
            raise ReviewsServiceError(f"Review summarization failed: {exc}") from exc

        return ReviewSummary(
            overall_rating=data.get("overall_rating"),
            pros=data.get("pros") or [],
            cons=data.get("cons") or [],
            common_complaints=data.get("common_complaints") or [],
            common_positive_feedback=data.get("common_positive_feedback") or [],
            recommended_for=data.get("recommended_for"),
            final_verdict=data.get("final_verdict"),
            source_count=len(review_results),
            has_data=True,
        )
