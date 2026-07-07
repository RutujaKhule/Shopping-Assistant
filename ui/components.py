"""
ui/components.py
=========================================================
Streamlit UI Components
=========================================================
Implements FEATURE 10: the dashboard's visual building blocks.

Design direction: a "digital receipt / price-tag" aesthetic — deep
charcoal-navy panels, an amber accent for prices and highlights, and a
teal accent for positive/best-value signals — rather than a generic
default Streamlit look. Space Grotesk is used for headings (a display
face with a slightly technical, price-tag character) paired with
Inter for body text.

Every function here is a pure render function: it takes plain data
(dicts, dataclasses, lists) produced by the services/ layer and draws
it. No service calls happen in this module — that keeps app.py as the
only orchestration point, and these components easily testable/reusable.
"""

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from services.comparison import PriceInfo
from services.reviews import ReviewSummary
from services.search import SearchResult

# -------------------------------------------------------
# Design tokens
# -------------------------------------------------------

COLOR_BG = "#12181F"
COLOR_PANEL = "#1B2430"
COLOR_PANEL_ALT = "#212B39"
COLOR_TEXT = "#E9EDF2"
COLOR_MUTED = "#8B98A9"
COLOR_AMBER = "#F2B134"   # price / highlight accent
COLOR_TEAL = "#2EC4B6"    # best value / positive accent
COLOR_CORAL = "#EF6461"   # cons / warnings accent
COLOR_BORDER = "#2C3847"


def inject_custom_css() -> None:
    """
    Inject the app's custom theme. Call this once, near the top of app.py,
    right after st.set_page_config().
    """
    st.markdown(
        f"""
        <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
        <style>
            html, body, [class*="css"] {{
                font-family: 'Inter', sans-serif;
                color: {COLOR_TEXT};
            }}
            .stApp {{
                background-color: {COLOR_BG};
            }}
            h1, h2, h3, h4 {{
                font-family: 'Space Grotesk', sans-serif !important;
                letter-spacing: -0.01em;
            }}
            /* Sidebar */
            section[data-testid="stSidebar"] {{
                background-color: {COLOR_PANEL};
                border-right: 1px solid {COLOR_BORDER};
            }}
            /* Card container used throughout the dashboard */
            .sa-card {{
                background-color: {COLOR_PANEL};
                border: 1px solid {COLOR_BORDER};
                border-radius: 14px;
                padding: 1.1rem 1.3rem;
                margin-bottom: 1rem;
            }}
            .sa-card-alt {{
                background-color: {COLOR_PANEL_ALT};
                border: 1px solid {COLOR_BORDER};
                border-radius: 14px;
                padding: 1.1rem 1.3rem;
                margin-bottom: 1rem;
            }}
            .sa-eyebrow {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 0.72rem;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: {COLOR_AMBER};
                margin-bottom: 0.3rem;
            }}
            .sa-price {{
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.4rem;
                font-weight: 700;
                color: {COLOR_AMBER};
            }}
            .sa-badge {{
                display: inline-block;
                font-size: 0.68rem;
                font-weight: 600;
                letter-spacing: 0.03em;
                padding: 0.15rem 0.55rem;
                border-radius: 999px;
                margin-right: 0.3rem;
                margin-bottom: 0.2rem;
            }}
            .sa-badge-teal {{
                background-color: rgba(46, 196, 182, 0.15);
                color: {COLOR_TEAL};
                border: 1px solid rgba(46, 196, 182, 0.4);
            }}
            .sa-badge-amber {{
                background-color: rgba(242, 177, 52, 0.15);
                color: {COLOR_AMBER};
                border: 1px solid rgba(242, 177, 52, 0.4);
            }}
            .sa-badge-coral {{
                background-color: rgba(239, 100, 97, 0.15);
                color: {COLOR_CORAL};
                border: 1px solid rgba(239, 100, 97, 0.4);
            }}
            .sa-muted {{
                color: {COLOR_MUTED};
                font-size: 0.85rem;
            }}
            .sa-divider {{
                border-top: 1px solid {COLOR_BORDER};
                margin: 0.8rem 0;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------
# Sidebar
# -------------------------------------------------------

def render_sidebar(chat_turns: List[Tuple[str, str]]) -> Dict[str, Any]:
    """
    Render the sidebar: image upload, settings, and chat history.

    Args:
        chat_turns: List of (question, answer) tuples from ConversationMemory.

    Returns:
        A dict with keys:
            "uploaded_file": the UploadedFile object or None
            "top_k": int, number of chunks to retrieve per query
            "reset_clicked": bool, whether the "Start Over" button was pressed
    """
    with st.sidebar:
        st.markdown("### 🛍️ Shopping Assistant")
        st.markdown(
            '<div class="sa-muted">Upload a product photo to get real-time '
            "prices, reviews, and AI buying advice.</div>",
            unsafe_allow_html=True,
        )
        st.markdown("<div class='sa-divider'></div>", unsafe_allow_html=True)

        st.markdown("#### Upload Image")
        uploaded_file = st.file_uploader(
            "Product photo",
            type=["png", "jpg", "jpeg", "webp"],
            label_visibility="collapsed",
        )

        st.markdown("<div class='sa-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Settings")
        top_k = st.slider(
            "Retrieval depth (context chunks)",
            min_value=2,
            max_value=8,
            value=4,
            help="How many retrieved chunks Gemini sees per answer. Higher = more context, slower.",
        )

        reset_clicked = st.button("🔄 Start Over (new product)", use_container_width=True)

        st.markdown("<div class='sa-divider'></div>", unsafe_allow_html=True)
        st.markdown("#### Chat History")
        if not chat_turns:
            st.markdown(
                '<div class="sa-muted">No questions asked yet.</div>',
                unsafe_allow_html=True,
            )
        else:
            for question, _ in reversed(chat_turns[-10:]):
                st.markdown(f"<div class='sa-muted'>💬 {question}</div>", unsafe_allow_html=True)

    return {
        "uploaded_file": uploaded_file,
        "top_k": top_k,
        "reset_clicked": reset_clicked,
    }


# -------------------------------------------------------
# Product identification card
# -------------------------------------------------------

def render_product_identification(product_info: Dict[str, Any]) -> None:
    """
    Render the "Detected Product" card from GeminiService.identify_product()
    output.
    """
    st.markdown('<div class="sa-eyebrow">Detected Product</div>', unsafe_allow_html=True)
    st.markdown(f"### {product_info.get('product_name', 'Unknown product')}")

    cols = st.columns(4)
    fields = [
        ("Brand", product_info.get("brand", "Unknown")),
        ("Category", product_info.get("category", "Unknown")),
        ("Model", product_info.get("model_number", "Unknown")),
        ("Color", product_info.get("color", "Unknown")),
    ]
    for col, (label, value) in zip(cols, fields):
        with col:
            st.markdown(f'<div class="sa-eyebrow">{label}</div>', unsafe_allow_html=True)
            st.markdown(f"**{value}**")

    confidence_notes = product_info.get("confidence_notes")
    if confidence_notes:
        st.markdown(
            f'<div class="sa-muted">ℹ️ {confidence_notes}</div>', unsafe_allow_html=True
        )


# -------------------------------------------------------
# Price comparison table (Feature 2)
# -------------------------------------------------------

def render_price_comparison(price_table: List[PriceInfo]) -> None:
    """
    Render the retailer price comparison table with badges for cheapest,
    best-rated, fastest-delivery, and best-value listings.
    """
    st.markdown('<div class="sa-eyebrow">Real-Time Price Comparison</div>', unsafe_allow_html=True)

    if not price_table:
        st.markdown('<div class="sa-muted">No pricing data found yet.</div>', unsafe_allow_html=True)
        return

    rows = []
    for p in price_table:
        badges = []
        if p.is_cheapest:
            badges.append("🏷️ Cheapest")
        if p.is_best_rated:
            badges.append("⭐ Best Rated")
        if p.is_fastest_delivery:
            badges.append("🚚 Fastest")
        if p.is_best_value:
            badges.append("💎 Best Value")

        rows.append(
            {
                "Website": p.retailer,
                "Product": p.product_title or "—",
                "Price (₹)": p.price if p.available else None,
                "Discount %": p.discount_percent,
                "Delivery": p.delivery_time_text or (
                    f"{p.delivery_days_estimate:.0f} days" if p.delivery_days_estimate else "—"
                ),
                "Rating": p.rating,
                "Availability": "In Stock" if p.available else "Unavailable",
                "Badges": " ".join(badges) if badges else "",
                "Link": p.url or "",
            }
        )

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Link": st.column_config.LinkColumn("Product Link", display_text="Visit ↗"),
            "Price (₹)": st.column_config.NumberColumn(format="₹%.0f"),
            "Rating": st.column_config.NumberColumn(format="%.1f ⭐"),
        },
    )

    # Savings callout
    available = [p for p in price_table if p.available and p.price is not None]
    if len(available) >= 2:
        cheapest = min(available, key=lambda p: p.price)
        most_expensive = max(available, key=lambda p: p.price)
        savings = round(most_expensive.price - cheapest.price, 2)
        if savings > 0:
            st.markdown(
                f"""
                <div class="sa-card-alt">
                    <span class="sa-badge sa-badge-teal">SAVINGS</span>
                    Buying from <b>{cheapest.retailer}</b> instead of
                    <b>{most_expensive.retailer}</b> saves you
                    <span class="sa-price">₹{savings:,.0f}</span>.
                </div>
                """,
                unsafe_allow_html=True,
            )


# -------------------------------------------------------
# AI Buying Recommendation (Feature 3)
# -------------------------------------------------------

def render_buying_recommendation(recommendation_text: str) -> None:
    """Render Gemini's buying recommendation as a highlighted callout."""
    st.markdown(
        f"""
        <div class="sa-card">
            <div class="sa-eyebrow">🤖 AI Buying Recommendation</div>
            {recommendation_text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -------------------------------------------------------
# Review summary (Feature 7)
# -------------------------------------------------------

def render_review_summary(summary: ReviewSummary) -> None:
    """Render the structured AI review summary."""
    st.markdown('<div class="sa-eyebrow">Customer Review Summary</div>', unsafe_allow_html=True)

    if not summary.has_data:
        st.markdown(
            '<div class="sa-muted">No customer reviews were found for this product yet.</div>',
            unsafe_allow_html=True,
        )
        return

    if summary.overall_rating is not None:
        st.markdown(
            f'<span class="sa-price">{summary.overall_rating:.1f}</span> '
            f'<span class="sa-muted">/ 5 overall</span>',
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**✅ Pros**")
        for item in summary.pros or ["Not enough data"]:
            st.markdown(f"- {item}")
    with col2:
        st.markdown("**⚠️ Cons**")
        for item in summary.cons or ["Not enough data"]:
            st.markdown(f"- {item}")

    if summary.common_positive_feedback:
        st.markdown("**💬 Common Positive Feedback**")
        for item in summary.common_positive_feedback:
            st.markdown(
                f'<span class="sa-badge sa-badge-teal">{item}</span>', unsafe_allow_html=True
            )

    if summary.common_complaints:
        st.markdown("**🗣️ Common Complaints**")
        for item in summary.common_complaints:
            st.markdown(
                f'<span class="sa-badge sa-badge-coral">{item}</span>', unsafe_allow_html=True
            )

    if summary.recommended_for:
        st.markdown(f"**Recommended for:** {summary.recommended_for}")

    if summary.final_verdict:
        st.markdown(f"**Final Verdict:** {summary.final_verdict}")

    st.markdown(
        f'<div class="sa-muted">Based on {summary.source_count} online source(s).</div>',
        unsafe_allow_html=True,
    )


# -------------------------------------------------------
# Similar products / budget alternatives (Features 4 & 5)
# -------------------------------------------------------

def render_result_cards(results: List[SearchResult], title: str) -> None:
    """
    Render a horizontal row of simple cards for similar products or
    budget alternatives. Since we have no product database, each card
    shows the search result's title/snippet/link rather than a
    structured price — the AI recommendation elsewhere provides the
    structured comparison.
    """
    st.markdown(f'<div class="sa-eyebrow">{title}</div>', unsafe_allow_html=True)

    if not results:
        st.markdown('<div class="sa-muted">No results found.</div>', unsafe_allow_html=True)
        return

    top_results = results[:5]
    cols = st.columns(len(top_results))
    for col, result in zip(cols, top_results):
        with col:
            snippet = (result.content or "")[:120]
            st.markdown(
                f"""
                <div class="sa-card-alt" style="min-height: 160px;">
                    <b>{result.title[:60]}</b>
                    <div class="sa-muted">{snippet}...</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if result.url:
                st.link_button("View ↗", result.url, use_container_width=True)


# -------------------------------------------------------
# Chat interface (Feature 9)
# -------------------------------------------------------

def render_quick_actions() -> Optional[str]:
    """
    Render a row of quick-action buttons for common questions (per the
    project's example question list). Returns the clicked question's
    text, or None if nothing was clicked this run.
    """
    st.markdown('<div class="sa-eyebrow">Quick Questions</div>', unsafe_allow_html=True)
    questions = [
        "Should I buy this?",
        "Summarize customer reviews",
        "What are the pros and cons?",
        "Show cheaper alternatives",
        "Which website offers the best deal?",
    ]
    cols = st.columns(len(questions))
    for col, question in zip(cols, questions):
        with col:
            if st.button(question, use_container_width=True, key=f"quick_{question}"):
                return question
    return None


def render_chat_history(chat_turns: List[Tuple[str, str]]) -> None:
    """Render the running chat conversation using Streamlit's chat UI."""
    for question, answer in chat_turns:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)


def render_sources(urls: List[str]) -> None:
    """Render a collapsed list of source URLs an answer was grounded in."""
    if not urls:
        return
    with st.expander(f"📎 Sources ({len(urls)})"):
        for url in urls:
            st.markdown(f"- [{url}]({url})")


# -------------------------------------------------------
# Generic status helpers
# -------------------------------------------------------

def render_error(message: str) -> None:
    st.markdown(
        f"""
        <div class="sa-card" style="border-color: {COLOR_CORAL};">
            <span class="sa-badge sa-badge-coral">ERROR</span> {message}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_divider() -> None:
    st.markdown("<div class='sa-divider'></div>", unsafe_allow_html=True)