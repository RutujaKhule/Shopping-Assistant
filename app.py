"""
app.py
=========================================================
Multimodal AI Shopping Assistant — Main Entrypoint
=========================================================
Wires together the full workflow described in the project spec:

  1. User uploads a product image.
  2. Gemini Vision identifies the MAIN product (ignoring accessories).
  3. app.py builds a concise search query manually from the structured
     Brand + Product Name + Category fields (NEVER from Gemini's raw
     `search_query` text — see utils.build_manual_search_query()).
  4. Tavily searches the live web for specs, prices (per retailer),
     and reviews, with accessory results filtered out.
  5-7. Retrieved text is chunked (accessory/low-quality content
       skipped), embedded, and stored in a temporary, in-memory FAISS
       index; LangChain handles retrieval.
  8-9. Gemini Flash generates grounded answers, buying recommendations,
       review summaries, and product comparisons.

This file is the ONLY orchestration point in the app — it calls into
services/ and rag/ for all logic, and ui/components.py for all
rendering. Nothing here talks to Gemini/Tavily/FAISS directly.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

import streamlit as st
from PIL import Image

from services.gemini import GeminiService, GeminiServiceError
from services.search import SearchService, SearchServiceError
from services.embeddings import EmbeddingService, EmbeddingServiceError
from services.comparison import ComparisonService, ComparisonServiceError
from services.reviews import ReviewsService, ReviewsServiceError
from rag.vector_store import VectorStoreManager, VectorStoreError
from rag.retrieval import RAGPipeline, ConversationMemory, RetrievalError

from ui import components as ui
from utils import build_manual_search_query

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


# =========================================================
# Service initialization (cached — these wrappers are stateless
# aside from config, so it's safe to share one instance app-wide)
# =========================================================

@st.cache_resource(show_spinner=False)
def get_services():
    """
    Instantiate the stateless service layer once per app process.
    Raises immediately (surfaced via st.error in main()) if API keys
    are missing, rather than failing confusingly deep in a callback.
    """
    gemini_service = GeminiService()
    search_service = SearchService()
    embedding_service = EmbeddingService()
    comparison_service = ComparisonService(gemini_service)
    reviews_service = ReviewsService(gemini_service)
    return gemini_service, search_service, embedding_service, comparison_service, reviews_service


# =========================================================
# Session state
# =========================================================

def init_session_state(embedding_service: EmbeddingService, gemini_service: GeminiService) -> None:
    """Ensure all per-session objects exist before first render."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = VectorStoreManager(embedding_service)
    if "rag_pipeline" not in st.session_state:
        st.session_state.rag_pipeline = RAGPipeline(st.session_state.vector_store, gemini_service)
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationMemory()

    st.session_state.setdefault("product_info", None)
    st.session_state.setdefault("search_bundle", None)
    st.session_state.setdefault("price_table", None)
    st.session_state.setdefault("review_summary", None)
    st.session_state.setdefault("buying_recommendation", None)
    st.session_state.setdefault("uploaded_image", None)
    st.session_state.setdefault("current_image_hash", None)


def reset_session() -> None:
    """Clear everything so a new product can be loaded from scratch."""
    st.session_state.vector_store.reset()
    st.session_state.memory.clear()
    st.session_state.product_info = None
    st.session_state.search_bundle = None
    st.session_state.price_table = None
    st.session_state.review_summary = None
    st.session_state.buying_recommendation = None
    st.session_state.uploaded_image = None
    st.session_state.current_image_hash = None


# =========================================================
# Core pipeline: image -> identified product -> search -> RAG index
# =========================================================

def process_uploaded_image(
    uploaded_file,
    gemini_service: GeminiService,
    search_service: SearchService,
    embedding_service: EmbeddingService,
    comparison_service: ComparisonService,
    reviews_service: ReviewsService,
) -> None:
    """
    Run the full STEP 1-9 pipeline for a newly uploaded product image
    and populate st.session_state with everything the dashboard needs.
    """
    image = Image.open(uploaded_file).convert("RGB")
    st.session_state.uploaded_image = image

    # STEP 2: Gemini Vision identifies the MAIN product (ignoring
    # accessories bundled alongside it - see services/gemini.py prompt).
    with st.spinner("Identifying product..."):
        product_info = gemini_service.identify_product(image)
    st.session_state.product_info = product_info

    product_name = product_info.get("product_name", "this product")
    brand = product_info.get("brand")
    category = product_info.get("category")

    # NEVER use Gemini's own `search_query` field to search the web.
    # Build the query manually from Brand + Product Name + Category.
    search_query = build_manual_search_query(brand, product_name, category)

    # STEP 3-4: real-time web search for specs, prices, reviews
    with st.spinner(f"Searching the web for {product_name}..."):
        bundle = search_service.build_product_bundle(search_query, category=category)

    # FEATURE 5 (improved): ask Gemini for structured similar-product
    # suggestions, then search the live web for each one individually.
    # Falls back to the bundle's legacy similar-products search (already
    # populated by build_product_bundle) if this improved flow fails.
    with st.spinner("Finding similar products..."):
        try:
            suggestions = gemini_service.suggest_similar_products(
                product_name, category=category, brand=brand
            )
            improved_similar = search_service.search_similar_products_by_suggestions(
                suggestions, category=category
            )
            if improved_similar:
                bundle.similar_products = improved_similar
        except (GeminiServiceError, SearchServiceError) as exc:
            logger.warning(
                "Improved similar-product suggestion flow failed, keeping "
                "legacy fallback results: %s", exc
            )

    st.session_state.search_bundle = bundle

    # STEP 5-6: chunk + embed + store in FAISS (per-result, to preserve
    # source URLs for citation — see services/embeddings.py)
    with st.spinner("Indexing retrieved data..."):
        documents = []
        documents += embedding_service.chunk_search_results(bundle.general_info, product_name)
        for retailer_results in bundle.prices_by_retailer.values():
            documents += embedding_service.chunk_search_results(retailer_results, product_name)
        documents += embedding_service.chunk_search_results(bundle.reviews, product_name)
        documents += embedding_service.chunk_search_results(bundle.similar_products, product_name)

        if documents:
            st.session_state.vector_store.build_index(documents)
        else:
            logger.warning("No documents retrieved for %s; RAG chat will have no context.", product_name)

    # Feature 2: structured price comparison table
    with st.spinner("Comparing prices across retailers..."):
        price_table = comparison_service.build_price_comparison_table(bundle.prices_by_retailer)
        price_table = comparison_service.highlight_best_options(price_table)
    st.session_state.price_table = price_table

    # Feature 7: review summary
    with st.spinner("Summarizing customer reviews..."):
        try:
            review_summary = reviews_service.summarize_reviews(product_name, bundle.reviews)
        except ReviewsServiceError as exc:
            logger.warning("Review summary failed: %s", exc)
            review_summary = None
    st.session_state.review_summary = review_summary

    # Feature 3: AI buying recommendation
    with st.spinner("Generating buying recommendation..."):
        try:
            verdict_text = review_summary.final_verdict if review_summary else None
            recommendation = comparison_service.generate_buying_recommendation(
                product_name, price_table, review_summary_text=verdict_text
            )
        except ComparisonServiceError as exc:
            logger.warning("Buying recommendation failed: %s", exc)
            recommendation = "Unable to generate a recommendation right now."
    st.session_state.buying_recommendation = recommendation


# =========================================================
# Chat handling (Features 5, 6, 9)
# =========================================================

def _maybe_augment_context(
    question: str,
    gemini_service: GeminiService,
    search_service: SearchService,
    embedding_service: EmbeddingService,
) -> None:
    """
    Feature 5 support: if the question implies budget alternatives
    ("show laptops under 60000", "cheaper alternatives"), run a fresh,
    targeted search and add the results into the FAISS index before
    answering, so the RAG answer is actually grounded in that data.
    """
    lowered = question.lower()
    product_info = st.session_state.product_info or {}
    category = product_info.get("category", "product")
    product_name = product_info.get("product_name", "product")
    brand = product_info.get("brand")

    price_match = re.search(r"under\s*[₹]?\s*([\d,]+)", lowered)
    if price_match and any(k in lowered for k in ["under", "budget", "cheap"]):
        max_price = int(price_match.group(1).replace(",", ""))
        with st.spinner(f"Searching for {category} under ₹{max_price:,}..."):
            results = search_service.search_budget_alternatives(category, max_price)
        if results:
            docs = embedding_service.chunk_search_results(results, product_name)
            st.session_state.vector_store.add_documents(docs)
        return

    if any(k in lowered for k in ["cheaper alternative", "similar product", "alternatives"]):
        with st.spinner("Searching for alternatives..."):
            try:
                suggestions = gemini_service.suggest_similar_products(
                    product_name, category=category, brand=brand
                )
                results = search_service.search_similar_products_by_suggestions(
                    suggestions, category=category
                )
            except (GeminiServiceError, SearchServiceError) as exc:
                logger.warning("Similar-product chat augmentation failed: %s", exc)
                results = []
            if not results:
                # Fall back to the legacy broad search rather than
                # returning nothing.
                search_query = build_manual_search_query(brand, product_name, category)
                results = search_service.search_similar_products(search_query, category)
        if results:
            docs = embedding_service.chunk_search_results(results, product_name)
            st.session_state.vector_store.add_documents(docs)


def _detect_comparison_target(question: str, gemini_service: GeminiService) -> Optional[str]:
    """
    Feature 6 support: detect whether the user wants a head-to-head
    comparison with a different, named product, and extract that
    product's name. Returns None if this isn't a comparison request.
    """
    lowered = question.lower()
    if "compar" not in lowered and " vs " not in lowered and " versus " not in lowered:
        return None

    prompt = f"""
The user asked: "{question}"

Does this question ask to compare the current product with a DIFFERENT,
specifically named product? Respond with ONLY a JSON object:
{{"is_comparison": true or false, "other_product": "string or null"}}

Only set is_comparison to true if another specific product is named.
General questions like "compare prices across websites" should be
is_comparison: false (that's a retailer price comparison, not a
product-vs-product one).
"""
    try:
        raw = gemini_service.generate_text(prompt, temperature=0.0)
        data = GeminiService.parse_json_response(raw)
        if data.get("is_comparison") and data.get("other_product"):
            return data["other_product"]
    except GeminiServiceError as exc:
        logger.warning("Comparison detection failed: %s", exc)
    return None


def handle_chat_question(
    question: str,
    gemini_service: GeminiService,
    search_service: SearchService,
    embedding_service: EmbeddingService,
    comparison_service: ComparisonService,
) -> None:
    """
    Answer one chat question, updating st.session_state.memory in place.
    Routes to product-vs-product comparison (Feature 6) when detected,
    otherwise runs the standard RAG pipeline (with optional budget/
    alternative augmentation for Feature 5).
    """
    product_info = st.session_state.product_info or {}
    product_name = product_info.get("product_name", "this product")
    rag_pipeline: RAGPipeline = st.session_state.rag_pipeline
    memory: ConversationMemory = st.session_state.memory

    other_product = _detect_comparison_target(question, gemini_service)

    if other_product:
        with st.spinner(f"Researching {other_product} for comparison..."):
            try:
                other_results = search_service.search_product_details(other_product)
                context_b = "\n---\n".join(r.to_text() for r in other_results)
                context_a = rag_pipeline.build_context(
                    rag_pipeline.retrieve(product_name, k=6)
                )
                answer = comparison_service.compare_products(
                    product_name, context_a, other_product, context_b
                )
            except (SearchServiceError, ComparisonServiceError, RetrievalError) as exc:
                logger.warning("Comparison flow failed: %s", exc)
                answer = f"I couldn't complete that comparison right now ({exc})."
        memory.add_turn(question, answer)
        return

    _maybe_augment_context(question, gemini_service, search_service, embedding_service)

    with st.spinner("Thinking..."):
        try:
            answer, _sources = rag_pipeline.answer_with_sources(question, memory=memory)
        except (RetrievalError, GeminiServiceError) as exc:
            logger.warning("RAG answer failed: %s", exc)
            memory.add_turn(
                question, "Sorry, I ran into an issue answering that. Please try again."
            )


# =========================================================
# Dashboard rendering
# =========================================================

def render_dashboard() -> None:
    """Render the main-page dashboard once a product has been identified."""
    product_info: Dict[str, Any] = st.session_state.product_info
    bundle = st.session_state.search_bundle
    price_table = st.session_state.price_table
    review_summary = st.session_state.review_summary
    recommendation = st.session_state.buying_recommendation

    col_image, col_details = st.columns([1, 2])
    with col_image:
        st.image(st.session_state.uploaded_image,width=350)
    with col_details:
        ui.render_product_identification(product_info)

    ui.render_section_divider()

    if recommendation:
        ui.render_buying_recommendation(recommendation)

    ui.render_price_comparison(price_table or [])

    ui.render_section_divider()

    if review_summary:
        ui.render_review_summary(review_summary)

    ui.render_section_divider()

    if bundle and bundle.similar_products:
        ui.render_result_cards(bundle.similar_products, "Similar Products")

    ui.render_section_divider()

    st.markdown("### 💬 Ask About This Product")
    quick_question = ui.render_quick_actions()

    ui.render_chat_history(st.session_state.memory.turns)

    user_question = st.chat_input(f"Ask anything about {product_info.get('product_name', 'this product')}...")
    question = quick_question or user_question

    if question:
        gemini_service, search_service, embedding_service, comparison_service, _ = get_services()
        handle_chat_question(question, gemini_service, search_service, embedding_service, comparison_service)
        st.rerun()


# =========================================================
# Main
# =========================================================

def main() -> None:
    st.set_page_config(
        page_title="Multimodal AI Shopping Assistant",
        page_icon="🛍️",
        layout="wide",
    )
    ui.inject_custom_css()

    try:
        gemini_service, search_service, embedding_service, comparison_service, reviews_service = get_services()
    except (GeminiServiceError, SearchServiceError, EmbeddingServiceError) as exc:
        st.error(
            f"⚠️ Setup error: {exc}\n\nAdd your API keys to the `.env` file "
            "(GOOGLE_API_KEY and TAVILY_API_KEY) and restart the app."
        )
        st.stop()

    init_session_state(embedding_service, gemini_service)

    sidebar_state = ui.render_sidebar(st.session_state.memory.turns)

    if sidebar_state["reset_clicked"]:
        reset_session()
        st.rerun()

    uploaded_file = sidebar_state["uploaded_file"]

    if uploaded_file is not None:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if file_hash != st.session_state.current_image_hash:
            st.session_state.current_image_hash = file_hash
            try:
                process_uploaded_image(
                    uploaded_file,
                    gemini_service,
                    search_service,
                    embedding_service,
                    comparison_service,
                    reviews_service,
                )
            except (
                GeminiServiceError,
                SearchServiceError,
                EmbeddingServiceError,
                VectorStoreError,
            ) as exc:
                ui.render_error(f"Something went wrong processing that image: {exc}")
                st.stop()

    if st.session_state.product_info is None:
        st.title("🛍️ Multimodal AI Shopping Assistant")
        st.markdown(
            "Upload a product photo in the sidebar to get **real-time prices**, "
            "**AI buying advice**, **review summaries**, and a **chat assistant** "
            "for follow-up questions — all powered by live web search, no static "
            "product database."
        )
    else:
        render_dashboard()


if __name__ == "__main__":
    main()