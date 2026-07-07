"""
rag/retrieval.py
=========================================================
Retrieval-Augmented Generation Pipeline
=========================================================
Implements STEP 7, 8, and 9 of the workflow:

  STEP 7 - Use LangChain retrieval to pull the most relevant chunks
           from the FAISS index (rag/vector_store.py) for a given query.
  STEP 8 - Pass that retrieved context into Gemini Flash.
  STEP 9 - Generate an intelligent, grounded answer.

It also implements FEATURE 9 (Chatbot): follow-up questions are
supported via `ConversationMemory`, and ambiguous follow-ups
("what about that one?", "is it worth it?") are rewritten into
standalone questions before retrieval, so pronouns don't break search
relevance.

Quality improvement (Feature 10 - "Improve Retrieval"): even though
accessory content is already filtered out before it's ever embedded
(see services/embeddings.py), build_context() applies the same
accessory blocklist a second time at answer-time, as defense in depth.
This protects against any chunk that slipped through (e.g. from an
older/blocked index) and guarantees accessory-related text never makes
it into a Gemini prompt.

This module deliberately does NOT talk to FAISS or Gemini directly at
the SDK level — it composes the two service layers
(services.gemini.GeminiService and rag.vector_store.VectorStoreManager)
that already encapsulate that.
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from services.gemini import GeminiService, GeminiServiceError
from rag.vector_store import VectorStoreManager, VectorStoreError
from utils import contains_accessory_keyword

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_TOP_K = 4
MAX_CONTEXT_CHARS = 6000  # keep the Gemini prompt focused and cost-efficient
MAX_MEMORY_TURNS = 8      # how many past exchanges to keep for follow-ups


class RetrievalError(Exception):
    """Raised when retrieval or grounded answer generation fails."""
    pass


# -------------------------------------------------------
# Conversation memory (Feature 9: chatbot with memory)
# -------------------------------------------------------

@dataclass
class ConversationMemory:
    """
    Simple rolling chat memory for the shopping assistant's chatbot.

    app.py should keep one instance of this per session (e.g. in
    st.session_state) and reset it whenever a new product is loaded.
    """
    turns: List[Tuple[str, str]] = field(default_factory=list)  # (question, answer)
    max_turns: int = MAX_MEMORY_TURNS

    def add_turn(self, question: str, answer: str) -> None:
        """Record a completed Q&A exchange, trimming old history if needed."""
        self.turns.append((question, answer))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def as_text(self) -> str:
        """Render the conversation so far as plain text, oldest first."""
        if not self.turns:
            return ""
        lines = []
        for question, answer in self.turns:
            lines.append(f"User: {question}")
            lines.append(f"Assistant: {answer}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Reset memory (call when the user uploads a new product image)."""
        self.turns = []

    @property
    def has_history(self) -> bool:
        return len(self.turns) > 0


# -------------------------------------------------------
# RAG Pipeline
# -------------------------------------------------------

class RAGPipeline:
    """
    Ties together FAISS retrieval and Gemini generation into a single
    "ask a question, get a grounded answer" interface.

    Usage:
        pipeline = RAGPipeline(vector_store, gemini_service)
        answer = pipeline.answer("What are the pros and cons?", memory)
    """

    def __init__(
        self,
        vector_store: VectorStoreManager,
        gemini_service: GeminiService,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        self.vector_store = vector_store
        self.gemini_service = gemini_service
        self.top_k = top_k

    # -----------------------------------------------------
    # Question rewriting for follow-ups
    # -----------------------------------------------------

    def _condense_question(self, question: str, memory: Optional[ConversationMemory]) -> str:
        """
        If there is prior conversation history, rewrite the (possibly
        ambiguous) follow-up question into a standalone question that
        will retrieve well from FAISS on its own.

        E.g. history: "What phone is this?" -> "iPhone 15"
             follow-up: "Is it worth buying?"
             becomes:   "Is the iPhone 15 worth buying?"

        If there's no history yet, the original question is returned
        unchanged (no need to burn an extra Gemini call).
        """
        if memory is None or not memory.has_history:
            return question

        prompt = f"""
Given the conversation history below, rewrite the user's latest question
into a fully standalone question that makes sense without the history.
Preserve the original intent exactly. Respond with ONLY the rewritten
question, no explanation, no quotes.

CONVERSATION HISTORY:
{memory.as_text()}

LATEST QUESTION:
{question}
"""
        try:
            rewritten = self.gemini_service.generate_text(prompt, temperature=0.0)
            # Guard against the model returning something unusable/empty.
            return rewritten if rewritten.strip() else question
        except GeminiServiceError as exc:
            logger.warning(
                "Question condensing failed, falling back to raw question: %s", exc
            )
            return question

    # -----------------------------------------------------
    # Retrieval
    # -----------------------------------------------------

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Fetch the top-k most relevant chunks from the FAISS index for a
        given (standalone) query.
        """
        try:
            return self.vector_store.similarity_search(query, k=k or self.top_k)
        except VectorStoreError as exc:
            raise RetrievalError(f"Retrieval failed: {exc}") from exc

    @staticmethod
    def build_context(documents: List[Document], max_chars: int = MAX_CONTEXT_CHARS) -> str:
        """
        Join retrieved document chunks into one context string for
        Gemini, deduplicating identical chunks, dropping any
        accessory-related chunk (Feature 10 - "Improve Retrieval": ignore
        chunks mentioning cover/case/adapter/charger/accessory), and
        truncating to a reasonable length so prompts stay fast and cheap.
        """
        seen = set()
        blocks: List[str] = []
        total_len = 0

        for doc in documents:
            content = doc.page_content.strip()
            if not content or content in seen:
                continue
            if contains_accessory_keyword(content):
                continue
            seen.add(content)

            if total_len + len(content) > max_chars:
                remaining = max_chars - total_len
                if remaining > 200:  # only add a truncated block if it's still useful
                    blocks.append(content[:remaining] + " [truncated]")
                break

            blocks.append(content)
            total_len += len(content)

        return "\n---\n".join(blocks)

    @staticmethod
    def extract_sources(documents: List[Document]) -> List[str]:
        """
        Pull unique source URLs out of retrieved documents' metadata, so
        the UI can show "Sources" links alongside an answer. Skips
        accessory-related chunks, consistent with build_context().
        """
        sources = []
        for doc in documents:
            if contains_accessory_keyword(doc.page_content):
                continue
            url = doc.metadata.get("url")
            if url and url not in sources:
                sources.append(url)
        return sources

    # -----------------------------------------------------
    # End-to-end answer generation
    # -----------------------------------------------------

    def answer(
        self,
        question: str,
        memory: Optional[ConversationMemory] = None,
        k: Optional[int] = None,
    ) -> str:
        """
        Full RAG turn: condense question (if follow-up) -> retrieve ->
        build context -> generate grounded answer -> update memory.

        Args:
            question: The user's raw question (from chat input or a
                quick-action button like "Should I buy this?").
            memory: Optional ConversationMemory for follow-up handling.
                If provided, the exchange is recorded automatically.
            k: Optional override for number of chunks retrieved.

        Returns:
            The generated answer as a plain string.

        Raises:
            RetrievalError: if retrieval fails.
            GeminiServiceError: if answer generation fails.
        """
        standalone_question = self._condense_question(question, memory)

        documents = self.retrieve(standalone_question, k=k)
        if not documents:
            logger.warning(
                "No relevant chunks retrieved for question: %s", standalone_question
            )
            context = ""
        else:
            context = self.build_context(documents)

        answer_text = self.gemini_service.generate_grounded_answer(
            question=question,  # use the user's original phrasing for the final answer
            context=context,
        )

        if memory is not None:
            memory.add_turn(question, answer_text)

        return answer_text

    def answer_with_sources(
        self,
        question: str,
        memory: Optional[ConversationMemory] = None,
        k: Optional[int] = None,
    ) -> Tuple[str, List[str]]:
        """
        Same as answer(), but also returns the list of source URLs the
        answer was grounded in, for display in the UI.
        """
        standalone_question = self._condense_question(question, memory)
        documents = self.retrieve(standalone_question, k=k)
        context = self.build_context(documents) if documents else ""

        answer_text = self.gemini_service.generate_grounded_answer(
            question=question,
            context=context,
        )

        if memory is not None:
            memory.add_turn(question, answer_text)

        return answer_text, self.extract_sources(documents)