"""
rag/vector_store.py
=========================================================
FAISS Vector Store Manager
=========================================================
Implements STEP 6 of the workflow: storing embeddings inside FAISS.

Important design decision: the FAISS index here is IN-MEMORY ONLY and
scoped to a single product/session. Per project requirements, this app
uses no static datasets — every index is built fresh from whatever
SearchService just retrieved from the live web, and is discarded (or
rebuilt) when the user uploads a new product image or starts a new
session. Nothing is written to disk.

app.py is expected to hold one VectorStoreManager instance per session
(e.g. in st.session_state), rebuilding its index each time a new
product is identified.
"""

import logging
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from services.embeddings import EmbeddingService, EmbeddingServiceError

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_TOP_K = 4


class VectorStoreError(Exception):
    """Raised when the FAISS index cannot be built or queried."""
    pass


class VectorStoreManager:
    """
    Wraps a single in-memory FAISS index for the current product session.

    Usage:
        embedder = EmbeddingService()
        store = VectorStoreManager(embedder)
        store.build_index(documents)          # from chunk_text() output
        results = store.similarity_search("battery life")
        store.add_documents(more_documents)   # e.g. after a follow-up search
        store.reset()                         # before loading a new product
    """

    def __init__(self, embedding_service: EmbeddingService) -> None:
        self.embedding_service = embedding_service
        self._index: Optional[FAISS] = None

    # -----------------------------------------------------
    # Index lifecycle
    # -----------------------------------------------------

    def build_index(self, documents: List[Document]) -> None:
        """
        Build a brand-new FAISS index from a list of Documents, replacing
        any existing index. Call this each time a new product is
        identified and its search bundle has been chunked.

        Args:
            documents: List of Document chunks (see EmbeddingService.chunk_text).

        Raises:
            VectorStoreError: if documents is empty or embedding fails.
        """
        if not documents:
            raise VectorStoreError(
                "Cannot build a FAISS index from zero documents. Ensure "
                "SearchService returned results and chunk_text() was called "
                "before build_index()."
            )

        try:
            embeddings_model = self.embedding_service.get_embeddings_model()
            self._index = FAISS.from_documents(documents, embeddings_model)
            logger.info(
                "Built new FAISS index with %d document chunks", len(documents)
            )
        except EmbeddingServiceError as exc:
            raise VectorStoreError(f"Embedding failure while building index: {exc}") from exc
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to build FAISS index")
            raise VectorStoreError(f"Failed to build FAISS index: {exc}") from exc

    def add_documents(self, documents: List[Document]) -> None:
        """
        Add additional documents to the existing index (e.g. if the user
        asks a follow-up question that triggers a fresh, targeted search,
        such as "show cheaper alternatives").

        If no index exists yet, this behaves like build_index().

        Args:
            documents: List of Document chunks to add.
        """
        if not documents:
            logger.warning("add_documents called with an empty list; skipping.")
            return

        if self._index is None:
            self.build_index(documents)
            return

        try:
            self._index.add_documents(documents)
            logger.info("Added %d document chunks to existing FAISS index", len(documents))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to add documents to FAISS index")
            raise VectorStoreError(f"Failed to add documents to index: {exc}") from exc

    def reset(self) -> None:
        """Discard the current index (call before loading a new product)."""
        self._index = None
        logger.info("FAISS index reset.")

    @property
    def is_ready(self) -> bool:
        """Whether an index currently exists and can be queried."""
        return self._index is not None

    # -----------------------------------------------------
    # Querying
    # -----------------------------------------------------

    def similarity_search(self, query: str, k: int = DEFAULT_TOP_K) -> List[Document]:
        """
        Retrieve the top-k most relevant chunks for a query.

        Args:
            query: The user's question or search phrase.
            k: Number of chunks to retrieve.

        Returns:
            List of Document objects, most relevant first.

        Raises:
            VectorStoreError: if no index has been built yet.
        """
        self._ensure_ready()
        try:
            return self._index.similarity_search(query, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Similarity search failed")
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

    def similarity_search_with_score(
        self, query: str, k: int = DEFAULT_TOP_K
    ) -> List[Tuple[Document, float]]:
        """
        Same as similarity_search, but also returns a distance score per
        result (lower = more similar for FAISS's default L2 metric).
        Useful for debugging retrieval quality or filtering out weak
        matches before passing context to Gemini.
        """
        self._ensure_ready()
        try:
            return self._index.similarity_search_with_score(query, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Similarity search with score failed")
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

    def as_retriever(self, k: int = DEFAULT_TOP_K):
        """
        Expose the index as a LangChain retriever object, for use in
        rag/retrieval.py's retrieval chain construction.

        Args:
            k: Number of chunks the retriever should fetch per query.

        Returns:
            A LangChain VectorStoreRetriever.
        """
        self._ensure_ready()
        return self._index.as_retriever(search_kwargs={"k": k})

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------

    def _ensure_ready(self) -> None:
        if self._index is None:
            raise VectorStoreError(
                "No FAISS index has been built yet. Call build_index() "
                "with product search results before querying."
            )
