"""
services/embeddings.py
=========================================================
Embedding & Chunking Service
=========================================================
Implements STEP 5 of the workflow: turning raw search-result text into
chunked, embeddable Documents for FAISS.

Quality improvements:
- Before chunking, each search result is checked against the shared
  accessory blocklist (utils.contains_accessory_keyword) and against a
  minimal "is this actually useful content" check. Accessory pages,
  empty/near-empty snippets, and boilerplate navigation/ad-style
  fragments are skipped entirely rather than embedded - so the FAISS
  index (and therefore every RAG answer) only ever contains real
  product specs, prices, reviews, and features.

Performance improvements:
- embed_query() now keeps a small in-memory cache keyed by the exact
  query text, since the same question text (e.g. after question
  condensing) is sometimes embedded more than once in a session.
"""

import os
import logging
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from utils import contains_accessory_keyword

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# ✅ Latest Gemini Embedding Model
EMBEDDING_MODEL_NAME = "models/gemini-embedding-001"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 120

# A result's content shorter than this is almost certainly boilerplate
# (nav labels, "Add to Cart", ad snippets) rather than real product
# information, and isn't worth embedding.
MIN_USEFUL_CONTENT_CHARS = 40

# Cap on how many query embeddings we keep cached in memory.
MAX_QUERY_CACHE_ENTRIES = 256


class EmbeddingServiceError(Exception):
    pass


class EmbeddingService:

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):

        if not GOOGLE_API_KEY:
            raise EmbeddingServiceError(
                "GOOGLE_API_KEY not found in .env"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n---\n", "\n\n", "\n", ". ", " ", ""],
        )

        self._embeddings_model = None

        # Simple in-memory cache for repeated query embeddings (e.g. the
        # same follow-up question text embedded more than once).
        self._query_embedding_cache: Dict[str, List[float]] = {}

    # -------------------------------------------------

    def get_embeddings_model(self):

        if self._embeddings_model is None:

            try:

                print(f"Using Embedding Model : {EMBEDDING_MODEL_NAME}")

                self._embeddings_model = GoogleGenerativeAIEmbeddings(
                    model=EMBEDDING_MODEL_NAME,
                    google_api_key=GOOGLE_API_KEY,
                )

            except Exception as exc:

                logger.exception(exc)

                raise EmbeddingServiceError(
                    f"Could not initialize embedding model : {exc}"
                )

        return self._embeddings_model

    # -------------------------------------------------

    def chunk_text(
        self,
        text: str,
        metadata: Optional[dict] = None,
    ) -> List[Document]:

        if not text.strip():
            raise EmbeddingServiceError("Empty text.")

        base_metadata = dict(metadata or {})

        raw_chunks = self._splitter.split_text(text)

        documents = []

        for idx, chunk in enumerate(raw_chunks):
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **base_metadata,
                        "chunk_index": idx,
                    },
                )
            )

        logger.info(f"Created {len(documents)} chunks")

        return documents

    # -------------------------------------------------

    @staticmethod
    def _is_useful_content(title: str, content: str) -> bool:
        """
        Decide whether a search result is worth embedding at all.

        Skips:
        - Accessory listings (covers, cases, chargers, cables, etc.) -
          RAG storage should only ever contain the real product's data.
        - Empty or too-short snippets, which are almost always
          navigation/header/ad boilerplate rather than real content.
        """
        if contains_accessory_keyword(title) or contains_accessory_keyword(content):
            return False
        if not content or len(content.strip()) < MIN_USEFUL_CONTENT_CHARS:
            return False
        return True

    def chunk_search_results(
        self,
        results: List[Any],
        product_name: str,
    ) -> List[Document]:

        documents = []

        if not results:
            return documents

        skipped = 0

        for result in results:

            content = getattr(result, "content", "")
            title = getattr(result, "title", "")

            if not content:
                continue

            if not self._is_useful_content(title, content):
                skipped += 1
                continue

            metadata = {
                "product_name": product_name,
                "url": getattr(result, "url", ""),
                "title": title,
                "source": getattr(result, "source", ""),
            }

            if len(content) <= self.chunk_size:

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            **metadata,
                            "chunk_index": 0,
                        },
                    )
                )

            else:

                chunks = self._splitter.split_text(content)

                for idx, chunk in enumerate(chunks):

                    documents.append(
                        Document(
                            page_content=chunk,
                            metadata={
                                **metadata,
                                "chunk_index": idx,
                            },
                        )
                    )

        if skipped:
            logger.info(
                f"Skipped {skipped} low-quality/accessory search result(s) "
                "before chunking"
            )

        logger.info(
            f"Chunked {len(results)} search results into {len(documents)} documents"
        )

        return documents

    # -------------------------------------------------

    def embed_texts(self, texts: List[str]):

        try:

            model = self.get_embeddings_model()

            return model.embed_documents(texts)

        except Exception as exc:

            raise EmbeddingServiceError(
                f"Embedding generation failed : {exc}"
            )

    # -------------------------------------------------

    def embed_query(self, query: str):

        cached = self._query_embedding_cache.get(query)
        if cached is not None:
            return cached

        try:

            model = self.get_embeddings_model()

            embedding = model.embed_query(query)

        except Exception as exc:

            raise EmbeddingServiceError(
                f"Query embedding failed : {exc}"
            )

        if len(self._query_embedding_cache) >= MAX_QUERY_CACHE_ENTRIES:
            self._query_embedding_cache.pop(next(iter(self._query_embedding_cache)))
        self._query_embedding_cache[query] = embedding

        return embedding