"""
services/gemini.py
=========================================================
Gemini Service Layer
=========================================================
This module wraps all interactions with Google's Gemini models:

1. Gemini Vision  -> identifies a product from an uploaded image
   (name, brand, category, model number, color, variant). Tuned to
   identify ONLY the main, largest, purchasable product in the frame
   and ignore accessories bundled alongside it (covers, cases, boxes,
   cables, etc.).

2. Gemini Flash    -> general-purpose text generation, used for:
   - RAG-grounded question answering
   - Buying recommendations
   - Review summaries
   - Product comparisons
   - Similar-product suggestions (Feature 5)

Design notes:
- All Gemini calls are centralized here so the rest of the app
  (services/comparison.py, services/reviews.py, rag/retrieval.py, app.py)
  never talks to the Gemini SDK directly. This keeps the model provider
  swappable and the error handling consistent in one place.
- Vision output is requested in strict JSON so downstream code can rely
  on structured fields instead of parsing free text.
- NOTE ON search_query: identify_product() still returns a `search_query`
  field for backward compatibility / debugging, but per project
  requirements the rest of the app (app.py) NEVER uses it to actually
  search the web. Instead, app.py builds the search query itself from
  the structured brand / product_name / category fields via
  utils.build_manual_search_query().
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# -------------------------------------------------------
# Setup
# -------------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_gemini_api_key_here":
    # We don't raise here at import time because that would crash the whole
    # app (e.g. during tests or docs generation). Instead we raise a clear
    # error the moment someone actually tries to call the API.
    logger.warning(
        "GOOGLE_API_KEY is not set. Set it in your .env file before "
        "calling any GeminiService methods."
    )
else:
    genai.configure(api_key=GOOGLE_API_KEY)


class GeminiServiceError(Exception):
    """Raised when a Gemini API call fails or returns an unusable response."""
    pass


class GeminiService:
    """
    Thin, reusable wrapper around the Gemini SDK.

    Usage:
        gemini = GeminiService()
        product_info = gemini.identify_product(image)
        answer = gemini.generate_text("Summarize these reviews...")
        similar = gemini.suggest_similar_products("iPhone 15", category="Smartphone")
    """

    def __init__(
        self,
        vision_model_name: str = GEMINI_VISION_MODEL,
        text_model_name: str = GEMINI_TEXT_MODEL,
    ) -> None:
        self._ensure_api_key()
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name

        # Separate model handles. Gemini's newer models are multimodal, so in
        # practice these may point to the same underlying model, but keeping
        # them separate lets us swap either independently later (e.g. a
        # cheaper text model vs a stronger vision model).
        self.vision_model = genai.GenerativeModel(self.vision_model_name)
        self.text_model = genai.GenerativeModel(self.text_model_name)

    @staticmethod
    def _ensure_api_key() -> None:
        """Fail fast with a clear message if no API key is configured."""
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_google_gemini_api_key_here":
            raise GeminiServiceError(
                "GOOGLE_API_KEY is missing or unset. Add a valid key to your "
                ".env file (GOOGLE_API_KEY=...) before using GeminiService."
            )

    # -----------------------------------------------------
    # FEATURE 1: Product Identification (Gemini Vision)
    # -----------------------------------------------------

    def identify_product(self, image: Image.Image) -> Dict[str, Any]:
        """
        Identify a product from an uploaded image using Gemini Vision.

        Tuned to solve the "accessory confusion" problem: if the photo
        contains a main product alongside an accessory (e.g. an iPhone
        next to/inside its cover, a laptop next to a mouse), Gemini is
        explicitly instructed to identify only the main, largest,
        purchasable product and ignore the accessory entirely.

        Args:
            image: A PIL.Image instance of the uploaded product photo.

        Returns:
            A dictionary with keys:
                product_name, brand, category, model_number, color,
                variant, search_query, confidence_notes

            NOTE: `search_query` is kept in the schema for backward
            compatibility and debugging only. app.py does NOT use it to
            perform searches — see utils.build_manual_search_query().

        Raises:
            GeminiServiceError: if the model call fails or the response
                cannot be parsed into usable structured data.
        """
        prompt = """
You are a product identification expert for an e-commerce visual search
assistant. Look carefully at the product image provided and identify
the MAIN PURCHASABLE PRODUCT ONLY.

CRITICAL RULE - Ignore accessories:
If the image shows multiple objects, identify ONLY the largest, main,
purchasable product. Completely ignore any accessories bundled with it,
such as: phone covers/cases, tempered glass/screen guards, cables,
chargers/adapters, skins, pouches, keyboard covers, watch straps, boxes,
or packaging.

Examples of correct behavior:
- iPhone + Cover           -> identify the iPhone (NOT the cover)
- Laptop + Mouse           -> identify the Laptop (NOT the mouse)
- Camera + Bag             -> identify the Camera (NOT the bag)
- Headphones + Box         -> identify the Headphones (NOT the box)
- Shoes + Shoe Box         -> identify the Shoes (NOT the box)

If you are looking at an accessory that has NO larger main product
visible in the frame (e.g. a photo of just a phone case by itself),
then identify that accessory as the product itself - the rule above
only applies when a larger, more significant product is also visible.

Respond with ONLY a valid JSON object (no markdown, no code fences, no
extra commentary) using exactly this schema:

{
  "product_name": "string, best guess of the full product name (main product only, no accessories)",
  "brand": "string, brand/manufacturer name of the MAIN product, or 'Unknown' if unclear",
  "category": "string, e.g. Smartphone, Laptop, Headphones, Shoes, etc.",
  "model_number": "string, model/version identifier, or 'Unknown'",
  "color": "string, dominant color/variant of the MAIN product",
  "variant": "string, storage/size/edition variant if visible, or 'Unknown'",
  "search_query": "string, a concise web-search-ready query for this exact MAIN product (brand + product name + category only, no accessory terms) - kept for reference only",
  "confidence_notes": "string, one short sentence on how confident you are and why"
}

If the image does not clearly show a purchasable product, still make your
best reasonable guess rather than refusing, and lower the confidence note
accordingly.
"""
        try:
            response = self.vision_model.generate_content(
                [prompt, image],
                request_options={"timeout": 30},
            )
            raw_text = self._extract_text(response)
            return self.parse_json_response(raw_text)
        except GeminiServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 - surface any SDK error uniformly
            logger.exception("Gemini Vision call failed")
            raise GeminiServiceError(f"Product identification failed: {exc}") from exc

    # -----------------------------------------------------
    # FEATURE 5: Similar-product suggestions (structured JSON)
    # -----------------------------------------------------

    def suggest_similar_products(
        self,
        product_name: str,
        category: Optional[str] = None,
        brand: Optional[str] = None,
        count: int = 5,
    ) -> List[Dict[str, str]]:
        """
        Ask Gemini for a short list of real, standalone products similar
        to `product_name`, restricted to the same category, and
        explicitly excluding accessories. The caller (services/search.py,
        via app.py) then searches the live web for each suggested name -
        replacing the old, low-quality "best alternatives to X" query.

        Args:
            product_name: The identified product's name.
            category: Optional category to keep suggestions relevant
                (e.g. "Smartphone", "Laptop").
            brand: Optional brand of the original product, purely for
                prompt context (suggestions are not limited to the same
                brand - that's the point of "similar products").
            count: How many suggestions to request (default 5, per spec).

        Returns:
            A list of dicts like [{"name": "...", "brand": "..."}, ...].
            Returns an empty list (never raises) if Gemini's response
            can't be parsed, so a flaky suggestion call never breaks the
            dashboard - callers should treat an empty list as "fall back
            to the legacy search".
        """
        category_line = f"Only products from the same category: {category}." if category else \
            "Only products from the same general category as the original product."

        prompt = f"""
Suggest {count} real, currently-available products similar to:

{product_name}

{category_line}

Do NOT include accessories such as: phone covers, cases, tempered glass,
screen guards, cables, chargers, adapters, skins, pouches, keyboard
covers, or watch straps. Only suggest standalone products a shopper
could buy INSTEAD of "{product_name}".

Respond with ONLY a valid JSON array (no markdown, no code fences, no
commentary) using exactly this schema:

[
  {{"name": "Samsung Galaxy S24", "brand": "Samsung"}},
  {{"name": "...", "brand": "..."}}
]
"""
        try:
            raw = self.generate_text(prompt, temperature=0.3)
            data = self.parse_json_array_response(raw)
        except GeminiServiceError as exc:
            logger.warning("Similar product suggestion failed for %s: %s", product_name, exc)
            return []

        suggestions: List[Dict[str, str]] = []
        for item in data:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            suggestions.append({"name": name, "brand": str(item.get("brand") or "").strip()})
        return suggestions[:count]

    # -----------------------------------------------------
    # Generic text generation (used by RAG, recommendations,
    # review summaries, comparisons, and the chatbot)
    # -----------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
    ) -> str:
        """
        Generate free-form text with Gemini Flash.

        Args:
            prompt: The user-facing prompt / question / instruction.
            system_instruction: Optional system-level instruction to steer
                tone, role, or output format.
            temperature: Sampling temperature (lower = more factual/consistent,
                which is preferred here since answers are grounded in
                retrieved real-time data).

        Returns:
            The generated text as a plain string.

        Raises:
            GeminiServiceError: if the model call fails.
        """
        try:
            model = self.text_model
            if system_instruction:
                # Re-instantiate with a system instruction when provided,
                # since GenerativeModel binds it at construction time.
                model = genai.GenerativeModel(
                    self.text_model_name,
                    system_instruction=system_instruction,
                )

            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                ),
                request_options={"timeout": 30},
            )
            return self._extract_text(response)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Gemini text generation failed")
            raise GeminiServiceError(f"Text generation failed: {exc}") from exc

    def generate_grounded_answer(self, question: str, context: str) -> str:
        """
        RAG-style answer generation: answers a user's question strictly
        using the provided retrieved context (specs/prices/reviews pulled
        from FAISS retrieval), rather than the model's own prior knowledge.

        Args:
            question: The user's question (may be a follow-up in a chat).
            context: Concatenated text chunks retrieved from the vector
                store (see rag/retrieval.py).

        Returns:
            The generated answer as a plain string.
        """
        system_instruction = (
            "You are a helpful, honest AI shopping assistant. Answer the "
            "user's question using ONLY the context provided below, which "
            "was retrieved in real time from the web. If the context does "
            "not contain enough information to answer confidently, say so "
            "plainly instead of guessing. Be concise, specific, and use "
            "bullet points for comparisons or lists where helpful."
        )
        prompt = f"""
CONTEXT (retrieved in real time):
---
{context}
---

USER QUESTION:
{question}

Answer the question using the context above.
"""
        return self.generate_text(prompt, system_instruction=system_instruction)

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        """
        Safely extract plain text from a Gemini SDK response object.
        Raises GeminiServiceError if the response was blocked or empty
        (e.g. due to safety filters).
        """
        try:
            text = response.text
        except Exception as exc:  # noqa: BLE001
            raise GeminiServiceError(
                "Gemini returned no usable text (it may have been blocked "
                f"by safety filters or the request failed): {exc}"
            ) from exc

        if not text or not text.strip():
            raise GeminiServiceError("Gemini returned an empty response.")

        return text.strip()

    @staticmethod
    def _strip_code_fences(raw_text: str) -> str:
        """
        Strip accidental markdown code fences from a model response
        (e.g. ```json ... ``` or ``` ... ```), despite instructions not
        to include them. Shared by the object and array JSON parsers.
        """
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        return cleaned

    @staticmethod
    def parse_json_response(raw_text: str) -> Dict[str, Any]:
        """
        Parse a JSON object out of a model response, tolerating common
        formatting quirks like accidental markdown code fences.
        """
        cleaned = GeminiService._strip_code_fences(raw_text)

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON response: %s", raw_text)
            raise GeminiServiceError(
                "Could not parse product identification response as JSON. "
                f"Raw response: {raw_text[:300]}"
            ) from exc

    @staticmethod
    def parse_json_array_response(raw_text: str) -> List[Any]:
        """
        Same as parse_json_response(), but expects (and validates) a
        top-level JSON array. Used by suggest_similar_products().
        """
        cleaned = GeminiService._strip_code_fences(raw_text)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Gemini JSON array response: %s", raw_text)
            raise GeminiServiceError(
                f"Could not parse response as a JSON array. Raw response: {raw_text[:300]}"
            ) from exc

        if not isinstance(data, list):
            raise GeminiServiceError(
                f"Expected a JSON array but got {type(data).__name__}: {raw_text[:300]}"
            )
        return data