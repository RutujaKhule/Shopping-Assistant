"""
services/gemini.py
=========================================================
Gemini Service Layer
=========================================================
This module wraps all interactions with Google's Gemini models:

1. Gemini Vision  -> identifies a product from an uploaded image
2. Gemini Flash   -> general-purpose text generation

Modified to dynamically fetch Streamlit Secrets at runtime.
"""

import os
import json
import logging
from typing import Any, Dict, List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image
import streamlit as st

# -------------------------------------------------------
# Setup
# -------------------------------------------------------

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GEMINI_VISION_MODEL = os.getenv("GEMINI_VISION_MODEL", "gemini-2.0-flash")
GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.0-flash")


def get_google_api_key() -> Optional[str]:
    """Dynamically fetches the API Key from Streamlit Secrets or environment variables."""
    return st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")


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
    """

    def __init__(
        self,
        vision_model_name: str = GEMINI_VISION_MODEL,
        text_model_name: str = GEMINI_TEXT_MODEL,
    ) -> None:
        self._ensure_api_key()
        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name

        self.vision_model = genai.GenerativeModel(self.vision_model_name)
        self.text_model = genai.GenerativeModel(self.text_model_name)

    @staticmethod
    def _ensure_api_key() -> None:
        """Fail fast with a clear message if no API key is configured at runtime."""
        api_key = get_google_api_key()
        
        if not api_key or api_key == "your_google_gemini_api_key_here":
            raise GeminiServiceError(
                "GOOGLE_API_KEY is missing or unset. Add a valid key to your "
                "Streamlit Secrets or .env file before using GeminiService."
            )
        else:
            genai.configure(api_key=api_key)

    # -----------------------------------------------------
    # FEATURE 1: Product Identification (Gemini Vision)
    # -----------------------------------------------------

    def identify_product(self, image: Image.Image) -> Dict[str, Any]:
        """Identify a product from an uploaded image using Gemini Vision."""
        prompt = """
You are a product identification expert for an e-commerce visual search
assistant. Look carefully at the product image provided and identify
the MAIN PURCHASABLE PRODUCT ONLY.

CRITICAL RULE - Ignore accessories:
If the image shows multiple objects, identify ONLY the largest, main,
purchasable product. Completely ignore any accessories bundled with it.

Respond with ONLY a valid JSON object (no markdown, no code fences, no
extra commentary) using exactly this schema:

{
  "product_name": "string",
  "brand": "string",
  "category": "string",
  "model_number": "string",
  "color": "string",
  "variant": "string",
  "search_query": "string",
  "confidence_notes": "string"
}
"""
        try:
            self._ensure_api_key()
            response = self.vision_model.generate_content(
                [prompt, image],
                request_options={"timeout": 30},
            )
            raw_text = self._extract_text(response)
            return self.parse_json_response(raw_text)
        except GeminiServiceError:
            raise
        except Exception as exc:
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
        """Suggest similar products without accessories."""
        category_line = f"Only products from the same category: {category}." if category else \
            "Only products from the same general category as the original product."

        prompt = f"""
Suggest {count} real, currently-available products similar to:
{product_name}
{category_line}

Respond with ONLY a valid JSON array using exactly this schema:
[
  {{"name": "Product Name", "brand": "Brand Name"}},
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
    # Generic text generation (used by RAG, recommendations)
    # -----------------------------------------------------

    def generate_text(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
    ) -> str:
        """Generate free-form text with Gemini Flash."""
        try:
            self._ensure_api_key()
            model = self.text_model
            if system_instruction:
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
        except Exception as exc:
            logger.exception("Gemini text generation failed")
            raise GeminiServiceError(f"Text generation failed: {exc}") from exc

    def generate_grounded_answer(self, question: str, context: str) -> str:
        """RAG-style answer generation using context."""
        system_instruction = (
            "You are a helpful, honest AI shopping assistant. Answer the "
            "user's question using ONLY the context provided below."
        )
        prompt = f"""
CONTEXT:
{context}

USER QUESTION:
{question}
"""
        return self.generate_text(prompt, system_instruction=system_instruction)

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------

    @staticmethod
    def _extract_text(response: Any) -> str:
        try:
            text = response.text
        except Exception as exc:
            raise GeminiServiceError(
                f"Gemini returned no usable text: {exc}"
            ) from exc

        if not text or not text.strip():
            raise GeminiServiceError("Gemini returned an empty response.")

        return text.strip()

    @staticmethod
    def _strip_code_fences(raw_text: str) -> str:
        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:]
            cleaned = cleaned.strip()
        return cleaned

    @staticmethod
    def parse_json_response(raw_text: str) -> Dict[str, Any]:
        cleaned = GeminiService._strip_code_fences(raw_text)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GeminiServiceError(
                f"Could not parse response as JSON. Raw response: {raw_text[:300]}"
            ) from exc

    @staticmethod
    def parse_json_array_response(raw_text: str) -> List[Any]:
        cleaned = GeminiService._strip_code_fences(raw_text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise GeminiServiceError(
                f"Could not parse response as a JSON array. Raw response: {raw_text[:300]}"
            ) from exc

        if not isinstance(data, list):
            raise GeminiServiceError(
                f"Expected a JSON array but got {type(data).__name__}"
            )
        return data
