# ai_utils.py
"""Model integrations: Gemini (captioning + image editing) and OpenAI (embeddings)."""
import os
from io import BytesIO

from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# ---- Model names (override via .env if needed) ----
CAPTION_MODEL = os.getenv("GENEDIT_CAPTION_MODEL", "gemini-2.5-flash")
EDIT_MODEL = os.getenv("GENEDIT_EDIT_MODEL", "gemini-2.5-flash-image")
EMBED_MODEL = os.getenv("GENEDIT_EMBED_MODEL", "text-embedding-3-small")

# ---- Prompts ----
CAPTION_PROMPT = """You are a helpful assistant that describes images.
Provide a concise, natural-sounding caption (20-30 words) for the image.
Capture the key subjects, setting and style without being too generic.
"""

EDIT_PROMPT_TEMPLATE = """You are an image editing assistant.
Apply the user's instruction to the provided image and return the edited image.
Change only what the instruction asks for and keep everything else unchanged.

Instruction: {user_prompt}
"""


class ImageEditError(RuntimeError):
    """Raised when the model does not return an edited image."""


# ---- Lazy clients: importing this module never fails because of a missing key ----
_gemini_client = None
_openai_client = None


def _gemini():
    global _gemini_client
    if _gemini_client is None:
        from google import genai

        api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GEN_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY. Add it to your .env file.")
        _gemini_client = genai.Client(api_key=api_key)
    return _gemini_client


def _openai():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY. Add it to your .env file.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def to_png_bytes(image_bytes: bytes) -> bytes:
    """Normalise any image bytes (JPEG, WebP, PNG...) to PNG so stored files match their .png names."""
    with Image.open(BytesIO(image_bytes)) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGBA")
        out = BytesIO()
        img.save(out, format="PNG")
        return out.getvalue()


# ========== GEMINI ==========

def generate_caption(image_path: str) -> str:
    """Generate a short caption for an image with Gemini."""
    with Image.open(image_path) as img:
        img.load()
        response = _gemini().models.generate_content(
            model=CAPTION_MODEL, contents=[CAPTION_PROMPT, img]
        )
    text = (response.text or "").strip()
    if not text:
        raise RuntimeError("Caption model returned an empty response.")
    return text


def edit_image(image_path: str, prompt_text: str) -> bytes:
    """Edit an image from a natural-language instruction. Returns PNG bytes.

    Raises ImageEditError (instead of silently saving a broken file) when the
    model returns no image, e.g. because it refused or replied with text only.
    """
    if not prompt_text or not prompt_text.strip():
        raise ImageEditError("Edit instruction is empty.")

    with Image.open(image_path) as img:
        img.load()
        response = _gemini().models.generate_content(
            model=EDIT_MODEL,
            contents=[EDIT_PROMPT_TEMPLATE.format(user_prompt=prompt_text.strip()), img],
        )

    model_text = []
    for candidate in response.candidates or []:
        parts = candidate.content.parts if candidate.content else None
        for part in parts or []:
            if part.inline_data and part.inline_data.data:
                return to_png_bytes(part.inline_data.data)
            if part.text:
                model_text.append(part.text.strip())

    detail = f" Model said: {' '.join(model_text)}" if model_text else ""
    raise ImageEditError("The model did not return an edited image." + detail)


# ========== OPENAI ==========

def embed_text(text: str) -> list[float]:
    """Embed text with OpenAI for semantic search."""
    response = _openai().embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding
