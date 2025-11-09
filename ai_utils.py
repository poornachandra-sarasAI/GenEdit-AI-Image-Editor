# ai_utils.py
import os
import base64
from io import BytesIO
from dotenv import load_dotenv
import requests
from PIL import Image

# ---- Load environment variables ----
load_dotenv()
GEN_API_KEY = os.getenv("GEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---- Initialize Gemini ----
import google.generativeai as genai
genai.configure(api_key=GEN_API_KEY)

# ---- Initialize OpenAI ----
from openai import OpenAI
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Model names ----
CAPTION_MODEL = "gemini-1.5-pro-vision"      # Check available model
EDIT_MODEL    = "gemini-2.5-flash-image"         # Use your latest image-capable Gemini model

# ---- Prompts ----
CAPTION_PROMPT = """You are a helpful assistant that describes images.
Provide a concise, natural-sounding caption (20-30 words) for the image.
Make sure to capture key elements and context without being too generic.
"""

EDIT_PROMPT_TEMPLATE = """You are an image editing assistant.
Given the user's instruction below, apply it and return the edited image.

Instruction: {user_prompt}
"""

# ========== GEMINI FUNCTIONS ==========

def generate_caption_gemini(image_path):
    """Use Gemini Vision to generate an image caption."""
    try:
        GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
        if not GEMINI_API_KEY:
            raise ValueError("Missing GEMINI_API_KEY. Add it to your .env file.")

        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.5-flash-image-preview")
        img = Image.open(image_path)

        response = model.generate_content([CAPTION_PROMPT, img])
        return response.text.strip()
    
    except Exception as e:
        print("Caption generation error:", e)
        return "No caption available."

def edit_image_gemini(image_path, prompt_text, mask_path=None):
    """Use Gemini to edit an image based on natural language instruction."""
    try:
        from google import genai
        from google.genai import types
        from PIL import Image
        img = Image.open(image_path)
        full_prompt = EDIT_PROMPT_TEMPLATE.format(user_prompt=prompt_text)
        client = genai.Client(api_key=GEN_API_KEY)

        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[full_prompt, img],
        )

        edited_bytes = None

        if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    edited_bytes = part.inline_data.data
                    return edited_bytes
        
        # fallback
        return img.tobytes()
    except Exception as e:
        print("Image edit failed:", e)
        with open(image_path, "rb") as f:
            return f.read()

# ========== OPENAI EMBEDDING ==========

def embed_text_openai(text):
    """Use OpenAI embeddings for natural language search."""
    try:
        response = openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print("Embedding error:", e)
        return []

# ========== UNIFIED PUBLIC FUNCTIONS ==========

def generate_caption(image_path):
    return generate_caption_gemini(image_path)

def edit_image(image_path, prompt, mask_path=None):
    return edit_image_gemini(image_path, prompt, mask_path)

def embed_text(text):
    return embed_text_openai(text)