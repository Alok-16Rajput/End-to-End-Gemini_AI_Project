import os
import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

# Load API key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("API key is missing. Please set GOOGLE_API_KEY in .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# ✅ FIXED MODEL NAMES
MODEL_NAME = "gemini-1.5-pro"  # Check available models using genai.list_models()
VISION_MODEL_NAME = "gemini-1.5-vision"
EMBEDDING_MODEL_NAME = "models/embedding-001"

# Load Gemini Pro model
def load_gemini_pro_model():
    return genai.GenerativeModel(MODEL_NAME)

# Get response from Gemini-Pro Vision (image/text to text)
def gemini_pro_vision_response(prompt, image):
    try:
        model = genai.GenerativeModel(VISION_MODEL_NAME)
        response = model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error generating image caption: {e}"

# Get response from Embeddings model (text to embeddings)
def embeddings_model_response(input_text):
    try:
        embedding = genai.embed_content(
            model=EMBEDDING_MODEL_NAME,
            content=input_text,
            task_type="retrieval_document"
        )
        return embedding["embedding"]
    except Exception as e:
        return f"Error generating embeddings: {e}"

# Get response from Gemini-Pro model (text to text)
def gemini_pro_response(user_prompt):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(user_prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {e}"
