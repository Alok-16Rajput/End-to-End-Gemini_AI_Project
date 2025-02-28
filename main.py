import os
from PIL import Image
import streamlit as st
from streamlit_option_menu import option_menu
import google.generativeai as genai
from dotenv import load_dotenv
from gemini_utility import (
    load_gemini_pro_model,
    gemini_pro_response,
    gemini_pro_vision_response,
    embeddings_model_response,
)

# Load API Key from .env file
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Ensure API key is set
if not GOOGLE_API_KEY:
    st.error("API key is missing. Please set GOOGLE_API_KEY in .env file.")
else:
    genai.configure(api_key=GOOGLE_API_KEY)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Gemini AI",
    page_icon="🧠",
    layout="centered",
)

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        'Gemini AI',
        ['ChatBot', 'Image Captioning', 'Embed text', 'Ask me anything'],
        menu_icon='robot',
        icons=['chat-dots-fill', 'image-fill', 'textarea-t', 'patch-question-fill'],
        default_index=0
    )

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    return "assistant" if user_role == "model" else user_role

# ChatBot Page
if selected == 'ChatBot':
    try:
        model = load_gemini_pro_model()

        # Initialize chat session if not already present
        if "chat_session" not in st.session_state:
            st.session_state.chat_session = model.start_chat(history=[])
  
        st.title("🤖 ChatBot")

        # Display chat history
        for message in st.session_state.chat_session.history:
            with st.chat_message(translate_role_for_streamlit(message.role)):
                st.markdown(message.parts[0].text)

        # Input field for user message
        user_prompt = st.chat_input("Ask Gemini-Pro...")
        if user_prompt:
            st.chat_message("user").markdown(user_prompt)

            # Send user's message to Gemini-Pro
            gemini_response = st.session_state.chat_session.send_message(user_prompt)

            # Display response
            with st.chat_message("assistant"):
                st.markdown(gemini_response.text)

    except Exception as e:
        st.error(f"Error: {e}")

# Configure the correct model name
VISION_MODEL = "gemini-1.5-flash"  # Use the latest available model for vision tasks

# Image Captioning Page
def generate_image_caption(image, prompt="Write a short caption for this image"):
    try:
        model = genai.GenerativeModel(VISION_MODEL)
        response = model.generate_content([prompt, image])
        return response.text if response else "No caption generated."
    except Exception as e:
        return f"Error generating image caption: {e}"

# Image Captioning Page
if selected == "Image Captioning":
    st.title("📷 Snap Narrate")

    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image and st.button("Generate Caption"):
        try:
            image = Image.open(uploaded_image)

            col1, col2 = st.columns(2)

            with col1:
                resized_img = image.resize((800, 500))
                st.image(resized_img)

            # Get caption from the updated model
            caption = generate_image_caption(image)

            with col2:
                st.info(caption)

        except Exception as e:
            st.error(f"Error processing image: {e}")


# Text Embedding Model
if selected == "Embed text":
    st.title("🔡 Embed Text")

    user_prompt = st.text_area(label='', placeholder="Enter the text to get embeddings")

    if user_prompt and st.button("Get Response"):
        try:
            response = embeddings_model_response(user_prompt)
            st.markdown(response)
        except Exception as e:
            st.error(f"Error: {e}")

# Ask Me Anything Page
if selected == "Ask me anything":
    st.title("❓ Ask me a question")

    user_prompt = st.text_area(label='', placeholder="Ask me anything...")

    if user_prompt and st.button("Get Response"):
        try:
            response = gemini_pro_response(user_prompt)
            st.markdown(response)
        except Exception as e:
            st.error(f"Error: {e}")
