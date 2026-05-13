import os
import streamlit as st
from huggingface_hub import InferenceClient

# =========================================================
# CONFIGURATION
# =========================================================

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"

st.set_page_config(
    page_title="Hugging Face Chatbot",
    page_icon="🤖",
    layout="centered"
)

# =========================================================
# LOAD HUGGING FACE TOKEN
# =========================================================

try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    st.error("Hugging Face token not found.")
    st.stop()

# =========================================================
# SIDEBAR SETTINGS
# =========================================================

with st.sidebar:

    st.header("⚙️ Model Settings")

    model_id = st.text_input(
        "Model ID",
        value=DEFAULT_MODEL,
        help="Enter any Hugging Face Instruct/Text Generation model"
    )

    max_new_tokens = st.slider(
        "Max New Tokens",
        min_value=32,
        max_value=512,
        value=160,
        step=16
    )

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.5,
        value=0.7,
        step=0.1
    )

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# =========================================================
# INITIALIZE INFERENCE CLIENT
# =========================================================

client = InferenceClient(
    model=model_id,
    token=HF_TOKEN
)

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# PAGE TITLE
# =========================================================

st.title("🤖 Hugging Face Chatbot")

st.markdown(
    """
    Chat with Hugging Face open-source language models using Streamlit.
    """
)

# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# BUILD PROMPT
# =========================================================

def build_prompt(messages):

    prompt = ""

    for message in messages:

        if message["role"] == "user":
            prompt += f"User: {message['content']}\n"

        elif message["role"] == "assistant":
            prompt += f"Assistant: {message['content']}\n"

    prompt += "Assistant:"

    return prompt

# =========================================================
# GENERATE RESPONSE
# =========================================================

def generate_response(messages):

    prompt = build_prompt(messages)

    response = client.text_generation(
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )

    return response.strip()

# =========================================================
# USER INPUT
# =========================================================

user_input = st.chat_input("Type your message...")

if user_input:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                assistant_response = generate_response(
                    st.session_state.messages
                )

            except Exception as e:

                assistant_response = f"❌ Error: {str(e)}"

        st.markdown(assistant_response)

    # Store assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response
        }
    )
