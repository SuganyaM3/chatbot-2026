import os
import streamlit as st
from huggingface_hub import InferenceClient

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Hugging Face Chatbot",
    page_icon="🤖",
    layout="centered"
)

# =========================================================
# DEFAULT MODEL
# =========================================================

DEFAULT_MODEL = "microsoft/Phi-3-mini-4k-instruct"

# =========================================================
# LOAD HUGGING FACE TOKEN
# =========================================================

HF_TOKEN = None

# Try Streamlit secrets first
try:
    HF_TOKEN = st.secrets["HF_TOKEN"]
except Exception:
    pass

# Fallback to environment variable
if not HF_TOKEN:
    HF_TOKEN = os.getenv("HF_TOKEN")

# Stop app if token missing
if not HF_TOKEN:
    st.error("❌ Hugging Face token not found.")
    st.info("Add HF_TOKEN inside Streamlit secrets.")
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("⚙️ Settings")

    model_id = st.text_input(
        "Model ID",
        value=DEFAULT_MODEL
    )

    max_tokens = st.slider(
        "Max Tokens",
        min_value=64,
        max_value=1024,
        value=256,
        step=32
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
# INITIALIZE CLIENT
# =========================================================

client = InferenceClient(
    token=HF_TOKEN
)

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# TITLE
# =========================================================

st.title("🤖 Hugging Face AI Chatbot")

st.caption("Built with Streamlit + Hugging Face Inference API")

# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# =========================================================
# USER INPUT
# =========================================================

user_prompt = st.chat_input("Type your message...")

# =========================================================
# GENERATE RESPONSE
# =========================================================

if user_prompt:

    # Store user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_prompt
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_prompt)

    # Assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                completion = client.chat_completion(
                    model=model_id,
                    messages=st.session_state.messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                assistant_response = (
                    completion.choices[0]
                    .message
                    .content
                )

            except Exception as e:

                assistant_response = f"❌ Error:\n\n{str(e)}"

        st.markdown(assistant_response)

    # Store assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": assistant_response
        }
    )
