import os
import streamlit as st

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


# Page Config
st.set_page_config(page_title="MuseMate 🎨🤖", page_icon="🤖", layout="centered")


# ---------- SIDEBAR: API KEYS ----------
with st.sidebar:
    st.subheader("🔑 API Keys")

    gemini_key = st.text_input(
        "Gemini API Key",
        type="password",
        placeholder="AIza...",
        value=st.session_state.get("gemini_key", "")
    )

    openrouter_key = st.text_input(
        "OpenRouter API Key",
        type="password",
        placeholder="sk-or-...",
        value=st.session_state.get("openrouter_key", "")
    )

    fallback_model = st.text_input(
        "OpenRouter Fallback Model",
        value=st.session_state.get("fallback_model", "deepseek/deepseek-r1:free")
    )

    apply_keys = st.button("Apply Keys")


# ---------- MODEL INIT ----------
@st.cache_resource
def init_models(gemini_key: str, openrouter_key: str, fallback_model: str):
    if not gemini_key or not openrouter_key:
        raise RuntimeError("Missing API keys")

    primary = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        api_key=gemini_key,
    )

    fallback = ChatOpenAI(
        model=fallback_model,
        api_key=openrouter_key,
        base_url="https://openrouter.ai/api/v1",
        default_headers={
            "X-Title": "MuseMate",
        },
        temperature=0.7,
    )

    return primary, fallback


def invoke_with_fallback(primary_llm, fallback_llm, messages):
    try:
        return primary_llm.invoke(messages)
    except Exception:
        st.warning("Gemini failed — switching to fallback model.")
        return fallback_llm.invoke(messages)


# ---------- APPLY KEYS ----------
if apply_keys:
    st.session_state.gemini_key = gemini_key
    st.session_state.openrouter_key = openrouter_key
    st.session_state.fallback_model = fallback_model

    # Clear cached models so they rebuild with new keys
    st.cache_resource.clear()
    st.success("Keys applied successfully!")

# ---------- LOAD MODELS ----------
models_ready = (
    "gemini_key" in st.session_state
    and "openrouter_key" in st.session_state
    and st.session_state.gemini_key
    and st.session_state.openrouter_key
)

primary_model = fallback_model_obj = None

if models_ready:
    try:
        primary_model, fallback_model_obj = init_models(
            st.session_state.gemini_key,
            st.session_state.openrouter_key,
            st.session_state.fallback_model,
        )
    except Exception as e:
        st.error(str(e))


# ---------- CHAT STATE ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        SystemMessage(
            content="You are MuseMate 🎨🤖 a friendly, playful, and creative AI assistant."
        )
    ]


# ---------- UI ----------
st.title("MuseMate 🎨🤖")
st.caption("Your friendly & creative AI chat companion")

if not models_ready:
    st.info("Enter your API keys in the sidebar to start chatting.")


# Display Chat Messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)


# ---------- CHAT INPUT ----------
if models_ready and (prompt := st.chat_input("Type your message... 💬")):
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("MuseMate is thinking... 🤔"):
            result = invoke_with_fallback(
                primary_model,
                fallback_model_obj,
                st.session_state.chat_history,
            )
            st.markdown(result.content)

    st.session_state.chat_history.append(AIMessage(content=result.content))
