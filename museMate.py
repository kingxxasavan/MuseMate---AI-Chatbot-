import os
import json
import time
from datetime import datetime

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI


# --------------------
# Page Config
# --------------------
st.set_page_config(page_title="MuseMate 🎨🤖", page_icon="🤖", layout="centered")


# --------------------
# Helpers: Storage
# --------------------
CHAT_DIR = "data/chats"

def ensure_chat_dir():
    os.makedirs(CHAT_DIR, exist_ok=True)

def msg_to_dict(m):
    if isinstance(m, SystemMessage):
        return {"type": "system", "content": m.content}
    if isinstance(m, HumanMessage):
        return {"type": "human", "content": m.content}
    if isinstance(m, AIMessage):
        return {"type": "ai", "content": m.content}
    return {"type": "unknown", "content": getattr(m, "content", "")}

def dict_to_msg(d):
    t = d.get("type")
    c = d.get("content", "")
    if t == "system":
        return SystemMessage(content=c)
    if t == "human":
        return HumanMessage(content=c)
    if t == "ai":
        return AIMessage(content=c)
    return AIMessage(content=c)

def save_chat(chat_id: str, history):
    ensure_chat_dir()
    payload = {
        "chat_id": chat_id,
        "updated_at": datetime.now().isoformat(),
        "messages": [msg_to_dict(m) for m in history],
    }
    path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

def load_chat(chat_id: str):
    path = os.path.join(CHAT_DIR, f"{chat_id}.json")
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    messages = payload.get("messages", [])
    return [dict_to_msg(m) for m in messages]

def list_chats():
    ensure_chat_dir()
    items = []
    for name in os.listdir(CHAT_DIR):
        if name.endswith(".json"):
            chat_id = name[:-5]
            path = os.path.join(CHAT_DIR, name)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    payload = json.load(f)
                updated_at = payload.get("updated_at", "")
            except Exception:
                updated_at = ""
            items.append((chat_id, updated_at))
    # sort newest first
    items.sort(key=lambda x: x[1], reverse=True)
    return items


# --------------------
# Secrets + Models
# --------------------
def get_keys_from_secrets():
    """
    Streamlit secrets live in st.secrets when deployed (and locally if you create .streamlit/secrets.toml).
    """
    gem = st.secrets.get("GEMINI_API_KEY")
    or_key = st.secrets.get("OPENROUTER_API_KEY")
    fallback_model = st.secrets.get("OPENROUTER_FALLBACK_MODEL", "deepseek/deepseek-r1:free")
    site_url = st.secrets.get("OPENROUTER_SITE_URL", "")
    app_name = st.secrets.get("OPENROUTER_APP_NAME", "MuseMate")
    return gem, or_key, fallback_model, site_url, app_name


@st.cache_resource
def init_models(gemini_key: str, openrouter_key: str, fallback_model: str, site_url: str, app_name: str):
    if not gemini_key:
        raise RuntimeError("Missing GEMINI_API_KEY in Streamlit secrets.")
    if not openrouter_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in Streamlit secrets.")

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
            "HTTP-Referer": site_url,
            "X-Title": app_name,
        },
        temperature=0.7,
    )

    return primary, fallback


def invoke_with_fallback(primary_llm, fallback_llm, messages):
    try:
        return primary_llm.invoke(messages)
    except Exception as e:
        st.warning(f"Gemini failed — using fallback. ({type(e).__name__})")
        return fallback_llm.invoke(messages)


# --------------------
# Chat State Init
# --------------------
DEFAULT_SYSTEM = "You are MuseMate 🎨🤖 a friendly, playful, and creative AI assistant."

def new_chat():
    chat_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(int(time.time() * 1000))[-6:]
    st.session_state.chat_id = chat_id
    st.session_state.chat_history = [SystemMessage(content=DEFAULT_SYSTEM)]
    # auto-save immediately so it appears in list
    save_chat(chat_id, st.session_state.chat_history)

# Ensure base state
if "chat_id" not in st.session_state:
    new_chat()


# --------------------
# Sidebar: Chat Manager (no keys UI if secrets exist)
# --------------------
with st.sidebar:
    st.subheader("💬 Chats")

    chats = list_chats()
    chat_labels = []
    chat_ids = []

    for cid, updated in chats:
        label = cid
        if updated:
            # keep it short
            label = f"{cid}  •  {updated.split('T')[0]}"
        chat_labels.append(label)
        chat_ids.append(cid)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New chat"):
            new_chat()
            st.rerun()

    with col2:
        autosave = st.toggle("Auto-save", value=st.session_state.get("autosave", True))
        st.session_state.autosave = autosave

    if chat_ids:
        # current selection index
        try:
            current_idx = chat_ids.index(st.session_state.chat_id)
        except ValueError:
            current_idx = 0

        selected = st.selectbox("Load chat", options=list(range(len(chat_ids))), format_func=lambda i: chat_labels[i], index=current_idx)

        load_btn = st.button("📂 Load selected")
        if load_btn:
            cid = chat_ids[selected]
            st.session_state.chat_id = cid
            st.session_state.chat_history = load_chat(cid)
            st.rerun()

    st.divider()
    st.caption("Keys are pulled from Streamlit Secrets (no key UI).")


# --------------------
# Load Models (from secrets)
# --------------------
gemini_key, openrouter_key, fallback_model, site_url, app_name = get_keys_from_secrets()

try:
    primary_model, fallback_model_obj = init_models(gemini_key, openrouter_key, fallback_model, site_url, app_name)
except Exception as e:
    st.error(str(e))
    st.stop()


# --------------------
# UI
# --------------------
st.title("MuseMate 🎨🤖")
st.caption("Your friendly & creative AI chat companion")

# Display messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# User input
if prompt := st.chat_input("Type your message... 💬"):
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("MuseMate is thinking... 🤔"):
            result = invoke_with_fallback(primary_model, fallback_model_obj, st.session_state.chat_history)
            st.markdown(result.content)

    st.session_state.chat_history.append(AIMessage(content=result.content))

    # Save chat
    if st.session_state.get("autosave", True):
        save_chat(st.session_state.chat_id, st.session_state.chat_history)
