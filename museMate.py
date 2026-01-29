import os
import json
import time
import base64
from io import BytesIO
from datetime import datetime

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

# For file parsing (add to requirements)
from docx import Document
from pypdf import PdfReader
from PIL import Image


# --------------------
# Page Config
# --------------------
st.set_page_config(page_title="MuseMate 🎨🤖", page_icon="🤖", layout="centered")


# --------------------
# Paths / Storage
# --------------------
DATA_DIR = "data"
CHAT_DIR = os.path.join(DATA_DIR, "chats")
INDEX_PATH = os.path.join(CHAT_DIR, "_index.json")


def ensure_dirs():
    os.makedirs(CHAT_DIR, exist_ok=True)


def load_index():
    ensure_dirs()
    if not os.path.exists(INDEX_PATH):
        return {"next_chat_num": 1, "chats": {}}  # chats: {chat_id: {"name":..., "updated_at":...}}
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"next_chat_num": 1, "chats": {}}


def save_index(index_data):
    ensure_dirs()
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)


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


def chat_path(chat_id: str):
    return os.path.join(CHAT_DIR, f"{chat_id}.json")


def save_chat(chat_id: str, display_name: str, history, context_pack: str):
    ensure_dirs()
    payload = {
        "chat_id": chat_id,
        "name": display_name,
        "updated_at": datetime.now().isoformat(),
        "context_pack": context_pack or "",
        "messages": [msg_to_dict(m) for m in history],
    }
    with open(chat_path(chat_id), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    idx = load_index()
    idx["chats"][chat_id] = {"name": display_name, "updated_at": payload["updated_at"]}
    save_index(idx)


def load_chat(chat_id: str):
    with open(chat_path(chat_id), "r", encoding="utf-8") as f:
        payload = json.load(f)
    history = [dict_to_msg(m) for m in payload.get("messages", [])]
    context_pack = payload.get("context_pack", "")
    name = payload.get("name", "Chat")
    return name, history, context_pack


def list_chats_newest_first():
    idx = load_index()
    items = []
    for cid, meta in idx.get("chats", {}).items():
        items.append((cid, meta.get("name", "Chat"), meta.get("updated_at", "")))
    items.sort(key=lambda x: x[2], reverse=True)  # newest first
    return items


def new_chat():
    idx = load_index()
    n = idx.get("next_chat_num", 1)

    display_name = f"Chat {n}"
    idx["next_chat_num"] = n + 1
    save_index(idx)

    chat_id = f"chat_{n}"  # simple + stable

    DEFAULT_SYSTEM = "You are MuseMate 🎨🤖 a friendly, playful, and creative AI assistant."
    st.session_state.chat_id = chat_id
    st.session_state.chat_name = display_name
    st.session_state.chat_history = [SystemMessage(content=DEFAULT_SYSTEM)]
    st.session_state.context_pack = ""  # extracted doc text + image analysis summaries

    save_chat(chat_id, display_name, st.session_state.chat_history, st.session_state.context_pack)


# --------------------
# Secrets + Models
# --------------------
def get_secrets():
    gem = st.secrets.get("GEMINI_API_KEY")
    or_key = st.secrets.get("OPENROUTER_API_KEY")
    fallback_model = st.secrets.get("OPENROUTER_FALLBACK_MODEL", "deepseek/deepseek-r1:free")
    site_url = st.secrets.get("OPENROUTER_SITE_URL", "")
    app_name = st.secrets.get("OPENROUTER_APP_NAME", "MuseMate")
    return gem, or_key, fallback_model, site_url, app_name


@st.cache_resource
def init_models(gemini_key: str, openrouter_key: str, fallback_model: str, site_url: str, app_name: str):
    if not gemini_key:
        raise RuntimeError("Missing GEMINI_API_KEY in Streamlit Secrets.")
    if not openrouter_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY in Streamlit Secrets.")

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
# File extraction
# --------------------
def extract_text_from_txt(file_bytes: bytes) -> str:
    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def extract_text_from_docx(file_bytes: bytes) -> str:
    bio = BytesIO(file_bytes)
    doc = Document(bio)
    parts = []
    for p in doc.paragraphs:
        if p.text.strip():
            parts.append(p.text.strip())
    return "\n".join(parts)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    bio = BytesIO(file_bytes)
    reader = PdfReader(bio)
    parts = []
    for page in reader.pages:
        t = page.extract_text() or ""
        t = t.strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts)


def image_to_data_url(file_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def analyze_image_with_gemini(primary_llm, file_bytes: bytes, mime: str) -> str:
    """
    Uses Gemini multimodal via LangChain.
    If your installed versions don’t support this message format, it will fail gracefully.
    """
    data_url = image_to_data_url(file_bytes, mime)

    # OpenAI-style multimodal content often works in modern LC wrappers
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze this image in detail. Describe what's in it and extract any readable text."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    result = primary_llm.invoke([msg])
    return (result.content or "").strip()


# --------------------
# Ensure chat session exists
# --------------------
if "chat_id" not in st.session_state:
    new_chat()


# --------------------
# Load models
# --------------------
gemini_key, openrouter_key, fallback_model, site_url, app_name = get_secrets()
try:
    primary_model, fallback_model_obj = init_models(gemini_key, openrouter_key, fallback_model, site_url, app_name)
except Exception as e:
    st.error(str(e))
    st.stop()


# --------------------
# Sidebar: Chat Manager + Uploads
# --------------------
with st.sidebar:
    st.subheader("💬 Chats")

    chats = list_chats_newest_first()
    chat_ids = [c[0] for c in chats]
    chat_labels = [f"{c[1]}  •  {c[2].split('T')[0] if c[2] else ''}".strip() for c in chats]

    c1, c2 = st.columns(2)
    with c1:
        if st.button("➕ New chat"):
            new_chat()
            st.rerun()

    with c2:
        st.session_state.autosave = st.toggle("Auto-save", value=st.session_state.get("autosave", True))

    # Load selected chat
    if chat_ids:
        # find current index
        try:
            current_idx = chat_ids.index(st.session_state.chat_id)
        except ValueError:
            current_idx = 0

        selected_idx = st.selectbox("Select chat", options=list(range(len(chat_ids))), index=current_idx, format_func=lambda i: chat_labels[i])
        if st.button("📂 Load"):
            cid = chat_ids[selected_idx]
            name, history, context_pack = load_chat(cid)
            st.session_state.chat_id = cid
            st.session_state.chat_name = name
            st.session_state.chat_history = history
            st.session_state.context_pack = context_pack or ""
            st.rerun()

    st.divider()

    st.subheader("📎 Files & Images")
    uploads = st.file_uploader(
        "Upload PDF/TXT/DOCX or images (PNG/JPG/WebP)",
        type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
    )

    if st.button("🧼 Clear uploaded context"):
        st.session_state.context_pack = ""
        # save immediately
        save_chat(st.session_state.chat_id, st.session_state.chat_name, st.session_state.chat_history, st.session_state.context_pack)
        st.success("Cleared.")
        st.rerun()

    if uploads:
        added_any = False
        for f in uploads:
            name = f.name
            bytes_data = f.getvalue()
            ext = name.lower().split(".")[-1]

            # Images
            if ext in ("png", "jpg", "jpeg", "webp"):
                # preview
                try:
                    img = Image.open(BytesIO(bytes_data))
                    st.image(img, caption=name, use_container_width=True)
                except Exception:
                    pass

                # analyze image (Gemini)
                try:
                    mime = f.type or "image/png"
                    analysis = analyze_image_with_gemini(primary_model, bytes_data, mime)
                    if analysis:
                        st.session_state.context_pack += f"\n\n[Image: {name}]\n{analysis}"
                        added_any = True
                except Exception as e:
                    st.warning(f"Couldn’t analyze image {name} ({type(e).__name__}).")

            # Documents
            else:
                extracted = ""
                try:
                    if ext == "txt":
                        extracted = extract_text_from_txt(bytes_data)
                    elif ext == "docx":
                        extracted = extract_text_from_docx(bytes_data)
                    elif ext == "pdf":
                        extracted = extract_text_from_pdf(bytes_data)
                except Exception as e:
                    st.warning(f"Couldn’t read {name} ({type(e).__name__}).")

                extracted = (extracted or "").strip()
                if extracted:
                    # keep it from exploding context (basic limit)
                    MAX_CHARS = 20000
                    if len(extracted) > MAX_CHARS:
                        extracted = extracted[:MAX_CHARS] + "\n\n[Truncated]"
                    st.session_state.context_pack += f"\n\n[File: {name}]\n{extracted}"
                    added_any = True

        if added_any:
            save_chat(st.session_state.chat_id, st.session_state.chat_name, st.session_state.chat_history, st.session_state.context_pack)
            st.success("Added to chat context ✅")


# --------------------
# UI
# --------------------
st.title("MuseMate 🎨🤖")
st.caption("Upload files/images, then ask questions about them.")

# Display chat messages
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)


def build_messages_for_model():
    """
    Inject file/image context as a hidden SystemMessage so user can ask questions about uploads.
    """
    history = list(st.session_state.chat_history)

    ctx = (st.session_state.get("context_pack") or "").strip()
    if ctx:
        ctx_msg = SystemMessage(
            content=(
                "You have access to the following user-provided context from uploaded files/images.\n"
                "Use it to answer questions. If the user asks something not supported by this context, say so.\n\n"
                f"{ctx}"
            )
        )
        # Insert right after the first SystemMessage if present
        if history and isinstance(history[0], SystemMessage):
            return [history[0], ctx_msg] + history[1:]
        return [ctx_msg] + history

    return history


# User input
if prompt := st.chat_input("Type your message... 💬"):
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("MuseMate is thinking... 🤔"):
            model_messages = build_messages_for_model()
            result = invoke_with_fallback(primary_model, fallback_model_obj, model_messages)
            st.markdown(result.content)

    st.session_state.chat_history.append(AIMessage(content=result.content))

    # auto-save
    if st.session_state.get("autosave", True):
        save_chat(st.session_state.chat_id, st.session_state.chat_name, st.session_state.chat_history, st.session_state.get("context_pack", ""))
