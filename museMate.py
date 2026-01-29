import os
import json
import base64
import hashlib
from io import BytesIO
from datetime import datetime

import streamlit as st
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from docx import Document
from pypdf import PdfReader
from PIL import Image

# --------------------
# Page Config & CSS
# --------------------
st.set_page_config(
    page_title="MuseMate 🎨🤖",
    page_icon="🤖",
    layout="wide", 
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* --- Global Background --- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        background-attachment: fixed;
        color: #ffffff;
    }

    /* Hide defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* --- Sidebar Glass --- */
    [data-testid="stSidebar"] {
        background: rgba(20, 20, 40, 0.6);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    /* --- Chat Messages Glass --- */
    .stChatMessage {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        margin-bottom: 1rem;
        backdrop-filter: blur(5px);
    }

    /* --- The Glass Pill Input Container --- */
    .glass-pill-container {
        background: rgba(30, 10, 60, 0.7);
        border: 1px solid rgba(168, 85, 247, 0.3); /* Purple glow */
        border-radius: 50px;
        padding: 5px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        display: flex;
        align-items: center;
        transition: border-color 0.3s;
    }
    .glass-pill-container:focus-within {
        border-color: rgba(168, 85, 247, 0.8);
        box-shadow: 0 0 15px rgba(168, 85, 247, 0.3);
    }

    /* --- File Uploader Styling (The + Icon) --- */
    [data-testid="stFileUploader"] {
        flex: 0 0 auto; /* Don't grow */
    }
    /* Hide the 'Browse Files' text and box */
    [data-testid="stFileUploader"] > section > label > span {
        display: none !important;
    }
    [data-testid="stFileUploader"] > section > label {
        width: 40px;
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(168, 85, 247, 0.2);
        border-radius: 50%;
        color: #d8b4fe;
        font-size: 1.5rem;
        cursor: pointer;
        border: none;
        margin: 0;
        padding: 0;
    }
    [data-testid="stFileUploader"] > section > label:hover {
        background: rgba(168, 85, 247, 0.5);
        color: white;
    }

    /* --- Chat Input Styling (Transparent to blend) --- */
    .stChatInput {
        border: none !important;
        background: transparent !important;
        box-shadow: none !important;
        padding: 0 !important;
    }
    .stChatInput > div {
        background: transparent !important;
        box-shadow: none !important;
    }
    /* Text styling */
    .stChatInput textarea {
        color: white !important;
        background: transparent !important;
    }
    .stChatInput textarea::placeholder {
        color: rgba(255, 255, 255, 0.5);
    }

    /* --- Buttons --- */
    .stButton > button {
        background: rgba(168, 85, 247, 0.2);
        border: 1px solid rgba(168, 85, 247, 0.4);
        color: white;
        border-radius: 8px;
    }
    .stButton > button:hover {
        background: rgba(168, 85, 247, 0.5);
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
    ::-webkit-scrollbar-thumb { background: rgba(168, 85, 247, 0.5); border-radius: 4px; }

</style>
""", unsafe_allow_html=True)


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
        return {"next_chat_num": 1, "chats": {}}
    try:
        with open(INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"next_chat_num": 1, "chats": {}}


def save_index(index_data):
    ensure_dirs()
    with open(INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index_data, f, ensure_ascii=False, indent=2)


def chat_path(chat_id: str):
    return os.path.join(CHAT_DIR, f"{chat_id}.json")


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
    items.sort(key=lambda x: x[2], reverse=True)
    return items


DEFAULT_SYSTEM = "You are MuseMate 🎨🤖 a friendly, playful, and creative AI assistant."


def new_chat():
    idx = load_index()
    n = idx.get("next_chat_num", 1)
    idx["next_chat_num"] = n + 1
    save_index(idx)

    display_name = f"Chat {n}"
    chat_id = f"chat_{n}"

    st.session_state.chat_id = chat_id
    st.session_state.chat_name = display_name
    st.session_state.chat_history = [SystemMessage(content=DEFAULT_SYSTEM)]
    st.session_state.context_pack = ""
    st.session_state.uploaded_fingerprints = set()

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
# Upload extraction
# --------------------
def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode("utf-8", errors="ignore").strip()


def extract_text_from_docx(file_bytes: bytes) -> str:
    bio = BytesIO(file_bytes)
    doc = Document(bio)
    parts = [p.text.strip() for p in doc.paragraphs if p.text and p.text.strip()]
    return "\n".join(parts).strip()


def extract_text_from_pdf(file_bytes: bytes) -> str:
    bio = BytesIO(file_bytes)
    reader = PdfReader(bio)
    parts = []
    for page in reader.pages:
        t = (page.extract_text() or "").strip()
        if t:
            parts.append(t)
    return "\n\n".join(parts).strip()


def image_to_data_url(file_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(file_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def analyze_image_with_gemini(primary_llm, file_bytes: bytes, mime: str) -> str:
    data_url = image_to_data_url(file_bytes, mime)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze this image. Describe what's in it and extract any readable text."},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
    )
    result = primary_llm.invoke([msg])
    return (result.content or "").strip()


def fingerprint_file(name: str, file_bytes: bytes) -> str:
    h = hashlib.sha256()
    h.update(name.encode("utf-8", errors="ignore"))
    h.update(file_bytes[:200000])
    return h.hexdigest()


# --------------------
# Session Init
# --------------------
if "chat_id" not in st.session_state:
    new_chat()

if "autosave" not in st.session_state:
    st.session_state.autosave = True

if "uploaded_fingerprints" not in st.session_state:
    st.session_state.uploaded_fingerprints = set()

if "context_pack" not in st.session_state:
    st.session_state.context_pack = ""


# --------------------
# Load Models
# --------------------
gemini_key, openrouter_key, fallback_model, site_url, app_name = get_secrets()
try:
    primary_model, fallback_model_obj = init_models(gemini_key, openrouter_key, fallback_model, site_url, app_name)
except Exception as e:
    st.error(str(e))
    st.stop()


# --------------------
# Sidebar
# --------------------
with st.sidebar:
    st.markdown("### 🎨 MuseMate")
    st.markdown("<div style='font-size:0.8rem; color:#d8b4fe;'>Create. Analyze. Dream.</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("➕ New Chat", use_container_width=True):
            new_chat()
            st.rerun()
    with col2:
        st.session_state.autosave = st.toggle("Auto-save", value=st.session_state.autosave)

    st.markdown("### 💬 History")
    
    chats = list_chats_newest_first()
    
    with st.container():
        for cid, name, updated_at in chats:
            is_current = (cid == st.session_state.chat_id)
            
            if is_current:
                label = f"✨ {name}"
                style = "background: rgba(168, 85, 247, 0.3); border: 1px solid rgba(168, 85, 247, 0.5);"
            else:
                label = name
                style = "background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(255, 255, 255, 0.1);"
            
            c = st.container()
            c.markdown(
                f"""
                <div style='{style} padding: 10px; border-radius: 8px; margin-bottom: 8px; color: white;'>
                    <strong>{label}</strong>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            if c.button(label, key=f"chatbtn_{cid}", use_container_width=True):
                if cid != st.session_state.chat_id:
                    chat_name, history, context_pack = load_chat(cid)
                    st.session_state.chat_id = cid
                    st.session_state.chat_name = chat_name
                    st.session_state.chat_history = history
                    st.session_state.context_pack = context_pack or ""
                    st.session_state.uploaded_fingerprints = set()
                    st.rerun()
                    
    st.divider()
    st.caption("Powered by Gemini & OpenRouter")


# --------------------
# Main Chat Area
# --------------------

st.markdown(
    f"""
    <div style="background: rgba(255, 255, 255, 0.05); padding: 15px; text-align: center; border-radius: 15px; margin-bottom: 20px; backdrop-filter: blur(5px); border: 1px solid rgba(255,255,255,0.1);">
        <h2 style="margin:0; color:white;">{st.session_state.chat_name}</h2>
    </div>
    """, 
    unsafe_allow_html=True
)

# Display History
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(msg.content)

# Helper to build messages
def build_messages_for_model():
    history = list(st.session_state.chat_history)
    ctx = (st.session_state.context_pack or "").strip()
    if ctx:
        ctx_msg = SystemMessage(
            content=(
                "You have access to the following user-provided context from uploaded files/images.\n"
                "Use it to answer questions. If the user asks something not supported by this context, say so.\n\n"
                f"{ctx}"
            )
        )
        if history and isinstance(history[0], SystemMessage):
            return [history[0], ctx_msg] + history[1:]
        return [ctx_msg] + history
    return history


# --------------------
# Input Area (Fixed Layout)
# --------------------
# Context Indicator
if st.session_state.context_pack:
    file_count = st.session_state.context_pack.count("[File:") + st.session_state.context_pack.count("[Image:")
    st.info(f"📎 Context: {file_count} file(s) attached.", icon="✅")

# The Glass Pill Container (Using HTML wrapper around the columns)
st.markdown(
    """
    <div class="glass-pill-container" style="margin-bottom: 20px;">
    """, 
    unsafe_allow_html=True
)

col_upload, col_chat = st.columns([0.5, 10], gap="small")

with col_upload:
    # The "Plus" icon uploader
    uploads = st.file_uploader(
        "➕", 
        type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "webp"],
        accept_multiple_files=True,
        label_visibility="visible"
    )
    
    # Clear button (Small X next to +)
    if st.button("✖", key="clear_ctx", help="Clear Context"):
        st.session_state.context_pack = ""
        st.session_state.uploaded_fingerprints = set()
        st.toast("Context cleared.")
        st.rerun()

with col_chat:
    # The Text Input
    prompt = st.chat_input("Ask MuseMate anything...")

st.markdown("</div>", unsafe_allow_html=True)


# --------------------
# Logic
# --------------------

# 1. Chatting
if prompt:
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    with st.chat_message("user", avatar="👤"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            model_messages = build_messages_for_model()
            result = invoke_with_fallback(primary_model, fallback_model_obj, model_messages)
            st.markdown(result.content)

    st.session_state.chat_history.append(AIMessage(content=result.content))

    if st.session_state.autosave:
        save_chat(
            st.session_state.chat_id,
            st.session_state.chat_name,
            st.session_state.chat_history,
            st.session_state.context_pack,
        )

# 2. Uploading
if 'uploads' in locals() and uploads:
    added_any = False
    for f in uploads:
        name = f.name
        file_bytes = f.getvalue()
        ext = name.lower().split(".")[-1]
        fp = fingerprint_file(name, file_bytes)

        if fp in st.session_state.uploaded_fingerprints:
            continue
        st.session_state.uploaded_fingerprints.add(fp)

        # Images
        if ext in ("png", "jpg", "jpeg", "webp"):
            try:
                img = Image.open(BytesIO(file_bytes))
                st.markdown(
                    f"<div style='text-align:center; margin-bottom:10px;'>🖼️ <b>{name}</b> attached</div>", 
                    unsafe_allow_html=True
                )
                st.image(img, width=100, use_container_width=True)
            except Exception:
                pass

            try:
                mime = f.type or "image/png"
                analysis = analyze_image_with_gemini(primary_model, file_bytes, mime)
                if analysis:
                    st.session_state.context_pack += f"\n\n[Image: {name}]\n{analysis}"
                    added_any = True
            except Exception as e:
                st.warning(f"Image analysis failed: {e}")

        # Docs
        else:
            extracted = ""
            try:
                if ext == "txt":
                    extracted = extract_text_from_txt(file_bytes)
                elif ext == "docx":
                    extracted = extract_text_from_docx(file_bytes)
                elif ext == "pdf":
                    extracted = extract_text_from_pdf(file_bytes)
            except Exception as e:
                st.warning(f"Read error for {name}: {e}")

            extracted = (extracted or "").strip()
            if extracted:
                MAX_CHARS = 20000
                if len(extracted) > MAX_CHARS:
                    extracted = extracted[:MAX_CHARS] + "\n\n[Truncated]"
                st.session_state.context_pack += f"\n\n[File: {name}]\n{extracted}"
                added_any = True

    if added_any:
        st.toast("Files processed! ✨")
        if st.session_state.autosave:
            save_chat(
                st.session_state.chat_id,
                st.session_state.chat_name,
                st.session_state.chat_history,
                st.session_state.context_pack,
            )
        st.rerun()
