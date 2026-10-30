from __future__ import annotations

import csv
import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

from rag_engine import (  # noqa: E402
    DocumentIngestionError,
    MAX_AUDIO_SIZE_MB,
    MAX_IMAGE_SIZE_MB,
    MAX_PDF_FILES,
    MAX_PDF_PAGES,
    MAX_PDF_SIZE_MB,
    MAX_TOTAL_UPLOAD_MB,
    MAX_VIDEO_SIZE_MB,
    build_messages,
    build_or_load_vector_store,
    chunk_text_for_debug,
    clean_filename,
    retrieve_documents,
    rewrite_followup,
    safe_collection_name,
    source_records,
    stream_text,
    verify_answer,
)

CHROMA_PATH = Path(
    os.getenv("CHROMA_PATH", str(BASE_DIR / "chroma_db"))
)
METRICS_PATH = Path(
    os.getenv("METRICS_PATH", str(BASE_DIR / "metrics.csv"))
)
FEEDBACK_PATH = Path(
    os.getenv("FEEDBACK_PATH", str(BASE_DIR / "feedback.csv"))
)
CHAT_PATH = BASE_DIR / "chat_history"

DEFAULT_MODEL = (
    os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
    or "openai/gpt-oss-20b"
)
DEFAULT_VISION_MODEL = (
    os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b").strip()
    or "qwen/qwen3.8-27b"
)
AUTH_ENABLED = os.getenv(
    "DOCUMIND_AUTH_ENABLED", "false"
).strip().lower() in {"1", "true", "yes", "on"}

st.set_page_config(
    page_title="DocuMind AI | Multimodal RAG",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 1. GENERAL HELPERS
# ============================================================

def show_html(markup: str) -> None:
    """Render custom HTML without Markdown code-block indentation."""
    cleaned = "\n".join(
        line.lstrip()
        for line in markup.strip().splitlines()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


def password_matches(
    password: str,
    salt_hex: str,
    digest_hex: str,
) -> bool:
    """Verify a PBKDF2-SHA256 password hash."""
    try:
        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            260_000,
        ).hex()

        return hmac.compare_digest(actual, digest_hex)

    except (ValueError, TypeError):
        return False


def load_users() -> dict[str, dict[str, str]]:
    """Load user hashes from the ignored local users.json file."""
    config_path = BASE_DIR / "users.json"

    if config_path.is_file():
        try:
            data = json.loads(
                config_path.read_text(encoding="utf-8")
            )
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    try:
        users = st.secrets.get("users", {})
        return dict(users) if users else {}
    except Exception:
        return {}


def require_login() -> str:
    """
    Optional local login.

    For a public application, use managed OIDC authentication.
    """
    if not AUTH_ENABLED:
        # Convenient single-user mode for local development.
        # Enable authentication before exposing the app publicly.
        st.session_state.workspace_id = "local-default"
        return "Local workspace"

    if st.session_state.get("authenticated_user"):
        username = st.session_state.authenticated_user
        st.session_state.workspace_id = (
            "user-" + username.casefold()
        )
        return username

    st.markdown("## Sign in to DocuMind AI")
    st.caption(
        "Access is restricted. Document indexes and saved "
        "conversations are separated by username."
    )

    users = load_users()

    if not users:
        st.error(
            "Authentication is enabled, but no users were found. "
            "Run `python make_user_hash.py` to create users.json."
        )
        st.stop()

    with st.form("login_form"):
        username = st.text_input(
            "Username",
            max_chars=80,
        ).strip()

        password = st.text_input(
            "Password",
            type="password",
        )

        submitted = st.form_submit_button(
            "Sign in",
            use_container_width=True,
        )

    if submitted:
        record = users.get(username)

        if isinstance(record, dict) and password_matches(
            password,
            str(record.get("salt", "")),
            str(record.get("password_hash", "")),
        ):
            st.session_state.authenticated_user = username
            st.session_state.workspace_id = (
                "user-" + username.casefold()
            )
            st.session_state.history_loaded = False
            st.rerun()

        st.error("Invalid username or password.")

    st.stop()
    return ""


def make_llm(api_key: str, model_name: str) -> ChatGroq:
    """Construct the Groq chat model."""
    return ChatGroq(
        model=model_name,
        api_key=api_key,
        temperature=0.2,
        max_retries=1,
        timeout=60,
    )


def friendly_error(
    error: Exception,
    model_name: str,
) -> str:
    """Translate common failures into user-friendly messages."""
    if isinstance(error, DocumentIngestionError):
        return str(error)

    detail = str(error).lower()

    if any(
        value in detail
        for value in (
            "401",
            "invalid api key",
            "authentication",
            "unauthorized",
            "api_key",
        )
    ):
        return (
            "Groq rejected the API key. Check GROQ_API_KEY "
            "in your .env file or Streamlit Secrets."
        )

    if any(
        value in detail
        for value in (
            "429",
            "rate limit",
            "rate_limit",
            "quota",
            "too many requests",
        )
    ):
        return (
            "The Groq rate limit or usage quota was reached. "
            "Check your account limits and try again later."
        )

    if any(
        value in detail
        for value in (
            "model_not_found",
            "model not found",
            "decommissioned",
            "unsupported model",
        )
    ):
        return (
            f"Groq does not recognise model '{model_name}'. "
            "Update GROQ_MODEL to a model available to your account."
        )

    if any(
        value in detail
        for value in (
            "timeout",
            "timed out",
            "connection error",
            "network is unreachable",
            "connection refused",
        )
    ):
        return (
            "The request timed out or could not connect. "
            "Check your internet connection and try again."
        )

    return (
        "The request failed. Expand Technical details to inspect "
        "the error, then check configuration and connectivity."
    )


def safe_workspace_token() -> str:
    """Create a non-identifying token for the current workspace."""
    workspace = str(
        st.session_state.get("workspace_id", "local")
    )
    return hashlib.sha256(
        workspace.encode("utf-8")
    ).hexdigest()[:24]


def chat_file() -> Path:
    """Return the current workspace's conversation file."""
    return CHAT_PATH / f"{safe_workspace_token()}.json"


def load_chat_history() -> None:
    """Load persisted history when authentication is enabled."""
    if st.session_state.get("history_loaded"):
        return

    st.session_state.history_loaded = True

    if not AUTH_ENABLED:
        return

    try:
        path = chat_file()

        if path.exists():
            saved = json.loads(
                path.read_text(encoding="utf-8")
            )

            if isinstance(saved, list):
                st.session_state.messages = saved[-100:]

    except Exception:
        st.session_state.messages = []


def persist_chat_history() -> None:
    """Save recent messages for an authenticated workspace."""
    if not AUTH_ENABLED:
        return

    try:
        CHAT_PATH.mkdir(parents=True, exist_ok=True)

        path = chat_file()
        temp_path = path.with_suffix(".tmp")

        temp_path.write_text(
            json.dumps(
                st.session_state.messages[-100:],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        temp_path.replace(path)

    except Exception:
        # Session memory remains usable if disk persistence fails.
        pass


def append_metric(row: dict[str, Any]) -> None:
    """Log hashed question identifiers and performance metrics."""
    columns = [
        "timestamp_utc",
        "workspace_hash",
        "question_hash",
        "model",
        "retrieval_mode",
        "top_k",
        "reranking",
        "verification",
        "retrieval_ms",
        "generation_ms",
        "total_ms",
        "sources_count",
        "success",
    ]

    try:
        METRICS_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        exists = METRICS_PATH.exists()

        with METRICS_PATH.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:

            writer = csv.DictWriter(
                handle,
                fieldnames=columns,
                extrasaction="ignore",
            )

            if not exists:
                writer.writeheader()

            writer.writerow({
                key: row.get(key, "")
                for key in columns
            })

    except Exception:
        pass


def save_feedback(
    message: dict[str, Any],
    rating: str,
    collection_name: str | None,
) -> None:
    """Store rating metadata without writing raw question content."""
    fields = [
        "timestamp_utc",
        "interaction_id",
        "rating",
        "question_sha256",
        "collection",
    ]

    FEEDBACK_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    needs_header = not FEEDBACK_PATH.exists()

    question_hash = hashlib.sha256(
        str(message.get("question", "")).encode("utf-8")
    ).hexdigest()

    with FEEDBACK_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
        )

        if needs_header:
            writer.writeheader()

        writer.writerow({
            "timestamp_utc": datetime.now(
                timezone.utc
            ).isoformat(),
            "interaction_id": message.get("id", ""),
            "rating": rating,
            "question_sha256": question_hash,
            "collection": collection_name or "",
        })


def transcript_markdown(
    messages: list[dict[str, Any]],
) -> str:
    """Export the conversation and any available source references."""
    result = [
        "# DocuMind AI conversation",
        "",
    ]

    for message in messages:
        role = (
            "You"
            if message.get("role") == "user"
            else "DocuMind AI"
        )

        result.extend([
            f"## {role}",
            "",
            str(message.get("content", "")),
            "",
        ])

        for source in message.get("sources", []):
            result.append(
                f"- {source.get('label', 'Source')}: "
                f"{source.get('file_name', 'file')}, "
                f"page {source.get('page', '—')}, "
                f"timestamp {source.get('timestamp_start', '')}"
            )

        result.append("")

    return "\n".join(result)


def reset_collection() -> None:
    """Delete the current Chroma collection and reset the UI state."""
    store = st.session_state.get("vector_store")
    target_key = st.session_state.get("active_index_key")

    try:
        if store is not None:
            store.delete_collection()
    except Exception as exc:
        st.session_state.last_index_error = (
            f"Could not delete collection: {exc}"
        )

    st.session_state.deleted_index_key = target_key
    st.session_state.vector_store = None
    st.session_state.collection_name = None
    st.session_state.active_index_key = None
    st.session_state.doc_stats = None
    st.session_state.messages = []

    persist_chat_history()


def init_state() -> None:
    """Initialise session variables."""
    defaults = {
        "messages": [],
        "vector_store": None,
        "collection_name": None,
        "active_index_key": None,
        "doc_stats": None,
        "last_index_error": None,
        "failed_index_key": None,
        "deleted_index_key": None,
        "feedback_status": {},
        "history_loaded": False,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ============================================================
# 2. AUTHENTICATION AND STATE
# ============================================================

current_user = require_login()

init_state()
load_chat_history()


# ============================================================
# 3. STYLING
# ============================================================

show_html("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --ink:#20243A;
    --muted:#747B91;
    --line:#E7E9F2;
}

html,
body,
[data-testid="stAppViewContainer"] {
    font-family:'DM Sans',sans-serif;
    background:
        radial-gradient(
            circle at 85% 0%,
            rgba(116,103,240,.08),
            transparent 28%
        ),
        #F7F8FC;
    color:var(--ink);
}

[data-testid="stHeader"] {
    background:transparent;
}

[data-testid="stMainBlockContainer"] {
    max-width:1240px;
    padding-top:1.35rem;
    padding-bottom:2.5rem;
}

h1,h2,h3 {
    font-family:'Manrope',sans-serif!important;
    color:var(--ink);
    letter-spacing:-.5px;
}

p,li {
    color:#535B73;
    line-height:1.65;
}

/* Sidebar visibility and layout */

[data-testid="stSidebar"] {
    display:block!important;
    visibility:visible!important;
    opacity:1!important;
    width:19rem!important;
    min-width:19rem!important;
    max-width:19rem!important;
    flex:0 0 19rem!important;
    transform:translateX(0)!important;
    margin-left:0!important;
    background:linear-gradient(
        180deg,
        #171A2E,
        #131626
    )!important;
    border-right:1px solid rgba(255,255,255,.07);
}

[data-testid="stSidebarContent"] {
    visibility:visible!important;
    opacity:1!important;
    width:100%!important;
}

/* Apply the custom font to text, without overriding icon fonts */

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] textarea,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-family:'DM Sans',sans-serif;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] label {
    color:#F5F6FF!important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color:#B7BCD0;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] section {
    background:rgba(255,255,255,.045);
    border:1px dashed rgba(194,190,255,.38);
    border-radius:14px;
}

[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background:#7569EF!important;
    color:white!important;
    border:0!important;
    border-radius:9px!important;
}

.sidebar-label {
    color:#999FBE;
    font-size:10px;
    font-weight:700;
    letter-spacing:1.4px;
    text-transform:uppercase;
    margin:8px 0;
}

.sidebar-status {
    display:flex;
    gap:9px;
    align-items:center;
    padding:12px;
    border:1px solid rgba(255,255,255,.09);
    background:rgba(255,255,255,.045);
    border-radius:13px;
    margin:10px 0 12px;
}

.status-dot {
    width:8px;
    height:8px;
    border-radius:50%;
    background:#52D3A0;
    box-shadow:0 0 0 4px rgba(82,211,160,.12);
    flex-shrink:0;
}

.status-dot.off {
    background:#E5B45C;
    box-shadow:0 0 0 4px rgba(229,180,92,.12);
}

.status-title {
    color:#F4F5FF;
    font-size:12px;
    font-weight:700;
}

.status-sub {
    color:#9FA6C2;
    font-size:10px;
    margin-top:2px;
    line-height:1.45;
}

/* Hero */

.hero {
    padding:30px 32px;
    border-radius:25px;
    border:1px solid #E9E7FF;
    background:linear-gradient(
        118deg,
        #fff 0%,
        #F9F8FF 58%,
        #F0EEFF 100%
    );
    margin-bottom:20px;
    box-shadow:0 12px 40px rgba(33,32,78,.045);
}

.hero-tag {
    display:inline-block;
    padding:7px 12px;
    background:#EEECFF;
    color:#6355D9;
    font-size:10px;
    font-weight:700;
    text-transform:uppercase;
    letter-spacing:1.1px;
    border-radius:100px;
    margin-bottom:15px;
}

.hero h1 {
    font-size:clamp(30px,4vw,45px);
    line-height:1.12;
    margin:0;
    color:#20213A;
    font-weight:800;
    letter-spacing:-1.5px;
}

.hero p {
    font-size:14px;
    max-width:730px;
    color:#747990;
    margin-top:14px;
    margin-bottom:0;
}

.hero-bottom {
    display:flex;
    flex-wrap:wrap;
    gap:9px;
    margin-top:22px;
}

.hero-chip {
    font-size:11px;
    font-weight:600;
    color:#595D78;
    background:rgba(255,255,255,.85);
    border:1px solid #E9E7F6;
    border-radius:10px;
    padding:8px 11px;
}

/* Metrics */

[data-testid="stMetric"] {
    background:white;
    border:1px solid #E8EAF2;
    border-radius:15px;
    padding:14px 16px;
    box-shadow:0 5px 18px rgba(26,29,57,.025);
}

/* Chat messages */

[data-testid="stChatMessage"] {
    background:#FAFBFF;
    border:1px solid #E8EAF2;
    border-radius:16px;
    padding:15px 17px;
    margin:8px 0 12px;
    box-shadow:0 4px 16px rgba(26,29,57,.025);
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] h1,
[data-testid="stChatMessage"] h2,
[data-testid="stChatMessage"] h3 {
    color:#29314A!important;
    opacity:1!important;
}

[data-testid="stChatMessage"] table {
    width:100%;
    border-collapse:collapse;
    background:#fff;
    color:#29314A;
}

[data-testid="stChatMessage"] th {
    background:#F0EEFF!important;
    color:#40358F!important;
    padding:9px;
    border:1px solid #E1E5F0!important;
}

[data-testid="stChatMessage"] td {
    background:#fff!important;
    color:#29314A!important;
    padding:9px;
    border:1px solid #E1E5F0!important;
}

/* Chat input */

[data-testid="stChatInput"] {
    border:1px solid #DDDFEC;
    border-radius:15px;
    background:#252630;
    box-shadow:0 8px 25px rgba(30,32,72,.06);
}

[data-testid="stChatInput"] textarea,
[data-testid="stChatInput"] textarea:focus {
    color:#fff!important;
    -webkit-text-fill-color:#fff!important;
    caret-color:#fff!important;
    background:transparent!important;
    opacity:1!important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color:#C5C9D8!important;
    -webkit-text-fill-color:#C5C9D8!important;
    opacity:1!important;
}

.stButton>button {
    border-radius:11px;
    min-height:39px;
    font-weight:600;
    border:1px solid #E1E3EF;
}

.stButton>button:hover {
    border-color:#A9A1FF;
    color:#5D4EDB;
}

[data-testid="stExpander"] {
    border:1px solid #E7E9F2;
    border-radius:13px;
    background:#fff;
}

.footer-note {
    text-align:center;
    color:#9399AC;
    font-size:11px;
    margin-top:28px;
    padding-top:15px;
    border-top:1px solid #E7E9F2;
}

@media(max-width:768px) {
    .hero {
        padding:22px 18px;
    }

    .hero h1 {
        letter-spacing:-.8px;
    }
}
</style>
""")


# ============================================================
# 4. SIDEBAR
# ============================================================

api_key = os.getenv("GROQ_API_KEY", "").strip()

try:
    api_key = api_key or str(
        st.secrets.get("GROQ_API_KEY", "")
    ).strip()
except Exception:
    pass

with st.sidebar:

    show_html("""
    <div style="padding:8px 0 18px">
      <div style="display:flex;align-items:center;gap:11px">
        <div style="
          width:43px;height:43px;
          background:linear-gradient(135deg,#9A8FFF,#6255DC);
          border-radius:14px;
          display:flex;align-items:center;justify-content:center;
          color:white;font-size:22px;font-weight:800">✦</div>

        <div>
          <div style="color:#fff;font-size:20px;font-weight:800">
            DocuMind AI
          </div>
          <div style="
            color:#A7AED0;font-size:10px;font-weight:600;
            letter-spacing:1.2px;text-transform:uppercase">
            Multimodal Intelligence
          </div>
        </div>
      </div>
    </div>

    <div style="
      height:1px;background:rgba(255,255,255,.1);
      margin-bottom:17px"></div>

    <div class="sidebar-label">
      Workspace · Files
    </div>
    """)

    # One upload widget accepts all supported file formats.
    uploaded_files = st.file_uploader(
        "Upload documents, images, audio or video",
        type=[
            "pdf",
            "png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff",
            "mp4", "mov", "m4v", "webm", "mkv",
            "mp3", "wav", "m4a", "ogg", "flac", "weba", "mpga",
        ],
        accept_multiple_files=True,
        key="documind_multimodal_uploader",
        max_upload_size=MAX_VIDEO_SIZE_MB,
        help=(
            "PDF, image, video and audio files. "
            "Large videos can take longer to process."
        ),
    )

    st.caption(
        f"Up to {MAX_PDF_FILES} files · "
        f"Combined limit {MAX_TOTAL_UPLOAD_MB} MB"
    )

    use_ocr = st.checkbox(
        "OCR scanned PDFs and images",
        value=True,
    )

    use_vision = st.checkbox(
        "AI image/frame descriptions",
        value=True,
        help=(
            "Uses a Groq vision model for images and sampled video frames."
        ),
    )

    use_reranker = st.checkbox(
        "Use cross-encoder reranking",
        value=False,
    )

    use_query_rewrite = st.checkbox(
        "Rewrite follow-up queries",
        value=True,
    )

    use_verification = st.checkbox(
        "Verify answer against sources",
        value=True,
    )

    top_k = st.slider(
        "Source chunks to retrieve",
        2,
        10,
        5,
    )

    st.markdown("---")

    show_html(
        '<div class="sidebar-label">AI connection</div>'
    )

    if api_key:
        show_html("""
        <div class="sidebar-status">
          <span class="status-dot"></span>
          <div>
            <div class="status-title">Groq is configured</div>
            <div class="status-sub">
              Key loaded from environment or Streamlit Secrets.
              Not shown here.
            </div>
          </div>
        </div>
        """)
    else:
        show_html("""
        <div class="sidebar-status">
          <span class="status-dot off"></span>
          <div>
            <div class="status-title">Groq key not found</div>
            <div class="status-sub">
              Set GROQ_API_KEY in .env.
            </div>
          </div>
        </div>
        """)

    model_name = (
        st.text_input(
            "Text model",
            value=DEFAULT_MODEL,
        ).strip()
        or DEFAULT_MODEL
    )

    vision_model = (
        st.text_input(
            "Vision model",
            value=DEFAULT_VISION_MODEL,
        ).strip()
        or DEFAULT_VISION_MODEL
    )

    if st.button(
        "Test Groq connection",
        use_container_width=True,
    ):
        if not api_key:
            st.error(
                "Add GROQ_API_KEY to .env, then restart Streamlit."
            )
        else:
            try:
                with st.spinner("Testing model..."):
                    make_llm(
                        api_key,
                        model_name,
                    ).invoke("Reply OK.")

                st.success("Text model connection successful.")

            except Exception as exc:
                st.error(
                    friendly_error(exc, model_name)
                )

                with st.expander("Technical details"):
                    st.code(str(exc)[:1500])

    st.markdown("---")

    if (
        AUTH_ENABLED
        and st.session_state.get("authenticated_user")
    ):
        st.caption(f"Signed in as **{current_user}**")

        if st.button(
            "Log out",
            use_container_width=True,
        ):
            persist_chat_history()
            st.session_state.authenticated_user = None
            st.session_state.workspace_id = "logged-out"
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.collection_name = None
            st.session_state.active_index_key = None
            st.session_state.history_loaded = False
            st.rerun()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        persist_chat_history()
        st.rerun()

    if st.session_state.messages:
        st.download_button(
            "Download conversation",
            transcript_markdown(
                st.session_state.messages
            ),
            "documind_chat.md",
            "text/markdown",
            use_container_width=True,
        )

    if st.session_state.vector_store is not None:
        with st.expander("Index management"):
            if st.button(
                "Delete this saved index",
                use_container_width=True,
            ):
                reset_collection()
                st.rerun()

    show_html("""
    <div style="
      margin-top:13px;padding:13px;
      background:rgba(255,255,255,.045);
      border:1px solid rgba(255,255,255,.08);
      border-radius:13px">

      <div style="
        color:#F2F3FF;font-size:12px;
        font-weight:700;margin-bottom:7px">
        Pipeline
      </div>

      <div style="
        color:#A7AED0;font-size:11px;line-height:1.9">
        01 · Extract PDFs and media<br>
        02 · OCR, vision and transcription<br>
        03 · Hybrid Chroma + BM25 search<br>
        04 · Rerank and verify answers
      </div>
    </div>
    """)


# ============================================================
# 5. MAIN PAGE
# ============================================================

show_html("""
<div class="hero">
  <div class="hero-tag">
    ✦ AI-powered document intelligence
  </div>

  <h1>Talk to your documents.</h1>

  <p>
    Ask questions across PDFs, images, audio and video.
    Inspect evidence, timestamps and source pages.
  </p>

  <div class="hero-bottom">
    <span class="hero-chip">✧ Hybrid Retrieval</span>
    <span class="hero-chip">◉ Multimodal RAG</span>
    <span class="hero-chip">⌕ Answer Verification</span>
  </div>
</div>
""")

chat_tab, analytics_tab = st.tabs([
    "💬 Chat",
    "📈 Analytics & Evaluation",
])


# ============================================================
# 6. CHAT TAB: FILE INGESTION
# ============================================================

with chat_tab:

    incoming: list[tuple[str, bytes]] = []

    if uploaded_files:
        incoming = [
            (
                clean_filename(item.name),
                item.getvalue(),
            )
            for item in uploaded_files
        ]

    effective_vision = use_vision and bool(api_key)

    current_key = (
        safe_collection_name(
            incoming,
            st.session_state.workspace_id,
            use_ocr,
            effective_vision,
            vision_model,
        )
        if incoming
        else None
    )

    if not incoming:
        st.session_state.vector_store = None
        st.session_state.collection_name = None
        st.session_state.active_index_key = None
        st.session_state.doc_stats = None
        # Keep saved history visible when no file is selected.
        st.session_state.failed_index_key = None
        st.session_state.last_index_error = None
        st.session_state.deleted_index_key = None

    elif current_key == st.session_state.deleted_index_key:
        st.info(
            "This index was deleted. Remove the files and upload "
            "them again to rebuild it."
        )

    elif (
        current_key != st.session_state.active_index_key
        and current_key != st.session_state.failed_index_key
    ):

        had_active_index = (
            st.session_state.active_index_key is not None
        )

        if had_active_index:
            st.session_state.messages = []
            persist_chat_history()

        st.session_state.vector_store = None
        st.session_state.doc_stats = None
        st.session_state.last_index_error = None

        try:
            with st.status(
                "Processing your files...",
                expanded=True,
            ) as status:

                status.write(
                    f"Preparing {len(incoming)} file(s)."
                )

                status.write(
                    "Extracting text, OCR, vision descriptions "
                    "and transcripts where enabled."
                )

                store, collection_name, stats = (
                    build_or_load_vector_store(
                        incoming,
                        workspace_id=st.session_state.workspace_id,
                        use_ocr=use_ocr,
                        use_vision=effective_vision,
                        api_key=api_key,
                        vision_model=vision_model,
                        persist_directory=CHROMA_PATH,
                    )
                )

                status.update(
                    label="Knowledge base ready",
                    state="complete",
                    expanded=False,
                )

            st.session_state.vector_store = store
            st.session_state.collection_name = collection_name
            st.session_state.active_index_key = current_key
            st.session_state.doc_stats = stats
            st.session_state.failed_index_key = None
            st.session_state.deleted_index_key = None

        except Exception as exc:
            st.session_state.vector_store = None
            st.session_state.collection_name = None
            st.session_state.active_index_key = None
            st.session_state.failed_index_key = current_key
            st.session_state.last_index_error = friendly_error(
                exc,
                model_name,
            )

    if st.session_state.last_index_error:
        st.error(st.session_state.last_index_error)

        with st.expander("Troubleshooting"):
            st.markdown(
                "- PDF OCR requires Tesseract and its language data.\n"
                "- Image descriptions require a Groq vision model; local OCR may still work.\n"
                "- Video processing uses FFmpeg; imageio-ffmpeg is included.\n"
                "- Audio transcription requires a Groq API key.\n"
                "- Check the terminal for the full technical traceback."
            )

        if st.button(
            "Retry file processing",
            key="retry_ingest",
        ):
            st.session_state.failed_index_key = None
            st.session_state.last_index_error = None
            st.rerun()

    store = st.session_state.vector_store
    stats = st.session_state.doc_stats

    if store is not None and stats:
        columns = st.columns(4)

        columns[0].metric(
            "Files",
            len(stats.get("files", [])),
        )
        columns[1].metric(
            "Pages / media units",
            stats.get("pages", 0),
        )
        columns[2].metric(
            "Chunks",
            stats.get("chunks", 0),
        )
        columns[3].metric(
            "New chunks indexed",
            stats.get("new_chunks_indexed", 0),
        )

        for item in stats.get("files", []):
            file_name = item.get("filename", "file")
            summary = (
                f"{item.get('modality', 'file')} · "
                f"{item.get('text_pages', 0)} text units"
            )

            if item.get("ocr_pages"):
                summary += f" · OCR {item['ocr_pages']} page(s)"

            if item.get("frames_indexed"):
                summary += (
                    f" · {item['frames_indexed']} video frames"
                )

            if item.get("segments"):
                summary += (
                    f" · {item['segments']} transcript segments"
                )

            st.caption(f"📄 **{file_name}** — {summary}")

            if item.get("warning"):
                st.warning(
                    f"{file_name}: {item['warning']}"
                )

        if not st.session_state.messages:
            st.markdown("#### Start with a question")

            s1, s2, s3 = st.columns(3)

            with s1:
                if st.button(
                    "Summarize files",
                    use_container_width=True,
                ):
                    st.session_state.next_prompt = (
                        "Summarize the uploaded files and cite the sources."
                    )
                    st.rerun()

            with s2:
                if st.button(
                    "Compare documents",
                    use_container_width=True,
                ):
                    st.session_state.next_prompt = (
                        "Compare the uploaded documents. Highlight "
                        "agreements, differences, dates, figures, "
                        "and cite the relevant sources."
                    )
                    st.rerun()

            with s3:
                if st.button(
                    "Find key details",
                    use_container_width=True,
                ):
                    st.session_state.next_prompt = (
                        "Extract the most important details, names, "
                        "dates and figures from the uploaded files "
                        "with citations."
                    )
                    st.rerun()

    else:
        show_html("""
        <div style="
          padding:20px;background:#fff;
          border:1px solid #E7E9F2;border-radius:17px">

          <b>Upload files to begin</b>

          <div style="
            font-size:13px;color:#7C8298;margin-top:5px">
            Supported: PDF, PNG/JPG/WEBP, MP4/MOV/WEBM/MKV,
            MP3/WAV/M4A/OGG/FLAC.
          </div>
        </div>
        """)


# ============================================================
# 7. CHAT HISTORY, SOURCES AND FEEDBACK
# ============================================================

with chat_tab:

    st.markdown("### Conversation")

    if not st.session_state.messages:
        if store is None:
            st.caption(
                "Upload a file in the sidebar to enable chat."
            )
        elif not api_key:
            st.warning(
                "Your index is ready, but GROQ_API_KEY was not found."
            )
        else:
            st.caption(
                "Your conversation will appear here. "
                "Ask your first question below."
            )

    for message in st.session_state.messages:
        role = message.get("role", "assistant")

        if role not in ("user", "assistant"):
            continue

        with st.chat_message(role):
            st.markdown(
                str(message.get("content", ""))
            )

            sources = message.get("sources", [])

            if role == "assistant" and sources:
                with st.expander(
                    f"📚 Sources used ({len(sources)})"
                ):
                    for source in sources:
                        location = (
                            f"page {source.get('page', '—')}"
                        )

                        if source.get("timestamp_start"):
                            location += (
                                f", timestamp "
                                f"{source['timestamp_start']}"
                            )

                        st.markdown(
                            f"**{source.get('label', 'Source')} · "
                            f"{source.get('file_name', 'file')} · "
                            f"{location}**"
                        )

                        st.caption(
                            f"{source.get('modality', 'file')} · "
                            f"{source.get('content_kind', 'text')}"
                        )

                        st.markdown(
                            f"> {source.get('excerpt', '')}"
                        )
                        st.divider()

            debug_rows = message.get("retrieval_debug", [])

            if role == "assistant" and debug_rows:
                with st.expander("🔎 Retrieval inspector"):
                    st.caption(
                        "Hybrid search combines semantic and lexical ranking. "
                        "Vector distance is not a probability."
                    )

                    st.dataframe(
                        pd.DataFrame(debug_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

            if (
                role == "assistant"
                and message.get("technical_error")
            ):
                with st.expander("Technical details"):
                    st.code(
                        str(message["technical_error"])[:1800]
                    )

            if (
                role == "assistant"
                and message.get("question")
            ):
                msg_id = str(message.get("id", "unknown"))

                rated = st.session_state.feedback_status.get(
                    msg_id
                )

                if rated:
                    st.caption(f"Feedback saved: {rated}")
                else:
                    f1, f2, _ = st.columns([1, 1, 5])

                    if f1.button(
                        "👍 Helpful",
                        key=f"up_{msg_id}",
                    ):
                        try:
                            save_feedback(
                                message,
                                "helpful",
                                st.session_state.collection_name,
                            )
                        except Exception:
                            pass

                        st.session_state.feedback_status[
                            msg_id
                        ] = "helpful"
                        st.rerun()

                    if f2.button(
                        "👎 Not helpful",
                        key=f"down_{msg_id}",
                    ):
                        try:
                            save_feedback(
                                message,
                                "not_helpful",
                                st.session_state.collection_name,
                            )
                        except Exception:
                            pass

                        st.session_state.feedback_status[
                            msg_id
                        ] = "not_helpful"
                        st.rerun()


# ============================================================
# 8. CHAT INPUT AND ANSWER GENERATION
# ============================================================

with chat_tab:

    can_chat = store is not None and bool(api_key)

    prompt = st.chat_input(
        "Ask across your documents, images, or videos...",
        disabled=not can_chat,
    )

    if st.session_state.get("next_prompt"):
        prompt = st.session_state.pop("next_prompt")

    if not api_key and store is not None:
        st.warning(
            "The knowledge base is ready, but GROQ_API_KEY "
            "is not configured."
        )

    if prompt and can_chat:
        prompt = prompt.strip()

        if prompt:
            llm = make_llm(api_key, model_name)
            old_history = list(st.session_state.messages)
            turn_id = uuid.uuid4().hex

            st.session_state.messages.append({
                "id": turn_id,
                "role": "user",
                "content": prompt,
            })

            with st.chat_message("user"):
                st.markdown(prompt)

            total_start = time.perf_counter()
            retrieval_start = time.perf_counter()

            retrieval_query = prompt

            try:
                if use_query_rewrite:
                    retrieval_query = rewrite_followup(
                        llm,
                        prompt,
                        old_history,
                    )

                retrieval = retrieve_documents(
                    retrieval_query,
                    store,
                    top_k=top_k,
                    use_reranker=use_reranker,
                    hybrid=True,
                )

                retrieval_ms = (
                    time.perf_counter() - retrieval_start
                ) * 1000

                sources = source_records(retrieval)
                debug_rows = chunk_text_for_debug(retrieval)
                generation_start = time.perf_counter()

                with st.chat_message("assistant"):

                    if not retrieval:
                        answer = (
                            "I could not find relevant evidence in the "
                            "uploaded files to answer this question."
                        )
                        st.markdown(answer)

                    else:
                        messages = build_messages(
                            prompt,
                            old_history,
                            retrieval,
                        )

                        if use_verification:
                            with st.spinner(
                                "Generating and verifying the answer "
                                "against sources..."
                            ):
                                first = llm.invoke(messages)

                                answer = str(
                                    getattr(first, "content", first) or ""
                                ).strip()

                                if not answer:
                                    answer = (
                                        "The model returned an empty response. "
                                        "Please rephrase the question."
                                    )

                                answer = verify_answer(
                                    llm,
                                    prompt,
                                    answer,
                                    retrieval,
                                )

                            st.markdown(answer)

                        else:
                            answer = st.write_stream(
                                stream_text(llm, messages)
                            )
                            answer = str(answer or "").strip()

                            if not answer:
                                answer = (
                                    "The model returned an empty response. "
                                    "Please rephrase the question."
                                )
                                st.markdown(answer)

                        if sources:
                            with st.expander(
                                f"📚 Sources used ({len(sources)})"
                            ):
                                for source in sources:
                                    location = (
                                        f"page {source.get('page', '—')}"
                                    )

                                    if source.get("timestamp_start"):
                                        location += (
                                            f", timestamp "
                                            f"{source['timestamp_start']}"
                                        )

                                    st.markdown(
                                        f"**{source['label']} · "
                                        f"{source['file_name']} · "
                                        f"{location}**"
                                    )

                                    st.markdown(
                                        f"> {source['excerpt']}"
                                    )
                                    st.divider()

                generation_ms = (
                    time.perf_counter() - generation_start
                ) * 1000

                assistant_message = {
                    "id": uuid.uuid4().hex,
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "retrieval_debug": debug_rows,
                    "question": prompt,
                    "retrieval_query": retrieval_query,
                    "verification_enabled": use_verification,
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                }

                st.session_state.messages.append(
                    assistant_message
                )

                append_metric({
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "workspace_hash": safe_workspace_token(),
                    "question_hash": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "model": model_name,
                    "retrieval_mode": "hybrid_bm25_dense_rrf",
                    "top_k": top_k,
                    "reranking": use_reranker,
                    "verification": use_verification,
                    "retrieval_ms": round(retrieval_ms, 2),
                    "generation_ms": round(generation_ms, 2),
                    "total_ms": round(
                        (
                            time.perf_counter() - total_start
                        ) * 1000,
                        2,
                    ),
                    "sources_count": len(sources),
                    "success": True,
                })

            except Exception as exc:
                answer = friendly_error(exc, model_name)

                with st.chat_message("assistant"):
                    st.error(answer)

                    with st.expander("Technical details"):
                        st.code(str(exc)[:1800])

                st.session_state.messages.append({
                    "id": uuid.uuid4().hex,
                    "role": "assistant",
                    "content": answer,
                    "sources": [],
                    "question": prompt,
                    "technical_error": str(exc),
                })

                append_metric({
                    "timestamp_utc": datetime.now(
                        timezone.utc
                    ).isoformat(),
                    "workspace_hash": safe_workspace_token(),
                    "question_hash": hashlib.sha256(
                        prompt.encode("utf-8")
                    ).hexdigest(),
                    "model": model_name,
                    "retrieval_mode": "hybrid_bm25_dense_rrf",
                    "top_k": top_k,
                    "reranking": use_reranker,
                    "verification": use_verification,
                    "retrieval_ms": "",
                    "generation_ms": "",
                    "total_ms": round(
                        (
                            time.perf_counter() - total_start
                        ) * 1000,
                        2,
                    ),
                    "sources_count": 0,
                    "success": False,
                })

            persist_chat_history()
            st.rerun()


# ============================================================
# 9. ANALYTICS AND EVALUATION DASHBOARD
# ============================================================

with analytics_tab:

    st.markdown("### Usage and latency")

    st.caption(
        "Metrics store timestamps, hashed question IDs and timings, "
        "not raw question text."
    )

    if METRICS_PATH.exists():
        try:
            metrics_df = pd.read_csv(METRICS_PATH)

            if not metrics_df.empty:
                if "workspace_hash" in metrics_df.columns:
                    metrics_df = metrics_df[
                        metrics_df["workspace_hash"]
                        == safe_workspace_token()
                    ]

                c1, c2, c3, c4 = st.columns(4)

                c1.metric(
                    "Logged queries",
                    len(metrics_df),
                )

                success_values = pd.to_numeric(
                    metrics_df.get(
                        "success",
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                ).fillna(0)

                c2.metric(
                    "Success rate",
                    f"{100 * success_values.mean():.1f}%"
                    if len(metrics_df)
                    else "—",
                )

                retrieval_values = pd.to_numeric(
                    metrics_df.get(
                        "retrieval_ms",
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                )

                c3.metric(
                    "Avg retrieval",
                    f"{retrieval_values.mean():.0f} ms"
                    if retrieval_values.notna().any()
                    else "—",
                )

                total_values = pd.to_numeric(
                    metrics_df.get(
                        "total_ms",
                        pd.Series(dtype=float),
                    ),
                    errors="coerce",
                )

                c4.metric(
                    "Avg total",
                    f"{total_values.mean():.0f} ms"
                    if total_values.notna().any()
                    else "—",
                )

                for column in (
                    "retrieval_ms",
                    "generation_ms",
                    "total_ms",
                ):
                    if column in metrics_df:
                        metrics_df[column] = pd.to_numeric(
                            metrics_df[column],
                            errors="coerce",
                        )

                chart_columns = [
                    column
                    for column in (
                        "retrieval_ms",
                        "generation_ms",
                        "total_ms",
                    )
                    if column in metrics_df
                ]

                if chart_columns and metrics_df[
                    chart_columns
                ].notna().any().any():
                    st.line_chart(
                        metrics_df[chart_columns].reset_index(
                            drop=True
                        )
                    )

                with st.expander("View metric records"):
                    st.dataframe(
                        metrics_df.tail(100),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.download_button(
                    "Download metrics CSV",
                    metrics_df.to_csv(index=False),
                    "documind_metrics.csv",
                    "text/csv",
                )

            else:
                st.info(
                    "No metrics yet. Ask questions in the Chat tab."
                )

        except Exception as exc:
            st.warning(
                f"Metrics file could not be read: {exc}"
            )

    else:
        st.info(
            "No metrics yet. Ask questions in the Chat tab."
        )

    st.markdown("### Retrieval evaluation results")

    results_path = (
        BASE_DIR / "evaluation" / "results.csv"
    )

    if results_path.exists():
        try:
            results_df = pd.read_csv(results_path)

            st.dataframe(
                results_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download evaluation results",
                results_df.to_csv(index=False),
                "documind_evaluation.csv",
                "text/csv",
            )

        except Exception as exc:
            st.warning(
                f"Could not read evaluation results: {exc}"
            )

    else:
        st.info(
            "Run `python evaluate_rag.py --top-k 5` after putting "
            "test PDFs in data/ and editing evaluation/questions.json."
        )

    st.markdown("### Feedback summary")

    if FEEDBACK_PATH.exists():
        try:
            feedback_df = pd.read_csv(FEEDBACK_PATH)

            if "collection" in feedback_df:
                feedback_df = feedback_df[
                    feedback_df["collection"].astype(str)
                    == str(
                        st.session_state.collection_name or ""
                    )
                ]

            if (
                not feedback_df.empty
                and "rating" in feedback_df
            ):
                st.bar_chart(
                    feedback_df["rating"].value_counts()
                )
            else:
                st.caption(
                    "No feedback recorded for this collection."
                )

        except Exception:
            st.caption(
                "Feedback file is not readable yet."
            )

    else:
        st.caption(
            "Rate assistant answers in the Chat tab to collect feedback."
        )

# ============================================================
# 11. FOOTER
# ============================================================

show_html("""
<div class="footer-note">
  DOCUMIND AI · Hybrid Search · Multimodal RAG ·
  Chroma · Groq · Streamlit
</div>
""")
