"""
streamlit_app.py
----------------
Streamlit web UI for the YouTube QA Bot.

Layout:
  Sidebar  — mode info, corpus video browser, session controls
  Main     — chat interface with streaming output and source citations

Session state keys:
  agent             — YouTubeQAAgent singleton (persists across reruns)
  messages          — list of {"role": "user"|"assistant", "content": str, "sources": list}
  mode              — "corpus" | "live" (informational only; agent always queries both)
  last_embed_source — top-scoring RAG source chunk dict for video embed (None if not RAG)

Run locally:
  streamlit run app/streamlit_app.py

Deploy:
  Push to GitHub → Streamlit Community Cloud → set env vars in dashboard
"""

from __future__ import annotations

import html
import json
import os
import re
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT         = Path(__file__).resolve().parent.parent
_AGENT_DIR    = _ROOT / "agent"
_PIPELINE_DIR = _ROOT / "pipeline"

for p in [str(_AGENT_DIR), str(_PIPELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agent import YouTubeQAAgent, _classify_intent_fast  # noqa: E402

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="YouTube QA Bot",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Tighten chat bubbles */
  .stChatMessage { padding: 0.5rem 0; }
  /* Source pill styling */
  .source-pill {
    display: inline-block;
    background: #1e3a5f;
    color: #a8d8f0;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.75rem;
    margin: 2px 3px;
    text-decoration: none;
  }
  .source-pill:hover { background: #2a5080; }
  /* Sidebar section headers */
  .sidebar-section {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #888;
    margin: 1rem 0 0.3rem;
  }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ───────────────────────────────────────────────

def _init_session() -> None:
    if "agent" not in st.session_state:
        st.session_state.agent             = YouTubeQAAgent()
        st.session_state.messages          = []   # list of dicts: role / content / sources
        st.session_state.mode              = "corpus"
        st.session_state.last_embed_source = None  # top RAG chunk for video embed


# ── Metadata loader (corpus video catalog) ────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load_corpus_metadata() -> list[dict]:
    """Load data/metadata.json and return a flat list of video dicts."""
    meta_path = _ROOT / "data" / "metadata.json"
    if not meta_path.exists():
        return []
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
        videos = raw.get("videos", raw) if isinstance(raw, dict) else raw
        if isinstance(videos, dict):
            return list(videos.values())
        return videos
    except Exception:
        return []
    
# -- Rewrite the Youtube URLs (safeguard) ---------------------------------------

def _safe_yt_url(video_id: str, start: int = 0) -> str:
    """Construct YouTube URL from video_id rather than trusting stored URLs."""
    vid_id_clean = re.sub(r"[^A-Za-z0-9_-]", "", video_id)[:11]
    return f"https://www.youtube.com/watch?v={vid_id_clean}&t={start}"

# ── Source rendering ───────────────────────────────────────────────────────────

def _render_sources(sources: list[dict]) -> None:
    """Render source citations as clickable pills under an assistant message."""
    if not sources:
        return
    pills_html = ""
    seen = set()
    for s in sources:
        vid_id = s.get("video_id", "")
        start  = int(s.get("start", 0))
        title  = html.escape(s.get("title", vid_id)[:40])
        key    = f"{vid_id}_{start}"
        if key in seen:
            continue
        seen.add(key)
        ts_label = f"{start // 60}:{start % 60:02d}"
        url = _safe_yt_url(vid_id, start)
        pills_html += (
            f'<a class="source-pill" href="{url}" target="_blank">'
            f"▶ {title} @ {ts_label}"
            f"</a>"
        )
    if pills_html:
        st.markdown(
            f'<div style="margin-top:4px">{pills_html}</div>',
            unsafe_allow_html=True,
        )


# ── Video embed ───────────────────────────────────────────────────────────────

def _render_video_embed(sources: list[dict]) -> None:
    """
    Render a st.video() embed for the top-scoring source chunk.

    Displayed outside the chat bubble as a separate block, inside a collapsed
    expander so it doesn't dominate the layout by default.

    Called from main() using st.session_state["last_embed_source"], so it
    survives st.rerun() and persists until the next RAG answer or reset.
    Never called from _render_history().

    Source dict shape expected (streaming path from agent.last_sources):
        title, video_id, start, end, chunk_text, channel
    The blocking path (source_chunks_for_display) lacks raw video_id + start
    integers and is never used for RAG in the Streamlit UI, so it is not handled.

    Top chunk selection:
        Pinecone returns results sorted by score descending, so sources[0] is
        already the highest scorer. A defensive fallback to max(score) is used
        in case order is ever disrupted (e.g. multi-namespace merge reordering).
    """
    if not sources:
        return

    # Defensive top-chunk selection: trust index 0 (Pinecone score-sorted) but
    # fall back to explicit max if score field is present.
    if all("score" in s for s in sources):
        top = max(sources, key=lambda s: s.get("score", 0))
    else:
        top = sources[0]

    video_id = re.sub(r"[^A-Za-z0-9_-]", "", str(top.get("video_id", "")))[:11]
    start    = int(top.get("start", 0))
    title    = top.get("title", video_id)[:50]

    if not video_id:
        return  # no valid video_id — skip silently

    ts_label = f"{start // 60}:{start % 60:02d}"

    with st.expander(f"▶ Watch: {title} @ {ts_label}", expanded=False):
        st.video(
            f"https://www.youtube.com/watch?v={video_id}",
            start_time=start,
        )

# ── Sidebar ────────────────────────────────────────────────────────────────────

def _render_sidebar() -> None:
    with st.sidebar:
        st.title("🎬 YouTube QA Bot")
        st.caption("Ask questions about science videos from Veritasium, Kurzgesagt, and Big Think.")

        st.markdown('<div class="sidebar-section">Mode</div>', unsafe_allow_html=True)
        mode = st.radio(
            label     = "Query mode",
            options   = ["Corpus (pre-indexed)", "Live URL (paste below)"],
            index     = 0 if st.session_state.mode == "corpus" else 1,
            label_visibility = "collapsed",
        )
        st.session_state.mode = "corpus" if "Corpus" in mode else "live"

        if st.session_state.mode == "live":
            st.info(
                "Paste a YouTube URL in the chat to ingest a new video. "
                "The bot will index it and then answer questions about it.",
                icon="ℹ️",
            )

        # ── Corpus browser ─────────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Corpus — 28 videos</div>', unsafe_allow_html=True)
        videos = _load_corpus_metadata()
        if videos:
            topic_filter = st.selectbox(
                "Filter by topic",
                options=["All"] + sorted({v.get("topic", "Other") for v in videos}),
                label_visibility="collapsed",
            )
            filtered = (
                videos if topic_filter == "All"
                else [v for v in videos if v.get("topic") == topic_filter]
            )
            for v in sorted(filtered, key=lambda x: x.get("title", "")):
                title   = html.escape(v.get("title", v.get("video_id", "Unknown"))[:50])
                channel = html.escape(v.get("channel", ""))
                vid_id  = v.get("video_id", "")
                yt_url  = _safe_yt_url(vid_id) if vid_id else "#"
                st.markdown(
                    f'<a href="{yt_url}" target="_blank" style="font-size:0.8rem; '
                    f'text-decoration:none; color:#a8d8f0;">▶ {title}</a>'
                    f'<div style="font-size:0.7rem; color:#888; margin-bottom:6px">{channel}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("metadata.json not found — run bootstrap_metadata.py first.")

        # ── Session controls ───────────────────────────────────────────────────
        st.markdown('<div class="sidebar-section">Session</div>', unsafe_allow_html=True)
        if st.button("🗑️ Clear conversation", use_container_width=True):
            st.session_state.agent.reset()
            st.session_state.messages          = []
            st.session_state.last_embed_source = None
            st.rerun()

        st.markdown("---")
        st.caption("Built with LangChain · Groq · Pinecone · Streamlit")


# ── Chat history rendering ─────────────────────────────────────────────────────

def _render_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                _render_sources(msg["sources"])


# ── Suggested starter questions ────────────────────────────────────────────────

def _render_starters() -> None:
    if st.session_state.messages:
        return  # only show on empty chat

    st.markdown("#### What would you like to explore?")
    cols = st.columns(2)
    starters = [
        "What is entropy and why does it increase?",
        "How does natural selection actually work?",
        "What happens at the edge of the observable universe?",
        "What videos do you have on mathematics?",
    ]
    for i, q in enumerate(starters):
        if cols[i % 2].button(q, use_container_width=True, key=f"starter_{i}"):
            _handle_user_input(q)
            st.rerun()


# ── Input handler ──────────────────────────────────────────────────────────────

def _handle_user_input(user_input: str) -> None:
    """
    Append user message, stream assistant response, append assistant message.
    Called both from chat_input and starter buttons.
    """
    # Append user message to history
    st.session_state.messages.append({
        "role":    "user",
        "content": user_input,
        "sources": [],
    })

    # Display it immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # Stream the assistant response
    with st.chat_message("assistant"):
        agent: YouTubeQAAgent = st.session_state.agent

        # Use blocking chat() for ingest + metadata (no streaming needed)
        # Use stream_chat() for RAG (token-by-token)
        intent = _classify_intent_fast(user_input)

        if intent == "rag":
            # Streaming path
            placeholder  = st.empty()
            full_answer  = ""
            with st.spinner("Searching..."):
                try:
                    for token in agent.stream_chat(user_input):
                        full_answer += token
                        for char in token:
                            placeholder.markdown(full_answer + "▌")
                except Exception:
                    # Fall back to blocking
                    resp        = agent.chat(user_input)
                    full_answer = resp.answer
                    placeholder.markdown(full_answer)
            sources = agent.last_sources
        else:
            # Blocking path (ingest / metadata)
            with st.spinner(
                "Ingesting video..." if intent == "ingest" else "Looking up..."
            ):
                resp = agent.chat(user_input)
            full_answer = resp.answer
            st.markdown(full_answer)
            sources = resp.sources

        _render_sources(sources)

    # Store top source chunk for video embed — rendered in main() to survive rerun
    if intent == "rag" and sources:
        st.session_state["last_embed_source"] = sources[0]
    elif intent != "rag":
        st.session_state.pop("last_embed_source", None)

    # Persist assistant message
    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "sources": sources,
    })


# ── Main layout ────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session()
    _render_sidebar()

    st.header("🎬 YouTube QA Bot", divider="blue")

    _render_history()
    _render_starters()

    # Video embed for last RAG answer — rendered here so it survives st.rerun()
    if st.session_state.get("last_embed_source"):
        _render_video_embed([st.session_state["last_embed_source"]])

    # Chat input pinned to bottom
    user_input = st.chat_input(
        placeholder=(
            "Ask a question about the corpus videos, or paste a YouTube URL to ingest a new one..."
        )
    )
    if user_input and user_input.strip():
        _handle_user_input(user_input.strip())
        st.rerun()


if __name__ == "__main__":
    main()
