"""
agent.py
--------
LangGraph agent for YouTube QA.

Graph: [START] → classify_intent → [rag | metadata | ingest] → respond → [END]

Architecture:
  - Three-way intent routing (classify_intent node, keyword-based, zero LLM cost):
      URL detected       → ingest_node  (live_ingest pipeline)
      Metadata keywords  → metadata_node (VideoMetadataTool)
      Everything else    → rag_node (RAGRetrieverTool, multi_namespace=True)
  - Custom ConversationMemory (5-turn sliding window) from memory.py
  - LangSmith tracing configured via .env (LANGSMITH_TRACING, LANGSMITH_ENDPOINT)

Key classes:
  YouTubeQAAgent  — owns graph + memory + tools per session.
                    Use one instance per browser session (store in st.session_state).

CLI:
  python agent.py
  Commands during interactive session:
    reset   — clear conversation memory
    sources — show source chunks from last RAG answer
    quit    — exit

Changes from Day 3:
  - Added 'ingest' intent classification and ingest node
  - multi_namespace=True by default (corpus + live queried together)
  - ingest_url() imported from pipeline/live_ingest.py
  - Last ingest result stored on agent for Streamlit access
"""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field

from dotenv import load_dotenv                         # metadata resolution fallback

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_PIPELINE_DIR = _ROOT / "pipeline"
_AGENT_DIR    = _ROOT / "agent"

for p in [str(_PIPELINE_DIR), str(_AGENT_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── Local imports ──────────────────────────────────────────────────────────────
from rag_chain import answer, stream_answer, RAGResponse  # noqa: E402
from memory import ConversationMemory                      # noqa: E402
from tools import get_tools                                # noqa: E402
from live_ingest import ingest_url, IngestResult           # noqa: E402

# ── LangGraph / LangChain ──────────────────────────────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# ── Intent keywords ────────────────────────────────────────────────────────────

METADATA_INTENT_KEYWORDS = [
    # original
    "what videos", "which videos", "list videos", "show videos",
    "what topics", "what channels", "catalog", "what do you know about",
    "what's in", "what is in", "available videos", "indexed videos",
    # additions
    "what do you have", "full list", "everything you have",
    "what's indexed", "what is indexed", "all videos",
    "show me everything", "browse", "what can i ask",
]

# Matches any YouTube URL pattern
_YT_URL_PATTERN = re.compile(
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/)[A-Za-z0-9_-]{11}"
)


# ── LangGraph state ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages:     list        # full conversation so far as LangChain message objects
    question:     str         # current user question / raw input
    intent:       str         # 'rag' | 'metadata' | 'ingest'
    answer:       str         # final answer to return to user
    rag_response: Any         # RAGResponse or None


# ── Intent classification ──────────────────────────────────────────────────────

def classify_intent(state: AgentState) -> AgentState:
    """
    Zero-cost keyword routing. Order matters:
      1. URL detected → 'ingest'
      2. Metadata keywords → 'metadata'
      3. Everything else → 'rag'
    """
    question = state["question"].lower().strip()

    if _YT_URL_PATTERN.search(state["question"]):
        intent = "ingest"
    elif any(kw in question for kw in METADATA_INTENT_KEYWORDS):
        intent = "metadata"
    else:
        intent = "rag"

    log.info(f"Intent classified: {intent}")
    return {**state, "intent": intent}


def _route_after_classify(state: AgentState) -> str:
    return state["intent"]


# ── RAG node ───────────────────────────────────────────────────────────────────

def rag_node(state: AgentState) -> AgentState:
    """
    Call rag_chain.answer() with the current question + conversation history.
    multi_namespace=True queries both corpus and live namespaces.
    """
    history  = state["messages"][:-1]  # exclude current HumanMessage
    response: RAGResponse = answer(
        question        = state["question"],
        history         = history,
        multi_namespace = True,
    )
    return {
        **state,
        "answer":       response.answer,
        "rag_response": response,
    }


# ── Metadata node ──────────────────────────────────────────────────────────────

def metadata_node(state: AgentState) -> AgentState:
    """
    Use VideoMetadataTool to answer catalog/listing queries.
    Falls back gracefully if metadata.json is missing.
    Resolves vague references ("that topic") using history + fast LLM.
    """
    tools      = get_tools()
    meta_tool  = next(t for t in tools if t.name == "video_metadata")
    question   = state["question"]

    # History-aware resolution: if question contains vague references,
    # ask the fast LLM to produce a concrete search term
    history = state["messages"][:-1]
    if history and any(
        phrase in question.lower()
        for phrase in ["that topic", "that video", "those videos", "it", "them", "that"]
    ):
        try:
            from groq import Groq
            client = Groq(api_key=os.environ["GROQ_API_KEY"])
            history_text = "\n".join(
                f"{'User' if isinstance(m, HumanMessage) else 'Bot'}: {m.content}"
                for m in history[-6:]
            )
            prompt = (
                f"Given this conversation:\n{history_text}\n\n"
                f"The user now asks: \"{question}\"\n"
                "What concrete topic or video title are they referring to? "
                "Reply with only the search term, nothing else."
            )
            resp = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
            resolved = resp.choices[0].message.content.strip()
            log.info(f"Metadata query resolved: '{question}' → '{resolved}'")
            question = resolved
        except Exception as e:
            log.warning(f"Metadata resolution failed, using original query: {e}")

    raw_result = meta_tool.run(question)
    # Strip internal prefix if present
    answer_text = re.sub(r"^METADATA RESULT:\s*", "", raw_result, flags=re.IGNORECASE)

    return {**state, "answer": answer_text, "rag_response": None}


# ── Ingest node ────────────────────────────────────────────────────────────────

_TECHNICAL_ERRORS = ["Missing environment variable", "Unexpected error"] 

def ingest_node(state: AgentState) -> AgentState:
    """
    Extract the YouTube URL from the user's message, run live_ingest.ingest_url(),
    and return a natural-language status message as the answer.
    """
    raw_message = state["question"]

    # Pull the URL out of the message (user may write "analyse this: <url>")
    url_match = _YT_URL_PATTERN.search(raw_message)
    if not url_match:
        return {
            **state,
            "answer": "I couldn't find a valid YouTube URL in your message. Please paste the full URL.",
            "rag_response": None,
        }

    url = url_match.group(0)
    if not url.startswith("http"):
        url = "https://" + url

    log.info(f"Ingesting URL: {url}")
    result: IngestResult = ingest_url(url)


    if result.already_indexed:
        answer_text = (
            f"I already have **{result.title or result.video_id}** in my knowledge base. "
            "You can ask questions about it now."
        )
    elif result.success:
        answer_text = (
            f"✅ Done! I've added **{result.title}** by {result.channel} to your knowledge base. You can now ask me questions about this video."
        )

    elif result.error and any(result.error.startswith(t) for t in _TECHNICAL_ERRORS):
        answer_text = "❌ Something went wrong while adding that video. Please try again or try a different URL."
    else:
        answer_text = f"❌ {result.error or 'Something went wrong. Please try again.'}"

    return {
        **state,
        "answer":       answer_text,
        "rag_response": None,
        "_last_ingest": result,  # available on state for Streamlit to inspect
    }


# ── Respond node ───────────────────────────────────────────────────────────────

def respond_node(state: AgentState) -> AgentState:
    """
    Passthrough — answer is already assembled by the routing node.
    Exists as an explicit termination point so the graph topology is clear.
    """
    return state


# ── Graph construction ─────────────────────────────────────────────────────────

def _build_graph() -> Any:
    builder = StateGraph(AgentState)

    builder.add_node("classify_intent", classify_intent)
    builder.add_node("rag",             rag_node)
    builder.add_node("metadata",        metadata_node)
    builder.add_node("ingest",          ingest_node)
    builder.add_node("respond",         respond_node)

    builder.add_edge(START, "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {"rag": "rag", "metadata": "metadata", "ingest": "ingest"},
    )
    builder.add_edge("rag",      "respond")
    builder.add_edge("metadata", "respond")
    builder.add_edge("ingest",   "respond")
    builder.add_edge("respond",  END)

    return builder.compile()


# ── Public agent class ─────────────────────────────────────────────────────────

class YouTubeQAAgent:
    """
    One instance per user session. Store in st.session_state in Streamlit.

    Usage:
        agent = YouTubeQAAgent()
        response = agent.chat("What causes black holes?")
        print(response.answer)
        print(response.sources)   # list of source dicts with timestamps
    """

    def __init__(self) -> None:
        self.graph          = _build_graph()
        self.memory         = ConversationMemory(k=5)
        self._last_response: RAGResponse | None = None
        self._last_ingest:   IngestResult | None = None
        self._streamed_chunks: list = []

    def chat(self, question: str) -> "_ChatResponse":
        """
        Process a user message and return a _ChatResponse.
        Always updates memory regardless of intent.
        """
        history = self.memory.to_history()

        initial_state: AgentState = {
            "messages":     history + [HumanMessage(content=question)],
            "question":     question,
            "intent":       "rag",   # overwritten by classify_intent
            "answer":       "",
            "rag_response": None,
        }

        final_state = self.graph.invoke(initial_state)

        self._last_response = final_state.get("rag_response")
        self._last_ingest   = final_state.get("_last_ingest")
        answer_text         = final_state["answer"]

        # Update memory
        self.memory.save_turn(question, answer_text)

        return _ChatResponse(
            answer       = answer_text,
            rag_response = self._last_response,
            ingest_result= self._last_ingest,
            intent       = final_state["intent"],
        )

    def stream_chat(self, question: str):
        """
        Generator yielding answer tokens for Streamlit st.write_stream.
        Only streams for RAG intent; other intents yield the full answer at once.
        Falls back to blocking chat() on any streaming error.

        Yields: str tokens
        """
        self._streamed_chunks = []
        history = self.memory.to_history()
        intent  = _classify_intent_fast(question)

        if intent != "rag":
            # Non-RAG intents: run blocking and yield full answer
            resp = self.chat(question)
            yield resp.answer
            return

        # Stream RAG answer token by token
        full_answer = ""
        try:
            token_stream, chunks = stream_answer(
                question=        question,
                history=         history,
                multi_namespace= True,
            )
            for token in token_stream:
                full_answer += token
                yield token
            self._streamed_chunks = chunks
        except Exception as e:
            log.error(f"Streaming failed, falling back to blocking chat(): {e}")
            resp = self.chat(question)
            yield resp.answer
            return

        # Update memory with streamed answer
        self.memory.save_turn(question, full_answer)

    def reset(self) -> None:
        """Clear conversation memory. Does not affect Pinecone."""
        self.memory.clear()
        self._last_response = None
        self._last_ingest   = None
        log.info("Conversation memory reset.")

    @property
    def last_sources(self) -> list[dict]:
        """
        Return source chunks from the last RAG answer.
        Each dict has: title, video_id, start, end, chunk_text.

        Checks streaming path (_streamed_chunks) first, then blocking
        path (_last_response). Returns empty list if last intent was
        not RAG or no sources were retrieved.
        """

        if self._streamed_chunks:
            # format chunks the same way RAGResponse does
            return [
                {
                    "title":      c.title,
                    "video_id":   c.video_id,
                    "channel":    c.channel,
                    "start":      c.start,
                    "end":        c.end,
                    "chunk_text": c.text,
                }
                for c in self._streamed_chunks
            ]
        if self._last_response is None:
            return []
        return self._last_response.source_chunks_for_display


# ── Helper: fast intent check without full graph invocation ───────────────────

def _classify_intent_fast(question: str) -> str:
    """Mirrors classify_intent node logic without touching the graph."""
    q = question.lower().strip()
    if _YT_URL_PATTERN.search(question):
        return "ingest"
    if any(kw in q for kw in METADATA_INTENT_KEYWORDS):
        return "metadata"
    return "rag"


# ── Response dataclass ─────────────────────────────────────────────────────────

from dataclasses import dataclass, field  # noqa: E402

@dataclass
class _ChatResponse:
    answer:        str
    intent:        str
    rag_response:  Any | None = None
    ingest_result: Any | None = None

    @property
    def sources(self) -> list[dict]:
        if self.rag_response is None:
            return []
        return self.rag_response.source_chunks_for_display


# ── CLI interactive loop ───────────────────────────────────────────────────────

def _cli() -> None:
    print("\n🎬 YouTube QA Bot — interactive session")
    print("Commands: 'reset' | 'sources' | 'quit'\n")

    agent = YouTubeQAAgent()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "reset":
            agent.reset()
            print("Bot: Memory cleared.\n")
            continue
        if user_input.lower() == "sources":
            sources = agent.last_sources
            if not sources:
                print("Bot: No RAG sources from the last response.\n")
            else:
                print("Bot: Sources from last answer:")
                for s in sources:
                    ts = f"{int(s.get('start', 0))}s–{int(s.get('end', 0))}s"
                    print(f"  [{s.get('title', '?')}]  {ts}  — {s.get('chunk_text', '')[:80]}...")
            print()
            continue

        response = agent.chat(user_input)
        print(f"\nBot: {response.answer}\n")
        if response.intent == "rag" and response.sources:
            titles = list({s.get("title", "?") for s in response.sources})
            print(f"  Sources: {', '.join(titles)}\n")


if __name__ == "__main__":
    _cli()
