"""
rag_chain.py
------------
Core RAG chain for the YouTube QA Bot.

Wires together:
  - retriever.py       → Pinecone similarity search
  - prompts.py         → system + QA prompt templates  (Day 5: extracted to own module)
  - Groq LLM           → Llama 3.1 70B (fast, free)
  - LangSmith tracing  → automatic via LANGCHAIN_* env vars

Exposes two public functions:
  answer(question, namespace, top_k, ...)  → RAGResponse
  stream_answer(question, ...)             → tuple[Iterator[str], list[RetrievedChunk]]  (for Streamlit)

Usage (standalone smoke test):
  python rag_chain.py --question "How does a neural network learn?"
  python rag_chain.py --question "What is dark matter?" --threshold 0.35

Changes from Day 4:
  - All prompt strings + build_prompt() extracted to prompts.py
  - Imports: SYSTEM_PROMPT, NO_CONTEXT_RESPONSE, REWRITE_SYSTEM, build_prompt
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_groq import ChatGroq

from retriever import (
    RetrievedChunk,
    format_context_for_llm,
    retrieve,
    retrieve_multi_namespace,
    PINECONE_NAMESPACE_CORPUS,
)
from prompts import SYSTEM_PROMPT, NO_CONTEXT_RESPONSE, REWRITE_SYSTEM, build_prompt

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Env ────────────────────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY      = os.getenv("GROQ_API_KEY", "")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "youtube-qa-bot")

# LangSmith tracing is configured via .env (LANGSMITH_TRACING, LANGSMITH_API_KEY,
# LANGSMITH_ENDPOINT). LangChain reads these automatically — no manual setup needed.
if LANGSMITH_API_KEY:
    log.info(f"LangSmith tracing enabled → project: {LANGSMITH_PROJECT}")
else:
    log.warning("LANGSMITH_API_KEY not set — tracing disabled.")

# ── LLM config ─────────────────────────────────────────────────────────────────
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_TEMPERATURE = 0.2      # Low temp: factual, grounded responses
GROQ_MAX_TOKENS  = 1024

# -- Rewrite Model config ----------------------------------------------------

REWRITE_MODEL       = "llama-3.1-8b-instant"
REWRITE_TEMPERATURE = 0.0   # Temp set to 0 to prevent any creative deviations
REWRITE_MAX_TOKENS  = 128

# ── Retrieval config ───────────────────────────────────────────────────────────
DEFAULT_TOP_K           = 5
DEFAULT_SCORE_THRESHOLD = 0.28   # Below this → "I don't know" guard
                                    # Reduced from 0.35 on Day 4 (asymmetric embedding fix)

# ── Groq retry config ──────────────────────────────────────────────────────────
GROQ_MAX_RETRIES   = 3      # Maximum attempts before giving up
GROQ_RETRY_BASE    = 2.0    # Base delay in seconds (doubles each retry)
GROQ_RETRY_MAX     = 16.0   # Cap on retry delay


def _invoke_with_retry(chain, prompt_input: dict) -> str:
    """
    Invoke a LangChain chain with exponential backoff on Groq 429 errors.

    Retries up to GROQ_MAX_RETRIES times. Respects Retry-After header when
    present in the exception message. Falls back to base delay otherwise.
    """
    delay = GROQ_RETRY_BASE
    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            return chain.invoke(prompt_input)
        except Exception as e:
            err = str(e)
            is_rate_limit = "429" in err or "rate_limit" in err.lower() or "too many requests" in err.lower()
            if is_rate_limit and attempt < GROQ_MAX_RETRIES:
                # Honour Retry-After if Groq includes it in the error message
                import re as _re
                match = _re.search(r"retry.after[^\d]*(\d+)", err, _re.IGNORECASE)
                wait = float(match.group(1)) if match else min(delay, GROQ_RETRY_MAX)
                log.warning(f"Groq 429 — attempt {attempt}/{GROQ_MAX_RETRIES}, retrying in {wait:.1f}s")
                time.sleep(wait)
                delay = min(delay * 2, GROQ_RETRY_MAX)
            else:
                raise


# ── Response dataclass ─────────────────────────────────────────────────────────

@dataclass
class RAGResponse:
    """
    Structured response from the RAG chain.
    Carries both the answer text and the source chunks for UI rendering.
    """
    answer:    str
    chunks:    list[RetrievedChunk]
    question:  str
    namespace: str
    grounded:  bool = True    # False if no relevant chunks were found

    @property
    def sources(self) -> list[dict]:
        """
        Deduplicated source list for rendering in the UI.
        Groups chunks by video to avoid showing the same video multiple times.
        """
        seen: set[str] = set()
        sources = []
        for chunk in self.chunks:
            if chunk.video_id not in seen:
                seen.add(chunk.video_id)
                sources.append({
                    "title":   chunk.title,
                    "channel": chunk.channel,
                    "topic":   chunk.topic,
                    "link":    f"https://www.youtube.com/watch?v={chunk.video_id}",
                })
        return sources

    @property
    def source_chunks_for_display(self) -> list[dict]:
        """Per-chunk source info for detailed citation display in the UI."""
        return [
            {
                "title":     chunk.title,
                "timestamp": chunk.timestamp_label,
                "link":      chunk.youtube_link,
                "score":     chunk.score,
                "text":      chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
            }
            for chunk in self.chunks
        ]


# ── Singleton LLM ──────────────────────────────────────────────────────────────

_llm: Optional[ChatGroq] = None

def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file."
            )
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=GROQ_MODEL,
            temperature=GROQ_TEMPERATURE,
            max_tokens=GROQ_MAX_TOKENS,
        )
        log.info(f"Groq LLM initialised: {GROQ_MODEL}")
    return _llm

# ── Rewrite LLM singleton ──────────────────────────────────────────────────────

_rewrite_llm: Optional[ChatGroq] = None


def _get_rewrite_llm() -> ChatGroq:
    global _rewrite_llm
    if _rewrite_llm is None:
        _rewrite_llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=REWRITE_MODEL,
            temperature=REWRITE_TEMPERATURE,
            max_tokens=REWRITE_MAX_TOKENS,
        )
        log.info(f"Rewrite LLM initialised: {REWRITE_MODEL}")
    return _rewrite_llm


def rewrite_query(question: str, history: Optional[list] = None) -> str:
    if not history:
        return question

    history_text = ""
    for msg in history:
        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    user_prompt = (
        f"Conversation history:\n{history_text.strip()}\n\n"
        f"Latest question: {question}\n"
        f"Rewritten question:"
    )

    try:
        llm = _get_rewrite_llm()
        from langchain_core.messages import HumanMessage as HMsg, SystemMessage as SMsg
        response  = llm.invoke([SMsg(content=REWRITE_SYSTEM), HMsg(content=user_prompt)])
        rewritten = response.content.strip()
        if rewritten and rewritten != question:
            log.info(f"Query rewritten: {question!r} → {rewritten!r}")
        return rewritten or question
    except Exception as e:
        log.warning(f"Query rewrite failed ({e}) — using original question.")
        return question

# ── Core RAG function ──────────────────────────────────────────────────────────

def answer(
    question: str,
    *,
    namespace: str = PINECONE_NAMESPACE_CORPUS,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    history: Optional[list] = None,
    multi_namespace: bool = False,
) -> RAGResponse:
    """
    Run the full RAG pipeline for a single question.

    Args:
        question:        The user's question.
        namespace:       Pinecone namespace to query ("corpus" or "live").
        top_k:           Number of chunks to retrieve.
        score_threshold: Minimum similarity score — chunks below this are
                         discarded and the "I don't know" response is used.
        history:         Optional list of LangChain message objects for
                         multi-turn conversation context.
        multi_namespace: If True, queries both corpus and live namespaces
                         and merges results. Overrides `namespace`.

    Returns:
        RAGResponse with answer text, source chunks, and metadata.
    """
    if not question.strip():
        return RAGResponse(
            answer="Please ask a question.",
            chunks=[],
            question=question,
            namespace=namespace,
            grounded=False,
        )

    # ── Step 1: Rewrite query for better retrieval ──────────────────────────
    retrieval_query = rewrite_query(question, history)

    # ── Step 2: Retrieve relevant chunks ──────────────────────────────────────
    if multi_namespace:
        chunks = retrieve_multi_namespace(
            retrieval_query,
            top_k=top_k,
            score_threshold=score_threshold,
        )
    else:
        chunks = retrieve(
            retrieval_query,
            namespace=namespace,
            top_k=top_k,
            score_threshold=score_threshold,
        )

    # ── Step 2b: Rewrite fallback — retry with original if rewrite hurt retrieval ─
    if not chunks and retrieval_query != question:
        log.info(f"Rewritten query returned 0 chunks — retrying with original: {question!r}")
        if multi_namespace:
            chunks = retrieve_multi_namespace(
                question,
                top_k=top_k,
                score_threshold=score_threshold,
            )
        else:
            chunks = retrieve(
                question,
                namespace=namespace,
                top_k=top_k,
                score_threshold=score_threshold,
            )

    # ── Step 3: Guard — no relevant context found ──────────────────────────────
    if not chunks:
        log.info(f"No chunks above threshold ({score_threshold}) — returning no-context response.")
        # Manually log to LangSmith so failed retrievals are visible in traces
        try:
            from langsmith import Client
            ls_client = Client()
            ls_client.create_run(
                name="no_context_guard",
                run_type="chain",
                inputs={"question": question, "namespace": namespace,
                        "score_threshold": score_threshold},
                outputs={"answer": NO_CONTEXT_RESPONSE, "grounded": False},
                tags=["no_context", "grounded:false"],
            )
        except Exception:
            pass  # Never let tracing failure break the main flow
        return RAGResponse(
            answer=NO_CONTEXT_RESPONSE,
            chunks=[],
            question=question,
            namespace=namespace,
            grounded=False,
        )

    # ── Step 4: Format context for LLM ────────────────────────────────────────
    context = format_context_for_llm(chunks)

    # ── Step 5: Build prompt and call LLM ─────────────────────────────────────
    prompt   = build_prompt()
    llm      = _get_llm()
    chain    = prompt | llm | StrOutputParser()

    prompt_input = {
        "context":  context,
        "question": question,
        "history":  history or [],
    }

    log.info(f"Calling Groq ({GROQ_MODEL}) with {len(chunks)} context chunks...")
    response_text = _invoke_with_retry(chain, prompt_input)

    return RAGResponse(
        answer=response_text,
        chunks=chunks,
        question=question,
        namespace=namespace,
        grounded=True,
    )


def stream_answer(
    question: str,
    *,
    namespace: str = PINECONE_NAMESPACE_CORPUS,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    history: Optional[list] = None,
    multi_namespace: bool = False,
) -> tuple[Iterator[str], list[RetrievedChunk]]:
    """
    Streaming version of answer() for Streamlit's st.write_stream().

    Returns a tuple of:
      - token_iterator: yields string tokens as they stream from the LLM
      - chunks:         the retrieved source chunks (available immediately,
                        before streaming completes)

    Usage in Streamlit:
        token_stream, chunks = stream_answer(question)
        with st.chat_message("assistant"):
            response_text = st.write_stream(token_stream)
        render_sources(chunks)
    """
    # Rewrite query, then retrieve (blocking) — chunks available before stream starts
    retrieval_query = rewrite_query(question, history)
    if multi_namespace:
        chunks = retrieve_multi_namespace(
            retrieval_query, top_k=top_k, score_threshold=score_threshold
        )
    else:
        chunks = retrieve(
            retrieval_query, namespace=namespace, top_k=top_k,
            score_threshold=score_threshold
        )

    # Rewrite fallback — retry with original if rewrite hurt retrieval
    if not chunks and retrieval_query != question:
        log.info(f"Rewritten query returned 0 chunks — retrying with original: {question!r}")
        if multi_namespace:
            chunks = retrieve_multi_namespace(
                question, top_k=top_k, score_threshold=score_threshold
            )
        else:
            chunks = retrieve(
                question, namespace=namespace, top_k=top_k,
                score_threshold=score_threshold
            )

    if not chunks:
        def _no_context_stream():
            yield NO_CONTEXT_RESPONSE
        return _no_context_stream(), []

    context  = format_context_for_llm(chunks)
    prompt   = build_prompt()
    llm      = _get_llm()
    chain    = prompt | llm | StrOutputParser()

    def _token_stream_with_retry() -> Iterator[str]:
        """Yield tokens from Groq, retrying the stream on 429 errors."""
        delay = GROQ_RETRY_BASE
        for attempt in range(1, GROQ_MAX_RETRIES + 1):
            try:
                for token in chain.stream({
                    "context":  context,
                    "question": question,
                    "history":  history or [],
                }):
                    yield token
                return  # stream completed successfully
            except Exception as e:
                err = str(e)
                is_rate_limit = "429" in err or "rate_limit" in err.lower() or "too many requests" in err.lower()
                if is_rate_limit and attempt < GROQ_MAX_RETRIES:
                    import re as _re
                    match = _re.search(r"retry.after[^\d]*(\d+)", err, _re.IGNORECASE)
                    wait = float(match.group(1)) if match else min(delay, GROQ_RETRY_MAX)
                    log.warning(f"Groq 429 (stream) — attempt {attempt}/{GROQ_MAX_RETRIES}, retrying in {wait:.1f}s")
                    time.sleep(wait)
                    delay = min(delay * 2, GROQ_RETRY_MAX)
                else:
                    raise

    return _token_stream_with_retry(), chunks


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke-test the RAG chain.")
    parser.add_argument("--question",  type=str, required=True)
    parser.add_argument("--namespace", type=str, default="corpus",
                        choices=["corpus", "live"])
    parser.add_argument("--k",         type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--threshold", type=float, default=DEFAULT_SCORE_THRESHOLD)
    parser.add_argument("--multi",     action="store_true",
                        help="Query both corpus and live namespaces")
    args = parser.parse_args()

    print(f"\n{'═'*60}")
    print(f"Question: {args.question}")
    print(f"{'═'*60}\n")

    result = answer(
        args.question,
        namespace=args.namespace,
        top_k=args.k,
        score_threshold=args.threshold,
        multi_namespace=args.multi,
    )

    print("── Answer ──────────────────────────────────────────────")
    print(result.answer)

    if result.chunks:
        print(f"\n── Sources ({len(result.chunks)} chunks) ─────────────────────────")
        for src in result.source_chunks_for_display:
            print(f"  • {src['title']}  [{src['timestamp']}]  score={src['score']}")
            print(f"    {src['link']}")

    print(f"\n── Grounded: {result.grounded}")
    print(f"{'═'*60}\n")
