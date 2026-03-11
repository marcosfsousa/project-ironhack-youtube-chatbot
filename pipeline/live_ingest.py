"""
live_ingest.py
--------------
On-the-fly ingestion of a YouTube URL into the Pinecone 'live' namespace.

Pipeline (in order):
  1. Parse video ID from URL
  2. Check Pinecone for existing vectors → skip if already indexed (Option A)
  3. Fetch real metadata via yt-dlp (title, channel, duration)
  4. Extract transcript via youtube-transcript-api
  5. Clean transcript using cleaner.clean_text()
  6. Chunk transcript using chunker.chunk_segments()
  7. Infer topic label via llama-3.1-8b-instant on first 500 words
  8. Embed chunks via embedder.get_model()
  9. Upsert to Pinecone 'live' namespace

Returns an IngestResult dataclass — never raises; errors are captured in .error.

Edge cases handled:
  - Duplicate URL: detected via Pinecone fetch before any extraction work
  - No captions available: caught, returns descriptive error
  - Private / unavailable video: caught, returns descriptive error
  - yt-dlp unavailable: falls back to LLM-inferred title/channel
  - Any unexpected exception: caught, logged, returned in .error

Usage (standalone):
  python live_ingest.py https://www.youtube.com/watch?v=VIDEO_ID
  python live_ingest.py https://youtu.be/VIDEO_ID --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq                          # LLM fallback + topic inference
from pinecone import Pinecone                  # duplicate check + upsert
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)
from youtube_transcript_api.proxies import GenericProxyConfig

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Path setup — make pipeline/ siblings importable ───────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from cleaner import clean_text, is_sponsor_segment   # noqa: E402
from chunker import chunk_segments                    # noqa: E402
from embedder import get_model                        # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────
CHUNK_WINDOW_SECONDS = 60

TOPIC_CHOICES = [
    "Physics", "Biology", "Chemistry", "Cosmology", "Mathematics",
    "Neuroscience", "Psychology", "Cognitive Science", "Philosophy",
    "Technology", "History", "Education", "Other",
]

# ── Proxy config ───────────────────────────────────────────────────────────────

def _get_proxy_config() -> tuple[str | None, GenericProxyConfig | None]:
    """
    Returns (proxy_url, proxy_config) if IPROYAL_PROXY_URL is set, else (None, None).

    - proxy_url is used by yt-dlp via --proxy flag
    - proxy_config is used by YouTubeTranscriptApi via proxy_config=

    Expected format: http://username:password@geo.iproyal.com:12321
    Set via .env locally or Streamlit secrets on cloud.
    """
    proxy_url = os.environ.get("IPROYAL_PROXY_URL")
    if proxy_url:
        log.info("  Proxy configured — routing YouTube requests via residential proxy.")
        proxy_config = GenericProxyConfig(
            http_url=proxy_url,
            https_url=proxy_url,
        )
        return proxy_url, proxy_config
    return None, None


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class IngestResult:
    video_id:    str
    url:         str
    title:       str        = "Unknown"
    channel:     str        = "Unknown"
    topic:       str        = "Other"
    duration:    int        = 0          # seconds
    chunk_count: int        = 0
    already_indexed: bool   = False
    success:     bool       = False
    error:       str | None = None
    chunks:      list[dict] = field(default_factory=list)

    @property
    def youtube_url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


# ── Video ID parsing ───────────────────────────────────────────────────────────

def parse_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


# ── Duplicate check via Pinecone ───────────────────────────────────────────────

def _is_already_indexed(video_id: str, index, corpus_namespace: str, live_namespace: str) -> tuple[bool, str]:
    """Returns (is_duplicate, namespace_where_found). Checks corpus first."""
    for ns in [corpus_namespace, live_namespace]:
        result = index.fetch(ids=[f"{video_id}_000"], namespace=ns)
        if result.vectors:
            return True, ns
    return False, ""

# ── Real metadata via yt-dlp ───────────────────────────────────────────────────

def _fetch_metadata_yt_dlp(url: str) -> dict | None:
    """
    Run: yt-dlp --dump-json --no-download URL
    Returns dict with keys: title, channel, duration (seconds).
    Returns None if yt-dlp is not installed or the call fails.
    """
    try:
        cmd = ["yt-dlp", "--dump-json", "--no-playlist", url]
        proxy_url, _ = _get_proxy_config()
        if proxy_url:
            cmd += ["--proxy", proxy_url]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            log.warning(f"yt-dlp exited {result.returncode}: {result.stderr[:200]}")
            return None
        data = json.loads(result.stdout)
        return {
            "title":    data.get("title", "Unknown"),
            "channel":  data.get("channel") or data.get("uploader", "Unknown"),
            "duration": int(data.get("duration") or 0),
        }
    except FileNotFoundError:
        log.warning("yt-dlp not found — falling back to LLM metadata inference.")
        return None
    except Exception as e:
        log.warning(f"yt-dlp metadata fetch failed: {e} — falling back to LLM.")
        return None


# ── LLM metadata fallback ──────────────────────────────────────────────────────

def _infer_metadata_llm(first_500_words: str) -> dict:
    """
    Ask llama-3.1-8b-instant to infer title and channel from transcript text.
    Returns dict with keys: title, channel. Topic is inferred separately.
    Falls back to generic strings on any error.
    """
    try:
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        prompt = (
            "Below is the opening of a YouTube video transcript. "
            "Infer a concise, descriptive title (max 10 words) and the most likely "
            "channel name or creator name. Reply ONLY as JSON: "
            '{"title": "...", "channel": "..."}\n\n'
            f"Transcript excerpt:\n{first_500_words}"
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0.1,
        )
        raw = response.choices[0].message.content.strip()
        # Strip markdown fences if present
        raw = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(raw)
        return {
            "title":   data.get("title", "Unknown Video"),
            "channel": data.get("channel", "Unknown Channel"),
        }
    except Exception as e:
        log.warning(f"LLM metadata inference failed: {e}")
        return {"title": "Unknown Video", "channel": "Unknown Channel"}


# ── Topic inference ────────────────────────────────────────────────────────────

def _infer_topic_llm(first_500_words: str) -> str:
    """
    Ask llama-3.1-8b-instant to classify the topic from TOPIC_CHOICES.
    Returns the topic string, defaults to "Other" on any failure.
    """
    try:
        from groq import Groq
        client = Groq(api_key=os.environ["GROQ_API_KEY"])
        choices_str = ", ".join(TOPIC_CHOICES)
        prompt = (
            f"Classify this YouTube video transcript excerpt into EXACTLY ONE of these topics: {choices_str}.\n"
            "Reply with only the topic name, nothing else.\n\n"
            f"Transcript excerpt:\n{first_500_words}"
        )
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0.0,
        )
        raw = response.choices[0].message.content.strip()
        # Validate against known choices
        for choice in TOPIC_CHOICES:
            if choice.lower() in raw.lower():
                return choice
        return "Other"
    except Exception as e:
        log.warning(f"LLM topic inference failed: {e}")
        return "Other"


# ── Transcript extraction ──────────────────────────────────────────────────────

def _extract_transcript(video_id: str) -> tuple[list[dict], str, bool]:
    """
    Extract transcript using youtube-transcript-api v1.x.
    Returns (segments, language_code, is_generated).
    Raises NoTranscriptFound / TranscriptsDisabled / VideoUnavailable on failure.
    """

    _, proxy_config = _get_proxy_config()
    ytt = YouTubeTranscriptApi(proxy_config=proxy_config) if proxy_config else YouTubeTranscriptApi()
    transcript_list = ytt.list(video_id)

    # Preference: human English → any human → auto English → any auto
    try:
        t = transcript_list.find_manually_created_transcript(["en"])
        fetched = t.fetch()
        return _normalise_segments(fetched), t.language_code, False
    except NoTranscriptFound:
        pass

    for t in transcript_list:
        if not t.is_generated:
            fetched = t.fetch()
            return _normalise_segments(fetched), t.language_code, False

    try:
        t = transcript_list.find_generated_transcript(["en"])
        fetched = t.fetch()
        return _normalise_segments(fetched), t.language_code, True
    except NoTranscriptFound:
        pass

    for t in transcript_list:
        if t.is_generated:
            fetched = t.fetch()
            return _normalise_segments(fetched), t.language_code, True

    raise NoTranscriptFound(video_id, ["en"])


def _normalise_segments(fetched) -> list[dict]:
    segments = []
    for item in fetched:
        if hasattr(item, "text"):
            segments.append({
                "start":    round(item.start, 3),
                "duration": round(item.duration, 3),
                "end":      round(item.start + item.duration, 3),
                "text":     item.text,
            })
        else:
            start = item.get("start", 0)
            dur   = item.get("duration", 0)
            segments.append({
                "start":    round(start, 3),
                "duration": round(dur, 3),
                "end":      round(start + dur, 3),
                "text":     item.get("text", ""),
            })
    return segments


# ── Cleaning (reuses cleaner.py logic on raw segments) ────────────────────────

def _clean_segments(segments: list[dict]) -> list[dict]:
    """Apply cleaner.clean_text() to each segment; add flags, never delete."""
    cleaned = []
    for seg in segments:
        text = clean_text(seg["text"])
        out  = {**seg, "text": text}
        flags = []
        if not text:
            flags.append("empty_after_clean")
        if is_sponsor_segment(text):
            flags.append("potential_sponsor")
        if flags:
            out["flags"] = flags
        cleaned.append(out)
    return cleaned


# ── Embedding + Pinecone upsert ────────────────────────────────────────────────

def _embed_and_upsert(
    chunks: list[dict],
    title: str,
    channel: str,
    topic: str,
    index,
    live_namespace: str,
    dry_run: bool = False,
) -> int:
    """
    Embed enriched text ("{title} | {chunk_text}") and upsert to Pinecone.
    Mirrors the asymmetric embedding strategy from indexer.py.
    Returns number of vectors upserted.
    """
    model = get_model()

    # Build enriched texts for embedding
    enriched_texts = [
        f"{title} | {c['text']}" for c in chunks
    ]
    embeddings = model.encode(enriched_texts, normalize_embeddings=True).tolist()

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        vectors.append({
            "id": chunk["chunk_id"],
            "values": embedding,
            "metadata": {
                "chunk_id":   chunk["chunk_id"],
                "video_id":   chunk["video_id"],
                "title":      title,
                "channel":    channel,
                "topic":      topic,
                "start":      chunk["start"],
                "end":        chunk["end"],
                "chunk_text": chunk["text"],
            },
        })

    if dry_run:
        log.info(f"  [dry-run] Would upsert {len(vectors)} vectors to '{live_namespace}'")
        return len(vectors)

    # Upsert in batches of 100 (Pinecone recommended batch size)
    batch_size = 100
    total_upserted = 0
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i : i + batch_size]
        index.upsert(vectors=batch, namespace=live_namespace)
        total_upserted += len(batch)

    return total_upserted


# ── Main public API ────────────────────────────────────────────────────────────

def ingest_url(url: str, dry_run: bool = False) -> IngestResult:
    """
    Full ingestion pipeline for a single YouTube URL.

    This is the function called by agent.py and streamlit_app.py.
    Never raises — all errors are captured in IngestResult.error.

    Args:
        url:     YouTube URL (any standard format)
        dry_run: If True, run all steps except Pinecone upsert

    Returns:
        IngestResult with .success=True on success, .error set on failure
    """
    # ── 1. Parse video ID ──────────────────────────────────────────────────────
    video_id = parse_video_id(url)
    if not video_id:
        return IngestResult(
            video_id="unknown", url=url,
            error=f"Could not parse a valid YouTube video ID from: {url}"
        )

    result = IngestResult(video_id=video_id, url=url)
    log.info(f"Starting ingestion for video: {video_id}")

    try:
        # ── 2. Pinecone client setup ───────────────────────────────────────────
        pc               = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name       = os.environ["PINECONE_INDEX_NAME"]
        live_namespace   = os.environ.get("PINECONE_NAMESPACE_LIVE", "live")
        corpus_namespace = os.environ.get("PINECONE_NAMESPACE_CORPUS", "corpus")
        index            = pc.Index(index_name)

        # ── 3. Duplicate check ─────────────────────────────────────────────────
        if not dry_run:
            is_duplicate, found_in_ns = _is_already_indexed(video_id, index, corpus_namespace, live_namespace)
            if is_duplicate:
                try:
                    fetch_result = index.fetch(ids=[f"{video_id}_000"], namespace=found_in_ns)
                    vector = fetch_result.vectors.get(f"{video_id}_000")
                    if vector and vector.metadata:
                        result.title   = vector.metadata.get("title",   video_id)
                        result.channel = vector.metadata.get("channel", "Unknown")
                        result.topic   = vector.metadata.get("topic",   "Other")
                except Exception:
                    result.title = video_id
                result.already_indexed = True
                result.success         = True
                result.message = (
                    f'"{result.title}" is already in your knowledge base.'
                    if found_in_ns == corpus_namespace
                    else f'"{result.title}" was already ingested previously.'
                )
                return result

        # ── 4. Fetch real metadata via yt-dlp ──────────────────────────────────
        log.info("  Fetching metadata via yt-dlp...")
        meta = _fetch_metadata_yt_dlp(url)

        # ── 5. Extract transcript ──────────────────────────────────────────────
        log.info("  Extracting transcript...")
        try:
            segments, language, is_generated = _extract_transcript(video_id)
        except (NoTranscriptFound, TranscriptsDisabled):
            result.error = (
                "No captions are available for this video. "
                "Try a video with auto-generated or manual subtitles enabled."
            )
            return result
        except VideoUnavailable:
            result.error = "This video is private or has been removed."
            return result

        log.info(
            f"  ✓ {len(segments)} segments | language: {language} | "
            f"{'auto-generated' if is_generated else 'human'} captions"
        )

        # ── 6. LLM fallback metadata if yt-dlp failed ─────────────────────────
        first_words = " ".join(
            seg["text"] for seg in segments[:40]  # ~first 500 words
        )
        if meta is None:
            log.info("  Inferring title/channel via LLM fallback...")
            meta = _infer_metadata_llm(first_words)
            meta["duration"] = 0  # unknown without yt-dlp

        result.title    = meta["title"]
        result.channel  = meta["channel"]
        result.duration = meta.get("duration", 0)
        log.info(f"  Title:   {result.title}")
        log.info(f"  Channel: {result.channel}")

        # ── 7. Infer topic ─────────────────────────────────────────────────────
        log.info("  Inferring topic via LLM...")
        result.topic = _infer_topic_llm(first_words)
        log.info(f"  Topic:   {result.topic}")

        # ── 8. Clean segments ─────────────────────────────────────────────────
        cleaned_segments = _clean_segments(segments)

        # ── 9. Chunk ──────────────────────────────────────────────────────────
        chunks = chunk_segments(
            segments=cleaned_segments,
            window=CHUNK_WINDOW_SECONDS,
            video_id=video_id,
        )
        log.info(f"  ✓ {len(chunks)} chunks @ {CHUNK_WINDOW_SECONDS}s windows")

        if not chunks:
            result.error = "Transcript produced no usable chunks after cleaning."
            return result

        # ── 10. Embed + upsert ────────────────────────────────────────────────
        log.info(f"  Embedding and upserting to namespace '{live_namespace}'...")
        n_upserted = _embed_and_upsert(
            chunks         = chunks,
            title          = result.title,
            channel        = result.channel,
            topic          = result.topic,
            index          = index,
            live_namespace = live_namespace,
            dry_run        = dry_run,
        )

        result.chunk_count = n_upserted
        result.chunks      = chunks
        result.success     = True
        log.info(f"  ✓ Ingestion complete — {n_upserted} vectors in '{live_namespace}'")

    except KeyError as e:
        result.error = f"Missing environment variable: {e}. Check your .env file."
        log.error(result.error)
    except Exception as e:
        result.error = f"Unexpected error during ingestion: {e}"
        log.error(result.error, exc_info=True)

    return result


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest a YouTube URL into the Pinecone live namespace."
    )
    parser.add_argument("url", help="YouTube URL to ingest")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run all steps except Pinecone upsert",
    )
    args = parser.parse_args()

    res = ingest_url(args.url, dry_run=args.dry_run)

    if res.already_indexed:
        print(f"\n✓ Already indexed — {res.video_id} is ready to query.")
    elif res.success:
        print(f"\n✓ Ingestion complete")
        print(f"  Title:   {res.title}")
        print(f"  Channel: {res.channel}")
        print(f"  Topic:   {res.topic}")
        print(f"  Chunks:  {res.chunk_count}")
    else:
        print(f"\n✗ Ingestion failed: {res.error}")
        sys.exit(1)
