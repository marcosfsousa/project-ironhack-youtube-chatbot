"""
transcript_extractor.py
-----------------------
Extracts YouTube transcripts from a text file of URLs.
Compatible with youtube-transcript-api >= 1.2.0

Guardrails:
  - Random delay between requests (3–8s default)
  - Retry logic with exponential backoff
  - Skips & logs videos with no captions (no crash)
  - Deduplicates URLs in input file
  - Skips video IDs already extracted (resume support)
  - Progress log written to data/logs/extraction_log.json

Usage:
  python transcript_extractor.py                        # uses data/video_urls.txt
  python transcript_extractor.py --input my_urls.txt
  python transcript_extractor.py --delay-min 5 --delay-max 12
  python transcript_extractor.py --max-retries 5
"""

import argparse
import json
import logging
import random
import re
import time
from datetime import datetime
from pathlib import Path

from youtube_transcript_api import (
    YouTubeTranscriptApi,
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

# ── Logging setup ──────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
VIDEOS_DIR    = DATA_DIR / "videos"
LOGS_DIR      = DATA_DIR / "logs"
DEFAULT_INPUT = DATA_DIR / "video_urls.txt"
EXTRACT_LOG   = LOGS_DIR / "extraction_log.json"


# ── Helpers ────────────────────────────────────────────────────────────────────

def extract_video_id(url: str) -> str | None:
    """
    Parse a video ID from any common YouTube URL format.
    Handles:
      https://www.youtube.com/watch?v=VIDEO_ID
      https://youtu.be/VIDEO_ID
      https://www.youtube.com/embed/VIDEO_ID
    Returns None if no ID found.
    """
    patterns = [
        r"(?:v=)([A-Za-z0-9_-]{11})",
        r"(?:youtu\.be/)([A-Za-z0-9_-]{11})",
        r"(?:embed/)([A-Za-z0-9_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def load_urls(input_path: Path) -> list[str]:
    """Read URLs from file, strip blanks and comments (#)."""
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found: {input_path}\n"
            f"Create it with one YouTube URL per line."
        )
    lines = input_path.read_text().splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def deduplicate_urls(urls: list[str]) -> list[str]:
    """
    Remove duplicate URLs AND duplicate video IDs.
    Keeps first occurrence, logs any duplicates found.
    """
    seen_ids:  set[str] = set()
    seen_urls: set[str] = set()
    unique = []
    for url in urls:
        vid_id = extract_video_id(url)
        if url in seen_urls:
            log.warning(f"Duplicate URL skipped: {url}")
            continue
        if vid_id and vid_id in seen_ids:
            log.warning(f"Duplicate video ID ({vid_id}) skipped: {url}")
            continue
        seen_urls.add(url)
        if vid_id:
            seen_ids.add(vid_id)
        unique.append(url)
    return unique


def load_extraction_log() -> dict:
    """Load existing extraction log or return empty structure."""
    if EXTRACT_LOG.exists():
        return json.loads(EXTRACT_LOG.read_text())
    return {"extracted": [], "skipped": [], "failed": []}


def save_extraction_log(log_data: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACT_LOG.write_text(json.dumps(log_data, indent=2))


def already_extracted(video_id: str) -> bool:
    """Check if transcript file already exists on disk."""
    return (VIDEOS_DIR / video_id / "transcript_raw.json").exists()


def _to_segments(fetched) -> list[dict]:
    """
    Normalise FetchedTranscript (v1.x) to a consistent list of dicts.
    Handles both snippet objects and raw dicts gracefully.
    """
    segments = []
    for item in fetched:
        # v1.x returns FetchedTranscriptSnippet objects
        if hasattr(item, "text"):
            segments.append({
                "start":    round(item.start, 3),
                "duration": round(item.duration, 3),
                "text":     item.text,
            })
        else:
            # Fallback: raw dict (shouldn't happen in v1.x but safe to handle)
            segments.append({
                "start":    round(item.get("start", 0), 3),
                "duration": round(item.get("duration", 0), 3),
                "text":     item.get("text", ""),
            })
    return segments


def fetch_transcript(video_id: str, max_retries: int, backoff_base: float = 2.0):
    """
    Fetch transcript using youtube-transcript-api v1.2.x API.

    Preference order:
      1. Human-generated captions (prefer 'en')
      2. Any human-generated language
      3. Auto-generated 'en'
      4. Any auto-generated language

    Returns:
      (segments: list[dict], language_code: str, is_generated: bool)
    Raises:
      NoTranscriptFound | TranscriptsDisabled | VideoUnavailable | Exception
    """
    attempt = 0
    last_error = None

    # v1.x requires instantiation — no more static calls
    ytt = YouTubeTranscriptApi()

    while attempt <= max_retries:
        try:
            transcript_list = ytt.list(video_id)

            # ── Try human-generated English first ──────────────────────────
            try:
                transcript = transcript_list.find_manually_created_transcript(["en"])
                fetched = transcript.fetch()
                return _to_segments(fetched), transcript.language_code, False
            except NoTranscriptFound:
                pass

            # ── Any human-generated language ───────────────────────────────
            for t in transcript_list:
                if not t.is_generated:
                    fetched = t.fetch()
                    return _to_segments(fetched), t.language_code, False

            # ── Fall back to auto-generated English ────────────────────────
            try:
                transcript = transcript_list.find_generated_transcript(["en"])
                fetched = transcript.fetch()
                return _to_segments(fetched), transcript.language_code, True
            except NoTranscriptFound:
                pass

            # ── Any auto-generated language ────────────────────────────────
            for t in transcript_list:
                if t.is_generated:
                    fetched = t.fetch()
                    return _to_segments(fetched), t.language_code, True

            raise NoTranscriptFound(video_id, ["en"])

        except (NoTranscriptFound, TranscriptsDisabled, VideoUnavailable):
            raise  # Don't retry — these won't change on retry

        except Exception as e:
            attempt += 1
            last_error = e
            if attempt > max_retries:
                break
            wait = backoff_base ** attempt + random.uniform(0, 1)
            log.warning(
                f"  Attempt {attempt}/{max_retries} failed for {video_id}: {e}. "
                f"Retrying in {wait:.1f}s..."
            )
            time.sleep(wait)

    raise Exception(
        f"All {max_retries} retries exhausted for {video_id}. Last error: {last_error}"
    )


def save_transcript(video_id: str, url: str, segments: list,
                    language: str, is_generated: bool) -> Path:
    """Save raw transcript JSON to data/videos/{video_id}/transcript_raw.json."""
    out_dir = VIDEOS_DIR / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "transcript_raw.json"

    payload = {
        "video_id":      video_id,
        "url":           url,
        "language":      language,
        "is_generated":  is_generated,
        "extracted_at":  datetime.utcnow().isoformat() + "Z",
        "segment_count": len(segments),
        "transcript": [
            {
                "start":    seg["start"],
                "duration": seg["duration"],
                "end":      round(seg["start"] + seg["duration"], 3),
                "text": seg["text"].replace("\u00a0", " ").strip(),
            }
            for seg in segments
        ],
    }
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_path


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    input_path: Path,
    delay_min: float,
    delay_max: float,
    max_retries: int,
) -> None:

    # Load & deduplicate URLs
    raw_urls = load_urls(input_path)
    log.info(f"Loaded {len(raw_urls)} URLs from {input_path}")
    urls = deduplicate_urls(raw_urls)
    if len(urls) < len(raw_urls):
        log.info(f"After deduplication: {len(urls)} unique URLs")

    # Load existing progress log
    extraction_log = load_extraction_log()
    already_done_ids = {e["video_id"] for e in extraction_log["extracted"]}

    # Filter out already-extracted videos
    to_process = []
    for url in urls:
        vid_id = extract_video_id(url)
        if not vid_id:
            log.warning(f"Could not parse video ID from URL, skipping: {url}")
            continue
        if vid_id in already_done_ids or already_extracted(vid_id):
            log.info(f"Already extracted, skipping: {vid_id}")
            if vid_id not in already_done_ids:
                already_done_ids.add(vid_id)
            continue
        to_process.append((vid_id, url))

    if not to_process:
        log.info("Nothing new to extract. All videos already processed.")
        return

    log.info(f"{len(to_process)} videos to extract.\n")

    # ── Extraction loop ────────────────────────────────────────────────────────
    for idx, (video_id, url) in enumerate(to_process, 1):
        log.info(f"[{idx}/{len(to_process)}] Extracting: {video_id}  ({url})")

        try:
            segments, language, is_generated = fetch_transcript(
                video_id, max_retries=max_retries
            )
            out_path = save_transcript(video_id, url, segments, language, is_generated)

            caption_type = "auto-generated" if is_generated else "human"
            log.info(
                f"  ✓ {len(segments)} segments | language: {language} "
                f"| {caption_type} | saved → {out_path}"
            )

            extraction_log["extracted"].append({
                "video_id":     video_id,
                "url":          url,
                "language":     language,
                "is_generated": is_generated,
                "segments":     len(segments),
                "extracted_at": datetime.utcnow().isoformat() + "Z",
            })

        except (NoTranscriptFound, TranscriptsDisabled):
            log.warning(f"  ✗ No captions available for {video_id} — skipped")
            extraction_log["skipped"].append({
                "video_id":   video_id,
                "url":        url,
                "reason":     "no_captions",
                "skipped_at": datetime.utcnow().isoformat() + "Z",
            })

        except VideoUnavailable:
            log.warning(f"  ✗ Video unavailable (private/deleted): {video_id} — skipped")
            extraction_log["skipped"].append({
                "video_id":   video_id,
                "url":        url,
                "reason":     "video_unavailable",
                "skipped_at": datetime.utcnow().isoformat() + "Z",
            })

        except Exception as e:
            log.error(f"  ✗ Failed after retries: {video_id} — {e}")
            extraction_log["failed"].append({
                "video_id":  video_id,
                "url":       url,
                "error":     str(e),
                "failed_at": datetime.utcnow().isoformat() + "Z",
            })

        finally:
            # Always save progress after each video
            save_extraction_log(extraction_log)

        # ── Delay before next request (skip after last video) ──────────────
        if idx < len(to_process):
            delay = random.uniform(delay_min, delay_max)
            log.info(f"  Waiting {delay:.1f}s before next request...")
            time.sleep(delay)

    # ── Summary ────────────────────────────────────────────────────────────────
    n_ok      = len(extraction_log["extracted"])
    n_skipped = len(extraction_log["skipped"])
    n_failed  = len(extraction_log["failed"])
    log.info(
        f"\n── Extraction complete ──────────────────────\n"
        f"  ✓ Extracted : {n_ok}\n"
        f"  ✗ Skipped   : {n_skipped}  (no captions / unavailable)\n"
        f"  ✗ Failed    : {n_failed}  (errors after retries)\n"
        f"  Log saved → {EXTRACT_LOG}"
    )
    if n_skipped:
        log.info("  Skipped videos are candidates for Whisper transcription (offline).")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract YouTube transcripts from a list of URLs."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to URL list file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--delay-min",
        type=float,
        default=3.0,
        help="Minimum delay between requests in seconds (default: 3)",
    )
    parser.add_argument(
        "--delay-max",
        type=float,
        default=8.0,
        help="Maximum delay between requests in seconds (default: 8)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retry attempts per video on transient errors (default: 3)",
    )
    args = parser.parse_args()

    run(
        input_path=args.input,
        delay_min=args.delay_min,
        delay_max=args.delay_max,
        max_retries=args.max_retries,
    )