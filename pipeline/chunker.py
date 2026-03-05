"""
chunker.py
----------
Chunks cleaned transcripts produced by cleaner.py.
Reads  → data/videos/{video_id}/transcript_clean.json
Writes → data/videos/{video_id}/chunks.json

Chunking strategy:
  - Time-window based: target 60s windows (configurable via CLI)
  - No overlap between chunks (clean boundaries)
  - Sentence-respectful: closes a chunk at the nearest segment
    boundary once the window target is reached
  - Skips empty segments (flagged as empty_after_clean)
  - Preserves start/end timestamps accurately for deep links

Chunk format:
  {
    "chunk_id":  "aircAruvnKk_000",
    "video_id":  "aircAruvnKk",
    "start":     0.0,
    "end":       61.3,
    "duration":  61.3,
    "text":      "...",
    "segment_count": 12
  }

Guarantees:
  - Clean file is NEVER modified
  - chunk_id is always zero-padded and sortable
  - Empty segments are skipped silently
  - Running twice produces identical output

Usage:
  python chunker.py                            # process all videos
  python chunker.py --video-id aircAruvnKk     # process one video
  python chunker.py --window 90                # 90s windows
  python chunker.py --dry-run                  # print stats, write nothing
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

# ── Logging ────────────────────────────────────────────────────────────────────
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
DATA_DIR   = Path("data")
VIDEOS_DIR = DATA_DIR / "videos"
LOGS_DIR   = DATA_DIR / "logs"
CHUNK_LOG  = LOGS_DIR / "chunking_log.json"

# ── Default config ─────────────────────────────────────────────────────────────
DEFAULT_WINDOW_SECONDS = 60


# ── Core chunking logic ────────────────────────────────────────────────────────

def chunk_segments(segments: list[dict], window: float, video_id: str) -> list[dict]:
    """
    Split a list of cleaned segments into time-window chunks.

    Strategy:
      - Accumulate segments until the window target is reached
      - Close the chunk at the next segment boundary (never mid-segment)
      - Skip segments flagged as empty_after_clean
      - Always track start from first segment, end from last segment

    Args:
        segments:  list of cleaned transcript segments
        window:    target chunk duration in seconds
        video_id:  used to construct chunk_id

    Returns:
        list of chunk dicts
    """
    chunks      = []
    buffer      = []        # segments accumulated in current chunk
    chunk_index = 0

    for seg in segments:
        # Skip empty segments silently
        flags = seg.get("flags", [])
        if "empty_after_clean" in flags:
            continue

        # Skip segments with no meaningful text
        if not seg["text"].strip():
            continue

        buffer.append(seg)

        # Calculate current window duration
        window_duration = buffer[-1]["end"] - buffer[0]["start"]

        # Close chunk when window target is reached
        if window_duration >= window:
            chunks.append(_build_chunk(buffer, chunk_index, video_id))
            chunk_index += 1
            buffer = []

    # Flush any remaining segments as a final chunk
    if buffer:
        chunks.append(_build_chunk(buffer, chunk_index, video_id))

    return chunks


def _build_chunk(buffer: list[dict], index: int, video_id: str) -> dict:
    """
    Build a single chunk dict from a buffer of segments.
    chunk_id is zero-padded to 3 digits for reliable sorting.
    """
    start    = buffer[0]["start"]
    end      = buffer[-1]["end"]
    text     = " ".join(seg["text"].strip() for seg in buffer)
    # Collapse any double spaces that result from joining
    text     = " ".join(text.split())

    return {
        "chunk_id":      f"{video_id}_{index:03d}",
        "video_id":      video_id,
        "start":         round(start, 3),
        "end":           round(end, 3),
        "duration":      round(end - start, 3),
        "text":          text,
        "segment_count": len(buffer),
    }


# ── Per-video chunking ─────────────────────────────────────────────────────────

def chunk_transcript(
    video_id: str,
    window: float,
    dry_run: bool = False,
    force: bool = False,
) -> dict | None:
    """
    Load transcript_clean.json, chunk it, write chunks.json.
    Returns a stats dict, or None if clean file not found.
    """
    clean_path  = VIDEOS_DIR / video_id / "transcript_clean.json"
    chunks_path = VIDEOS_DIR / video_id / "chunks.json"

    if not clean_path.exists():
        log.warning(f"  Clean transcript not found, skipping: {clean_path}")
        return None
    
    # Resume support — skip if already chunked (unless --force)
    if chunks_path.exists() and not force:
        log.info(f"  Already chunked, skipping: {video_id}  (use --force to re-chunk)")
        return {"video_id": video_id, "skipped": True, "reason": "already_chunked"}

    clean = json.loads(clean_path.read_text(encoding="utf-8", errors="replace"))
    segments = clean["transcript"]

    chunks = chunk_segments(segments, window=window, video_id=video_id)

    # ── Stats ──────────────────────────────────────────────────────────────────
    durations    = [c["duration"] for c in chunks]
    avg_duration = sum(durations) / len(durations) if durations else 0
    avg_segments = (
        sum(c["segment_count"] for c in chunks) / len(chunks) if chunks else 0
    )

    stats = {
        "video_id":       video_id,
        "total_chunks":   len(chunks),
        "avg_duration_s": round(avg_duration, 1),
        "avg_segments":   round(avg_segments, 1),
        "min_duration_s": round(min(durations), 1) if durations else 0,
        "max_duration_s": round(max(durations), 1) if durations else 0,
        "window_s":       window,
        "chunked_at":     datetime.utcnow().isoformat() + "Z",
    }

    # ── Output payload ─────────────────────────────────────────────────────────
    output = {
        "video_id":    clean["video_id"],
        "url":         clean["url"],
        "language":    clean["language"],
        "chunked_at":  stats["chunked_at"],
        "window_s":    window,
        "chunk_count": len(chunks),
        "chunks":      chunks,
    }

    if not dry_run:
        chunks_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return stats


# ── Log helpers ────────────────────────────────────────────────────────────────

def load_chunk_log() -> dict:
    if CHUNK_LOG.exists():
        return json.loads(CHUNK_LOG.read_text())
    return {"chunked": [], "skipped": []}


def save_chunk_log(log_data: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CHUNK_LOG.write_text(json.dumps(log_data, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────

def run(video_id: str | None, window: float, dry_run: bool, force: bool) -> None:

    if dry_run:
        log.info("DRY RUN — no files will be written.")

    # Resolve which video(s) to process
    if video_id:
        targets = [video_id]
        log.info(f"Processing single video: {video_id}")
    else:
        targets = [p.name for p in VIDEOS_DIR.iterdir() if p.is_dir()]
        if not targets:
            log.warning(f"No video folders found in {VIDEOS_DIR}")
            return
        log.info(f"Found {len(targets)} video(s) to chunk.")

    chunk_log  = load_chunk_log()
    total_ok   = 0
    total_skip = 0

    for vid_id in sorted(targets):
        log.info(f"Chunking: {vid_id}")

        stats = chunk_transcript(vid_id, window=window, dry_run=dry_run, force=force)

        if stats is None:
            chunk_log["skipped"].append({
                "video_id":   vid_id,
                "reason":     "clean_file_missing",
                "skipped_at": datetime.utcnow().isoformat() + "Z",
            })
            total_skip += 1
            continue

        if stats.get("skipped"):
            total_skip += 1
            continue

        # only reaches here if actually chunked
        chunk_log["chunked"].append(stats)
        total_ok += 1

        log.info(
            f"  ✓ {stats['total_chunks']} chunks | "
            f"avg duration: {stats['avg_duration_s']}s | "
            f"avg segments/chunk: {stats['avg_segments']} | "
            f"range: {stats['min_duration_s']}s – {stats['max_duration_s']}s"
        )

    if not dry_run:
        save_chunk_log(chunk_log)

    log.info(
        f"\n── Chunking complete ────────────────────────\n"
        f"  ✓ Chunked : {total_ok}\n"
        f"  ✗ Skipped : {total_skip}  (already chunked or missing clean file)\n"
        + ("  (dry run — nothing written)" if dry_run else
           f"  Log saved → {CHUNK_LOG}")
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk cleaned YouTube transcripts into time windows."
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Process a single video ID (default: process all)",
    )
    parser.add_argument(
        "--window",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help=f"Target chunk window in seconds (default: {DEFAULT_WINDOW_SECONDS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only — do not write any files",
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Re-chunk already chunked videos"
    )

    args = parser.parse_args()

    run(video_id=args.video_id, window=args.window, dry_run=args.dry_run, force=args.force)
