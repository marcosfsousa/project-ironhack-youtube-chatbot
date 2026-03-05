"""
cleaner.py
----------
Cleans raw transcripts produced by transcript_extractor.py.
Reads  → data/videos/{video_id}/transcript_raw.json
Writes → data/videos/{video_id}/transcript_clean.json

Cleaning operations (in order):
  1. HTML entity decoding      (&amp; &#39; → & ')
  2. Strip non-speech tags     ([Music], [Applause], etc.)
  3. Remove filler words       (uh, um, uh-huh, etc.)
  4. Whitespace normalisation  (collapse multiple spaces)
  5. Flag sponsor segments     (mark, never delete)

Guarantees:
  - Raw file is NEVER modified (idempotent reads)
  - Timestamps are ALWAYS preserved as-is
  - Empty segments after cleaning are flagged, not deleted
  - Running twice produces identical output

Usage:
  python cleaner.py                        # process all videos
  python cleaner.py --video-id aircAruvnKk # process one video
  python cleaner.py --dry-run              # print stats, write nothing
"""

import argparse
import html
import json
import logging
import re
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
CLEAN_LOG  = LOGS_DIR / "cleaning_log.json"

# ── Patterns ───────────────────────────────────────────────────────────────────

# Non-speech tags: [Music], [Applause], [Laughter], [MUSIC], etc.
NON_SPEECH_PATTERN = re.compile(r"\[[\w\s]+\]", re.IGNORECASE)

# Filler words — whole word matches only, case-insensitive
# Uses word boundaries to avoid clipping "umbrella" → "brella"
FILLER_PATTERN = re.compile(
    r"\b(uh+|um+|uh-huh|mm-hmm|hmm+|mhm|uhm)\b[\s,]*",
    re.IGNORECASE,
)

# Sponsor signal phrases — used for FLAGGING only, never deletion
# Covers common YouTube sponsor patterns
SPONSOR_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        r"\bbrought to you by\b",
        r"\bsponsored by\b",
        r"\bthis (video|episode) (is )?sponsored\b",
        r"\buse (code|promo|coupon)\b",
        r"\bcheck out .{0,30} (in the description|below|link)\b",
        r"\bsign up .{0,30} free\b",
        r"\bdiscount\b.{0,30}\bcode\b",
        r"\baffiliate\b",
        r"\bsquarespace\b|\bskillshare\b|\baudible\b|\bnordvpn\b|"
        r"\bbrave browser\b|\bvpn\b|\bcurious(ity)? stream\b",
    ]
]


# ── Per-segment cleaning ───────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Apply all text cleaning operations to a single segment string.
    Order matters — decode first, strip tags, then fillers, then whitespace.
    """
    # 1. HTML entity decoding
    text = html.unescape(text)

    # 2. Strip non-speech tags  [Music], [Applause], etc.
    text = NON_SPEECH_PATTERN.sub("", text)

    # 3. Remove filler words
    text = FILLER_PATTERN.sub("", text)

    # 4. Normalise whitespace — replace non-breaking spaces, collapse multiples
    text = text.replace("\u00a0", " ")  # non-breaking space → regular space
    text = re.sub(r" {2,}", " ", text).strip()

    return text


def is_sponsor_segment(text: str) -> bool:
    """Return True if any sponsor signal pattern matches the text."""
    return any(p.search(text) for p in SPONSOR_PATTERNS)


# ── Per-video cleaning ─────────────────────────────────────────────────────────

def clean_transcript(video_id: str, dry_run: bool = False, force: bool = False) -> dict | None:
    """
    Load transcript_raw.json, clean each segment, write transcript_clean.json.
    Returns a stats dict, or None if raw file not found.
    """
    raw_path   = VIDEOS_DIR / video_id / "transcript_raw.json"
    clean_path = VIDEOS_DIR / video_id / "transcript_clean.json"

    if not raw_path.exists():
        log.warning(f"  Raw transcript not found, skipping: {raw_path}")
        return None
    
    # Resume support — skip if already cleaned (unless --force)
    if clean_path.exists() and not force:
        log.info(f"  Already cleaned, skipping: {video_id}  (use --force to re-clean)")
        return {"video_id": video_id, "skipped": True, "reason": "already_cleaned"}

    raw_text = raw_path.read_text(encoding="utf-8", errors="replace")
    raw_text = raw_text.replace("\u00a0", " ").replace("\ufffd", " ")
    raw = json.loads(raw_text)

    stats = {
        "video_id":          video_id,
        "total_segments":    len(raw["transcript"]),
        "empty_after_clean": 0,
        "sponsor_flagged":   0,
        "cleaned_at":        datetime.utcnow().isoformat() + "Z",
    }

    cleaned_segments = []

    for seg in raw["transcript"]:
        original_text = seg["text"]
        cleaned_text  = clean_text(original_text)

        segment = {
            "start":    seg["start"],
            "duration": seg["duration"],
            "end":      seg["end"],
            "text":     cleaned_text,
        }

        # Flag empty segments (e.g. segment was only [Music])
        if not cleaned_text:
            segment["flags"] = segment.get("flags", []) + ["empty_after_clean"]
            stats["empty_after_clean"] += 1

        # Flag potential sponsor segments
        if is_sponsor_segment(cleaned_text):
            segment["flags"] = segment.get("flags", []) + ["potential_sponsor"]
            stats["sponsor_flagged"] += 1
            log.info(f"  ⚑  Sponsor flag @ {seg['start']:.1f}s: \"{cleaned_text[:80]}\"")

        cleaned_segments.append(segment)

    # Build output payload — preserve all original metadata
    output = {
        "video_id":      raw["video_id"],
        "url":           raw["url"],
        "language":      raw["language"],
        "is_generated":  raw["is_generated"],
        "extracted_at":  raw["extracted_at"],
        "cleaned_at":    stats["cleaned_at"],
        "segment_count": len(cleaned_segments),
        "cleaning_stats": {
            "empty_after_clean": stats["empty_after_clean"],
            "sponsor_flagged":   stats["sponsor_flagged"],
        },
        "transcript": cleaned_segments,
    }

    if not dry_run:
        clean_path.write_text(
            json.dumps(output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return stats


# ── Logging helpers ────────────────────────────────────────────────────────────

def load_clean_log() -> dict:
    if CLEAN_LOG.exists():
        return json.loads(CLEAN_LOG.read_text())
    return {"cleaned": [], "skipped": []}


def save_clean_log(log_data: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    CLEAN_LOG.write_text(json.dumps(log_data, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────

def run(video_id: str | None, dry_run: bool, force: bool) -> None:

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
        log.info(f"Found {len(targets)} video(s) to clean.")

    clean_log  = load_clean_log()
    total_ok   = 0
    total_skip = 0

    for vid_id in sorted(targets):
        log.info(f"Cleaning: {vid_id}")

        stats = clean_transcript(vid_id, dry_run=dry_run, force=force)

        if stats is None:
            clean_log["skipped"].append({
                "video_id":   vid_id,
                "reason":     "raw_file_missing",
                "skipped_at": datetime.utcnow().isoformat() + "Z",
            })
            total_skip += 1
            continue

        if stats.get("skipped"):
            total_skip += 1
            continue

        # only reaches here if actually cleaned
        clean_log["cleaned"].append(stats)
        total_ok += 1

        log.info(
            f"  ✓ {stats['total_segments']} segments | "
            f"empty: {stats['empty_after_clean']} | "
            f"sponsor flags: {stats['sponsor_flagged']}"
        )

    if not dry_run:
        save_clean_log(clean_log)

    log.info(
        f"\n── Cleaning complete ────────────────────────\n"
        f"  ✓ Cleaned : {total_ok}\n"
        f"  ✗ Skipped : {total_skip}  (already cleaned or missing raw file)\n"
        + ("  (dry run — nothing written)" if dry_run else
           f"  Log saved → {CLEAN_LOG}")
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean raw YouTube transcripts."
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Process a single video ID (default: process all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print stats only — do not write any files",
    )
    parser.add_argument(
        "--force",
        action="store_true", 
        help="Re-clean already cleaned videos"
    )

    args = parser.parse_args()

    run(video_id=args.video_id, dry_run=args.dry_run, force=args.force)