"""
bootstrap_metadata.py
---------------------
Scans data/videos/ for existing video folders and generates
a starter data/metadata.json with video_ids pre-filled and
human fields left blank for manual completion.

Safe to re-run — existing entries are NEVER overwritten.
New video folders found on re-run are APPENDED with blank fields.

Usage:
  python bootstrap_metadata.py             # scan and generate/update
  python bootstrap_metadata.py --dry-run   # print what would be written

After running, open data/metadata.json and fill in:
  - title    : exact YouTube video title
  - channel  : channel name (Veritasium / Kurzgesagt / Big Think / etc.)
  - topic    : one of Physics / Biology / Cosmology / Psychology / Chemistry / Other
  - url      : full YouTube URL (https://www.youtube.com/watch?v=VIDEO_ID)

These fields are used by indexer.py to enrich embeddings and
populate Pinecone metadata for filtering.
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR      = Path("data")
VIDEOS_DIR    = DATA_DIR / "videos"
METADATA_PATH = DATA_DIR / "metadata.json"

BLANK_ENTRY_TEMPLATE = {
    "title":   "",       # ← fill in: exact YouTube video title
    "channel": "",       # ← fill in: Veritasium / Kurzgesagt / Big Think / etc.
    "topic":   "",       # ← fill in: Physics / Biology / Cosmology / Psychology / Other
    "url":     "",       # ← fill in: https://www.youtube.com/watch?v=VIDEO_ID
    "indexed": False,    # managed by indexer.py — do not edit manually
}

VALID_TOPICS = {
    "Physics", "Biology", "Cosmology", "Psychology",
    "Chemistry", "Mathematics", "Technology", "Other",
}


def load_existing_metadata() -> dict:
    """Load existing metadata.json or return empty structure."""
    if METADATA_PATH.exists():
        data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        log.info(f"Loaded existing metadata.json ({len(data['videos'])} entries)")
        return data
    return {
        "created_at":  datetime.utcnow().isoformat() + "Z",
        "updated_at":  datetime.utcnow().isoformat() + "Z",
        "video_count": 0,
        "videos":      {},
    }


def scan_video_folders() -> list[str]:
    """Return sorted list of video_ids found in data/videos/."""
    if not VIDEOS_DIR.exists():
        log.warning(f"Videos directory not found: {VIDEOS_DIR}")
        return []
    folders = sorted(p.name for p in VIDEOS_DIR.iterdir() if p.is_dir())
    log.info(f"Found {len(folders)} video folder(s) in {VIDEOS_DIR}")
    return folders


def has_chunks(video_id: str) -> bool:
    """Check if this video has been through the full pipeline."""
    return (VIDEOS_DIR / video_id / "chunks.json").exists()


def run(dry_run: bool) -> None:
    if dry_run:
        log.info("DRY RUN — nothing will be written.")

    metadata   = load_existing_metadata()
    video_ids  = scan_video_folders()

    if not video_ids:
        log.warning("No video folders found. Run transcript_extractor.py first.")
        return

    existing_ids = set(metadata["videos"].keys())
    new_ids      = [vid for vid in video_ids if vid not in existing_ids]
    skipped_ids  = [vid for vid in video_ids if vid in existing_ids]

    if skipped_ids:
        log.info(f"Skipping {len(skipped_ids)} already-present entries (no overwrite).")

    if not new_ids:
        log.info("No new video folders found. metadata.json is up to date.")
        _print_fill_reminder(metadata)
        return

    log.info(f"Adding {len(new_ids)} new entry/entries...")

    for video_id in new_ids:
        pipeline_complete = has_chunks(video_id)
        entry = {**BLANK_ENTRY_TEMPLATE, "video_id": video_id}

        # Try to recover URL from transcript_raw.json if it exists
        raw_path = VIDEOS_DIR / video_id / "transcript_raw.json"
        if raw_path.exists():
            try:
                raw = json.loads(raw_path.read_text(encoding="utf-8"))
                entry["url"] = raw.get("url", "")
            except Exception:
                pass

        metadata["videos"][video_id] = entry

        status = "✓ chunks ready" if pipeline_complete else "⚠ pipeline incomplete"
        log.info(f"  + {video_id}  [{status}]")

    metadata["video_count"] = len(metadata["videos"])
    metadata["updated_at"]  = datetime.utcnow().isoformat() + "Z"

    if not dry_run:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        METADATA_PATH.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(f"\n✓ metadata.json written → {METADATA_PATH}")

    _print_fill_reminder(metadata)


def _print_fill_reminder(metadata: dict) -> None:
    """Print a summary of which entries still need manual filling."""
    blank = [
        vid_id for vid_id, entry in metadata["videos"].items()
        if not entry.get("title") or not entry.get("channel") or not entry.get("topic")
    ]
    if blank:
        log.info(
            f"\n── Action required ──────────────────────────\n"
            f"  {len(blank)} entry/entries need manual fields filled in:\n"
            + "\n".join(f"    • {vid}" for vid in blank)
            + f"\n\n  Open data/metadata.json and fill in:\n"
            f"    title, channel, topic\n"
            f"  (url is auto-filled where possible)\n"
            f"  Valid topics: {', '.join(sorted(VALID_TOPICS))}\n"
        )
    else:
        log.info("  All entries have title, channel, and topic filled in. ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap metadata.json from existing video folders."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing anything",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
