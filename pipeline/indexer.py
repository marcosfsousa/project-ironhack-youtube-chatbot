"""
indexer.py
----------
Embeds enriched transcript chunks and upserts them to Pinecone.
Reads  → data/videos/{video_id}/chunks.json
Reads  → data/metadata.json
Writes → Pinecone index (namespace: 'corpus' or 'live')
Updates → data/metadata.json  (sets indexed: true on success)

Embedding text format:
  "{title} | {chunk_text}"
  Title is prepended to bake light topical context into each vector.
  This improves retrieval for queries that reference the video topic
  without needing explicit metadata filtering.

Pinecone vector record:
  {
    "id": "aircAruvnKk_000",
    "values": [...],          // 384-dim normalized float vector
    "metadata": {
      "chunk_id":   "aircAruvnKk_000",
      "video_id":   "aircAruvnKk",
      "title":      "What is Entropy?",
      "channel":    "Veritasium",
      "topic":      "Physics",
      "start":      0.0,
      "end":        61.3,
      "chunk_text": "Energy cannot be created or destroyed..."
    }
  }

Guarantees:
  - chunks.json and metadata.json are NEVER modified except
    the 'indexed' flag in metadata.json on successful upsert
  - Videos already marked indexed: true are skipped (use --force to override)
  - Index is auto-created if it doesn't exist
  - Upserts are batched (default: 100 vectors per request)
  - Progress log written to data/logs/indexing_log.json

Usage:
  python indexer.py                          # index all unindexed videos
  python indexer.py --video-id aircAruvnKk   # index one video
  python indexer.py --namespace live         # target live namespace
  python indexer.py --force                  # re-index already-indexed videos
  python indexer.py --dry-run                # print stats, write nothing
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# embedder.py lives in the same pipeline/ directory
# get_model() returns the singleton SentenceTransformer
import sys
sys.path.insert(0, str(Path(__file__).parent))
from embedder import get_model

# ── Env + Logging ──────────────────────────────────────────────────────────────
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
VIDEOS_DIR    = DATA_DIR / "videos"
LOGS_DIR      = DATA_DIR / "logs"
METADATA_PATH = DATA_DIR / "metadata.json"
INDEX_LOG     = LOGS_DIR / "indexing_log.json"

# ── Pinecone config ────────────────────────────────────────────────────────────
INDEX_NAME   = os.getenv("PINECONE_INDEX_NAME", "youtube-qa-bot")
DIMENSION    = 384
METRIC       = "cosine"
DEFAULT_NS   = os.getenv("PINECONE_NAMESPACE_CORPUS", "corpus")
LIVE_NS      = os.getenv("PINECONE_NAMESPACE_LIVE", "live")
UPSERT_BATCH = 100

# Pinecone Serverless spec — us-east-1 is free tier default
SERVERLESS_SPEC = ServerlessSpec(cloud="aws", region="us-east-1")


# ── Pinecone client (singleton) ────────────────────────────────────────────────

_pc    = None
_index = None

def get_pinecone_index():
    """
    Initialise Pinecone client and return the index handle.
    Creates the index if it doesn't exist.
    Called once per process.
    """
    global _pc, _index
    if _index is not None:
        return _index

    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY not found. "
            "Add it to your .env file: PINECONE_API_KEY=your_key_here"
        )

    _pc = Pinecone(api_key=api_key)

    existing = [idx.name for idx in _pc.list_indexes()]
    if INDEX_NAME not in existing:
        log.info(f"Index '{INDEX_NAME}' not found — creating ({DIMENSION}d, {METRIC})...")
        _pc.create_index(
            name=INDEX_NAME,
            dimension=DIMENSION,
            metric=METRIC,
            spec=SERVERLESS_SPEC,
        )
        log.info(f"Index '{INDEX_NAME}' created.")
    else:
        log.info(f"Index '{INDEX_NAME}' found.")

    _index = _pc.Index(INDEX_NAME)
    return _index


# ── Metadata helpers ───────────────────────────────────────────────────────────

def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {METADATA_PATH}. "
            "Run bootstrap_metadata.py first and fill in title/channel/topic."
        )
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


def save_metadata(metadata: dict) -> None:
    metadata["updated_at"] = datetime.utcnow().isoformat() + "Z"
    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_video_meta(metadata: dict, video_id: str) -> dict:
    """Return metadata entry for a video_id, with validation."""
    entry = metadata["videos"].get(video_id)
    if not entry:
        raise KeyError(
            f"video_id '{video_id}' not found in metadata.json. "
            "Run bootstrap_metadata.py and fill in the entry."
        )
    missing = [f for f in ("title", "channel", "topic") if not entry.get(f)]
    if missing:
        raise ValueError(
            f"metadata.json entry for '{video_id}' is missing: {missing}. "
            "Fill in these fields before indexing."
        )
    return entry


# ── Core indexing ──────────────────────────────────────────────────────────────

def index_video(
    video_id:  str,
    namespace: str,
    metadata:  dict,
    force:     bool,
    dry_run:   bool,
) -> dict | None:
    """
    Embed and upsert all chunks for a single video.

    Returns a stats dict on success, None if chunks.json not found.
    Raises on metadata validation errors.
    """
    chunks_path = VIDEOS_DIR / video_id / "chunks.json"

    if not chunks_path.exists():
        log.warning(f"  chunks.json not found, skipping: {chunks_path}")
        return None

    # Validate metadata entry
    meta_entry = get_video_meta(metadata, video_id)

    # Skip if already indexed (unless --force)
    if meta_entry.get("indexed") and not force:
        log.info(f"  Already indexed, skipping: {video_id}  (use --force to re-index)")
        return {"video_id": video_id, "skipped": True, "reason": "already_indexed"}

    title   = meta_entry["title"]
    channel = meta_entry["channel"]
    topic   = meta_entry["topic"]

    # Load chunks
    data   = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    if not chunks:
        log.warning(f"  No chunks found in {chunks_path}, skipping.")
        return None

    log.info(f"  {len(chunks)} chunks | title: \"{title}\" | ns: {namespace}")

    # Build enriched texts: "title | chunk_text"
    enriched_texts = [f"{title} | {c['text']}" for c in chunks]

    # Embed
    log.info(f"  Embedding enriched texts...")
    model   = get_model()
    vectors = model.encode(
        enriched_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Build Pinecone records
    records = []
    for chunk, vector in zip(chunks, vectors):
        records.append({
            "id":     chunk["chunk_id"],
            "values": vector.tolist(),
            "metadata": {
                "chunk_id":   chunk["chunk_id"],
                "video_id":   video_id,
                "title":      title,
                "channel":    channel,
                "topic":      topic,
                "start":      chunk["start"],
                "end":        chunk["end"],
                "chunk_text": chunk["text"],   # plain text, not enriched
            },
        })

    if dry_run:
        log.info(f"  DRY RUN — {len(records)} records built but not upserted.")
        return {
            "video_id":    video_id,
            "chunk_count": len(records),
            "namespace":   namespace,
            "dry_run":     True,
        }

    # Upsert in batches
    index = get_pinecone_index()
    total_upserted = 0

    for i in range(0, len(records), UPSERT_BATCH):
        batch = records[i : i + UPSERT_BATCH]
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)
        log.info(f"  Upserted {total_upserted}/{len(records)} vectors...")

    # Mark as indexed in metadata.json
    metadata["videos"][video_id]["indexed"] = True
    save_metadata(metadata)

    return {
        "video_id":    video_id,
        "chunk_count": total_upserted,
        "namespace":   namespace,
        "indexed_at":  datetime.utcnow().isoformat() + "Z",
    }


# ── Log helpers ────────────────────────────────────────────────────────────────

def load_index_log() -> dict:
    if INDEX_LOG.exists():
        return json.loads(INDEX_LOG.read_text())
    return {"indexed": [], "skipped": [], "failed": []}


def save_index_log(log_data: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_LOG.write_text(json.dumps(log_data, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    video_id:  str | None,
    namespace: str,
    force:     bool,
    dry_run:   bool,
) -> None:

    if dry_run:
        log.info("DRY RUN — no files or Pinecone records will be written.")

    metadata = load_metadata()

    # Resolve targets
    if video_id:
        targets = [video_id]
        log.info(f"Processing single video: {video_id}")
    else:
        targets = sorted(
            p.name for p in VIDEOS_DIR.iterdir() if p.is_dir()
        )
        if not targets:
            log.warning(f"No video folders found in {VIDEOS_DIR}")
            return
        log.info(f"Found {len(targets)} video folder(s) to process.")

    index_log  = load_index_log()
    total_ok   = 0
    total_skip = 0
    total_fail = 0

    for vid_id in targets:
        log.info(f"Indexing: {vid_id}")
        try:
            stats = index_video(
                vid_id,
                namespace=namespace,
                metadata=metadata,
                force=force,
                dry_run=dry_run,
            )

            if stats is None:
                index_log["skipped"].append({
                    "video_id":   vid_id,
                    "reason":     "chunks_file_missing",
                    "skipped_at": datetime.utcnow().isoformat() + "Z",
                })
                total_skip += 1
                continue

            if stats.get("skipped"):
                total_skip += 1
                continue

            log.info(
                f"  ✓ {stats['chunk_count']} vectors upserted "
                f"→ namespace: '{stats['namespace']}'"
            )
            index_log["indexed"].append(stats)
            total_ok += 1

        except (KeyError, ValueError) as e:
            # Metadata validation errors — actionable, not a crash
            log.error(f"  ✗ Metadata error for {vid_id}: {e}")
            index_log["failed"].append({
                "video_id":  vid_id,
                "error":     str(e),
                "failed_at": datetime.utcnow().isoformat() + "Z",
            })
            total_fail += 1

        except Exception as e:
            log.error(f"  ✗ Unexpected error for {vid_id}: {e}")
            index_log["failed"].append({
                "video_id":  vid_id,
                "error":     str(e),
                "failed_at": datetime.utcnow().isoformat() + "Z",
            })
            total_fail += 1

    if not dry_run:
        save_index_log(index_log)

    log.info(
        f"\n── Indexing complete ────────────────────────\n"
        f"  ✓ Indexed  : {total_ok}\n"
        f"  ↷ Skipped  : {total_skip}  (already indexed or missing chunks)\n"
        f"  ✗ Failed   : {total_fail}\n"
        f"  Namespace  : {namespace}\n"
        + ("  (dry run — nothing written)" if dry_run else
           f"  Log saved → {INDEX_LOG}")
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed and index YouTube transcript chunks into Pinecone."
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Index a single video ID (default: index all)",
    )
    parser.add_argument(
        "--namespace",
        type=str,
        default=DEFAULT_NS,
        choices=["corpus", "live"],
        help=f"Pinecone namespace to upsert into (default: {DEFAULT_NS})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-index videos already marked as indexed in metadata.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Embed and build records but do not upsert to Pinecone",
    )
    args = parser.parse_args()

    run(
        video_id=args.video_id,
        namespace=args.namespace,
        force=args.force,
        dry_run=args.dry_run,
    )
