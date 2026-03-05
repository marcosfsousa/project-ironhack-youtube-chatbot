"""
embedder.py
-----------
Embeds transcript chunks produced by chunker.py.
Reads  → data/videos/{video_id}/chunks.json
Writes → data/videos/{video_id}/embeddings.json

Why write embeddings to disk?
  Embedding is slow (~1–2 min per video on CPU).
  Persisting vectors lets you re-index into Pinecone
  without re-embedding — useful during corpus curation
  when you're tweaking namespaces or metadata.

Embedding model:
  sentence-transformers/all-MiniLM-L6-v2
  - 384-dimensional vectors
  - Fast on CPU (~2000 sentences/sec)
  - Good semantic quality for English science content
  - Free, no API key needed

Output format (embeddings.json):
  {
    "video_id":      "...",
    "model":         "all-MiniLM-L6-v2",
    "dimension":     384,
    "embedded_at":   "...",
    "chunk_count":   N,
    "embeddings": [
      {
        "chunk_id": "aircAruvnKk_000",
        "vector":   [0.123, ...]   // 384 floats
      },
      ...
    ]
  }

Guarantees:
  - chunks.json is NEVER modified
  - Running twice produces identical output (deterministic model)
  - Resume support: skips videos where embeddings.json already exists
  - Progress log written to data/logs/embedding_log.json

Usage:
  python embedder.py                        # embed all videos
  python embedder.py --video-id aircAruvnKk # embed one video
  python embedder.py --batch-size 64        # tune batch size
  python embedder.py --dry-run              # print stats, write nothing
  python embedder.py --force                # re-embed even if already done
"""

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

from sentence_transformers import SentenceTransformer

# ── Logging ────────────────────────────────────────────────────────────────────
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
EMBED_LOG     = LOGS_DIR / "embedding_log.json"

# ── Model config ───────────────────────────────────────────────────────────────
MODEL_NAME  = "sentence-transformers/all-MiniLM-L6-v2"
DIMENSION   = 384          # fixed for this model
DEFAULT_BATCH_SIZE = 32    # safe default for CPU; raise to 128 on GPU


# ── Model loader (singleton — load once per process) ──────────────────────────

_model: SentenceTransformer | None = None

def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        log.info(f"Loading embedding model: {MODEL_NAME}")
        _model = SentenceTransformer(MODEL_NAME)
        log.info(f"Model loaded. Embedding dimension: {DIMENSION}")
    return _model


# ── Per-video embedding ────────────────────────────────────────────────────────

def embed_video(
    video_id: str,
    batch_size: int,
    force: bool,
    dry_run: bool,
) -> dict | None:
    """
    Load chunks.json, embed all chunk texts, write embeddings.json.

    Args:
        video_id:   YouTube video ID (folder name under data/videos/)
        batch_size: number of texts to encode at once
        force:      re-embed even if embeddings.json already exists
        dry_run:    compute embeddings but write nothing to disk

    Returns:
        stats dict on success, None if chunks.json not found
    """
    chunks_path = VIDEOS_DIR / video_id / "chunks.json"
    embed_path  = VIDEOS_DIR / video_id / "embeddings.json"

    if not chunks_path.exists():
        log.warning(f"  chunks.json not found, skipping: {chunks_path}")
        return None

    # Resume support — skip if already done (unless --force)
    if embed_path.exists() and not force:
        log.info(f"  Already embedded, skipping: {video_id}  (use --force to re-embed)")
        return {"video_id": video_id, "skipped": True, "reason": "already_embedded"}

    # Load chunks
    data   = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    if not chunks:
        log.warning(f"  No chunks found in {chunks_path}, skipping.")
        return None

    # Extract texts preserving order
    chunk_ids = [c["chunk_id"] for c in chunks]
    texts     = [c["text"]     for c in chunks]

    log.info(f"  Embedding {len(texts)} chunks in batches of {batch_size}...")

    # Embed — show_progress_bar gives a tqdm bar per-video
    model   = get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine sim ≡ dot product → faster Pinecone queries
    )

    # Sanity check
    assert vectors.shape == (len(texts), DIMENSION), (
        f"Unexpected embedding shape: {vectors.shape}"
    )

    embedded_at = datetime.utcnow().isoformat() + "Z"

    stats = {
        "video_id":    video_id,
        "chunk_count": len(chunks),
        "dimension":   DIMENSION,
        "model":       MODEL_NAME,
        "embedded_at": embedded_at,
    }

    if dry_run:
        log.info(f"  DRY RUN — embeddings computed but not written.")
        return stats

    # Build output payload
    output = {
        "video_id":    video_id,
        "model":       MODEL_NAME,
        "dimension":   DIMENSION,
        "embedded_at": embedded_at,
        "chunk_count": len(chunks),
        "embeddings": [
            {
                "chunk_id": chunk_id,
                "vector":   vector.tolist(),   # numpy → plain list for JSON
            }
            for chunk_id, vector in zip(chunk_ids, vectors)
        ],
    }

    embed_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return stats


# ── Log helpers ────────────────────────────────────────────────────────────────

def load_embed_log() -> dict:
    if EMBED_LOG.exists():
        return json.loads(EMBED_LOG.read_text())
    return {"embedded": [], "skipped": [], "failed": []}


def save_embed_log(log_data: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_LOG.write_text(json.dumps(log_data, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    video_id:   str | None,
    batch_size: int,
    force:      bool,
    dry_run:    bool,
) -> None:

    if dry_run:
        log.info("DRY RUN — no files will be written.")

    # Resolve targets
    if video_id:
        targets = [video_id]
        log.info(f"Processing single video: {video_id}")
    else:
        targets = sorted(p.name for p in VIDEOS_DIR.iterdir() if p.is_dir())
        if not targets:
            log.warning(f"No video folders found in {VIDEOS_DIR}")
            return
        log.info(f"Found {len(targets)} video(s) to embed.")

    embed_log  = load_embed_log()
    total_ok   = 0
    total_skip = 0
    total_fail = 0

    for vid_id in targets:
        log.info(f"Embedding: {vid_id}")
        try:
            stats = embed_video(
                vid_id,
                batch_size=batch_size,
                force=force,
                dry_run=dry_run,
            )

            if stats is None:
                embed_log["skipped"].append({
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
                f"  ✓ {stats['chunk_count']} chunks embedded | "
                f"dim: {stats['dimension']} | model: {MODEL_NAME}"
            )
            embed_log["embedded"].append(stats)
            total_ok += 1

        except Exception as e:
            log.error(f"  ✗ Failed: {vid_id} — {e}")
            embed_log["failed"].append({
                "video_id":  vid_id,
                "error":     str(e),
                "failed_at": datetime.utcnow().isoformat() + "Z",
            })
            total_fail += 1

    if not dry_run:
        save_embed_log(embed_log)

    log.info(
        f"\n── Embedding complete ───────────────────────\n"
        f"  ✓ Embedded : {total_ok}\n"
        f"  ↷ Skipped  : {total_skip}  (already done or missing chunks)\n"
        f"  ✗ Failed   : {total_fail}\n"
        + ("  (dry run — nothing written)" if dry_run else
           f"  Log saved → {EMBED_LOG}")
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed YouTube transcript chunks using sentence-transformers."
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default=None,
        help="Embed a single video ID (default: embed all)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Encoding batch size (default: {DEFAULT_BATCH_SIZE}). "
             "Raise to 128 if running on GPU.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed videos that already have embeddings.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute embeddings but write nothing to disk",
    )
    args = parser.parse_args()

    run(
        video_id=args.video_id,
        batch_size=args.batch_size,
        force=args.force,
        dry_run=args.dry_run,
    )
