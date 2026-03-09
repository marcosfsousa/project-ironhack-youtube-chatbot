# Dataset — YouTube QA Bot Corpus

## Overview

The corpus consists of 29 curated science and education YouTube videos, totalling approximately 370 vectors in Pinecone. Videos were selected to maximize topic diversity, explanation density, and retrieval reliability.

---

## Corpus Statistics

| Attribute | Value |
|---|---|
| Total videos | 29 |
| Total chunks (vectors) | ~370 |
| Chunk window | ~60 seconds |
| Embedding dimensions | 384 |
| Pinecone namespace | `corpus` |

---

## Channel Selection

Videos were sourced from channels known for scripted, explanation-dense content with reliable closed captions:

| Channel | Videos | Notes |
|---|---|---|
| Veritasium | 6 | Physics, mathematics, cognitive science |
| Kurzgesagt – In a Nutshell | 4 | Biology, cosmology, philosophy, history |
| 3Blue1Brown | 3 | Mathematics, technology |
| PBS Space Time | 3 | Physics, cosmology |
| Big Think | 5 | Cosmology, psychology, biology, philosophy |
| TED / TEDx | 4 | Psychology, education, neuroscience, biology |
| Other (CGP Grey, IBM, HHMI, Science Time) | 4 | Philosophy, technology, biology, cosmology |

**Selection criteria:**
- One main concept per video (avoids retrieval ambiguity)
- Clear verbal explanations with causal reasoning
- Scripted or semi-scripted delivery (higher transcript quality)
- Human-generated captions preferred over auto-generated
- No pure opinion pieces or list-style ("Top 10...") videos

---

## Topic Distribution

| Topic | Videos |
|---|---|
| Physics | 4 |
| Biology | 5 |
| Cosmology | 6 |
| Mathematics | 3 |
| Psychology | 3 |
| Cognitive Science | 2 |
| Neuroscience | 1 |
| Philosophy | 2 |
| Technology | 2 |
| Education | 1 |
| History | 1 |

---

## Pipeline

The corpus was built using a 6-stage offline pipeline. Each stage reads from and writes to a per-video directory under `data/videos/{video_id}/`.

### Stage 1 — Transcript Extraction (`pipeline/transcript_extractor.py`)

Fetches closed captions via `youtube-transcript-api` v1.2.4 (instance-based API). Stores raw transcript segments with start/end timestamps.

```json
{
  "video_id": "pTn6Ewhb27k",
  "title": "Why No One Has Measured The Speed Of Light",
  "channel": "Veritasium",
  "transcript": [
    { "start": 0.0, "end": 4.2, "text": "..." }
  ]
}
```

Human-generated captions are preferred. Auto-generated captions are accepted as fallback. Videos with no available captions are excluded from the corpus.

**Note on live ingestion:** for on-the-fly URL ingestion, `yt-dlp` is used separately to fetch video title and channel before transcript extraction. The corpus pipeline does not use `yt-dlp` — titles and channels are manually curated in `metadata.json`.

### Stage 2 — Cleaning (`pipeline/cleaner.py`)

Normalizes raw transcripts without altering meaning:

- Merges broken subtitle lines into complete sentences
- Normalizes punctuation and casing
- Removes sponsor segment markers
- Handles `\u00a0` non-breaking spaces (common in auto-generated captions)
- Flags empty segments rather than deleting — downstream stages skip them

Timestamps are preserved exactly. The original transcript is never modified.

### Stage 3 — Chunking (`pipeline/chunker.py`)

Splits cleaned transcripts into time-window chunks:

- **Target window:** 60 seconds (configurable via `--window`)
- **Boundary logic:** closes a chunk at the nearest segment boundary once the window target is reached — never mid-sentence
- **No overlap** between chunks — clean boundaries for precise timestamp deep links
- Each chunk carries `chunk_id`, `video_id`, `start`, `end`, `duration`, `text`, and `segment_count`

```json
{
  "chunk_id": "pTn6Ewhb27k_003",
  "video_id": "pTn6Ewhb27k",
  "start": 180.4,
  "end": 241.1,
  "duration": 60.7,
  "text": "...",
  "segment_count": 14
}
```

### Stage 4 — Embedding (`pipeline/embedder.py`)

Embeds each chunk using `sentence-transformers/all-MiniLM-L6-v2`:

- **Dimensions:** 384
- **Similarity metric:** cosine
- **Encoding:** chunks are embedded as `"{title} | {chunk_text}"` — the video title is prepended to bake topical context into each vector. This improves retrieval precision for topic-adjacent queries where the chunk text alone is ambiguous.
- **Query encoding:** plain query text (no title prepend) — the asymmetry is intentional and works well with cosine similarity
- Vectors are persisted to `embeddings.json` per video to avoid re-embedding during re-indexing

### Stage 5 — Indexing (`pipeline/indexer.py`)

Upserts vectors to Pinecone Serverless (AWS us-east-1, cosine similarity):

- **Namespace:** `corpus` for the pre-built corpus, `live` for on-the-fly ingestion
- **Metadata stored per vector:** `chunk_id`, `video_id`, `title`, `channel`, `topic`, `start`, `end`, `chunk_text`
- Note: the embedded text includes the title prefix, but only the plain `chunk_text` is stored in metadata — this keeps LLM context clean
- Resume logic: skips videos already indexed unless `--force` flag is passed

### Stage 6 — Metadata Bootstrap (`pipeline/bootstrap_metadata.py`)

Builds `data/metadata.json` — a flat catalog of all indexed videos used by the `VideoMetadataTool` for catalog queries. Separate from Pinecone metadata; used for browsing and filtering without requiring a vector search.

---

## Retrieval Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `top_k` | 5 | Sufficient context without exceeding prompt budget |
| `score_threshold` | 0.28 | Lowered from 0.35 on Day 5 — asymmetric embedding (title-prepended chunks vs plain query text) depresses cosine scores. Single-word queries score 0.27–0.28, fuller questions score ~0.31 — both below the original gate. On-topic queries clear 0.28 reliably; off-topic fall below it. |
| Multi-namespace | True | Corpus + live queried together at runtime |

---

## Quality Control

Each video was verified through:

1. **Manual question testing** — 3–5 questions per video run against the retriever to confirm relevant chunks are returned above the score threshold
2. **Timestamp alignment** — source pills verified to deep-link to the correct moment in the video
3. **Hallucination risk check** — videos heavily dependent on visuals (without verbal explanation) were excluded
4. **Eval set coverage** — 20 of the 29 videos are covered by at least one case in `eval/eval_set.json`

---

## Eval Set

A separate evaluation dataset of 30 cases was constructed alongside the corpus:

| Type | Count |
|---|---|
| Factual RAG | 20 |
| Multi-turn (pronoun resolution) | 5 |
| Adversarial (prompt injection, out-of-scope) | 5 |

Cases cover all 29 corpus videos. Adversarial cases are excluded from automated scoring and reviewed manually in `eval/manual_review.json`.