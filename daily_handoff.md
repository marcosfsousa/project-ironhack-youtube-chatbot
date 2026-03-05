# Session Handoff — Day 3
## Status: Day 2 Complete ✓

---

## What's Done

### Day 1 (Previous Session)
- `transcript_extractor.py` — fetches captions via `youtube-transcript-api` v1.2.4
- `cleaner.py` — HTML decoding, filler removal, sponsor flagging
- `chunker.py` — 60s time-window chunking, no overlap
- 4 test videos extracted, cleaned, chunked into `data/videos/`
- Pipeline hardened against `\u00a0` and `\ufffd` encoding issues

### Day 2 (This Session)
- `embedder.py` — singleton `get_model()` utility, exposes `sentence-transformers/all-MiniLM-L6-v2`
- `indexer.py` — embeds enriched text (`title | chunk_text`), upserts to Pinecone
- `bootstrap_metadata.py` — scans `data/videos/`, generates blank `metadata.json` entries
- `metadata.json` — manually curated for all 28 videos
- Full corpus of 28 videos extracted, cleaned, chunked, and indexed
- Pinecone index created and populated: ~370 vectors in `corpus` namespace
- Resume support added to `cleaner.py` and `chunker.py` (`--force` flag)
- SSL cert issue resolved on Windows via `PYTHONUTF8=1` conda env var
- Charmap encoding issue resolved for `aeWyp2vXxqA`

---

## Corpus State

| Metric | Value |
|---|---|
| Total videos | 28 |
| Total vectors in Pinecone | ~370 |
| Namespace | `corpus` |
| Index name | `youtube-qa-bot` |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Embedding dimension | 384 |
| Chunk window | 60s, no overlap |

### Topic Distribution
| Topic | Count |
|---|---|
| Biology | 5 |
| Cognitive Science | 2 |
| Cosmology | 4 |
| Education | 1 |
| History | 1 |
| Mathematics | 3 |
| Neuroscience | 1 |
| Philosophy | 3 |
| Physics | 4 |
| Psychology | 2 |
| Technology | 2 |

### Known Corpus Issues
- `MBRqu0YOH14` (Optimistic Nihilism) — 6 chunks, ~6 min, below 10 min target. Harmless, keep.
- `cebFWOlx848_008` — 9.8s closing CTA chunk, will never surface in retrieval
- `tlTKTTt47WE_008` — 7.0s closing CTA chunk, will never surface in retrieval
- Future improvement: `--min-duration` flag in `chunker.py` to drop tail chunks

---

## Repo Structure
```
pipeline/
  transcript_extractor.py  ✓
  cleaner.py               ✓  (resume support added)
  chunker.py               ✓  (resume support added)
  embedder.py              ✓  (utility module, exposes get_model())
  indexer.py               ✓
  bootstrap_metadata.py    ✓
data/
  metadata.json            ✓  (28 videos, fully curated)
  video_urls.txt           ✓  (28 URLs)
  videos/                  ✓  (28 folders, each with raw/clean/chunks)
  logs/
    extraction_log.json    ✓
    cleaning_log.json      ✓
    chunking_log.json      ✓
    indexing_log.json      ✓
agent/                     ← Day 3 starts here
  agent.py
  rag_chain.py
  retriever.py
  tools.py
  memory.py
  prompts.py
```

---

## Key Architecture Decisions

### Embedding
- Model: `sentence-transformers/all-MiniLM-L6-v2` (384d, normalized, cosine similarity)
- Text embedded: `"{title} | {chunk_text}"` — title prepended for topical context
- `embeddings.json` dropped — `indexer.py` is the single embedder, one source of truth
- `embedder.py` kept as utility module only (exposes `get_model()` singleton)

### Pinecone
- Index: `youtube-qa-bot`, cosine metric, 384d, AWS us-east-1 serverless
- Namespaces: `corpus` (pre-built) and `live` (on-the-fly, Day 4)
- Re-run behaviour: skip if `indexed: true` in `metadata.json`, `--force` to override
- Metadata payload per vector:
  ```json
  {
    "chunk_id":   "...",
    "video_id":   "...",
    "title":      "...",
    "channel":    "...",
    "topic":      "...",
    "start":      0.0,
    "end":        61.3,
    "chunk_text": "..."
  }
  ```
- `chunk_text` stores plain text (not enriched) — clean readable output for the LLM

### Pipeline
- All scripts: resume support via output file existence check, `--force` to re-run
- `--dry-run` available on all scripts except `transcript_extractor.py`
- Single video processing via `--video-id` on all scripts
- Encoding: `PYTHONUTF8=1` set in conda env to handle Windows charmap issues

### Live URL Mode (Day 4)
- Will target `live` Pinecone namespace
- Metadata (title, channel, topic) to be inferred via Groq LLM call on first 500 words
- Channel name recovered from `transcript_raw.json` where available, LLM fallback only
- To be built into `live_ingest.py` as `infer_metadata()` function

---

## Environment
- Conda env: `youtube-qa-bot` (Python 3.11.9)
- `PYTHONUTF8=1` set in conda env vars
- Keys in `.env`: `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_NAMESPACE_CORPUS`, `PINECONE_NAMESPACE_LIVE`
- Still needed in `.env` for Day 3: `GROQ_API_KEY`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`

## Installed Packages (key)
- `youtube-transcript-api==1.2.4`
- `sentence-transformers`
- `pinecone`
- `python-dotenv`

## Still to Install for Day 3
```bash
pip install langchain langchain-groq langchain-pinecone langgraph langsmith
```

---

## Day 3 Goals
Build a working RAG chain answering questions from the terminal.

- [ ] `retriever.py` — Pinecone similarity search, returns top-k chunks with sources
- [ ] `rag_chain.py` — prompt template + LLM + retriever via LangChain LCEL
- [ ] System prompt: grounded, cites timestamps, says "I don't know" if off-topic
- [ ] LangSmith tracing — verify traces appear in dashboard
- [ ] `agent.py` — LangGraph agent with `RAGRetrieverTool` and `VideoMetadataTool`
- [ ] Conversation memory (`ConversationBufferWindowMemory`, last 5 turns)
- [ ] Manual smoke test: 10 questions across 3 videos

**Deliverable:** `python agent.py` answers questions correctly in terminal.

# Session Handoff — Day 2

## What's done
- Day 1 complete: transcript_extractor.py, cleaner.py, chunker.py
- 4 test videos extracted, cleaned, chunked into data/videos/
- Pipeline hardened against \u00a0 and \ufffd encoding issues
- conda env: youtube-qa-bot (Python 3.11.9)
- youtube-transcript-api==1.2.4 (v1.x API, instantiation required)

## Where we left off
- About to start Day 2: embedder.py + indexer.py
- Pinecone account ready, PINECONE_API_KEY in .env
- Will need: sentence-transformers, pinecone

## Repo structure
pipeline/
  transcript_extractor.py  ✓
  cleaner.py               ✓
  chunker.py               ✓
  embedder.py              ← next
  indexer.py               ← next

## Key decisions made
- Pinecone namespaces: 'corpus' and 'live'
- Embeddings: sentence-transformers/all-MiniLM-L6-v2
- Chunk window: 60s no overlap
- Deployment target: Streamlit Community Cloud
- LLM: Groq (Llama 3.1 70B), upgrade path to OpenAI/Claude