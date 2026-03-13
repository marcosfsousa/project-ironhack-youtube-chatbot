# ScienceQ

A RAG-based chatbot that answers questions grounded in YouTube science video transcripts. Ask anything about the pre-built corpus of 42 curated videos, or paste any YouTube URL to ingest and query it on the fly.

Built as the final project for the [Ironhack](https://www.ironhack.com) AI Engineering course.

**Live demo:** [scienceq.streamlit.app](https://scienceq.streamlit.app)

---

## Demo

![ScienceQ Demo](docs/demo.gif)

---

## What it does

- Answers factual questions from a corpus of 42 science explainer videos (Veritasium, Kurzgesagt, 3Blue1Brown, PBS Space Time, Big Think, and more)
- Pastes a YouTube URL → ingests it in real time → answers questions about it
- Streams answers token by token with clickable source timestamp pills
- Maintains 5-turn conversation memory with query rewriting for follow-up questions
- Stays grounded: if no relevant chunks are found above the confidence threshold, it says so rather than hallucinating

## Architecture

```
User query
    │
    ▼
LangGraph Agent  ── keyword routing ──►  RAG chain  ──►  Pinecone (corpus + live)
                                    │                          │
                                    └──►  Metadata tool        ▼
                                                          Groq LLM (llama-3.3-70b)
                                                               │
                                                               ▼
                                                     Streaming answer + sources
```

Full architecture details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)  
Corpus and pipeline details: [`docs/DATASET.md`](docs/DATASET.md)

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq — `llama-3.3-70b-versatile` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384d) |
| Vector DB | Pinecone Serverless (cosine, AWS us-east-1) |
| Orchestration | LangChain LCEL + LangGraph |
| Tracing | LangSmith |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

## Evaluation

Evaluated on a 30-case QA set (20 factual, 5 multi-turn, 5 adversarial) with GPT-4.1 as judge:

| Prompt version | Correctness | Tone | Grounding | Conciseness | Mean |
|---|---|---|---|---|---|
| v1 | 4.56 | 4.76 | 3.92 | 3.72 | 4.24 |
| v2 | 4.28 | 4.88 | 4.04 | 4.36 | **4.39** |

---

## Quickstart (run locally)

**Prerequisites:** Python 3.11, a Pinecone account, a Groq API key, a LangSmith account.

```bash
git clone https://github.com/<your-username>/project-ironhack-youtube-chatbot
cd project-ironhack-youtube-chatbot

pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys in .env

streamlit run app/streamlit_app.py
```

The app connects to the existing Pinecone corpus index — no pipeline run required to use the pre-built corpus.

### Required environment variables

```
PINECONE_API_KEY=
PINECONE_INDEX_NAME=youtube-qa-bot
PINECONE_NAMESPACE_CORPUS=corpus
PINECONE_NAMESPACE_LIVE=live
GROQ_API_KEY=
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=youtube-qa-bot
LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
```

---

## Rebuilding the corpus from scratch

If you want to index your own set of videos rather than using the pre-built Pinecone index, run the pipeline in order:

```bash
pip install -r requirements-dev.txt

# 1. Extract transcripts (edit data/video_urls.txt with your URLs first)
python pipeline/transcript_extractor.py

# 2. Clean transcripts
python pipeline/cleaner.py

# 3. Chunk into ~60s windows
python pipeline/chunker.py

# 4. Embed chunks
python pipeline/embedder.py

# 5. Upsert to Pinecone
python pipeline/indexer.py

# 6. Build metadata catalog
python pipeline/bootstrap_metadata.py
# Then manually fill in topic/difficulty fields in data/metadata.json
```

Each script supports `--video-id` to run on a single video and `--force` to re-run over existing output. See the docstring at the top of each file for full CLI options.

### Running tests

```bash
python tests/run_all_tests.py
```

76 unit tests, no live API calls required, full run in ~4s.

---

## Project structure

```
├── agent/              # LangGraph agent, RAG chain, retriever, tools, memory, prompts
├── app/                # Streamlit UI
├── data/               # metadata.json, eval_set.json, per-video transcript/chunk files
├── docs/               # ARCHITECTURE.md, DATASET.md
├── eval/               # LangSmith eval runner and results
├── pipeline/           # Offline corpus pipeline (extract → clean → chunk → embed → index)
├── tests/              # Unit tests
├── .env.example
├── requirements.txt        # Runtime (Streamlit Cloud)
└── requirements-dev.txt    # Full dev + pipeline + eval dependencies
```

---

## Known limitations

- Retrieval quality depends on transcript verbosity — visually-heavy videos without verbal explanation retrieve poorly
- Multi-turn pronoun resolution occasionally drifts on short follow-ups
- Live URL ingestion requires a video with available captions (auto-generated accepted). On Streamlit Community Cloud, a residential proxy is used to route transcript requests around YouTube's AWS IP blocks.

## Next steps

- Swap `all-MiniLM-L6-v2` for a natively asymmetric model (e.g. Cohere `embed-english-v3.0`) to improve retrieval scores without threshold tuning
- Add a reranker pass (cross-encoder) after initial retrieval for better precision
- Whisper integration for videos without captions
