# tests/test_ingest_node.py

import pytest
from unittest.mock import patch, MagicMock
from agent.agent import ingest_node, AgentState
from pipeline.live_ingest import IngestResult


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_state(question: str) -> AgentState:
    """Build a minimal AgentState for testing ingest_node."""
    return {
        "messages":     [],
        "question":     question,
        "intent":       "ingest",
        "answer":       "",
        "rag_response": None,
    }


def make_result(**kwargs) -> IngestResult:
    """Build an IngestResult with sensible defaults, overridable via kwargs."""
    defaults = {
        "video_id": "dQw4w9WgXcQ",
        "url":      "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "title":    "Test Video",
        "channel":  "Test Channel",
        "success":  False,
        "error":    None,
        "already_indexed": False,
    }
    defaults.update(kwargs)
    return IngestResult(**defaults)


# ── No URL in message ─────────────────────────────────────────────────────────

class TestIngestNodeNoUrl:

    def test_no_url_returns_helpful_message(self):
        state = make_state("can you add this video for me?")
        result = ingest_node(state)
        assert "couldn't find" in result["answer"].lower()
        assert "url" in result["answer"].lower()

    def test_no_url_rag_response_is_none(self):
        state = make_state("can you add this video for me?")
        result = ingest_node(state)
        assert result["rag_response"] is None


# ── Already indexed ───────────────────────────────────────────────────────────

class TestIngestNodeAlreadyIndexed:

    @patch("agent.agent.ingest_url")
    def test_already_indexed_message(self, mock_ingest):
        mock_ingest.return_value = make_result(
            already_indexed=True,
            success=True,
            title="Test Video",
        )
        state = make_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = ingest_node(state)
        assert "already have" in result["answer"].lower()
        assert "Test Video" in result["answer"]

    @patch("agent.agent.ingest_url")
    def test_already_indexed_uses_video_id_fallback(self, mock_ingest):
        mock_ingest.return_value = make_result(
            already_indexed=True,
            success=True,
            title="",
        )
        state = make_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = ingest_node(state)
        # Falls back to video_id when title is empty
        assert "dQw4w9WgXcQ" in result["answer"]


# ── Successful ingest ─────────────────────────────────────────────────────────

class TestIngestNodeSuccess:

    @patch("agent.agent.ingest_url")
    def test_success_message_contains_title(self, mock_ingest):
        mock_ingest.return_value = make_result(
            success=True,
            title="How Black Holes Form",
            channel="Veritasium",
        )
        state = make_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = ingest_node(state)
        assert "How Black Holes Form" in result["answer"]
        assert "Veritasium" in result["answer"]

    @patch("agent.agent.ingest_url")
    def test_success_rag_response_is_none(self, mock_ingest):
        mock_ingest.return_value = make_result(success=True)
        state = make_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = ingest_node(state)
        assert result["rag_response"] is None

    @patch("agent.agent.ingest_url")
    def test_success_last_ingest_stored_on_state(self, mock_ingest):
        ingest_result = make_result(success=True)
        mock_ingest.return_value = ingest_result
        state = make_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = ingest_node(state)
        assert result["_last_ingest"] == ingest_result


# ── Error handling ────────────────────────────────────────────────────────────

class TestIngestNodeErrors:

    @patch("agent.agent.ingest_url")
    def test_user_facing_error_passes_through(self, mock_ingest):
        mock_ingest.return_value = make_result(
            error="No captions are available for this video. "
                  "Try a video with auto-generated or manual subtitles enabled."
        )
        state = make_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = ingest_node(state)
        assert "No captions" in result["answer"]

    @patch("agent.agent.ingest_url")
    def test_technical_error_is_masked(self, mock_ingest):
        mock_ingest.return_value = make_result(
            error="Missing environment variable: PINECONE_API_KEY. Check your .env file."
        )
        state = make_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = ingest_node(state)
        assert "PINECONE_API_KEY" not in result["answer"]
        assert "something went wrong" in result["answer"].lower()

    @patch("agent.agent.ingest_url")
    def test_unexpected_error_is_masked(self, mock_ingest):
        mock_ingest.return_value = make_result(
            error="Unexpected error during ingestion: Connection timeout"
        )
        state = make_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = ingest_node(state)
        assert "Connection timeout" not in result["answer"]
        assert "something went wrong" in result["answer"].lower()

    @patch("agent.agent.ingest_url")
    def test_none_error_fallback_message(self, mock_ingest):
        mock_ingest.return_value = make_result(error=None)
        state = make_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = ingest_node(state)
        assert "something went wrong" in result["answer"].lower()

    @patch("agent.agent.ingest_url")
    def test_private_video_error_passes_through(self, mock_ingest):
        mock_ingest.return_value = make_result(
            error="This video is private or has been removed."
        )
        state = make_state("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        result = ingest_node(state)
        assert "private" in result["answer"].lower()