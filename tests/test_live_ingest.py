# tests/test_live_ingest.py

import pytest
from unittest.mock import MagicMock, patch
from pipeline.live_ingest import _is_already_indexed, parse_video_id, IngestResult


# ── parse_video_id (live_ingest version) ──────────────────────────────────────

class TestParseVideoId:
    """
    live_ingest.py has its own parse_video_id — verify it behaves
    identically to transcript_extractor.extract_video_id.
    """

    def test_standard_watch_url(self):
        assert parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_youtu_be_with_timestamp(self):
        assert parse_video_id("https://youtu.be/dQw4w9WgXcQ?t=43s") == "dQw4w9WgXcQ"

    def test_watch_url_with_timestamp(self):
        assert parse_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=43s") == "dQw4w9WgXcQ"

    def test_invalid_url_returns_none(self):
        assert parse_video_id("https://www.google.com") is None

    def test_empty_string_returns_none(self):
        assert parse_video_id("") is None


# ── _is_already_indexed ───────────────────────────────────────────────────────

def make_mock_index(found_in: str | None) -> MagicMock:
    """
    Build a mock Pinecone index.
    found_in: namespace where the vector exists, or None if not found anywhere.
    """
    def fetch_side_effect(ids, namespace):
        mock_result = MagicMock()
        if found_in and namespace == found_in:
            mock_vector = MagicMock()
            mock_vector.metadata = {"title": "Test Video", "channel": "Test Channel"}
            mock_result.vectors = {ids[0]: mock_vector}
        else:
            mock_result.vectors = {}
        return mock_result

    mock_index = MagicMock()
    mock_index.fetch.side_effect = fetch_side_effect
    return mock_index


class TestIsAlreadyIndexed:

    def test_found_in_corpus_namespace(self):
        index = make_mock_index(found_in="corpus")
        found, ns = _is_already_indexed("testvideo01", index, "corpus", "live")
        assert found is True
        assert ns == "corpus"

    def test_found_in_live_namespace(self):
        index = make_mock_index(found_in="live")
        found, ns = _is_already_indexed("testvideo01", index, "corpus", "live")
        assert found is True
        assert ns == "live"

    def test_not_found_in_either_namespace(self):
        index = make_mock_index(found_in=None)
        found, ns = _is_already_indexed("testvideo01", index, "corpus", "live")
        assert found is False
        assert ns == ""

    def test_corpus_checked_before_live(self):
        """Corpus namespace must be checked first — order matters."""
        index = make_mock_index(found_in="corpus")
        _is_already_indexed("testvideo01", index, "corpus", "live")
        calls = [call.kwargs["namespace"] for call in index.fetch.call_args_list]
        assert calls[0] == "corpus"

    def test_live_not_checked_if_found_in_corpus(self):
        """If found in corpus, live namespace should never be queried."""
        index = make_mock_index(found_in="corpus")
        _is_already_indexed("testvideo01", index, "corpus", "live")
        assert index.fetch.call_count == 1

    def test_live_checked_if_not_in_corpus(self):
        """If not in corpus, live namespace must be queried."""
        index = make_mock_index(found_in="live")
        _is_already_indexed("testvideo01", index, "corpus", "live")
        assert index.fetch.call_count == 2


# ── IngestResult dataclass ────────────────────────────────────────────────────

class TestIngestResult:

    def test_default_values(self):
        r = IngestResult(video_id="abc", url="https://youtu.be/abc")
        assert r.title == "Unknown"
        assert r.channel == "Unknown"
        assert r.topic == "Other"
        assert r.success is False
        assert r.error is None
        assert r.already_indexed is False
        assert r.chunk_count == 0

    def test_youtube_url_property(self):
        r = IngestResult(video_id="dQw4w9WgXcQ", url="https://youtu.be/dQw4w9WgXcQ")
        assert r.youtube_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_chunks_default_empty_list(self):
        r = IngestResult(video_id="abc", url="https://youtu.be/abc")
        assert r.chunks == []