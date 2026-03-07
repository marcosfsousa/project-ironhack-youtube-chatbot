# tests/test_chunker.py

import pytest
from pipeline.chunker import chunk_segments


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_segment(start: float, end: float, text: str, flags: list = None) -> dict:
    """Helper to build a minimal cleaned segment dict."""
    seg = {
        "start":    start,
        "end":      end,
        "duration": round(end - start, 3),
        "text":     text,
    }
    if flags:
        seg["flags"] = flags
    return seg


def make_segments(count: int, duration: float = 10.0) -> list[dict]:
    """Helper to build N segments of equal duration starting at 0."""
    return [
        make_segment(i * duration, (i + 1) * duration, f"segment {i}")
        for i in range(count)
    ]


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestChunkSegments:

    # ── Basic windowing ────────────────────────────────────────────────────────

    def test_single_chunk_when_under_window(self):
        segments = make_segments(3, duration=10.0)  # 30s total, window=60s
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert len(chunks) == 1

    def test_two_chunks_when_over_window(self):
        segments = make_segments(8, duration=10.0)  # 80s total, window=60s
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert len(chunks) == 2

    def test_exact_window_boundary(self):
        segments = make_segments(6, duration=10.0)  # exactly 60s
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert len(chunks) == 1

    # ── Chunk ID format ────────────────────────────────────────────────────────

    def test_chunk_id_zero_padded(self):
        segments = make_segments(8, duration=10.0)
        chunks = chunk_segments(segments, window=60.0, video_id="abc123")
        assert chunks[0]["chunk_id"] == "abc123_000"
        assert chunks[1]["chunk_id"] == "abc123_001"

    def test_chunk_id_contains_video_id(self):
        segments = make_segments(3, duration=10.0)
        chunks = chunk_segments(segments, window=60.0, video_id="myvideoXYZ")
        assert chunks[0]["chunk_id"].startswith("myvideoXYZ")

    # ── Timestamps ────────────────────────────────────────────────────────────

    def test_chunk_start_matches_first_segment(self):
        segments = make_segments(3, duration=10.0)
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert chunks[0]["start"] == 0.0

    def test_chunk_end_matches_last_segment(self):
        segments = make_segments(3, duration=10.0)
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert chunks[0]["end"] == 30.0

    def test_duration_is_end_minus_start(self):
        segments = make_segments(3, duration=10.0)
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert chunks[0]["duration"] == pytest.approx(chunks[0]["end"] - chunks[0]["start"])

    # ── Text joining ──────────────────────────────────────────────────────────

    def test_text_joins_segments(self):
        segments = [
            make_segment(0, 10, "hello"),
            make_segment(10, 20, "world"),
        ]
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert chunks[0]["text"] == "hello world"

    def test_text_collapses_double_spaces(self):
        segments = [
            make_segment(0, 10, "hello  "),
            make_segment(10, 20, "  world"),
        ]
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert "  " not in chunks[0]["text"]

    # ── Empty and flagged segment handling ────────────────────────────────────

    def test_skips_empty_after_clean_flag(self):
        segments = [
            make_segment(0, 10, "", flags=["empty_after_clean"]),
            make_segment(10, 20, "real content"),
        ]
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert len(chunks) == 1
        assert chunks[0]["text"] == "real content"

    def test_skips_blank_text_segment(self):
        segments = [
            make_segment(0, 10, "   "),
            make_segment(10, 20, "real content"),
        ]
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert chunks[0]["text"] == "real content"

    def test_all_empty_segments_returns_empty_list(self):
        segments = [
            make_segment(0, 10, "", flags=["empty_after_clean"]),
            make_segment(10, 20, "", flags=["empty_after_clean"]),
        ]
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert chunks == []

    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_empty_input_returns_empty_list(self):
        chunks = chunk_segments([], window=60.0, video_id="test")
        assert chunks == []

    def test_single_segment_becomes_one_chunk(self):
        segments = [make_segment(0, 5, "just one segment")]
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert len(chunks) == 1
        assert chunks[0]["segment_count"] == 1

    def test_segment_count_is_accurate(self):
        segments = make_segments(3, duration=10.0)
        chunks = chunk_segments(segments, window=60.0, video_id="test")
        assert chunks[0]["segment_count"] == 3