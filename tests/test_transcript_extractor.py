# tests/test_transcript_extractor.py

import pytest
from pipeline.transcript_extractor import extract_video_id


class TestParseVideoId:

    # ── Standard watch URLs ────────────────────────────────────────────────────

    def test_standard_watch_url(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_watch_url_with_timestamp(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=43s") == "dQw4w9WgXcQ"

    def test_watch_url_with_playlist(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLabc123") == "dQw4w9WgXcQ"

    def test_watch_url_with_timestamp_and_playlist(self):
        assert extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=43s&list=PLabc123") == "dQw4w9WgXcQ"

    # ── Short URLs ─────────────────────────────────────────────────────────────

    def test_youtu_be_short_url(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_youtu_be_with_timestamp(self):
        assert extract_video_id("https://youtu.be/dQw4w9WgXcQ?t=43s") == "dQw4w9WgXcQ"

    # ── Embed URLs ─────────────────────────────────────────────────────────────

    def test_embed_url(self):
        assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ") == "dQw4w9WgXcQ"

    def test_embed_url_with_params(self):
        assert extract_video_id("https://www.youtube.com/embed/dQw4w9WgXcQ?autoplay=1") == "dQw4w9WgXcQ"

    # ── Edge cases ─────────────────────────────────────────────────────────────

    def test_invalid_url_returns_none(self):
        assert extract_video_id("https://www.google.com") is None

    def test_empty_string_returns_none(self):
        assert extract_video_id("") is None

    def test_plain_text_returns_none(self):
        assert extract_video_id("not a url at all") is None

    def test_id_is_exactly_11_chars(self):
        result = extract_video_id("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
        assert len(result) == 11