# tests/test_cleaner.py

import pytest
from pipeline.cleaner import clean_text, is_sponsor_segment


class TestCleanText:

    # ── HTML entity decoding ───────────────────────────────────────────────────

    def test_decodes_html_ampersand(self):
        assert clean_text("rocks &amp; minerals") == "rocks & minerals"

    def test_decodes_html_apostrophe(self):
        assert clean_text("it&#39;s a test") == "it's a test"

    # ── Non-speech tag removal ─────────────────────────────────────────────────

    def test_removes_music_tag(self):
        assert clean_text("[Music]") == ""

    def test_removes_applause_tag(self):
        assert clean_text("[Applause]") == ""

    def test_removes_inline_music_tag(self):
        assert clean_text("hello [Music] world") == "hello world"

    def test_removes_uppercase_tag(self):
        assert clean_text("[MUSIC]") == ""

    # ── Filler word removal ────────────────────────────────────────────────────

    def test_removes_uh(self):
        assert clean_text("uh this is a test") == "this is a test"

    def test_removes_um(self):
        assert clean_text("this is um a test") == "this is a test"

    def test_removes_hmm(self):
        assert clean_text("hmm interesting") == "interesting"

    def test_does_not_clip_umbrella(self):
        assert "umbrella" in clean_text("under the umbrella")

    # ── Whitespace normalisation ───────────────────────────────────────────────

    def test_collapses_multiple_spaces(self):
        assert clean_text("too  many   spaces") == "too many spaces"

    def test_strips_leading_trailing_whitespace(self):
        assert clean_text("  hello world  ") == "hello world"

    def test_replaces_non_breaking_space(self):
        assert clean_text("hello\u00a0world") == "hello world"

    def test_empty_string_returns_empty(self):
        assert clean_text("") == ""

    # ── Combined operations ────────────────────────────────────────────────────

    def test_full_pipeline_order(self):
        # HTML decode → tag removal → filler removal → whitespace
        assert clean_text("&amp; uh [Music]  hello") == "& hello"


class TestIsSponsorSegment:

    def test_detects_brought_to_you_by(self):
        assert is_sponsor_segment("This video is brought to you by NordVPN") is True

    def test_detects_sponsored_by(self):
        assert is_sponsor_segment("sponsored by Squarespace") is True

    def test_detects_use_code(self):
        assert is_sponsor_segment("use code SCIENCE for 20% off") is True

    def test_detects_nordvpn(self):
        assert is_sponsor_segment("Check out NordVPN in the description below") is True

    def test_detects_skillshare(self):
        assert is_sponsor_segment("today's video is brought to you by Skillshare") is True

    def test_clean_segment_not_flagged(self):
        assert is_sponsor_segment("The speed of light is 299,792 km/s") is False

    def test_empty_string_not_flagged(self):
        assert is_sponsor_segment("") is False