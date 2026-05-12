"""
Unit tests for the SBD bridge's `split_complete` decision logic.

Inputs are taken from real 2026-05-11/12 captures where the regex-only
splitter produced poor flushes (orphan clauses, mid-sentence cuts).

Run:
    python -m unittest scripts/test_sbd_bridge.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sbd_bridge import load_pipelines, split_complete  # noqa: E402


class SbdEnTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipelines = load_pipelines()

    def test_orphan_fragment_without_verb_is_pending(self):
        # "really looks like." had a verb + period but was an orphan
        # in the original capture. With proper parsing it has finite
        # verb + subject ("it" implied) — still complete by rules.
        # The original PROBLEM was upstream (the accumulator split too
        # early); here we just verify the rules don't reject grammatical
        # sentences. Use a clearly incomplete fragment for the test:
        text = "out what that future"
        complete, rest = split_complete(text, "en", self.pipelines)
        self.assertEqual(complete, "")
        self.assertEqual(rest, text)

    def test_two_word_fragment_is_pending(self):
        # "is to" was flushed as its own phrase in the capture.
        text = "is to"
        complete, rest = split_complete(text, "en", self.pipelines)
        self.assertEqual(complete, "")
        self.assertEqual(rest, text)

    def test_complete_sentence_is_released(self):
        text = "Welcome to the first episode of our new series."
        complete, rest = split_complete(text, "en", self.pipelines)
        self.assertEqual(complete, text)
        self.assertEqual(rest, "")

    def test_holds_sentence_without_terminal_punctuation(self):
        # Streaming STT often emits without final punct until the next
        # token arrives. The bridge must hold these.
        text = "Welcome to the first episode of our new series"
        complete, rest = split_complete(text, "en", self.pipelines)
        self.assertEqual(complete, "")
        self.assertEqual(rest, text)

    def test_holds_run_on_open_clause(self):
        # The capture had "and two and a half times the height of the"
        # flushed mid-clause. With SBD: no period, no boundary, hold.
        text = "and two and a half times the height of the"
        complete, rest = split_complete(text, "en", self.pipelines)
        self.assertEqual(complete, "")
        self.assertEqual(rest, text)

    def test_complete_then_pending_splits(self):
        text = "Welcome to the first episode of our new series. And in every episode"
        complete, rest = split_complete(text, "en", self.pipelines)
        self.assertTrue(complete.startswith("Welcome"))
        self.assertTrue(complete.endswith("series."))
        self.assertTrue(rest.startswith("And in every"))


class SbdPtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipelines = load_pipelines()

    def test_full_sentence_with_thousands_separator_is_released(self):
        # "14.000" is pt-BR thousands separator; should NOT trip a
        # sentence boundary in the middle.
        text = "A plataforma Piper Alpha tinha 14.000 toneladas."
        complete, rest = split_complete(text, "pt", self.pipelines)
        self.assertEqual(complete, text)
        self.assertEqual(rest, "")

    def test_orphan_pt_fragment_is_pending(self):
        text = "para esta conversa"
        complete, rest = split_complete(text, "pt", self.pipelines)
        self.assertEqual(complete, "")
        self.assertEqual(rest, text)

    def test_holds_open_clause(self):
        text = "Meu objetivo para essa conversa é tentar descobrir o futuro"
        complete, rest = split_complete(text, "pt", self.pipelines)
        self.assertEqual(complete, "")
        self.assertEqual(rest, text)

    def test_complete_pt_sentence_is_released(self):
        text = "Meu objetivo para essa conversa é tentar descobrir o futuro."
        complete, rest = split_complete(text, "pt", self.pipelines)
        self.assertEqual(complete, text)
        self.assertEqual(rest, "")


class SbdEdgeCases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipelines = load_pipelines()

    def test_empty_returns_empty_pair(self):
        complete, rest = split_complete("", "en", self.pipelines)
        self.assertEqual(complete, "")
        self.assertEqual(rest, "")

    def test_unknown_language_holds_text(self):
        complete, rest = split_complete("Hello world.", "xx", self.pipelines)
        self.assertEqual(complete, "")
        self.assertEqual(rest, "Hello world.")


if __name__ == "__main__":
    unittest.main()
