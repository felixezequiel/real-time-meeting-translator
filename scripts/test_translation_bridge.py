"""
Unit tests for translation_bridge.split_commit_point.

The commit policy decides when a streaming Qwen translation fragment is
shipped to the TTS. The function is pure (string -> int), so we can
exercise every regression case without booting the LLM.

Run:
    python -m unittest scripts/test_translation_bridge.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from translation_bridge import (  # noqa: E402
    split_commit_point,
    MAX_BUFFER_CHARS,
    MIN_CLAUSE_FRAGMENT_CHARS,
)


class SentenceFinalCommitTests(unittest.TestCase):
    """`.!?…` at the tail SHOULD commit when unambiguous."""

    def test_period_after_word_commits(self):
        self.assertEqual(split_commit_point("Olá mundo."), len("Olá mundo."))

    def test_exclamation_commits(self):
        self.assertEqual(split_commit_point("Olá!"), len("Olá!"))

    def test_question_commits(self):
        self.assertEqual(split_commit_point("Tudo bem?"), len("Tudo bem?"))

    def test_ellipsis_commits(self):
        self.assertEqual(split_commit_point("Bom…"), len("Bom…"))


class NumericPeriodIsNotBoundaryTests(unittest.TestCase):
    """Regression: pt-BR thousands separator and decimals must NOT be
    treated as sentence boundaries. This was the bug that made XTTS
    synthesise `"14."` and `"000 toneladas."` as two separate fragments,
    audible as a stutter on the user's 2026-05-11 capture."""

    def test_thousands_separator_does_not_commit(self):
        # Mid-stream the LLM emits "14." before the rest of the number.
        # If we commit here, the TTS speaks "14" with sentence-ending
        # prosody and then "000 toneladas" as a separate fragment.
        self.assertEqual(split_commit_point("14."), -1)

    def test_thousands_separator_million_does_not_commit(self):
        self.assertEqual(split_commit_point("Custou 1."), -1)

    def test_decimal_does_not_commit(self):
        # English decimal "3.14" same shape mid-stream.
        self.assertEqual(split_commit_point("Cerca de 3."), -1)

    def test_full_number_then_word_does_commit(self):
        # Once the rest of the sentence finishes, the final period IS a
        # boundary (preceded by a letter, not a digit).
        text = "São 14.000 toneladas."
        self.assertEqual(split_commit_point(text), len(text))

    def test_number_dot_followed_by_digits_in_buffer_keeps_accumulating(self):
        # The buffer "14.000" doesn't end in punctuation at all — no commit.
        self.assertEqual(split_commit_point("14.000"), -1)

    def test_currency_thousands_does_not_commit_at_period(self):
        # "R$ 1." mid-stream — period preceded by digit.
        self.assertEqual(split_commit_point("R$ 1."), -1)


class ClauseCommitTests(unittest.TestCase):
    """Comma/semicolon/colon at the tail commit only above the floor."""

    def test_short_clause_does_not_commit(self):
        self.assertEqual(split_commit_point("Bem,"), -1)

    def test_long_clause_commits(self):
        buffer = "a" * (MIN_CLAUSE_FRAGMENT_CHARS - 1) + ","
        self.assertEqual(split_commit_point(buffer), len(buffer))


class LengthSafetyValveTests(unittest.TestCase):
    """When no punctuation arrives, commit at the last space past the cap."""

    def test_under_cap_does_not_commit(self):
        self.assertEqual(split_commit_point("a b c d"), -1)

    def test_over_cap_commits_at_last_space(self):
        # 80 chars of "word " pairs => last_space well past MAX_BUFFER_CHARS - 20.
        buffer = ("word " * 20).rstrip() + " more"
        self.assertGreater(len(buffer), MAX_BUFFER_CHARS)
        commit = split_commit_point(buffer)
        self.assertGreater(commit, 0)
        self.assertEqual(buffer[commit - 1], " ")


class EmptyBufferTests(unittest.TestCase):
    def test_empty_returns_minus_one(self):
        self.assertEqual(split_commit_point(""), -1)


if __name__ == "__main__":
    unittest.main()
