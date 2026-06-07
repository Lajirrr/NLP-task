import unittest

from code.text_utils import (
    corpus_bleu,
    detokenize_chinese,
    simple_english_tokenize,
)


class TextUtilsTest(unittest.TestCase):
    def test_simple_english_tokenize_lowercases_and_splits_punctuation(self):
        self.assertEqual(
            simple_english_tokenize("Tom is a student."),
            ["tom", "is", "a", "student", "."],
        )

    def test_simple_english_tokenize_handles_common_contractions(self):
        self.assertEqual(
            simple_english_tokenize("I'll go, don't worry."),
            ["i", "'ll", "go", ",", "do", "n't", "worry", "."],
        )

    def test_detokenize_chinese_removes_special_tokens_and_joins_words(self):
        self.assertEqual(
            detokenize_chinese(["<BOS>", "汤姆", "是", "学生", "。", "<EOS>", "<PAD>"]),
            "汤姆是学生。",
        )

    def test_corpus_bleu_returns_one_for_exact_match(self):
        score = corpus_bleu(
            predictions=[["汤姆", "是", "学生", "。"]],
            references=[["汤姆", "是", "学生", "。"]],
        )
        self.assertAlmostEqual(score, 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
