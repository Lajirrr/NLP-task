import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock


class EvaluateTest(unittest.TestCase):
    def test_parse_args_accepts_loss_only_and_decode_limit(self):
        from code.evaluate import parse_args

        args = parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--loss-only",
                "--decode-limit",
                "25",
            ]
        )

        self.assertTrue(args.loss_only)
        self.assertEqual(args.decode_limit, 25)
        self.assertEqual(len(args.checkpoint), 1)

    def test_parse_args_accepts_multiple_checkpoints_for_ensemble(self):
        from code.evaluate import parse_args

        args = parse_args(
            [
                "--checkpoint",
                "model-a.pt",
                "model-b.pt",
            ]
        )

        self.assertEqual([str(path) for path in args.checkpoint], ["model-a.pt", "model-b.pt"])

    def test_parse_args_accepts_translations_output(self):
        from code.evaluate import parse_args

        args = parse_args(
            [
                "--checkpoint",
                "model.pt",
                "--translations-output",
                "translations.txt",
            ]
        )

        self.assertEqual(args.translations_output, Path("translations.txt"))

    def test_write_translations_output_uses_utf8_text(self):
        from code import evaluate

        with TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "translations.txt"

            evaluate.write_translations_output(
                output_path,
                [
                    {
                        "index": 1,
                        "src": "tom is a student .",
                        "ref": "汤姆是学生。",
                        "pred": "汤姆是个学生。",
                    }
                ],
                metrics={"Corpus BLEU": "0.3844"},
            )

            raw = output_path.read_bytes()

        text = raw.decode("utf-8")
        self.assertIn("Corpus BLEU: 0.3844", text)
        self.assertIn("EN : tom is a student .", text)
        self.assertIn("REF: 汤姆是学生。", text)
        self.assertIn("PRED: 汤姆是个学生。", text)
        self.assertIn("汤姆".encode("utf-8"), raw)

    def test_evaluate_bleu_and_examples_respects_decode_limit(self):
        from code import evaluate

        dataset = mock.Mock()
        dataset.examples = [
            {"src": [10, 2], "src_tokens": ["one"], "tgt_tokens": ["一"]},
            {"src": [11, 2], "src_tokens": ["two"], "tgt_tokens": ["二"]},
            {"src": [12, 2], "src_tokens": ["three"], "tgt_tokens": ["三"]},
        ]

        calls = []

        def fake_decode_sequence(*args, **kwargs):
            calls.append(args[1])
            return [4, 2]

        with mock.patch.object(evaluate, "decode_sequence", side_effect=fake_decode_sequence):
            bleu, examples = evaluate.evaluate_bleu_and_examples(
                model=mock.Mock(),
                dataset=dataset,
                int2word_cn={4: "一", 2: "<EOS>"},
                device="cpu",
                max_len=8,
                num_examples=10,
                decode_limit=2,
            )

        self.assertEqual(calls, [[10, 2], [11, 2]])
        self.assertEqual(len(examples), 2)
        self.assertGreaterEqual(bleu, 0.0)


if __name__ == "__main__":
    unittest.main()
