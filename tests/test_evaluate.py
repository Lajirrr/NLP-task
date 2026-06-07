import unittest
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
