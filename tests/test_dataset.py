import tempfile
import unittest
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None

from code.config import BOS_ID, EOS_ID, PAD_ID, UNK_ID
from code.dataset import (
    TranslationDataset,
    collate_translation_batch,
    encode_tokens,
    parse_parallel_line,
)
from code import dataset as dataset_module


class DatasetTest(unittest.TestCase):
    def test_parse_parallel_line_splits_source_and_target_tokens(self):
        src, tgt = parse_parallel_line("he is a teacher .\t他 是 老师 。")
        self.assertEqual(src, ["he", "is", "a", "teacher", "."])
        self.assertEqual(tgt, ["他", "是", "老师", "。"])

    def test_parse_parallel_line_strips_utf8_bom(self):
        src, tgt = parse_parallel_line("\ufeffhe is a teacher .\t他 是 老师 。")
        self.assertEqual(src, ["he", "is", "a", "teacher", "."])
        self.assertEqual(tgt, ["他", "是", "老师", "。"])

    def test_encode_tokens_maps_unknown_to_unk_and_appends_eos(self):
        vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "he": 4}
        self.assertEqual(encode_tokens(["he", "missing"], vocab, add_eos=True), [4, 3, 2])

    def test_normalize_target_tokens_char_splits_joined_chinese_words(self):
        normalize_target_tokens = getattr(dataset_module, "normalize_target_tokens", None)
        self.assertTrue(callable(normalize_target_tokens))
        self.assertEqual(
            normalize_target_tokens(["他", "是", "老师", "。"], target_level="char"),
            ["他", "是", "老", "师", "。"],
        )

    def test_build_char_vocab_from_splits_keeps_special_ids_and_characters(self):
        build_char_vocab_from_splits = getattr(
            dataset_module, "build_char_vocab_from_splits", None
        )
        self.assertTrue(callable(build_char_vocab_from_splits))
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "training.txt").write_text(
                "he is a teacher .\t他 是 老师 。\n", encoding="utf-8"
            )
            (data_dir / "validation.txt").write_text(
                "tom is a student .\t汤姆 是 学生 。\n", encoding="utf-8"
            )
            (data_dir / "testing.txt").write_text(
                "flowers are beautiful .\t花 很 美丽 。\n", encoding="utf-8"
            )

            word2int, int2word = build_char_vocab_from_splits(data_dir)

        self.assertEqual(word2int["<PAD>"], PAD_ID)
        self.assertEqual(word2int["<BOS>"], BOS_ID)
        self.assertEqual(word2int["<EOS>"], EOS_ID)
        self.assertEqual(word2int["<UNK>"], UNK_ID)
        self.assertIn("老", word2int)
        self.assertIn("姆", word2int)
        self.assertEqual(int2word[word2int["师"]], "师")

    def test_translation_dataset_supports_char_targets(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "training.txt"
            split_path.write_text(
                "he is a teacher .\t他 是 老师 。\n", encoding="utf-8"
            )
            src_vocab = {
                "<PAD>": PAD_ID,
                "<BOS>": BOS_ID,
                "<EOS>": EOS_ID,
                "<UNK>": UNK_ID,
                "he": 4,
                "is": 5,
                "a": 6,
                "teacher": 7,
                ".": 8,
            }
            tgt_vocab = {
                "<PAD>": PAD_ID,
                "<BOS>": BOS_ID,
                "<EOS>": EOS_ID,
                "<UNK>": UNK_ID,
                "他": 4,
                "是": 5,
                "老": 6,
                "师": 7,
                "。": 8,
            }

            dataset = TranslationDataset(
                split_path, src_vocab, tgt_vocab, target_level="char"
            )

        self.assertEqual(dataset[0]["tgt_tokens"], ["他", "是", "老", "师", "。"])
        self.assertEqual(dataset[0]["tgt"], [4, 5, 6, 7, 8])

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_collate_translation_batch_pads_and_builds_decoder_sequences(self):
        batch = [
            {"src": [4, 5, 2], "tgt": [6, 7]},
            {"src": [4, 2], "tgt": [6]},
        ]
        collated = collate_translation_batch(batch)
        self.assertEqual(collated["src_ids"].tolist(), [[4, 5, 2], [4, 2, PAD_ID]])
        self.assertEqual(collated["tgt_input_ids"].tolist(), [[1, 6, 7], [1, 6, PAD_ID]])
        self.assertEqual(collated["tgt_output_ids"].tolist(), [[6, 7, 2], [6, 2, PAD_ID]])
        self.assertEqual(
            collated["src_padding_mask"].tolist(),
            [[False, False, False], [False, False, True]],
        )


if __name__ == "__main__":
    unittest.main()
