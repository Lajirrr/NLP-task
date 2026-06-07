import tempfile
import unittest
from pathlib import Path


class PretrainedUtilsTest(unittest.TestCase):
    def test_detokenize_english_tokens_restores_punctuation_and_contractions(self):
        from code.pretrained_utils import detokenize_english_tokens

        tokens = ["do", "n't", "underestimate", "my", "power", "."]
        self.assertEqual(detokenize_english_tokens(tokens), "don't underestimate my power.")

        tokens = ["she", "'s", "got", "a", "good", "eye", "for", "paintings", "."]
        self.assertEqual(
            detokenize_english_tokens(tokens),
            "she's got a good eye for paintings.",
        )

    def test_chinese_text_to_chars_removes_spaces(self):
        from code.pretrained_utils import chinese_text_to_chars

        self.assertEqual(chinese_text_to_chars("他 是 老师 。"), ["他", "是", "老", "师", "。"])

    def test_load_pretrained_parallel_examples_detokenizes_both_sides(self):
        from code.pretrained_utils import load_pretrained_parallel_examples

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "tiny.txt"
            split_path.write_text(
                "do n't go .\t不要 走 。\n"
                "she 's a student .\t她 是 学生 。\n",
                encoding="utf-8",
            )

            examples = load_pretrained_parallel_examples(split_path)

        self.assertEqual(len(examples), 2)
        self.assertEqual(examples[0]["src_text"], "don't go.")
        self.assertEqual(examples[0]["tgt_text"], "不要走。")
        self.assertEqual(examples[1]["src_text"], "she's a student.")
        self.assertEqual(examples[1]["tgt_chars"], ["她", "是", "学", "生", "。"])

    def test_load_pretrained_parallel_examples_supports_limit(self):
        from code.pretrained_utils import load_pretrained_parallel_examples

        with tempfile.TemporaryDirectory() as tmpdir:
            split_path = Path(tmpdir) / "tiny.txt"
            split_path.write_text(
                "i am here .\t我 在 这里 。\n"
                "you are here .\t你 在 这里 。\n",
                encoding="utf-8",
            )

            examples = load_pretrained_parallel_examples(split_path, limit=1)

        self.assertEqual([example["src_text"] for example in examples], ["i am here."])

    def test_resolve_pretrained_model_prefers_local_base_when_no_model_is_given(self):
        from code.pretrained_utils import (
            DEFAULT_PRETRAINED_MODEL,
            resolve_pretrained_model_name_or_path,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            local_base_dir = Path(tmpdir) / "checkpoints" / "pretrained-opus-base"
            local_base_dir.mkdir(parents=True)

            resolved = resolve_pretrained_model_name_or_path(
                None, local_base_dir=local_base_dir
            )
            explicit = resolve_pretrained_model_name_or_path(
                "custom-model", local_base_dir=local_base_dir
            )
            missing_local = resolve_pretrained_model_name_or_path(
                None, local_base_dir=Path(tmpdir) / "missing"
            )

        self.assertEqual(resolved, str(local_base_dir))
        self.assertEqual(explicit, "custom-model")
        self.assertEqual(missing_local, DEFAULT_PRETRAINED_MODEL)


if __name__ == "__main__":
    unittest.main()
