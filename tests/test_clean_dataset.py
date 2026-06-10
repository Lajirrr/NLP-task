import tempfile
import unittest
from pathlib import Path


class CleanDatasetTest(unittest.TestCase):
    def test_filter_atat_rows_preserves_retained_bytes_and_order(self):
        from code.clean_dataset import filter_atat_rows

        lines = [
            b"keep one\tbao yi\r\n",
            b"drop re@@ move\tshan chu\n",
            "keep 中文\t保留\n".encode("utf-8"),
            b"drop second@@\tshan\n",
        ]

        kept, removed = filter_atat_rows(lines)

        self.assertEqual(
            kept,
            [
                b"keep one\tbao yi\r\n",
                "keep 中文\t保留\n".encode("utf-8"),
            ],
        )
        self.assertEqual(removed, 2)

    def test_clean_dataset_splits_creates_backup_and_overwrites_active_splits(self):
        from code.clean_dataset import clean_dataset_splits

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            backup_dir = data_dir / "original_with_atat"
            data_dir.mkdir()
            Path(data_dir, "training.txt").write_bytes(b"a@@ b\tc\nkeep\tline\n")
            Path(data_dir, "validation.txt").write_bytes(b"valid\tok\n")
            Path(data_dir, "testing.txt").write_bytes(b"bad@@\trow\nfinal\tok\n")

            stats = clean_dataset_splits(data_dir, backup_dir)

            self.assertEqual(Path(data_dir, "training.txt").read_bytes(), b"keep\tline\n")
            self.assertEqual(Path(data_dir, "validation.txt").read_bytes(), b"valid\tok\n")
            self.assertEqual(Path(data_dir, "testing.txt").read_bytes(), b"final\tok\n")
            self.assertEqual(
                Path(backup_dir, "training.txt").read_bytes(),
                b"a@@ b\tc\nkeep\tline\n",
            )
            self.assertEqual(
                Path(backup_dir, "validation.txt").read_bytes(), b"valid\tok\n"
            )
            self.assertEqual(
                Path(backup_dir, "testing.txt").read_bytes(),
                b"bad@@\trow\nfinal\tok\n",
            )
            self.assertEqual(
                [(item.split_name, item.total, item.removed, item.kept) for item in stats],
                [
                    ("training.txt", 2, 1, 1),
                    ("validation.txt", 1, 0, 1),
                    ("testing.txt", 2, 1, 1),
                ],
            )

    def test_clean_dataset_splits_refuses_existing_backup_directory(self):
        from code.clean_dataset import clean_dataset_splits

        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir) / "data"
            backup_dir = data_dir / "original_with_atat"
            data_dir.mkdir()
            backup_dir.mkdir()
            Path(data_dir, "training.txt").write_text("a\tb\n", encoding="utf-8")
            Path(data_dir, "validation.txt").write_text("a\tb\n", encoding="utf-8")
            Path(data_dir, "testing.txt").write_text("a\tb\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                clean_dataset_splits(data_dir, backup_dir)


if __name__ == "__main__":
    unittest.main()
