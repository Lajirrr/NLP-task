import tempfile
import unittest
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None


class TrainingUtilsTest(unittest.TestCase):
    def test_parse_args_accepts_model_and_optimizer_tuning(self):
        from code.train import parse_args

        args = parse_args(
            [
                "--d-model",
                "384",
                "--nhead",
                "6",
                "--encoder-layers",
                "4",
                "--decoder-layers",
                "4",
                "--dim-feedforward",
                "1024",
                "--dropout",
                "0.2",
                "--tie-target-embeddings",
                "--adam-beta2",
                "0.98",
                "--adam-eps",
                "1e-9",
            ]
        )

        self.assertEqual(args.d_model, 384)
        self.assertEqual(args.nhead, 6)
        self.assertEqual(args.encoder_layers, 4)
        self.assertEqual(args.decoder_layers, 4)
        self.assertEqual(args.dim_feedforward, 1024)
        self.assertEqual(args.dropout, 0.2)
        self.assertTrue(args.tie_target_embeddings)
        self.assertEqual(args.adam_beta2, 0.98)
        self.assertEqual(args.adam_eps, 1e-9)

    def test_noam_learning_rate_warms_up_then_decays(self):
        from code.train import compute_noam_lr

        peak = compute_noam_lr(step=4000, d_model=256, warmup_steps=4000, factor=1.0)
        half_warmup = compute_noam_lr(
            step=2000, d_model=256, warmup_steps=4000, factor=1.0
        )
        after_warmup = compute_noam_lr(
            step=8000, d_model=256, warmup_steps=4000, factor=1.0
        )

        self.assertGreater(peak, half_warmup)
        self.assertGreater(peak, after_warmup)
        self.assertAlmostEqual(half_warmup, peak * 0.5, places=8)

    @unittest.skipIf(torch is None, "torch is not installed")
    def test_prune_best_checkpoints_keeps_lowest_losses_and_deletes_rest(self):
        from code.train import prune_best_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for epoch, loss in [(1, 2.0), (2, 1.5), (3, 1.7)]:
                path = Path(tmpdir) / f"best_epoch_{epoch:02d}.pt"
                path.write_text(str(loss), encoding="utf-8")
                paths.append({"path": path, "valid_loss": loss})

            kept = prune_best_checkpoints(paths, keep=2)

            kept_names = [record["path"].name for record in kept]
            self.assertEqual(kept_names, ["best_epoch_02.pt", "best_epoch_03.pt"])
            self.assertTrue((Path(tmpdir) / "best_epoch_02.pt").exists())
            self.assertTrue((Path(tmpdir) / "best_epoch_03.pt").exists())
            self.assertFalse((Path(tmpdir) / "best_epoch_01.pt").exists())


if __name__ == "__main__":
    unittest.main()
