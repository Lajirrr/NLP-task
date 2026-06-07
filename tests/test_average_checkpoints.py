import tempfile
import unittest
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class AverageCheckpointsTest(unittest.TestCase):
    def test_average_checkpoints_averages_float_tensors_and_preserves_metadata(self):
        from code.average_checkpoints import average_checkpoints

        with tempfile.TemporaryDirectory() as tmpdir:
            first = Path(tmpdir) / "first.pt"
            second = Path(tmpdir) / "second.pt"
            output = Path(tmpdir) / "averaged.pt"
            torch.save(
                {
                    "model_state_dict": {
                        "linear.weight": torch.tensor([[1.0, 3.0]]),
                        "counter": torch.tensor([1], dtype=torch.long),
                    },
                    "model_config": {"d_model": 16},
                    "target_level": "char",
                    "epoch": 3,
                },
                first,
            )
            torch.save(
                {
                    "model_state_dict": {
                        "linear.weight": torch.tensor([[3.0, 5.0]]),
                        "counter": torch.tensor([5], dtype=torch.long),
                    },
                    "model_config": {"d_model": 16},
                    "target_level": "char",
                    "epoch": 4,
                },
                second,
            )

            average_checkpoints([first, second], output)
            averaged = torch.load(output, map_location="cpu", weights_only=True)

        self.assertEqual(averaged["target_level"], "char")
        self.assertEqual(averaged["epoch"], "averaged")
        self.assertEqual(
            averaged["model_state_dict"]["linear.weight"].tolist(), [[2.0, 4.0]]
        )
        self.assertEqual(averaged["model_state_dict"]["counter"].tolist(), [1])
        self.assertEqual(averaged["averaged_from"], [str(first), str(second)])


if __name__ == "__main__":
    unittest.main()
