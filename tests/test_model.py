import unittest
import warnings


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class ModelTest(unittest.TestCase):
    def test_transformer_translator_returns_target_vocab_logits(self):
        from code.model import TransformerTranslator

        model = TransformerTranslator(
            src_vocab_size=20,
            tgt_vocab_size=30,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        src_ids = torch.tensor([[4, 5, 2], [4, 2, 0]])
        tgt_ids = torch.tensor([[1, 6], [1, 7]])
        logits = model(src_ids, tgt_ids)
        self.assertEqual(tuple(logits.shape), (2, 2, 30))

    def test_eval_forward_does_not_emit_nested_tensor_warning(self):
        from code.model import TransformerTranslator

        model = TransformerTranslator(
            src_vocab_size=20,
            tgt_vocab_size=30,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        model.eval()
        src_ids = torch.tensor([[4, 5, 2], [4, 2, 0]])
        tgt_ids = torch.tensor([[1, 6], [1, 7]])
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with torch.no_grad():
                model(src_ids, tgt_ids)
        nested_warnings = [
            warning
            for warning in caught
            if "nested tensors is in prototype stage" in str(warning.message)
        ]
        self.assertEqual(nested_warnings, [])

    def test_tie_target_embeddings_shares_output_projection_weight(self):
        from code.model import TransformerTranslator

        model = TransformerTranslator(
            src_vocab_size=20,
            tgt_vocab_size=30,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
            tie_target_embeddings=True,
        )

        self.assertIs(model.output_projection.weight, model.tgt_embedding.weight)


if __name__ == "__main__":
    unittest.main()
