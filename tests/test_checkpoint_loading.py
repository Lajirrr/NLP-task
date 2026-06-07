import tempfile
import unittest
import warnings
from pathlib import Path


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "torch is not installed")
class CheckpointLoadingTest(unittest.TestCase):
    def test_load_checkpoint_model_does_not_emit_torch_load_future_warning(self):
        from code.config import DATA_DIR, TransformerConfig
        from code.dataset import load_vocabularies
        from code.train import build_model, save_checkpoint
        from code.translate import load_checkpoint_model

        src_vocab, _, tgt_vocab, _ = load_vocabularies(DATA_DIR)
        config = TransformerConfig(
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        model = build_model(config, len(src_vocab), len(tgt_vocab))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "smoke.pt"
            save_checkpoint(checkpoint_path, model, config, epoch=0, valid_loss=0.0)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                load_checkpoint_model(checkpoint_path, DATA_DIR, torch.device("cpu"))

        future_warnings = [
            warning for warning in caught if issubclass(warning.category, FutureWarning)
        ]
        self.assertEqual(future_warnings, [])

    def test_char_checkpoint_restores_target_vocab_metadata(self):
        from code.config import DATA_DIR, TransformerConfig
        from code.dataset import load_vocabularies
        from code.train import build_model, save_checkpoint
        from code.translate import load_checkpoint_model

        src_vocab, _, _, _ = load_vocabularies(DATA_DIR)
        tgt_vocab = {"<PAD>": 0, "<BOS>": 1, "<EOS>": 2, "<UNK>": 3, "他": 4, "。": 5}
        int2word_tgt = {index: token for token, index in tgt_vocab.items()}
        config = TransformerConfig(
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            num_decoder_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )
        model = build_model(config, len(src_vocab), len(tgt_vocab))

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "char.pt"
            save_checkpoint(
                checkpoint_path,
                model,
                config,
                epoch=0,
                valid_loss=0.0,
                target_level="char",
                tgt_vocab=tgt_vocab,
                int2word_tgt=int2word_tgt,
            )
            loaded_model, _, _, loaded_tgt_vocab, loaded_int2word_tgt = load_checkpoint_model(
                checkpoint_path, DATA_DIR, torch.device("cpu")
            )

        self.assertEqual(getattr(loaded_model, "target_level"), "char")
        self.assertEqual(loaded_tgt_vocab, tgt_vocab)
        self.assertEqual(loaded_int2word_tgt[4], "他")

    def test_decode_constraints_can_suppress_unk(self):
        from code.config import UNK_ID
        from code.translate import apply_decode_constraints

        logits = torch.zeros(6)
        logits[UNK_ID] = 10.0
        constrained = apply_decode_constraints(logits.clone(), suppress_unk=True)
        self.assertLess(constrained[UNK_ID].item(), -1e8)

    def test_decode_constraints_can_allow_unk(self):
        from code.config import UNK_ID
        from code.translate import apply_decode_constraints

        logits = torch.zeros(6)
        logits[UNK_ID] = 10.0
        constrained = apply_decode_constraints(logits.clone(), suppress_unk=False)
        self.assertEqual(constrained[UNK_ID].item(), 10.0)

    def test_normalize_beam_score_prefers_stronger_length_normalized_score(self):
        from code.translate import normalize_beam_score

        short_score = normalize_beam_score(sequence_length=2, log_prob=-1.0, length_penalty=0.6)
        long_score = normalize_beam_score(sequence_length=5, log_prob=-2.0, length_penalty=0.6)
        self.assertGreater(short_score, long_score)

    def test_blocked_ngram_tokens_blocks_repeated_trigram_completion(self):
        from code.translate import blocked_ngram_tokens

        blocked = blocked_ngram_tokens([4, 5, 4, 5], no_repeat_ngram_size=3)
        self.assertEqual(blocked, {4})

    def test_decode_constraints_can_block_repeated_ngrams(self):
        from code.translate import apply_decode_constraints

        logits = torch.zeros(8)
        constrained = apply_decode_constraints(
            logits.clone(),
            suppress_unk=False,
            generated_ids=[4, 5, 4, 5],
            no_repeat_ngram_size=3,
        )
        self.assertLess(constrained[4].item(), -1e8)
        self.assertEqual(constrained[6].item(), 0.0)

    def test_beam_decode_reuses_encoded_source_and_batches_beams(self):
        from code.translate import beam_decode

        class FakeModel:
            def __init__(self):
                self.encode_calls = 0
                self.decode_batch_sizes = []

            def encode(self, src, src_padding_mask=None):
                self.encode_calls += 1
                return torch.zeros(src.size(0), src.size(1), 4)

            def decode_from_memory(
                self,
                tgt,
                memory,
                src_padding_mask=None,
                tgt_padding_mask=None,
            ):
                self.decode_batch_sizes.append(tgt.size(0))
                logits = torch.full((tgt.size(0), tgt.size(1), 6), -1e9)
                if tgt.size(1) == 1:
                    logits[:, -1, 4] = 0.0
                    logits[:, -1, 5] = -0.1
                else:
                    logits[:, -1, 2] = 0.0
                return logits

        model = FakeModel()
        prediction = beam_decode(
            model,
            src_ids=[4, 2],
            device=torch.device("cpu"),
            max_len=3,
            beam_size=2,
            length_penalty=0.0,
        )

        self.assertEqual(prediction, [4, 2])
        self.assertEqual(model.encode_calls, 1)
        self.assertEqual(model.decode_batch_sizes, [1, 2])

    def test_expand_memory_for_beams_supports_ensembles(self):
        from code.translate import expand_memory_for_beams

        first = torch.zeros(1, 3, 4)
        second = torch.ones(1, 3, 4)
        expanded = expand_memory_for_beams([first, second], beam_count=5)

        self.assertEqual(len(expanded), 2)
        self.assertEqual(expanded[0].shape, (5, 3, 4))
        self.assertEqual(expanded[1].shape, (5, 3, 4))


if __name__ == "__main__":
    unittest.main()
