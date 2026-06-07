from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"

PAD_TOKEN = "<PAD>"
BOS_TOKEN = "<BOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3


@dataclass
class TransformerConfig:
    data_dir: Path = DATA_DIR
    checkpoint_dir: Path = CHECKPOINT_DIR
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dim_feedforward: int = 512
    dropout: float = 0.1
    tie_target_embeddings: bool = False
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 1e-4
    grad_clip: float = 1.0
    max_decode_len: int = 64
