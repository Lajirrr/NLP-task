import math

import torch
from torch import nn

from .config import PAD_ID


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


def generate_square_subsequent_mask(size: int, device):
    return torch.triu(
        torch.ones((size, size), dtype=torch.bool, device=device),
        diagonal=1,
    )


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        pad_id: int = PAD_ID,
        tie_target_embeddings: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_id)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        if hasattr(self.transformer.encoder, "enable_nested_tensor"):
            self.transformer.encoder.enable_nested_tensor = False
        if hasattr(self.transformer.encoder, "use_nested_tensor"):
            self.transformer.encoder.use_nested_tensor = False
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        if tie_target_embeddings:
            self.output_projection.weight = self.tgt_embedding.weight

    def forward(
        self,
        src_ids,
        tgt_input_ids,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        if src_padding_mask is None:
            src_padding_mask = src_ids.eq(self.pad_id)
        if tgt_padding_mask is None:
            tgt_padding_mask = tgt_input_ids.eq(self.pad_id)

        memory = self.encode(src_ids, src_padding_mask)
        return self.decode_from_memory(
            tgt_input_ids,
            memory,
            src_padding_mask=src_padding_mask,
            tgt_padding_mask=tgt_padding_mask,
        )

    def encode(self, src_ids, src_padding_mask=None):
        if src_padding_mask is None:
            src_padding_mask = src_ids.eq(self.pad_id)
        src_emb = self.positional_encoding(self.src_embedding(src_ids) * math.sqrt(self.d_model))
        return self.transformer.encoder(
            src_emb,
            src_key_padding_mask=src_padding_mask,
        )

    def decode_from_memory(
        self,
        tgt_input_ids,
        memory,
        src_padding_mask=None,
        tgt_padding_mask=None,
    ):
        if tgt_padding_mask is None:
            tgt_padding_mask = tgt_input_ids.eq(self.pad_id)
        tgt_mask = generate_square_subsequent_mask(tgt_input_ids.size(1), tgt_input_ids.device)
        tgt_emb = self.positional_encoding(
            self.tgt_embedding(tgt_input_ids) * math.sqrt(self.d_model)
        )
        decoded = self.transformer.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask,
        )
        return self.output_projection(decoded)
