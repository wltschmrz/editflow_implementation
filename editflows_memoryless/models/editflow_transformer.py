import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: (B,) in [0,1]
        returns: (B, d_model)
        """
        half = self.d_model // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.d_model % 2 == 1:
            emb = torch.cat([emb, torch.zeros((t.shape[0], 1), device=t.device)], dim=1)
        return emb


class SimpleEditFlowsTransformer(nn.Module):
    """
    Paper-aligned heads (Fig.13-style):
      - rates (lambda_ins, lambda_del, lambda_sub): (B,L,3), positive
      - Q_ins over DNA vocab: (B,L,V)
      - Q_sub over DNA vocab: (B,L,V)

    IMPORTANT:
      This model is run on x_t (no EPS tokens). Length L corresponds to current x_t length.
      Variable-length batching uses PAD and an attention mask.
    """
    def __init__(
        self,
        vocab_size: int,
        dna_vocab: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
        max_pos: int = 4096,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.dna_vocab = dna_vocab
        self.d_model = d_model

        self.t_emb = SinusoidalTimeEmbedding(d_model)
        self.t_proj = nn.Linear(d_model, d_model)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

        # Heads
        self.rate_head = nn.Linear(d_model, 3)        # ins, del, sub
        self.ins_head  = nn.Linear(d_model, dna_vocab)
        self.sub_head  = nn.Linear(d_model, dna_vocab)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, x_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x_t: (B, L) tokens (DNA bases + optional specials, but NO EPS should appear here)
        t: (B,) in [0,1]
        x_mask: (B, L) boolean, True for valid (non-PAD)
        """
        B, L = x_t.shape
        device = x_t.device

        pos = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        h = self.token_emb(x_t) + self.pos_emb(pos)

        te = self.t_proj(self.t_emb(t)).unsqueeze(1)  # (B,1,D)
        h = h + te

        src_key_padding_mask = ~x_mask
        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        h = self.norm(h)

        # Positive rates (softplus)
        rates_raw = self.rate_head(h)
        rates = F.softplus(rates_raw)  # (B,L,3) >= 0

        ins_logits = self.ins_head(h)  # (B,L,V)
        sub_logits = self.sub_head(h)  # (B,L,V)

        # Q distributions
        q_ins = F.softmax(ins_logits, dim=-1)
        q_sub = F.softmax(sub_logits, dim=-1)

        return {
            "rates": rates,        # (B,L,3) [ins, del, sub]
            "q_ins": q_ins,        # (B,L,V)
            "q_sub": q_sub,        # (B,L,V)
        }
