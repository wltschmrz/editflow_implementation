from typing import List, Tuple

import torch

from .datasets import SpecialTokens


def pad_1d(seqs: List[torch.Tensor], pad_value: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pads a list of 1D LongTensors to (B, Lmax).
    Returns (padded, lengths).
    """
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item()) if len(seqs) > 0 else 0
    out = torch.full((len(seqs), max_len), pad_value, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
    return out, lengths


def collate_variable_length(batch_x0: List[torch.Tensor], batch_x1: List[torch.Tensor], tokens: SpecialTokens):
    """
    Collate for paired sequences (x0, x1) with variable length.
    PAD is only for batching; edit operations are defined in z-space (with EPS tokens).
    """
    x0, x0_len = pad_1d(batch_x0, tokens.PAD)
    x1, x1_len = pad_1d(batch_x1, tokens.PAD)
    x0_mask = (x0 != tokens.PAD)
    x1_mask = (x1 != tokens.PAD)
    return {
        "x0": x0,
        "x1": x1,
        "x0_len": x0_len,
        "x1_len": x1_len,
        "x0_mask": x0_mask,
        "x1_mask": x1_mask,
    }
