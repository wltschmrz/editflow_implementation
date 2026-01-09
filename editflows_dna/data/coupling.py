# data/coupling.py
import random
from typing import Optional, Tuple

import torch
from torch.utils.data import Dataset

from .datasets import SpecialTokens


def _rand_seq(min_len: int, max_len: int, vocab: int) -> torch.Tensor:
    L = random.randint(min_len, max_len)
    return torch.randint(low=0, high=vocab, size=(L,), dtype=torch.long)


def _crop_or_pad_to_range(
    x: torch.Tensor,
    min_len: int,
    max_len: int,
    vocab: int,
    crop_mode: str = "random_window",
) -> torch.Tensor:
    """
    Make x length fall in [min_len, max_len].
    - If longer than target L: crop (random window by default)
    - If shorter: pad with random bases
    """
    assert min_len <= max_len
    L_target = random.randint(min_len, max_len)

    L = int(x.numel())
    if L == L_target:
        return x
    if L > L_target:
        if crop_mode == "random_window":
            s = random.randint(0, L - L_target)
            return x[s : s + L_target]
        elif crop_mode == "center":
            s = (L - L_target) // 2
            return x[s : s + L_target]
        elif crop_mode == "left":
            return x[:L_target]
        elif crop_mode == "right":
            return x[-L_target:]
        else:
            raise ValueError(f"Unknown crop_mode: {crop_mode}")
    else:
        pad_len = L_target - L
        pad = torch.randint(0, vocab, (pad_len,), dtype=torch.long)
        return torch.cat([x, pad], dim=0)


def sample_pair(
    mode: str,
    x1: torch.Tensor,
    tokens: SpecialTokens,
    vocab: int,
    min_len: int,
    max_len: int,
    background_dataset: Optional[Dataset] = None,
    vary_x1: bool = True,
    x1_crop_mode: str = "random_window",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns (x0, x1) for Edit Flows training.

    mode:
      - random: x0 is random tokens (length in [min_len, max_len])
      - shuffle: x0 is shuffled x1 (same multiset of bases; same length)
      - background: x0 sampled from background_dataset

    Important for variable-length training:
      - If vary_x1=True, we also crop/pad x1 into [min_len, max_len] each time.
        (This is how you get real INS/DEL supervision, even if original data is fixed-length.)
    """
    # 0) make x1 variable-length (optional but recommended)
    if vary_x1:
        x1 = _crop_or_pad_to_range(
            x1,
            min_len=min_len,
            max_len=max_len,
            vocab=vocab,
            crop_mode=x1_crop_mode,
        )

    if mode == "random":
        x0 = _rand_seq(min_len, max_len, vocab=vocab)
        return x0, x1

    if mode == "shuffle":
        x0 = x1.clone()
        perm = torch.randperm(x0.numel())
        x0 = x0[perm]
        return x0, x1

    if mode == "background":
        if background_dataset is None:
            raise ValueError("background_dataset must be provided for mode='background'")
        x0 = background_dataset[random.randrange(len(background_dataset))]
        # background도 길이 범위로 정규화 (안 하면 배치 길이 분포가 튐)
        x0 = _crop_or_pad_to_range(
            x0,
            min_len=min_len,
            max_len=max_len,
            vocab=vocab,
            crop_mode="random_window",
        )
        return x0, x1

    raise ValueError(f"Unknown mode: {mode}")
