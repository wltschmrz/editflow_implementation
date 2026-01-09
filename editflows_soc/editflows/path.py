import torch
from data.datasets import SpecialTokens


def kappa_linear(t: torch.Tensor) -> torch.Tensor:
    """Linear schedule kappa(t)=t in [0,1]."""
    return t


def sample_zt(z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor, tokens: SpecialTokens) -> torch.Tensor:
    """
    Simple mixture path in z-space:
      z_t = z0 with prob (1-kappa), else z1
    This is the same discrete-mixture path used in discrete flow matching style.

    t: shape (B,) in [0,1]
    """
    B, L = z0.shape
    k = kappa_linear(t).clamp(0.0, 1.0).view(B, 1).expand(B, L)
    u = torch.rand((B, L), device=z0.device)
    zt = torch.where(u < k, z1, z0)

    # keep PAD fixed (batching)
    pad_mask = (z0 == tokens.PAD)
    zt = torch.where(pad_mask, z0, zt)
    return zt


def remove_epsilon(z: torch.Tensor, tokens: SpecialTokens) -> torch.Tensor:
    """
    Remove EPS tokens per sequence; returns ragged list in practice.
    Here we return a list of 1D tensors for convenience.
    """
    out = []
    for i in range(z.shape[0]):
        zi = z[i]
        zi = zi[(zi != tokens.PAD) & (zi != tokens.EPS0) & (zi != tokens.EPS1)]
        out.append(zi)
    return out
