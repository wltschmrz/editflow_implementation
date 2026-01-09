import torch
import torch.nn.functional as F
from typing import Dict

from data.datasets import SpecialTokens


def kappa_linear(t: torch.Tensor) -> torch.Tensor:
    return t


def dkappa_linear(t: torch.Tensor) -> torch.Tensor:
    # derivative of kappa(t)=t is 1
    return torch.ones_like(t)


def remove_eps_per_batch(zt: torch.Tensor, tokens: SpecialTokens) -> torch.Tensor:
    """
    Convert zt (B,Lz) into padded x_t (B,Lxmax) by removing EPS0/EPS1.
    Returns x_t and x_mask.
    """
    xs = []
    for i in range(zt.shape[0]):
        zi = zt[i]
        keep = (zi != tokens.PAD) & (zi != tokens.EPS0) & (zi != tokens.EPS1)
        xs.append(zi[keep])
    lengths = torch.tensor([x.numel() for x in xs], device=zt.device, dtype=torch.long)
    Lmax = int(lengths.max().item()) if len(xs) > 0 else 0
    x = torch.full((len(xs), Lmax), tokens.PAD, device=zt.device, dtype=torch.long)
    x_mask = torch.zeros((len(xs), Lmax), device=zt.device, dtype=torch.bool)
    for i, xi in enumerate(xs):
        L = xi.numel()
        if L > 0:
            x[i, :L] = xi
            x_mask[i, :L] = True
    return x, x_mask, lengths


def editflows_loss_fig13(
    zt: torch.Tensor,
    z1: torch.Tensor,
    z_mask: torch.Tensor,
    model_out: Dict[str, torch.Tensor],
    t: torch.Tensor,
    tokens: SpecialTokens,
    dna_vocab: int,
) -> torch.Tensor:
    """
    Fig.13-style Monte Carlo loss (paper-aligned proxy):

    Loss = sum_i (lambda_ins + lambda_del + lambda_sub)   [rate penalty]
           - w(t) * sum_{i in "remaining edits"} log( lambda_* * Q_* )

    where w(t) = dkappa(t)/(1-kappa(t)).

    We iterate along zt positions and maintain x_t_index that advances only on real tokens.
    This is exactly how the paper's simplified code indexes x_t when scanning z_t. (Fig.13)
    """
    rates = model_out["rates"]   # (B,Lx,3)
    q_ins = model_out["q_ins"]   # (B,Lx,V)
    q_sub = model_out["q_sub"]   # (B,Lx,V)

    B, Lz = zt.shape
    device = zt.device

    k = kappa_linear(t).clamp(0.0, 1.0)          # (B,)
    dk = dkappa_linear(t)                        # (B,)
    w = (dk / (1.0 - k).clamp_min(1e-6)).view(B, 1)  # (B,1)

    # term1: sum of rates over x positions (masked)
    # We assume caller used correct x_mask in model, but we can still clamp with PAD if needed.
    term1 = rates.sum(dim=-1).sum(dim=-1)  # (B,)

    # term2: accumulate log terms for remaining edits
    term2 = torch.zeros((B,), device=device)

    for b in range(B):
        x_idx = 0
        # x_len is inferred from model_out tensor length; we clamp x_idx accordingly
        Lx = rates.shape[1]

        for j in range(Lz):
            if not bool(z_mask[b, j].item()):
                continue

            tok_t = int(zt[b, j].item())
            tok_1 = int(z1[b, j].item())

            # Skip PAD
            if tok_t == tokens.PAD or tok_1 == tokens.PAD:
                continue

            # Insertion remaining: zt is EPS0 slot, z1 wants real token (not EPS1)
            if tok_t == tokens.EPS0 and tok_1 != tokens.EPS1:
                # idx for insertion uses current x_idx (position "before next real token")
                idx = min(x_idx, Lx - 1)
                lam_ins = rates[b, idx, 0].clamp_min(1e-12)
                # tok_1 must be in DNA vocab
                if 0 <= tok_1 < dna_vocab:
                    prob = q_ins[b, idx, tok_1].clamp_min(1e-12)
                    term2[b] += torch.log(lam_ins) + torch.log(prob)

            # Deletion remaining: zt has real token, z1 is EPS1
            elif tok_t != tokens.EPS0 and tok_1 == tokens.EPS1:
                idx = min(x_idx, Lx - 1)
                lam_del = rates[b, idx, 1].clamp_min(1e-12)
                term2[b] += torch.log(lam_del)
                x_idx += 1

            # Substitution remaining: both real tokens but different
            elif tok_t != tokens.EPS0 and tok_1 != tokens.EPS1 and tok_t != tok_1:
                idx = min(x_idx, Lx - 1)
                lam_sub = rates[b, idx, 2].clamp_min(1e-12)
                if 0 <= tok_1 < dna_vocab:
                    prob = q_sub[b, idx, tok_1].clamp_min(1e-12)
                    term2[b] += torch.log(lam_sub) + torch.log(prob)
                x_idx += 1

            # No edit (keep): advance index if tok_t is real token
            else:
                if tok_t != tokens.EPS0 and tok_t != tokens.EPS1:
                    x_idx += 1

    # Final loss (mean over batch)
    loss = (term1 - (w.squeeze(1) * term2)).mean()
    return loss


def build_xt_from_zt(zt: torch.Tensor, tokens: SpecialTokens):
    """Convenience wrapper for train.py."""
    return remove_eps_per_batch(zt, tokens)
