import torch
import torch.nn.functional as F
from typing import Dict, Tuple


def _stable_weighted_softmax(q: torch.Tensor, delta: torch.Tensor, dim: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (q_tilt, Z) where q_tilt ∝ q * exp(delta).
    q: prob simplex (>=0, sum=1), delta: same shape.
    Uses max-subtraction for numerical stability.

    Z is the normalizer: Z = sum_a q_a * exp(delta_a).
    """
    # delta_max over vocab
    dmax = delta.max(dim=dim, keepdim=True).values
    w = torch.exp((delta - dmax).clamp(min=-30.0, max=30.0))
    qw = q * w
    Z = qw.sum(dim=dim, keepdim=True).clamp_min(1e-12)
    q_tilt = qw / Z
    # return scalar Z without keepdim
    return q_tilt, Z.squeeze(dim)


def apply_doob_value_difference(
    base_out: Dict[str, torch.Tensor],
    v_logits: torch.Tensor,
    v_curr: torch.Tensor,
    dpsi_ins: torch.Tensor,
    dpsi_del: torch.Tensor,
    dna_vocab: int,
) -> Dict[str, torch.Tensor]:
    """Apply Doob/value-difference tilting to the *rate form* outputs.

    base_out must contain:
      - rates: (B,L,3) [ins, del, sub], nonnegative
      - q_ins: (B,L,V)
      - q_sub: (B,L,V)

    v_logits: (B,L,V) token potential logits phi_i(a)
    v_curr:   (B,L)   phi_i(x_i) for current token x_i (gathered)
    dpsi_ins: (B,)    psi(L+1,t)-psi(L,t)
    dpsi_del: (B,)    psi(L-1,t)-psi(L,t)

    Returns:
      out_u in the same format (rates,q_ins,q_sub) + debug tensors (Z_ins,Z_sub).
    """
    rates = base_out["rates"]
    q_ins = base_out["q_ins"][..., :dna_vocab]
    q_sub = base_out["q_sub"][..., :dna_vocab]

    lam_ins = rates[..., 0]
    lam_del = rates[..., 1]
    lam_sub = rates[..., 2]

    # --- substitution: Q(x->x^i=a) ∝ lam_sub * q_sub(a); tilt by exp(phi(a)-phi(curr)) ---
    delta_sub = v_logits - v_curr.unsqueeze(-1)  # (B,L,V)
    q_sub_u, Z_sub = _stable_weighted_softmax(q_sub, delta_sub, dim=-1)  # Z_sub: (B,L)
    lam_sub_u = lam_sub * Z_sub

    # --- insertion: Q(x->ins(a)) ∝ lam_ins * q_ins(a); tilt by exp(phi(a) + dpsi_ins) ---
    delta_ins = v_logits + dpsi_ins.view(-1, 1, 1)  # (B,L,V)
    q_ins_u, Z_ins = _stable_weighted_softmax(q_ins, delta_ins, dim=-1)
    lam_ins_u = lam_ins * Z_ins

    # --- deletion: tilt by exp(-phi(curr) + dpsi_del) ---
    delta_del = (-v_curr + dpsi_del.view(-1, 1))  # (B,L)
    lam_del_u = lam_del * torch.exp(delta_del.clamp(min=-30.0, max=30.0))

    out = dict(base_out)
    out["rates"] = torch.stack([lam_ins_u, lam_del_u, lam_sub_u], dim=-1)
    out["q_ins"] = q_ins_u
    out["q_sub"] = q_sub_u
    out["Z_ins"] = Z_ins
    out["Z_sub"] = Z_sub
    out["delta_del"] = delta_del
    return out
