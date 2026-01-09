import math
from typing import Dict, Optional

import torch


def _piecewise_cosine_shape(t: torch.Tensor, t0: float, t1: float) -> torch.Tensor:
    """
    Returns shape g(t) in [0,1]:
      g(t)=1                    for t<=t0
      g(t)=0.5(1+cos(pi*(t-t0)/(t1-t0))) for t0<t<t1
      g(t)=0                    for t>=t1
    t: (B,) in [0,1]
    """
    t0 = float(t0)
    t1 = float(t1)
    if t1 <= t0:
        # degenerate: step at t0
        return (t <= t0).float()

    g = torch.zeros_like(t)
    g = torch.where(t <= t0, torch.ones_like(t), g)
    mid = (t > t0) & (t < t1)
    phase = (t[mid] - t0) / (t1 - t0)
    g_mid = 0.5 * (1.0 + torch.cos(math.pi * phase))
    g[mid] = g_mid
    # t>=t1 stays 0
    return g


def _auto_scale_gamma_hi(
    rates: torch.Tensor,
    mix_rates: Dict[str, float],
    k_ratio: float,
    reduce: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    rates: (B,L,3) model edit rates (ins,del,sub)
    Returns gamma_hi: (B,1,1) scalar per-sample so that
      Gamma_mix ≈ k_ratio * Gamma_edit,
    where Gamma_edit is the *total* edit rate and Gamma_mix,unit is sum(mix_rates).
    We detach rates upstream to avoid backprop through gamma.
    """
    # total model edit rate per position
    edit_total = rates.sum(dim=-1, keepdim=True)  # (B,L,1)

    if reduce == "mean":
        edit_scalar = edit_total.mean(dim=1, keepdim=True)  # (B,1,1)
    elif reduce == "median":
        edit_scalar = edit_total.median(dim=1, keepdim=True).values
    else:
        raise ValueError(f"Unknown reduce='{reduce}'")

    mix_unit = float(mix_rates.get("ins", 0.0) + mix_rates.get("del", 0.0) + mix_rates.get("sub", 0.0))
    mix_unit = max(mix_unit, eps)

    gamma_hi = (float(k_ratio) * edit_scalar) / mix_unit  # (B,1,1)
    return gamma_hi


def apply_memoryless_mixing(
    out: Dict[str, torch.Tensor],
    t: torch.Tensor,
    dna_vocab: int,
    cfg_mem: Optional[dict],
) -> Dict[str, torch.Tensor]:
    """
    Apply memoryless base generator:
      Q_base(t) = Q_edit(t) + gamma(t) Q_mix

    - out: dict containing
        rates: (B,L,3)  [ins, del, sub] >=0
        q_ins: (B,L,V)
        q_sub: (B,L,V)
      (V can include extra tokens; we only mix first dna_vocab.)
    - t: (B,) in [0,1]
    - dna_vocab: 4 for DNA (ACGT)
    - cfg_mem: config dict under cfg["memoryless"]

    Returns a NEW dict with mixed "rates", "q_ins", "q_sub".
    """
    if (cfg_mem is None) or (not cfg_mem.get("enabled", False)):
        return out

    rates = out["rates"]
    q_ins = out["q_ins"]
    q_sub = out["q_sub"]

    device = rates.device
    B, L, _ = rates.shape

    # ---------- schedule params ----------
    sch = cfg_mem.get("schedule", {})
    t0 = float(sch.get("t0", 0.1))
    t1 = float(sch.get("t1", 0.5))

    auto = cfg_mem.get("auto_scale", {})
    k_ratio = float(auto.get("k_ratio", 10.0))
    reduce = str(auto.get("reduce", "mean"))
    gamma_min = float(auto.get("gamma_min", 0.0))
    gamma_max = float(auto.get("gamma_max", 50.0))

    mix_rates = cfg_mem.get("mix_rates", {"ins": 0.5, "del": 0.5, "sub": 1.0})
    mix_q_mode = str(cfg_mem.get("mix_q", "uniform"))

    # shape in [0,1] (B,)
    gshape = _piecewise_cosine_shape(t, t0=t0, t1=t1).view(B, 1, 1)

    # ---------- gamma_hi auto scaling ----------
    # detach to avoid gradients flowing into gamma
    gamma_hi = _auto_scale_gamma_hi(rates.detach(), mix_rates=mix_rates, k_ratio=k_ratio, reduce=reduce)  # (B,1,1)

    # clamp gamma_hi and apply shape
    gamma_hi = gamma_hi.clamp(min=gamma_min, max=gamma_max)
    gamma = gamma_hi * gshape  # (B,1,1)

    # ---------- mixing rates ----------
    mix_vec = torch.tensor(
        [float(mix_rates.get("ins", 0.0)), float(mix_rates.get("del", 0.0)), float(mix_rates.get("sub", 0.0))],
        device=device,
        dtype=rates.dtype,
    ).view(1, 1, 3)

    rates_mix = gamma * mix_vec  # (B,1,3) -> broadcast
    rates_base = rates + rates_mix  # (B,L,3)

    # ---------- mixing token distributions ----------
    if mix_q_mode == "uniform":
        q_mix = torch.full((B, L, dna_vocab), 1.0 / float(dna_vocab), device=device, dtype=q_ins.dtype)
    else:
        # default fallback
        q_mix = torch.full((B, L, dna_vocab), 1.0 / float(dna_vocab), device=device, dtype=q_ins.dtype)

    # For ins/sub heads: mix distributions weighted by their *total rates*
    lam_ins_model = rates[:, :, 0:1].clamp_min(1e-12)
    lam_sub_model = rates[:, :, 2:3].clamp_min(1e-12)
    lam_ins_mix = rates_mix[:, :, 0:1].expand(B, L, 1)
    lam_sub_mix = rates_mix[:, :, 2:3].expand(B, L, 1)

    denom_ins = (lam_ins_model + lam_ins_mix).clamp_min(1e-12)
    denom_sub = (lam_sub_model + lam_sub_mix).clamp_min(1e-12)

    q_ins_base = (lam_ins_model * q_ins[:, :, :dna_vocab] + lam_ins_mix * q_mix) / denom_ins
    q_sub_base = (lam_sub_model * q_sub[:, :, :dna_vocab] + lam_sub_mix * q_mix) / denom_sub

    # write back
    out2 = dict(out)
    out2["rates"] = rates_base

    q_ins2 = q_ins.clone()
    q_sub2 = q_sub.clone()
    q_ins2[:, :, :dna_vocab] = q_ins_base
    q_sub2[:, :, :dna_vocab] = q_sub_base
    out2["q_ins"] = q_ins2
    out2["q_sub"] = q_sub2

    # Optionally expose gamma for logging/debug
    out2["gamma"] = gamma.squeeze(-1).squeeze(-1)  # (B,)
    out2["gamma_hi"] = gamma_hi.squeeze(-1).squeeze(-1)  # (B,)
    out2["gshape"] = gshape.squeeze(-1).squeeze(-1)  # (B,)
    return out2
