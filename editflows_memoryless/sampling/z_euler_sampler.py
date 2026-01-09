import torch
from typing import List, Dict, Optional, Tuple, Union

from data.datasets import SpecialTokens

from editflows.memoryless import apply_memoryless_mixing


@torch.no_grad()
def init_z_from_x(x: torch.Tensor, tokens: SpecialTokens) -> torch.Tensor:
    """
    x: (L,) DNA tokens
    z: (2L+1,) = [EPS0, x0, EPS0, x1, ..., x_{L-1}, EPS0]
    """
    L = x.numel()
    z = torch.full((2 * L + 1,), tokens.EPS0, dtype=torch.long, device=x.device)
    if L > 0:
        z[1::2] = x
    return z


@torch.no_grad()
def remove_eps(z: torch.Tensor, tokens: SpecialTokens) -> torch.Tensor:
    keep = (z != tokens.EPS0) & (z != tokens.EPS1)
    return z[keep]


def _sample_categorical(probs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    probs: (..., V) (not necessarily normalized)
    return: (...,) sampled indices
    """
    p = probs.clamp_min(1e-12)
    logits = torch.log(p) / max(temperature, 1e-6)
    dist = torch.distributions.Categorical(logits=logits)
    return dist.sample()


@torch.no_grad()
def _build_padded_batch(
    x_list: List[torch.Tensor],
    tokens: SpecialTokens,
    device: str,
):
    B = len(x_list)
    lengths = torch.tensor([x.numel() for x in x_list], device=device, dtype=torch.long)
    Lmax = int(lengths.max().item()) if B > 0 else 0
    if Lmax == 0:
        x = torch.empty((B, 0), device=device, dtype=torch.long)
        x_mask = torch.empty((B, 0), device=device, dtype=torch.bool)
        return x, x_mask, lengths

    x = torch.full((B, Lmax), tokens.PAD, device=device, dtype=torch.long)
    x_mask = torch.zeros((B, Lmax), device=device, dtype=torch.bool)
    for i, xi in enumerate(x_list):
        L = xi.numel()
        if L > 0:
            x[i, :L] = xi
            x_mask[i, :L] = True
    return x, x_mask, lengths


@torch.no_grad()
def _normalize_len_tokens(
    x: torch.Tensor,
    dna_vocab: int,
    min_len: int,
    max_len: int,
) -> torch.Tensor:
    """
    Ensure length in [min_len, max_len].
    - if too long: center crop
    - if too short: pad with random bases
    """
    L = x.numel()
    if L > max_len:
        s = (L - max_len) // 2
        x = x[s : s + max_len]
        L = x.numel()
    if L < min_len:
        pad_len = min_len - L
        pad = torch.randint(0, dna_vocab, (pad_len,), device=x.device, dtype=torch.long)
        x = torch.cat([x, pad], dim=0)
    return x


@torch.no_grad()
def z_euler_sample_simultaneous(
    model,
    x0_list: List[torch.Tensor],
    tokens: SpecialTokens,
    dna_vocab: int,
    num_steps: int = 128,
    min_len: int = 1,
    max_len: int = 1001,
    temperature: float = 1.0,
    device: str = "cuda",
    memoryless_cfg: Optional[dict] = None,
    return_edit_stats: bool = False,
) -> Union[List[torch.Tensor], Tuple[List[torch.Tensor], Dict[str, torch.Tensor]]]:
    """
    z-space Euler sampler with *simultaneous* INS/DEL/SUB decisions per step.

    Model outputs (for padded x):
      rates: (B, Lmax, 3)  -> [lambda_ins, lambda_del, lambda_sub]
      q_ins: (B, Lmax, V)  -> token dist for insertion
      q_sub: (B, Lmax, V)  -> token dist for substitution

    Simultaneous update (per sequence i, length L):
      - sample insert mask on L+1 slots (before each token + after last)
      - sample edit mask on L tokens (either delete or substitute)
      - apply all decisions in one pass to construct new x
      - re-wrap to z

    If return_edit_stats=True:
      returns (final_list, stats_dict) where stats_dict has per-sample counts:
        ins, del, sub, total  (all torch.int64 on CPU)
    """
    # init z from x0
    z_states = [init_z_from_x(x.to(device), tokens) for x in x0_list]
    B = len(z_states)
    if B == 0:
        return [] if not return_edit_stats else ([], {"ins": torch.zeros((0,), dtype=torch.long),
                                                     "del": torch.zeros((0,), dtype=torch.long),
                                                     "sub": torch.zeros((0,), dtype=torch.long),
                                                     "total": torch.zeros((0,), dtype=torch.long)})

    h = 1.0 / float(max(num_steps, 1))

    # ---- edit counters (per sample) ----
    ins_counts = torch.zeros((B,), device=device, dtype=torch.long)
    del_counts = torch.zeros((B,), device=device, dtype=torch.long)
    sub_counts = torch.zeros((B,), device=device, dtype=torch.long)

    for step in range(num_steps):
        t_scalar = float(step) / float(max(num_steps - 1, 1))
        t = torch.full((B,), t_scalar, device=device)

        # build x_t from z
        x_list = [remove_eps(z, tokens) for z in z_states]
        x_pad, x_mask, lengths = _build_padded_batch(x_list, tokens, device=device)

        if x_pad.shape[1] == 0:
            # all empty -> return empties
            final = [torch.empty((0,), dtype=torch.long) for _ in range(B)]
            if not return_edit_stats:
                return final
            stats = {
                "ins": torch.zeros((B,), dtype=torch.long),
                "del": torch.zeros((B,), dtype=torch.long),
                "sub": torch.zeros((B,), dtype=torch.long),
                "total": torch.zeros((B,), dtype=torch.long),
            }
            return final, stats

        out = model(x_pad, t, x_mask)
        # ---- memoryless base: Q_base(t) = Q_edit(t) + gamma(t) Q_mix ----
        if memoryless_cfg is not None and memoryless_cfg.get("enabled", False):
            out = apply_memoryless_mixing(out, t=t, dna_vocab=dna_vocab, cfg_mem=memoryless_cfg)

        rates = out["rates"]   # (B,Lmax,3)
        q_ins = out["q_ins"]   # (B,Lmax,V)
        q_sub = out["q_sub"]   # (B,Lmax,V)

        new_x_list: List[torch.Tensor] = []

        for i in range(B):
            L = int(lengths[i].item())
            if L <= 0:
                # if empty, just random init a bit (or keep empty)
                xi_new = torch.randint(0, dna_vocab, (min_len,), device=device, dtype=torch.long)
                new_x_list.append(xi_new)
                continue

            lam = rates[i, :L, :]  # (L,3)
            lam_ins = lam[:, 0]
            lam_del = lam[:, 1]
            lam_sub = lam[:, 2]

            # ---------- insertion slots ----------
            slot_to_idx = torch.arange(L + 1, device=device)
            slot_to_idx = torch.clamp(slot_to_idx, 0, L - 1)

            lam_ins_slot = lam_ins[slot_to_idx]  # (L+1,)
            p_ins = (h * lam_ins_slot).clamp(min=0.0, max=1.0)
            ins_mask = (torch.rand((L + 1,), device=device) < p_ins)  # (L+1,)

            qins_slot = q_ins[i, slot_to_idx, :dna_vocab]  # (L+1, V)
            ins_tok = _sample_categorical(qins_slot, temperature=temperature)  # (L+1,)

            # ---------- delete/substitute on tokens ----------
            p_edit = (h * (lam_del + lam_sub)).clamp(min=0.0, max=1.0)  # (L,)
            edit_mask = (torch.rand((L,), device=device) < p_edit) & (p_edit > 0)

            total = (lam_del + lam_sub).clamp_min(1e-12)
            p_del_given_edit = (lam_del / total).clamp(0.0, 1.0)
            del_mask = edit_mask & (torch.rand((L,), device=device) < p_del_given_edit)
            sub_mask = edit_mask & (~del_mask)

            qsub_tok = q_sub[i, :L, :dna_vocab]  # (L, V)
            sub_tok = _sample_categorical(qsub_tok, temperature=temperature)  # (L,)

            # ---- count edits (THIS STEP) ----
            # insertion counts: number of True slots
            ins_counts[i] += ins_mask.long().sum()
            del_counts[i] += del_mask.long().sum()
            sub_counts[i] += sub_mask.long().sum()

            # ---------- apply simultaneously (construct new x) ----------
            xi = x_pad[i, :L]  # original tokens
            out_tokens = []

            for j in range(L):
                # slot before token j
                if ins_mask[j]:
                    out_tokens.append(ins_tok[j])
                # token j action
                if del_mask[j]:
                    pass  # delete => skip
                elif sub_mask[j]:
                    out_tokens.append(sub_tok[j])
                else:
                    out_tokens.append(xi[j])

            # slot after last token
            if ins_mask[L]:
                out_tokens.append(ins_tok[L])

            if len(out_tokens) == 0:
                xi_new = torch.empty((0,), device=device, dtype=torch.long)
            else:
                xi_new = torch.stack(out_tokens, dim=0).to(device)

            # enforce length bounds
            xi_new = _normalize_len_tokens(
                xi_new,
                dna_vocab=dna_vocab,
                min_len=min_len,
                max_len=max_len,
            )
            new_x_list.append(xi_new)

        # re-wrap x -> z for next step (regenerate EPS slots)
        z_states = [init_z_from_x(x, tokens) for x in new_x_list]

    # final x
    final = [remove_eps(z, tokens).detach().cpu() for z in z_states]

    if not return_edit_stats:
        return final

    stats = {
        "ins": ins_counts.detach().cpu(),
        "del": del_counts.detach().cpu(),
        "sub": sub_counts.detach().cpu(),
        "total": (ins_counts + del_counts + sub_counts).detach().cpu(),
    }
    return final, stats


@torch.no_grad()
def sample_sequences_zspace(
    model,
    tokens: SpecialTokens,
    device: str,
    n_samples: int,
    sampler_cfg: dict,
    return_edit_stats: bool = False,
):
    """
    Create x0_list based on cfg and sample with simultaneous sampler.
    sampler_cfg example:
      {
        "num_steps": 128,
        "min_len": 50,
        "max_len": 200,
        "temperature": 1.0,
        "init_mode": "random",
        "init_len": 50
      }

    If return_edit_stats=True:
      returns (final_list, stats_dict)
    """
    dna_vocab = int(sampler_cfg.get("dna_vocab", 4))
    init_mode = sampler_cfg.get("init_mode", "random")
    init_len = int(sampler_cfg.get("init_len", 50))

    x0_list = []
    for _ in range(n_samples):
        if init_mode == "random":
            x0 = torch.randint(0, dna_vocab, (init_len,), device=device, dtype=torch.long)
        else:
            # fallback: random
            x0 = torch.randint(0, dna_vocab, (init_len,), device=device, dtype=torch.long)
        x0_list.append(x0)

    return z_euler_sample_simultaneous(
        model=model,
        x0_list=x0_list,
        tokens=tokens,
        dna_vocab=dna_vocab,
        num_steps=int(sampler_cfg.get("num_steps", 128)),
        min_len=int(sampler_cfg.get("min_len", 1)),
        max_len=int(sampler_cfg.get("max_len", 1001)),
        temperature=float(sampler_cfg.get("temperature", 1.0)),
        device=device,
        memoryless_cfg=sampler_cfg.get("memoryless", None),
        return_edit_stats=return_edit_stats,
    )
