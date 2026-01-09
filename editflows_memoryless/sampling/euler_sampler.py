import torch
import torch.nn.functional as F
from typing import List

from data.datasets import SpecialTokens


def _sample_ops(op_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Sample operation category for each position.
    op_logits: (B, L, 4)
    returns op: (B, L) in {0,1,2,3}
    """
    probs = F.softmax(op_logits / max(temperature, 1e-6), dim=-1)
    B, L, _ = probs.shape
    op = torch.multinomial(probs.view(B * L, -1), num_samples=1).view(B, L)
    return op


def _sample_tokens(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    logits: (B,L,V)
    returns tok: (B,L)
    """
    probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
    B, L, V = probs.shape
    tok = torch.multinomial(probs.view(B * L, V), 1).view(B, L)
    return tok


def apply_edits_single(seq: List[int], ops: List[int], ins_tok: List[int], sub_tok: List[int], tokens: SpecialTokens, dna_vocab: int, max_len: int):
    """
    Apply INS/DEL/SUB edits to a single sequence (list of ints in DNA vocab).
    ops are per-position over current sequence positions; for variable length,
    we treat ops aligned to current sequence indices.
    This is a simplified sampler: it applies at most one edit per position.
    """
    out = []
    for i, base in enumerate(seq):
        op = ops[i]
        if op == 2:  # DEL
            continue
        elif op == 1:  # SUB
            out.append(int(sub_tok[i]))
        elif op == 3:  # INS
            # insert before current base
            out.append(int(ins_tok[i]))
            out.append(int(base))
        else:  # KEEP
            out.append(int(base))
        if len(out) >= max_len:
            out = out[:max_len]
            break
    return out


@torch.no_grad()
def euler_sample(
    model,
    x0_list: List[torch.Tensor],
    tokens: SpecialTokens,
    dna_vocab: int,
    num_steps: int = 128,
    temperature: float = 1.0,
    max_len: int = 256,
    device: str = "cuda",
):
    """
    Generate sequences starting from variable-length x0_list.
    This uses a simplified CTMJP-like Euler scheme directly in x-space.
    For a closer Edit Flows sampler, you would sample in z-space and remove EPS each step,
    but this is a good starting point and matches the same action heads.

    Returns list of tensors (variable length).
    """
    seqs = [x.clone().tolist() for x in x0_list]
    B = len(seqs)

    for step in range(num_steps):
        t = torch.full((B,), float(step) / max(num_steps - 1, 1), device=device)

        # build padded batch in x-space with PAD for batching
        lengths = torch.tensor([len(s) for s in seqs], device=device, dtype=torch.long)
        Lmax = int(lengths.max().item())
        x = torch.full((B, Lmax), tokens.PAD, device=device, dtype=torch.long)
        mask = torch.zeros((B, Lmax), device=device, dtype=torch.bool)
        for i, s in enumerate(seqs):
            x[i, :len(s)] = torch.tensor(s, device=device, dtype=torch.long)
            mask[i, :len(s)] = True

        # NOTE: model expects z-space tokens; here we treat x-space as z-space without EPS.
        out = model(x, t, mask)
        op = _sample_ops(out["op_logits"], temperature=temperature)
        ins_tok = _sample_tokens(out["ins_logits"], temperature=temperature)
        sub_tok = _sample_tokens(out["sub_logits"], temperature=temperature)

        # apply edits per sequence
        new_seqs = []
        for i in range(B):
            L = len(seqs[i])
            if L == 0:
                new_seqs.append(seqs[i])
                continue
            ops_i = op[i, :L].tolist()
            ins_i = ins_tok[i, :L].tolist()
            sub_i = sub_tok[i, :L].tolist()
            new = apply_edits_single(seqs[i], ops_i, ins_i, sub_i, tokens, dna_vocab=dna_vocab, max_len=max_len)
            new_seqs.append(new)
        seqs = new_seqs

    return [torch.tensor(s, dtype=torch.long) for s in seqs]
