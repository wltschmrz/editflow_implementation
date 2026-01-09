import torch
from data.datasets import SpecialTokens

# op ids
OP_KEEP = 0
OP_SUB  = 1
OP_DEL  = 2
OP_INS  = 3


def make_targets(z0: torch.Tensor, z1: torch.Tensor, tokens: SpecialTokens):
    """
    Create per-position supervision targets from aligned z0,z1.

    For each position:
      - if z0==z1 (including both EPS): KEEP
      - if z0==EPS and z1!=EPS: INS, ins_token=z1
      - if z0!=EPS and z1==EPS: DEL
      - if z0!=EPS and z1!=EPS and z0!=z1: SUB, sub_token=z1

    PAD positions are masked out by caller using z_mask.
    """
    B, L = z0.shape
    op = torch.full((B, L), OP_KEEP, dtype=torch.long, device=z0.device)
    ins_tok = torch.full((B, L), -1, dtype=torch.long, device=z0.device)
    sub_tok = torch.full((B, L), -1, dtype=torch.long, device=z0.device)

    eps0 = tokens.EPS0
    eps1 = tokens.EPS1
    pad = tokens.PAD

    z0_is_eps = (z0 == eps0) | (z0 == eps1)
    z1_is_eps = (z1 == eps0) | (z1 == eps1)
    z0_is_pad = (z0 == pad)

    # INS: z0 is eps, z1 is not eps/pad
    ins = z0_is_eps & (~z1_is_eps) & (~z0_is_pad) & (z1 != pad)
    op[ins] = OP_INS
    ins_tok[ins] = z1[ins]

    # DEL: z0 is not eps/pad, z1 is eps
    dele = (~z0_is_eps) & (~z0_is_pad) & z1_is_eps
    op[dele] = OP_DEL

    # SUB: both not eps/pad and different
    sub = (~z0_is_eps) & (~z1_is_eps) & (~z0_is_pad) & (z1 != pad) & (z0 != z1)
    op[sub] = OP_SUB
    sub_tok[sub] = z1[sub]

    # KEEP: everything else
    return {"op": op, "ins_tok": ins_tok, "sub_tok": sub_tok}
