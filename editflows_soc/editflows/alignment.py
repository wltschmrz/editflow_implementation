from typing import List, Tuple

import torch

from data.datasets import SpecialTokens


def align_pair_dp(a: List[int], b: List[int], eps: int) -> Tuple[List[int], List[int]]:
    """
    Classic DP alignment (minimum edit distance) producing aligned sequences with eps gaps.
    Cost: match/sub = 0/1, ins/del = 1. This is a reasonable default.
    """
    n, m = len(a), len(b)
    # dp[i][j] = cost to align a[:i], b[:j]
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]  # backtrace

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "del"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "ins"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost_sub = dp[i - 1][j - 1] + (0 if a[i - 1] == b[j - 1] else 1)
            cost_del = dp[i - 1][j] + 1
            cost_ins = dp[i][j - 1] + 1
            best = min(cost_sub, cost_del, cost_ins)
            dp[i][j] = best
            if best == cost_sub:
                bt[i][j] = "sub"
            elif best == cost_del:
                bt[i][j] = "del"
            else:
                bt[i][j] = "ins"

    # backtrace to build aligned sequences
    i, j = n, m
    aa, bb = [], []
    while i > 0 or j > 0:
        move = bt[i][j]
        if move == "sub":
            aa.append(a[i - 1]); bb.append(b[j - 1])
            i -= 1; j -= 1
        elif move == "del":
            aa.append(a[i - 1]); bb.append(eps)
            i -= 1
        elif move == "ins":
            aa.append(eps); bb.append(b[j - 1])
            j -= 1
        else:
            # should not happen
            break
    aa.reverse(); bb.reverse()
    return aa, bb


def align_batch(x0: torch.Tensor, x1: torch.Tensor, x0_len: torch.Tensor, x1_len: torch.Tensor, tokens: SpecialTokens):
    """
    Batch alignment: returns z0,z1 tensors of shape (B, Lzmax) with EPS and PAD.
    z-space contains EPS in internal positions; PAD only for batching.
    """
    B = x0.shape[0]
    z0_list, z1_list = [], []
    for i in range(B):
        a = x0[i, : int(x0_len[i].item())].tolist()
        b = x1[i, : int(x1_len[i].item())].tolist()
        aa, bb = align_pair_dp(a, b, eps=tokens.EPS0)
        z0_list.append(torch.tensor(aa, dtype=torch.long))
        z1_list.append(torch.tensor(bb, dtype=torch.long))

    z_lens = torch.tensor([len(z) for z in z0_list], dtype=torch.long)
    Lmax = int(z_lens.max().item())
    z0 = torch.full((B, Lmax), tokens.PAD, dtype=torch.long)
    z1 = torch.full((B, Lmax), tokens.PAD, dtype=torch.long)
    for i in range(B):
        L = len(z0_list[i])
        z0[i, :L] = z0_list[i]
        z1[i, :L] = z1_list[i]
    z_mask = (z0 != tokens.PAD)
    return {"z0": z0, "z1": z1, "z_len": z_lens, "z_mask": z_mask}
