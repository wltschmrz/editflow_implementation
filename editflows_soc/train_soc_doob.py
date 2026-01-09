import argparse
import os
import random
from typing import List, Dict, Tuple, Optional

import yaml
import math
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.distributed as dist

import wandb
wandb.login(key="d0eaf79bc1b9ac8962e21de0a5eebe8c129b2770")  # 보안상 env var 권장

from data.datasets import SpecialTokens
from sampling.z_euler_sampler import _build_padded_batch, _sample_categorical, _normalize_len_tokens
from editflows.memoryless import apply_memoryless_mixing
from editflows.doob_control import apply_doob_value_difference
from models.editflow_transformer import SimpleEditFlowsTransformer
from models.doob_value_net import DoobValueNet
from eval.oracles import OracleWrapper
from eval.metrics import jaspar_spearman_corr, kmer_pearson_corr


def build_tokens(vocab: int) -> SpecialTokens:
    BOS = vocab
    PAD = vocab + 1
    EPS0 = vocab + 2
    EPS1 = vocab + 3
    return SpecialTokens(BOS=BOS, PAD=PAD, EPS0=EPS0, EPS1=EPS1)


def is_ddp_launch() -> bool:
    return ("RANK" in os.environ) and ("WORLD_SIZE" in os.environ) and ("LOCAL_RANK" in os.environ)

def ddp_setup():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    return local_rank

def is_main_process() -> bool:
    return (not dist.is_initialized()) or dist.get_rank() == 0


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def only_main_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def _log_bernoulli(mask: torch.Tensor, p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(eps, 1.0 - eps)
    return mask.float() * torch.log(p) + (1.0 - mask.float()) * torch.log(1.0 - p)


def _gather_token_potential(v_logits: torch.Tensor, x_pad: torch.Tensor, dna_vocab: int) -> torch.Tensor:
    """v_curr = v_logits[..., x_pad] with safe clamp for PAD tokens."""
    x_clamped = x_pad.clamp(0, dna_vocab - 1)
    return v_logits.gather(-1, x_clamped.unsqueeze(-1)).squeeze(-1)


@torch.no_grad()
def _init_x0_list(batch_size: int, init_len: int, dna_vocab: int, device: str) -> List[torch.Tensor]:
    return [torch.randint(0, dna_vocab, (init_len,), device=device, dtype=torch.long) for _ in range(batch_size)]


def rollout_soc_doob(
    base_model,
    doob_net: DoobValueNet,
    tokens: SpecialTokens,
    cfg: dict,
    device: str,
) -> Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, float]]:
    """Roll out trajectories under the controlled CTMC (Euler discretization) and return:
      - xT_list: final sequences (list of 1D LongTensor, DNA tokens)
      - logp_u: (B,) log-prob under controlled policy (differentiable wrt doob_net)
      - logp_base: (B,) log-prob of same trajectory under base policy (no_grad)
      - debug stats (scalars)
    """

    dna_vocab = int(cfg["vocab"])
    mem_cfg = cfg.get("memoryless", {"enabled": False})
    soc_cfg = cfg.get("soc", {})

    B = int(soc_cfg.get("batch_size", 64))
    num_steps = int(soc_cfg.get("num_steps", cfg.get("eval", {}).get("sampler", {}).get("num_steps", 128)))
    temperature = float(soc_cfg.get("temperature", cfg.get("eval", {}).get("sampler", {}).get("temperature", 1.0)))

    min_len = int(soc_cfg.get("min_len", cfg.get("min_len", 1)))
    max_len = int(soc_cfg.get("max_len", cfg.get("max_len", 1001)))
    init_len = int(soc_cfg.get("init_len", cfg.get("eval", {}).get("sampler", {}).get("init_len", 100)))

    x_list = _init_x0_list(B, init_len, dna_vocab, device=device)

    logp_u = torch.zeros((B,), device=device, dtype=torch.float32)
    logp_base = torch.zeros((B,), device=device, dtype=torch.float32)

    # per-sample edit counts (for logging)
    ins_cnt = torch.zeros((B,), device=device, dtype=torch.float32)
    del_cnt = torch.zeros((B,), device=device, dtype=torch.float32)
    sub_cnt = torch.zeros((B,), device=device, dtype=torch.float32)

    # Euler step size in (0,1]
    h = 1.0 / float(max(num_steps - 1, 1))

    # debug accumulators
    Zins_acc = 0.0
    Zsub_acc = 0.0
    deltilt_acc = 0.0
    debug_count = 0

    doob_core = doob_net.module if hasattr(doob_net, "module") else doob_net

    base_model.eval()
    doob_net.train()

    for k in range(num_steps):
        t_scalar = float(k) / float(max(num_steps - 1, 1))
        t = torch.full((B,), t_scalar, device=device, dtype=torch.float32)

        x_pad, x_mask, lengths = _build_padded_batch(x_list=x_list, tokens=tokens, device=device)  # PAD-filled
        Lmax = x_pad.shape[1]
        if Lmax == 0:
            break

        # ---- base model forward (frozen; no_grad) ----
        with torch.no_grad():
            out_base = base_model(x_pad, t, x_mask, return_h=True)
            h_base = out_base["h"]  # (B,L,D) detached
            out_base = {"rates": out_base["rates"], "q_ins": out_base["q_ins"], "q_sub": out_base["q_sub"]}
            out_base = apply_memoryless_mixing(out_base, t=t, dna_vocab=dna_vocab, cfg_mem=mem_cfg)

        # ---- value net -> Doob tilt (grad flows only through doob_net) ----
        v_logits = doob_net(h_base)  # (B,L,V)
        v_curr = _gather_token_potential(v_logits, x_pad, dna_vocab=dna_vocab)  # (B,L)

        # psi length potentials
        L = lengths.clamp(min=1)
        psi_L  = doob_core.psi_len(t, L, max_len=max_len)
        psi_Lp = doob_core.psi_len(t, (L + 1).clamp(max=max_len), max_len=max_len)
        psi_Lm = doob_core.psi_len(t, (L - 1).clamp(min=1), max_len=max_len)
        dpsi_ins = psi_Lp - psi_L   # (B,)
        dpsi_del = psi_Lm - psi_L   # (B,)

        out_u = apply_doob_value_difference(
            base_out=out_base,
            v_logits=v_logits,
            v_curr=v_curr,
            dpsi_ins=dpsi_ins,
            dpsi_del=dpsi_del,
            dna_vocab=dna_vocab,
        )

        # ---- per-sample simultaneous INS/DEL/SUB decisions (match z_euler_sampler) ----
        rates_u = out_u["rates"]                # (B,L,3)
        q_ins_u = out_u["q_ins"]                # (B,L,V)
        q_sub_u = out_u["q_sub"]                # (B,L,V)

        rates_b = out_base["rates"]
        q_ins_b = out_base["q_ins"][..., :dna_vocab]
        q_sub_b = out_base["q_sub"][..., :dna_vocab]

        for i in range(B):
            Li = int(lengths[i].item())
            if Li <= 0:
                continue

            # slice valid region
            lam_ins_u = rates_u[i, :Li, 0]
            lam_del_u = rates_u[i, :Li, 1]
            lam_sub_u = rates_u[i, :Li, 2]
            qins_u_pos = q_ins_u[i, :Li, :]  # (Li,V)
            qsub_u_pos = q_sub_u[i, :Li, :]

            lam_ins_b = rates_b[i, :Li, 0]
            lam_del_b = rates_b[i, :Li, 1]
            lam_sub_b = rates_b[i, :Li, 2]
            qins_b_pos = q_ins_b[i, :Li, :]
            qsub_b_pos = q_sub_b[i, :Li, :]

            # slot mapping 0..Li -> idx 0..Li-1 (clamp)
            slot_to_idx = torch.arange(Li + 1, device=device).clamp(0, Li - 1)
            lam_ins_slot_u = lam_ins_u[slot_to_idx]
            lam_ins_slot_b = lam_ins_b[slot_to_idx]

            qins_slot_u = qins_u_pos[slot_to_idx, :]
            qins_slot_b = qins_b_pos[slot_to_idx, :]

            # INS Bernoulli per slot
            p_ins_u = (h * lam_ins_slot_u).clamp(0.0, 1.0)
            p_ins_b = (h * lam_ins_slot_b).clamp(0.0, 1.0)

            ins_mask = (torch.rand((Li + 1,), device=device) < p_ins_u)
            ins_cnt[i] += float(ins_mask.sum().item())

            # sample inserted tokens where needed
            ins_tok = torch.full((Li + 1,), 0, device=device, dtype=torch.long)
            if ins_mask.any():
                ins_tok[ins_mask] = _sample_categorical(qins_slot_u[ins_mask], temperature=temperature)

            # log-prob INS (u and base for same ins_mask/tok)
            logp_u[i] = logp_u[i] + _log_bernoulli(ins_mask, p_ins_u).sum()
            logp_base[i] = logp_base[i] + _log_bernoulli(ins_mask, p_ins_b).sum()

            if ins_mask.any():
                # token probs
                q_u = qins_slot_u[ins_mask]
                q_b = qins_slot_b[ins_mask]
                tok = ins_tok[ins_mask]
                logp_u[i] = logp_u[i] + torch.log(q_u.gather(-1, tok.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)).sum()
                logp_base[i] = logp_base[i] + torch.log(q_b.gather(-1, tok.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)).sum()

            # DEL/SUB Bernoulli per position
            lam_edit_u = lam_del_u + lam_sub_u
            lam_edit_b = lam_del_b + lam_sub_b

            p_edit_u = (h * lam_edit_u).clamp(0.0, 1.0)
            p_edit_b = (h * lam_edit_b).clamp(0.0, 1.0)

            edit_mask = (torch.rand((Li,), device=device) < p_edit_u)

            logp_u[i] = logp_u[i] + _log_bernoulli(edit_mask, p_edit_u).sum()
            logp_base[i] = logp_base[i] + _log_bernoulli(edit_mask, p_edit_b).sum()

            del_mask = torch.zeros((Li,), device=device, dtype=torch.bool)
            sub_mask = torch.zeros((Li,), device=device, dtype=torch.bool)
            sub_tok = torch.full((Li,), 0, device=device, dtype=torch.long)

            if edit_mask.any():
                # choose DEL vs SUB
                p_del_u = (lam_del_u / lam_edit_u.clamp_min(1e-12)).clamp(0.0, 1.0)
                p_del_b = (lam_del_b / lam_edit_b.clamp_min(1e-12)).clamp(0.0, 1.0)

                del_mask[edit_mask] = (torch.rand((int(edit_mask.sum().item()),), device=device) < p_del_u[edit_mask])
                sub_mask[edit_mask] = ~del_mask[edit_mask]

                del_cnt[i] += float(del_mask.sum().item())
                sub_cnt[i] += float(sub_mask.sum().item())

                # log-prob for DEL/SUB choice
                logp_u[i] = logp_u[i] + (del_mask[edit_mask].float() * torch.log(p_del_u[edit_mask].clamp_min(1e-6)) +
                                         sub_mask[edit_mask].float() * torch.log((1.0 - p_del_u[edit_mask]).clamp_min(1e-6))).sum()
                logp_base[i] = logp_base[i] + (del_mask[edit_mask].float() * torch.log(p_del_b[edit_mask].clamp_min(1e-6)) +
                                               sub_mask[edit_mask].float() * torch.log((1.0 - p_del_b[edit_mask]).clamp_min(1e-6))).sum()

                # sample SUB token
                if sub_mask.any():
                    sub_tok[sub_mask] = _sample_categorical(qsub_u_pos[sub_mask], temperature=temperature)
                    # token log-prob
                    q_u = qsub_u_pos[sub_mask]
                    q_b = qsub_b_pos[sub_mask]
                    tok = sub_tok[sub_mask]
                    logp_u[i] = logp_u[i] + torch.log(q_u.gather(-1, tok.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)).sum()
                    logp_base[i] = logp_base[i] + torch.log(q_b.gather(-1, tok.unsqueeze(-1)).squeeze(-1).clamp_min(1e-12)).sum()

            # ---- apply edits to sequence xi ----
            xi = x_list[i]

            # insertion: insert before position p (slot p) if ins_mask[p] True
            if ins_mask.any():
                outs = []
                for p in range(Li + 1):
                    if ins_mask[p]:
                        outs.append(ins_tok[p:p+1])
                    if p < Li:
                        outs.append(xi[p:p+1])
                xi = torch.cat(outs, dim=0)

            # after insertion, length changes
            # apply deletion/substitution by mapping to original indices (approx like simultaneous in sampler)
            # We'll apply DEL/SUB on the *prefix* of current xi corresponding to original Li positions after insertions.
            # This matches the existing sampler's order (INS first, then DEL/SUB on original positions).
            # To do so, we rebuild a pointer through the constructed sequence.
            if edit_mask.any():
                # locate original tokens within xi after insertions:
                # They appear in order with possible inserted tokens before each original index.
                # We can reconstruct by scanning again.
                new_seq = []
                orig_idx = 0
                pos_in_xi = 0
                while orig_idx < Li and pos_in_xi < xi.numel():
                    # There might be inserted tokens before this original position.
                    # We detect original tokens by counting: every time we pass a slot boundary.
                    # Easier: reconstruct indices during insertion above is heavy; for simplicity, redo insertion with tracking.
                    break

                # robust approach: re-run insertion construction with tracking indices
                if ins_mask.any():
                    outs = []
                    orig_positions = []
                    for p in range(Li + 1):
                        if ins_mask[p]:
                            outs.append(ins_tok[p:p+1])
                        if p < Li:
                            orig_positions.append(sum(o.numel() for o in outs))
                            outs.append(x_list[i][p:p+1])
                    xi2 = torch.cat(outs, dim=0)
                else:
                    xi2 = x_list[i]
                    orig_positions = list(range(Li))

                xi = xi2
                xi_list = xi.tolist()
                # apply DEL/SUB in reverse order to keep indices valid
                for j in reversed(range(Li)):
                    pos = orig_positions[j]
                    if del_mask[j]:
                        if 0 <= pos < len(xi_list):
                            xi_list.pop(pos)
                    elif sub_mask[j]:
                        if 0 <= pos < len(xi_list):
                            xi_list[pos] = int(sub_tok[j].item())
                xi = torch.tensor(xi_list, device=device, dtype=torch.long)

            xi = _normalize_len_tokens(xi, dna_vocab=dna_vocab, min_len=min_len, max_len=max_len)
            x_list[i] = xi

        # debug stats
        with torch.no_grad():
            Zins_acc += float(out_u["Z_ins"].mean().item())
            Zsub_acc += float(out_u["Z_sub"].mean().item())
            deltilt_acc += float(out_u["delta_del"].mean().item())
            debug_count += 1

    debug = {}
    if debug_count > 0:
        debug = {
            "Z_ins_mean": Zins_acc / debug_count,
            "Z_sub_mean": Zsub_acc / debug_count,
            "delta_del_mean": deltilt_acc / debug_count,
        }
    edit_stats = {
        "ins": ins_cnt.detach(),
        "del": del_cnt.detach(),
        "sub": sub_cnt.detach(),
        "total": (ins_cnt + del_cnt + sub_cnt).detach(),
    }
    return x_list, logp_u, logp_base, debug, edit_stats


def tokens_to_strings(x_list: List[torch.Tensor]) -> List[str]:
    lookup = np.array(["A", "C", "G", "T"])
    out = []
    for xi in x_list:
        arr = xi.detach().cpu().numpy()
        arr = arr[(arr >= 0) & (arr < 4)]
        out.append("".join(lookup[arr]))
    return out


def normalize_seq_random_pad(seq: str, L: int = 200) -> str:
    seq = seq.upper()
    curr_len = len(seq)
    if curr_len == L:
        return seq
    if curr_len > L:
        s = (curr_len - L) // 2
        return seq[s : s + L]
    pad_len = L - curr_len
    left_pad = pad_len // 2
    right_pad = pad_len - left_pad
    bases = ["A", "C", "G", "T"]
    prefix = "".join(random.choices(bases, k=left_pad))
    suffix = "".join(random.choices(bases, k=right_pad))
    return prefix + seq + suffix

def get_top01_reference_hepg2_seqs(
    gosai_csv: str,
    L: int = 200,
    seq_col: str = "seq",
    hepg2_col: str = "hepg2",
) -> List[str]:
    """DRAKES에서 3-mer/JASPAR corr 레퍼런스로 쓰는 top 0.1% HepG2 activity 서열."""
    df = pd.read_csv(gosai_csv)
    if seq_col not in df.columns or hepg2_col not in df.columns:
        raise ValueError(f"CSV must contain '{seq_col}' and '{hepg2_col}'. cols={list(df.columns)}")

    seqs = df[seq_col].astype(str).str.upper()
    hepg2 = df[hepg2_col].astype(float).to_numpy()

    valid_mask = seqs.apply(lambda s: all(c in "ACGTN" for c in s)).to_numpy()
    seqs = seqs[valid_mask].tolist()
    hepg2 = hepg2[valid_mask]

    thr = np.quantile(hepg2, 0.999)
    idx = np.where(hepg2 > thr)[0]

    ref = [normalize_seq_random_pad(seqs[i], L=L) for i in idx]
    return ref

@torch.no_grad()
def log_memoryless_diagnostics(
    model,
    cfg: dict,
    tokens: SpecialTokens,
    device: str,
    step: int,
):
    """train_resume.py와 동일한 개념의 memoryless 진단 로깅(JS divergence).

    - 서로 완전히 다른 x0 두 그룹을 만들고
    - memoryless mixing이 적용된 sampler를 t=t1까지만 굴려서
    - 토큰 분포가 얼마나 비슷해지는지(JS divergence) 측정

    해석:
      - memoryless/js_token_t1_bits 가 0에 가까울수록 "초기정보가 빨리 사라짐" (좋음)
      - 0.1 이상이면 mixing이 약하거나 t1이 너무 작을 가능성
    """
    mem = cfg.get("memoryless", {"enabled": False})
    if not mem.get("enabled", False):
        return

    from sampling.z_euler_sampler import z_euler_sample_simultaneous

    t1 = float(mem.get("schedule", {}).get("t1", 0.5))
    sampler_cfg = dict(cfg.get("eval", {}).get("sampler", {}))
    # eval.sampler가 없으면 SOC sampler의 값을 fallback
    soc_cfg = cfg.get("soc", {})
    num_steps_full = int(sampler_cfg.get("num_steps", soc_cfg.get("num_steps", 128)))
    init_len = int(sampler_cfg.get("init_len", soc_cfg.get("init_len", 100)))
    min_len = int(sampler_cfg.get("min_len", cfg.get("min_len", 1)))
    max_len = int(sampler_cfg.get("max_len", cfg.get("max_len", 1001)))
    temperature = float(sampler_cfg.get("temperature", soc_cfg.get("temperature", 1.0)))

    # t1까지만 실행되도록 step 수를 줄임
    num_steps_t1 = int(round(t1 * float(max(num_steps_full - 1, 1)))) + 1
    num_steps_t1 = max(2, min(num_steps_t1, num_steps_full))

    dna_vocab = int(cfg["vocab"])
    n_per_group = 64

    x0_A = [torch.zeros((init_len,), device=device, dtype=torch.long) for _ in range(n_per_group)]  # A...A
    x0_B = [torch.full((init_len,), fill_value=dna_vocab - 1, device=device, dtype=torch.long) for _ in range(n_per_group)]  # T...T

    model.eval()
    seqA = z_euler_sample_simultaneous(
        model=model,
        x0_list=x0_A,
        tokens=tokens,
        dna_vocab=dna_vocab,
        num_steps=num_steps_t1,
        min_len=min_len,
        max_len=max_len,
        temperature=temperature,
        device=device,
        memoryless_cfg=mem,
        return_edit_stats=False,
    )
    seqB = z_euler_sample_simultaneous(
        model=model,
        x0_list=x0_B,
        tokens=tokens,
        dna_vocab=dna_vocab,
        num_steps=num_steps_t1,
        min_len=min_len,
        max_len=max_len,
        temperature=temperature,
        device=device,
        memoryless_cfg=mem,
        return_edit_stats=False,
    )
    model.train()

    def _hist_and_meanlen(seq_list):
        vocab_size = tokens.vocab_size
        counts = torch.zeros(vocab_size, dtype=torch.float64, device="cpu")
        total_len = 0.0
        n = max(len(seq_list), 1)
        for s in seq_list:
            if isinstance(s, torch.Tensor):
                ids = s.detach().to("cpu").long().flatten()
                total_len += float(ids.numel())
            else:
                ids = torch.tensor(s, dtype=torch.long, device="cpu").flatten()
                total_len += float(ids.numel())
            bc = torch.bincount(ids, minlength=vocab_size).to(torch.float64)
            counts += bc
        denom = counts.sum().clamp_min(1.0)
        p = counts / denom
        mean_len = total_len / n
        return p, mean_len

    def _js_bits(p, q):
        eps = 1e-12
        p = p.clamp_min(eps)
        q = q.clamp_min(eps)
        m = 0.5 * (p + q)
        kl_pm = (p * (p / m).log()).sum()
        kl_qm = (q * (q / m).log()).sum()
        js = 0.5 * (kl_pm + kl_qm)
        return float(js / math.log(2.0))

    pA, lenA = _hist_and_meanlen(seqA)
    pB, lenB = _hist_and_meanlen(seqB)
    js_bits = _js_bits(pA, pB)
    len_gap = float(abs(lenA - lenB))

    if wandb.run is not None:
        wandb.log({
            "memoryless/js_token_t1_bits": js_bits,
            "memoryless/len_gap_t1": len_gap,
        }, step=step)

@torch.no_grad()
def _run_eval(
    step: int,
    base_model,
    doob_net,
    oracle: OracleWrapper,
    tokens: SpecialTokens,
    cfg: dict,
    device: str,
    reference_top01: Optional[List[str]],
    jaspar_meme: Optional[str],
    use_eval_oracle: bool,
    train_oracle_name: str,
):
    """train_resume.py에서 로깅하던 메트릭을 SOC 파인튜닝에서도 동일하게 찍기."""
    # SOC cfg를 eval sampler 설정으로 잠깐 override
    sampler_cfg = dict(cfg.get("eval", {}).get("sampler", {}))
    if len(sampler_cfg) == 0:
        sampler_cfg = {}

    # 임시 cfg 복사
    cfg_eval = copy.deepcopy(cfg)
    cfg_eval.setdefault("soc", {})
    # rollout이 soc cfg를 보므로 eval.sampler -> soc로 반영
    if "num_steps" in sampler_cfg:
        cfg_eval["soc"]["num_steps"] = int(sampler_cfg["num_steps"])
    if "temperature" in sampler_cfg:
        cfg_eval["soc"]["temperature"] = float(sampler_cfg["temperature"])
    if "init_len" in sampler_cfg:
        cfg_eval["soc"]["init_len"] = int(sampler_cfg["init_len"])
    if "min_len" in sampler_cfg:
        cfg_eval["soc"]["min_len"] = int(sampler_cfg["min_len"])
    if "max_len" in sampler_cfg:
        cfg_eval["soc"]["max_len"] = int(sampler_cfg["max_len"])
    cfg_eval["soc"]["batch_size"] = int(cfg_eval.get("eval", {}).get("n_samples", 32))

    xT_list, _, _, _, edit_stats = rollout_soc_doob(
        base_model=base_model,
        doob_net=doob_net,
        tokens=tokens,
        cfg=cfg_eval,
        device=device,
    )

    seqs = tokens_to_strings(xT_list)
    lens = np.array([len(s) for s in seqs], dtype=np.float32)

    # edit 통계(평균)
    ins_mean = float(edit_stats["ins"].double().mean().item()) if edit_stats is not None else float("nan")
    del_mean = float(edit_stats["del"].double().mean().item()) if edit_stats is not None else float("nan")
    sub_mean = float(edit_stats["sub"].double().mean().item()) if edit_stats is not None else float("nan")
    tot_mean = float(edit_stats["total"].double().mean().item()) if edit_stats is not None else float("nan")

    # oracle metric (Pred-Activity)
    model_name = "eval" if use_eval_oracle else train_oracle_name

    seqs200 = [normalize_seq_random_pad(s, L=200) for s in seqs]
    acts = oracle.predict_activity(seqs200, model_name)
    med_act = float(np.median(acts[:, 0])) if acts is not None and len(acts) > 0 else float("nan")

    # ATAC-Acc (optional; DRAKES style = fraction of HepG2 prob > 0.5, index=1)
    atac_acc = float("nan")
    if "atac" in oracle.models:
        atac_probs = oracle.predict_atac(seqs200)
        atac_idx = 1  # DRAKES eval.ipynb uses [:,1] > 0.5 for HepG2
        atac_acc = float((atac_probs[:, atac_idx] > 0.5).mean())

    # Correlations (optional; variable-length "true" metrics)
    k3_corr_val = float("nan")
    jaspar_corr_val = float("nan")
    if reference_top01 is not None and len(reference_top01) > 0:
        try:
            k3_corr_val = float(kmer_pearson_corr(seqs, reference_top01, k=3, scale_mode="kmers"))
        except Exception:
            k3_corr_val = float("nan")
        try:
            jaspar_corr_val = float(jaspar_spearman_corr(seqs, reference_top01, motif_db="jaspar", scale_mode="kmers"))
        except Exception:
            jaspar_corr_val = float("nan")

    # log (same keys as train_resume.py)
    if wandb.run is not None:
        wandb.log({
            "gen_len/raw_mean": float(lens.mean()) if lens.size > 0 else 0.0,
            "gen_len/raw_std": float(lens.std()) if lens.size > 0 else 0.0,
            "eval/edits/ins_mean": ins_mean,
            "eval/edits/del_mean": del_mean,
            "eval/edits/sub_mean": sub_mean,
            "eval/edits/total_mean": tot_mean,
            "eval/Pred-Activity_median": med_act,
            "eval/ATAC-Acc": atac_acc,
            "eval/3mer_Corr": k3_corr_val,
            "eval/JASPAR_Corr": jaspar_corr_val,
            "eval/App-Log-Lik_median": float("nan"),
        }, step=step)

    atac_print = atac_acc * 100.0 if not np.isnan(atac_acc) else float("nan")
    print(
        f"[Eval@{step}] Act:{med_act:.3f} ATAC:{atac_print:.1f}% "
        f"3mer:{k3_corr_val:.3f} JASPAR:{jaspar_corr_val:.3f} | "
        f"edits(ins/del/sub/total)={ins_mean:.2f}/{del_mean:.2f}/{sub_mean:.2f}/{tot_mean:.2f}"
    )


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: str):
    """Resume 시 optimizer state 텐서들이 CPU에 남아있는 경우가 많아서 GPU로 이동."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--base_ckpt", type=str, required=True, help="Base edit-flow checkpoint (.pt) to freeze.")
    parser.add_argument("--oracle_ckpt", type=str, required=True, help="Reward oracle ckpt (.ckpt) (DRAKES grelu).")
    parser.add_argument("--oracle_name", type=str, default="ft", help="Name for loaded oracle model.")
    parser.add_argument("--reward_index", type=int, default=0, help="Which output column to use as reward.")

    # ---- evaluation (same metrics as train_resume.py) ----
    parser.add_argument("--eval_oracle_ckpt", type=str, default=None,
                        help="(Optional) evaluation oracle ckpt (.ckpt). If None, reuse --oracle_ckpt.")
    parser.add_argument("--atac_ckpt", type=str, default=None,
                        help="(Optional) ATAC oracle ckpt (.ckpt) for eval/ATAC-Acc.")
    parser.add_argument("--gosai_csv", type=str, default=None,
                        help="(Optional) gosai_all.csv path for reference top0.1% sequences (3mer/JASPAR corr).")
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--hepg2_col", type=str, default="hepg2")
    parser.add_argument("--jaspar_meme", type=str, default=None,
                        help="(Optional) JASPAR MEME file path for eval/JASPAR_Corr.")

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--outdir", type=str, default="soc_checkpoints")
    parser.add_argument("--save_every", type=int, default=100)

    parser.add_argument("--resume_ckpt", type=str, default=None,
                    help="(Optional) resume SOC Doob checkpoint (.pt). Loads doob_net/optimizer/step/baseline/rng.")
    parser.add_argument("--wandb_resume_id", type=str, default=None,
                    help="(Optional) wandb run id to resume (only used if you want true wandb resume).")

    args = parser.parse_args()

    # -------------------------
    # DDP setup (safe)
    # -------------------------
    ddp = is_ddp_launch()
    if ddp:
        local_rank = ddp_setup()
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank, world = 0, 1

    # -------------------------
    # Seed per rank
    # -------------------------
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # -------------------------
    # Load config / tokens
    # -------------------------
    cfg = yaml.safe_load(open(args.config, "r"))
    vocab = int(cfg["vocab"])
    tokens = build_tokens(vocab)

    # -------------------------
    # Build/load base model (freeze)
    # -------------------------
    base_model = SimpleEditFlowsTransformer(
        vocab_size=tokens.vocab_size,
        dna_vocab=vocab,
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        max_pos=4096,
    ).to(device)

    ckpt = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    missing, unexpected = base_model.load_state_dict(state, strict=False)
    only_main_print(f"[BASE] loaded {args.base_ckpt} missing={len(missing)} unexpected={len(unexpected)}")

    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)

    # -------------------------
    # Doob value net (trainable)
    # -------------------------
    doob_net = DoobValueNet(d_model=int(cfg["model"]["d_model"]), dna_vocab=vocab).to(device)

    # optional DDP for doob_net only
    if ddp:
        doob_net = torch.nn.parallel.DistributedDataParallel(
            doob_net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    soc_cfg = cfg.get("soc", {})
    lr = float(soc_cfg.get("lr", 1e-4))
    num_updates = int(soc_cfg.get("num_updates", 20000))
    beta_kl = float(soc_cfg.get("beta_kl", 0.1))
    ema_baseline = float(soc_cfg.get("baseline_ema", 0.95))
    grad_clip = float(soc_cfg.get("grad_clip", 1.0))

    opt = torch.optim.AdamW(doob_net.parameters(), lr=lr)

    # -------------------------
    # Resume (optional)
    # -------------------------
    baseline = 0.0
    start_step = 1  # default
    if args.resume_ckpt is not None:
        if not os.path.isfile(args.resume_ckpt):
            raise FileNotFoundError(f"resume_ckpt not found: {args.resume_ckpt}")

        resume = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)
        baseline = float(resume.get("baseline", 0.0))

        # ---- load doob_net ----
        doob_state = resume.get("doob_net", None)
        if doob_state is None:
            raise KeyError(f"resume ckpt missing key 'doob_net': {args.resume_ckpt}")

        target = (doob_net.module if ddp else doob_net)
        missing, unexpected = target.load_state_dict(doob_state, strict=False)
        only_main_print(f"[RESUME] doob_net loaded from {args.resume_ckpt} missing={len(missing)} unexpected={len(unexpected)}")

        # ---- load optimizer (optional) ----
        if "opt" in resume and resume["opt"] is not None:
            try:
                opt.load_state_dict(resume["opt"])
                _move_optimizer_state_to_device(opt, device)
                only_main_print("[RESUME] optimizer state loaded")
            except Exception as e:
                only_main_print(f"[RESUME] optimizer load failed (will re-init opt state): {e}")

        # ---- load baseline/step ----
        baseline = float(resume.get("baseline", 0.0))
        ckpt_step = int(resume.get("step", 0))
        start_step = ckpt_step + 1
        only_main_print(f"[RESUME] step={ckpt_step} -> start_step={start_step}, baseline={baseline:.6f}")

        # ---- load RNG states (optional but recommended for exact reproducibility) ----
        rng = resume.get("rng", None)
        if rng is not None:
            try:
                if "python" in rng:
                    random.setstate(rng["python"])
                if "numpy" in rng:
                    np.random.set_state(rng["numpy"])
                if "torch" in rng:
                    torch.set_rng_state(rng["torch"])
                if torch.cuda.is_available() and ("cuda" in rng) and (rng["cuda"] is not None):
                    # cuda rng는 device count에 맞춰 들어있어야 함
                    torch.cuda.set_rng_state_all(rng["cuda"])
                only_main_print("[RESUME] RNG state restored")
            except Exception as e:
                only_main_print(f"[RESUME] RNG restore failed (continuing anyway): {e}")


    # -------------------------
    # Oracles
    # -------------------------
    oracle = OracleWrapper(device=device)

    # training reward oracle (needed on every rank because reward used in loss)
    oracle.load_model(args.oracle_name, args.oracle_ckpt)

    # eval oracle / atac oracle only on rank0 (only used for logging)
    if is_main_process() and (args.eval_oracle_ckpt is not None):
        oracle.load_model("eval", args.eval_oracle_ckpt)
    if is_main_process() and (args.atac_ckpt is not None):
        oracle.load_model("atac", args.atac_ckpt)

    # -------------------------
    # wandb
    # -------------------------
    if is_main_process():
        wandb.init(
            project=cfg.get("wandb", {}).get("project", "editflows-dna-soc-doob"),
            name=cfg.get("wandb", {}).get("name", None),
            config=cfg,
        )
        only_main_print(f"[DDP={ddp}] rank={rank}/{world} device={device}")

    os.makedirs(args.outdir, exist_ok=True)

    # -------------------------
    # reference sequences (optional, rank0 only)
    # -------------------------
    reference_top01: Optional[List[str]] = None
    if is_main_process() and (args.gosai_csv is not None):
        try:
            reference_top01 = get_top01_reference_hepg2_seqs(
                gosai_csv=args.gosai_csv,
                L=200,
                seq_col=args.seq_col,
                hepg2_col=args.hepg2_col,
            )
            only_main_print(f"[REF] Loaded {len(reference_top01)} top0.1% reference seqs from {args.gosai_csv}")
        except Exception as e:
            only_main_print(f"[REF] Failed to load reference seqs: {e}")
            reference_top01 = None

    # -------------------------
    # baseline + logging freqs
    # -------------------------

    memlog_every = int(cfg.get("train", {}).get("log_memoryless_every", 0) or 0)
    eval_every = int(cfg.get("eval", {}).get("every_steps", 0) or 0)
    eval_n_samples = int(cfg.get("eval", {}).get("n_samples", 0) or 0)

    # -------------------------
    # SOC training loop
    # -------------------------
    if start_step > num_updates:
        only_main_print(f"[RESUME] start_step({start_step}) > num_updates({num_updates}). Nothing to do.")
        if is_main_process():
            wandb.finish()
        ddp_cleanup()
        return

    pbar = tqdm(range(start_step, num_updates + 1), disable=not is_main_process())

    for step in pbar:

        # training rollout: DO NOT bypass DDP wrapper
        xT_list, logp_u, logp_b, dbg, edit_stats = rollout_soc_doob(
            base_model=base_model,
            doob_net=(doob_net if ddp else doob_net),
            tokens=tokens,
            cfg=cfg,
            device=device,
        )

        seqs = tokens_to_strings(xT_list)
        seqs200 = [normalize_seq_random_pad(s, L=200) for s in seqs]

        # reward (needs on every rank)
        with torch.no_grad():
            out = oracle.predict_activity(seqs200, model_name=args.oracle_name)  # (B,C)
            r = torch.tensor(out[:, args.reward_index], device=device, dtype=torch.float32)  # (B,)

        # KL proxy per sample: logpi - logp0
        with torch.no_grad():
            kl_proxy = (logp_u.detach() - logp_b.detach())

        # advantage with EMA baseline (baseline scalar is rank-local; we'll log rank-avg anyway)
        r_mean = float(r.mean().item())
        baseline = ema_baseline * baseline + (1.0 - ema_baseline) * r_mean

        adv = r - beta_kl * kl_proxy - float(baseline)

        # REINFORCE loss: maximize E[ adv * logp_u ]
        loss = -(adv.detach() * logp_u).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(doob_net.parameters(), grad_clip)
        opt.step()

        # ---- logging scalars (avg across ranks) ----
        if ddp:
            scalars = torch.tensor([
                loss.detach(),
                r.mean().detach(),
                r.std().detach(),
                kl_proxy.mean().detach(),
            ], device=device)
            dist.all_reduce(scalars, op=dist.ReduceOp.SUM)
            scalars = scalars / float(world)
            loss_v, r_m, r_s, kl_m = [float(x.item()) for x in scalars]
        else:
            loss_v = float(loss.item())
            r_m = float(r.mean().item())
            r_s = float(r.std().item())
            kl_m = float(kl_proxy.mean().item())

        if is_main_process():
            wandb.log({
                # keep key parity with train_resume.py
                "train/loss": loss_v,

                # SOC extras
                "soc/loss": loss_v,
                "soc/reward_mean": r_m,
                "soc/reward_std": r_s,
                "soc/kl_proxy_mean": kl_m,
                "soc/baseline": float(baseline),
                "soc/Z_ins_mean": float(dbg.get("Z_ins_mean", 0.0)),
                "soc/Z_sub_mean": float(dbg.get("Z_sub_mean", 0.0)),
                "soc/delta_del_mean": float(dbg.get("delta_del_mean", 0.0)),
            }, step=step)

            pbar.set_description(f"loss={loss_v:.4f} r={r_m:.4f} kl={kl_m:.4f}")

        # ---- sync ranks around rank0-only heavy blocks (memlog/eval/save) ----
        do_mem  = (memlog_every > 0) and (step % memlog_every == 0)
        do_eval = (eval_every > 0) and (eval_n_samples > 0) and (step % eval_every == 0)
        do_save = (int(args.save_every) > 0) and (step % int(args.save_every) == 0)

        need_barrier = do_mem or do_eval or do_save

        # 1) 무거운 작업 시작 전: 모두 같은 타이밍 맞추기
        if ddp and need_barrier:
            dist.barrier()

        # ---- memoryless diagnostics (rank0 only) ----
        if is_main_process() and do_mem:
            log_memoryless_diagnostics(
                model=base_model, cfg=cfg, tokens=tokens, device=device, step=step
            )

        # ---- periodic evaluation (rank0 only) ----
        if is_main_process() and do_eval:
            _run_eval(
                step=step,
                base_model=base_model,
                doob_net=(doob_net.module if ddp else doob_net),
                oracle=oracle,
                tokens=tokens,
                cfg=cfg,
                device=device,
                reference_top01=reference_top01,
                jaspar_meme=args.jaspar_meme,
                use_eval_oracle=(args.eval_oracle_ckpt is not None),
                train_oracle_name=args.oracle_name,
            )

        # ---- save (rank0 only) ----
        if is_main_process() and do_save:
            save = {
                "doob_net": (doob_net.module.state_dict() if ddp else doob_net.state_dict()),
                "opt": opt.state_dict(),
                "config": cfg,
                "step": step,
                "baseline": baseline,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            }
            path = os.path.join(args.outdir, f"doob_soc_{step:06d}.pt")
            torch.save(save, path)
            only_main_print(f"[SAVE] {path}")

        # 2) 무거운 작업 끝난 후: 다시 모두 같은 타이밍 맞추기
        if ddp and need_barrier:
            dist.barrier()


        # ---- save doob net ----
        if is_main_process() and (step % int(args.save_every) == 0):
            save = {
                "doob_net": (doob_net.module.state_dict() if ddp else doob_net.state_dict()),
                "opt": opt.state_dict(),
                "config": cfg,
                "step": step,
                "baseline": baseline,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                },
            }
            path = os.path.join(args.outdir, f"doob_soc_{step:06d}.pt")
            torch.save(save, path)
            only_main_print(f"[SAVE] {path}")

    # ---- final save ----
    if is_main_process():
        path = os.path.join(args.outdir, "doob_soc_final.pt")
        torch.save({
            "doob_net": (doob_net.module.state_dict() if ddp else doob_net.state_dict()),
            "config": cfg,
            "step": num_updates,
            "baseline": baseline,
        }, path)
        only_main_print(f"[SAVE] {path}")
        wandb.finish()

    ddp_cleanup()

if __name__ == "__main__":
    main()