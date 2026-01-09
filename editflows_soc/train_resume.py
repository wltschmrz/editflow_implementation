# train.py (Updated with Resume + Edit-Count Logging)
import argparse
import os
import random
import math
from typing import List, Optional, Dict, Tuple

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm


import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
wandb.login(key="d0eaf79bc1b9ac8962e21de0a5eebe8c129b2770")  # 보안상 env var 권장

from data.datasets_gosai_csv import GosaiCSVDataset
from data.datasets import TargetDataset, BackgroundDataset, SpecialTokens
from data.coupling import sample_pair
from data.collate import collate_variable_length

from editflows.alignment import align_batch
from editflows.path import sample_zt
from editflows.loss import build_xt_from_zt, editflows_loss_fig13
from editflows.memoryless import apply_memoryless_mixing
from models.editflow_transformer import SimpleEditFlowsTransformer

# ---- UPDATED EVAL IMPORTS ----
from eval.oracles import OracleWrapper
from eval.metrics import jaspar_spearman_corr, kmer_pearson_corr


def build_tokens(vocab: int) -> SpecialTokens:
    BOS = vocab
    PAD = vocab + 1
    EPS0 = vocab + 2
    EPS1 = vocab + 3
    return SpecialTokens(BOS=BOS, PAD=PAD, EPS0=EPS0, EPS1=EPS1)


def is_ddp() -> bool:
    return int(os.environ.get("RANK", "-1")) != -1


def ddp_setup():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank


def ddp_cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return (not is_ddp()) or dist.get_rank() == 0


def only_main_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def collate_as_list(batch):
    return batch  # List[Tensor]


def move_optimizer_to_device(optimizer: torch.optim.Optimizer, device: str):
    """
    After optimizer.load_state_dict, state tensors may remain on CPU.
    Move them onto the right device (important for CUDA training).
    """
    dev = torch.device(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(dev, non_blocking=True)


# ---- Helper: Random Padding for DNA (DRAKES Standard) ----
def normalize_seq_random_pad(seq: str, L: int = 200) -> str:
    """
    짧은 서열은 양옆에 랜덤 DNA를 붙여 200bp로 만들고,
    긴 서열은 중앙을 자릅니다. (Oracle 평가용)
    """
    seq = seq.upper()
    curr_len = len(seq)
    if curr_len == L:
        return seq

    if curr_len > L:
        start = (curr_len - L) // 2
        return seq[start : start + L]
    else:
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
    """
    DRAKES: 3-mer corr uses top 0.1% HepG2 activity sequences as reference.
    """
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
def sample_for_eval(model, cfg: dict, tokens: SpecialTokens, device: str, n_samples: int):
    from sampling.z_euler_sampler import sample_sequences_zspace

    model.eval()
    sampler_cfg = dict(cfg["eval"]["sampler"])
    sampler_cfg["memoryless"] = cfg.get("memoryless", {"enabled": False})
    seqs, edit_stats = sample_sequences_zspace(
        model=model,
        tokens=tokens,
        device=device,
        n_samples=n_samples,
        sampler_cfg=sampler_cfg,
        return_edit_stats=True,  # ✅ 핵심
    )

    # Convert token IDs to Strings
    if len(seqs) > 0 and not isinstance(seqs[0], str):
        lookup = np.array(["A", "C", "G", "T"])
        out = []
        for arr in seqs:
            arr = np.asarray(arr)
            valid_mask = (arr >= 0) & (arr < 4)
            arr_valid = arr[valid_mask]
            out.append("".join(lookup[arr_valid]))
        seqs = out

    model.train()
    return seqs, edit_stats
@torch.no_grad()
def log_memoryless_diagnostics(model, cfg: dict, tokens: SpecialTokens, device: str, step: int):
    from sampling.z_euler_sampler import z_euler_sample_simultaneous
    """
    "memoryless가 됐는지" 빠르게 확인하는 진단 로깅.

    아이디어:
      - 서로 완전히 다른 초기상태 x0 두 그룹(예: AAAAA..., TTTTT...)을 만든 뒤
      - sampler를 t=t1까지만 굴려서 나온 분포가 서로 얼마나 비슷해졌는지 측정

    로깅 지표:
      - memoryless/js_token_t1_bits : 두 그룹의 토큰 히스토그램(JS divergence, bits). 0에 가까울수록 "초기정보가 사라짐".
      - memoryless/len_gap_t1       : 두 그룹의 평균 길이 차이. 0에 가까울수록 좋음.

    해석 가이드:
      - js_token_t1_bits ~ 0.00~0.02  : 매우 잘 섞임(좋음)
      - js_token_t1_bits ~ 0.02~0.05  : 어느 정도 섞이지만 초기의 흔적이 남을 수 있음(γ↑ 또는 t1↑ 고려)
      - js_token_t1_bits > 0.05       : mixing 부족(γ↑ / t0↑ / t1↑ 또는 mix_rates↑ 필요)
    """

    mem = cfg.get("memoryless", {"enabled": False})
    if not mem.get("enabled", False):
        return

    # schedule params
    t1 = float(mem.get("schedule", {}).get("t1", 0.5))

    # sampler cfg
    sampler_cfg = dict(cfg["eval"]["sampler"])
    sampler_cfg["memoryless"] = mem

    dna_vocab = int(cfg["vocab"])
    init_len = int(sampler_cfg.get("init_len", 100))
    min_len = int(sampler_cfg.get("min_len", 1))
    max_len = int(sampler_cfg.get("max_len", 1001))
    temperature = float(sampler_cfg.get("temperature", 1.0))

    num_steps_full = int(sampler_cfg.get("num_steps", 128))
    num_steps_t1 = int(round(t1 * float(max(num_steps_full - 1, 1)))) + 1
    num_steps_t1 = max(2, min(num_steps_t1, num_steps_full))

    # two very different x0 groups
    n_per_group = 64
    x0_A = [torch.zeros((init_len,), device=device, dtype=torch.long) for _ in range(n_per_group)]             # all 'A'
    x0_B = [torch.full((init_len,), fill_value=dna_vocab - 1, device=device, dtype=torch.long) for _ in range(n_per_group)]  # all 'T' if dna_vocab=4

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

    def _hist_and_meanlen(seq_list, dna_vocab: int):
        counts = torch.zeros(dna_vocab, dtype=torch.float64)
        total_len = 0.0
        n = max(len(seq_list), 1)

        for s in seq_list:
            if isinstance(s, torch.Tensor):
                ids = s.detach().to("cpu").long().flatten()
            else:
                ids = torch.tensor(s, dtype=torch.long).flatten()

            # only count valid DNA ids
            ids = ids[(ids >= 0) & (ids < dna_vocab)]
            total_len += float(ids.numel())

            if ids.numel() > 0:
                counts += torch.bincount(ids, minlength=dna_vocab).to(torch.float64)

        p = counts / counts.sum().clamp_min(1.0)
        mean_len = total_len / n
        return p, mean_len

    def _js_div_bits(p, q, eps=1e-12):
        # p,q: torch float64 vectors on CPU
        p = (p + eps); p = p / p.sum()
        q = (q + eps); q = q / q.sum()
        m = 0.5 * (p + q)
        kl_pm = (p * (p / m).log()).sum()
        kl_qm = (q * (q / m).log()).sum()
        js_nat = 0.5 * (kl_pm + kl_qm)
        js_bits = js_nat / math.log(2.0)
        return float(js_bits)

    pA, lenA = _hist_and_meanlen(seqA, dna_vocab)
    pB, lenB = _hist_and_meanlen(seqB, dna_vocab)
    js_bits = _js_div_bits(pA, pB)
    len_gap = float(abs(lenA - lenB))

    # (선택) gamma 로그도 같이: memoryless.py에서 out에 gamma를 넣었다면 sampler가 edit_stats로 반환하도록 확장 가능
    wandb.log({
        "memoryless/js_token_t1_bits": js_bits,
        "memoryless/len_gap_t1": len_gap,
        "memoryless/t1": t1,
        "memoryless/num_steps_t1": num_steps_t1,
    }, step=step)

    model.train()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--gosai_csv", type=str, default="/mnt/data1/intern/jeongchan/enhancer_data/data_and_model/mdlm/gosai_data/processed_data/gosai_all.csv")
    parser.add_argument("--seq_col", type=str, default="seq")
    parser.add_argument("--expected_len", type=int, default=200)
    parser.add_argument("--background_fasta", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    # ---- ORACLE PATHS ----
    parser.add_argument("--gosai_eval_ckpt", type=str, default="/mnt/data1/intern/jeongchan/enhancer_data/data_and_model/mdlm/outputs_gosai/lightning_logs/reward_oracle_eval.ckpt")
    parser.add_argument("--gosai_ft_ckpt", type=str, default=None)
    parser.add_argument("--atac_ckpt", type=str, default="/mnt/data1/intern/jeongchan/enhancer_data/data_and_model/mdlm/gosai_data/binary_atac_cell_lines.ckpt")

    # ---- JASPAR ----
    parser.add_argument("--jaspar_meme", type=str, default="/mnt/data1/intern/jeongchan/enhancer_data/JASPAR2026_CORE_non-redundant_pfms_meme.txt")

    # ---- RESUME ----
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Path to .pt checkpoint to resume")
    parser.add_argument("--resume_strict", action="store_true", help="strict load_state_dict")

    args = parser.parse_args()

    ddp = is_ddp()
    if ddp:
        local_rank = ddp_setup()
        device = f"cuda:{local_rank}"
        rank = dist.get_rank()
        world = dist.get_world_size()
    else:
        local_rank = 0
        device = "cuda" if torch.cuda.is_available() else "cpu"
        rank, world = 0, 1

    # ---- seed ----
    seed = args.seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ---- config ----
    cfg = yaml.safe_load(open(args.config, "r"))
    vocab = int(cfg["vocab"])
    tokens = build_tokens(vocab)

    base_modes: List[str] = cfg["base_mode"]
    min_len = int(cfg["min_len"])
    max_len = int(cfg["max_len"])

    # ---- datasets ----
    target = GosaiCSVDataset(
        csv_path=args.gosai_csv,
        seq_col=args.seq_col,
        expected_len=args.expected_len,
        return_labels=False,
        drop_ambiguous=True,
    )
    bg = BackgroundDataset(args.background_fasta) if args.background_fasta else None

    sampler = None
    if ddp:
        sampler = DistributedSampler(target, shuffle=True, drop_last=True)

    batch_size = int(cfg["train"]["batch_size"])
    loader = DataLoader(
        target,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_as_list,
    )

    # ---- model ----
    model = SimpleEditFlowsTransformer(
        vocab_size=tokens.vocab_size,
        dna_vocab=vocab,
        d_model=int(cfg["model"]["d_model"]),
        n_heads=int(cfg["model"]["n_heads"]),
        n_layers=int(cfg["model"]["n_layers"]),
        dropout=float(cfg["model"]["dropout"]),
        max_pos=4096,
    ).to(device)

    if ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False
        )

    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg["train"]["lr"]))
    grad_clip = float(cfg["train"]["grad_clip"])
    num_steps = int(cfg["train"]["num_steps"])

    # ---- RESUME LOAD (model/opt/step) ----
    step = 0  # default
    if args.resume_ckpt is not None:
        ckpt = torch.load(args.resume_ckpt, map_location="cpu", weights_only=False)

        load_state = ckpt.get("model", None)
        if load_state is None:
            raise ValueError(f"[RESUME] checkpoint missing 'model' key: {args.resume_ckpt}")

        if ddp:
            missing, unexpected = model.module.load_state_dict(load_state, strict=args.resume_strict)
        else:
            missing, unexpected = model.load_state_dict(load_state, strict=args.resume_strict)

        only_main_print(f"[RESUME] loaded model from {args.resume_ckpt}")
        only_main_print(f"[RESUME] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")

        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            opt.load_state_dict(ckpt["optimizer"])
            move_optimizer_to_device(opt, device=device)
            only_main_print("[RESUME] optimizer state loaded + moved to device")

        if "step" in ckpt:
            step = int(ckpt["step"])
            only_main_print(f"[RESUME] starting step={step}")

        # RNG restore는 DDP에서 rank별로 다르기 때문에 여기서는 비-DDP만 복원(안전)
        if (not ddp) and ("rng" in ckpt) and (ckpt["rng"] is not None):
            rng = ckpt["rng"]
            try:
                random.setstate(rng["python"])
                np.random.set_state(rng["numpy"])
                torch.set_rng_state(rng["torch_cpu"])
                if torch.cuda.is_available() and rng.get("torch_cuda", None) is not None:
                    torch.cuda.set_rng_state_all(rng["torch_cuda"])
                only_main_print("[RESUME] RNG states restored (non-DDP)")
            except Exception as e:
                only_main_print(f"[RESUME] RNG restore failed: {e}")

    # ---- wandb & Oracle Init (rank0 only) ----
    oracle_wrapper = None
    reference_top01 = None

    if is_main_process():
        wandb.init(
            project=cfg.get("wandb", {}).get("project", "editflows-dna-memoryless"),
            name=cfg.get("wandb", {}).get("name", None),
            config=cfg,
        )

        only_main_print(f"gosai_csv={args.gosai_csv} seq_col={args.seq_col} expected_len={args.expected_len}")
        only_main_print(f"[DDP={ddp}] rank={rank}/{world} device={device}")
        if args.resume_ckpt is not None:
            only_main_print(f"[RESUME] wandb run is new by default. (원하면 run id 저장/복원도 추가 가능)")

        # 1. Oracle Wrapper Load
        if args.gosai_eval_ckpt or args.atac_ckpt:
            oracle_wrapper = OracleWrapper(device=device)
            if args.gosai_eval_ckpt:
                oracle_wrapper.load_model("eval", args.gosai_eval_ckpt)
            if args.atac_ckpt:
                oracle_wrapper.load_model("atac", args.atac_ckpt)

        # 2. Reference Sequences Load
        reference_top01 = get_top01_reference_hepg2_seqs(
            gosai_csv=args.gosai_csv,
            L=200,
            seq_col=args.seq_col,
            hepg2_col="hepg2",
        )
        only_main_print(f"Loaded {len(reference_top01)} reference sequences (Top 0.1% Activity)")

    eval_every = int(cfg.get("eval", {}).get("every_steps", 500))
    eval_n_samples = int(cfg.get("eval", {}).get("n_samples", 512))

    # ---- Training Loop ----
    save_every = int(cfg["train"].get("save_every", 10000))
    model.train()
    loader_len = max(len(loader), 1)

    while step < num_steps:
        if ddp:
            epoch = step // loader_len
            sampler.set_epoch(epoch)

        for batch_x1 in loader:
            pairs = []
            for x1 in batch_x1:
                mode = random.choice(base_modes)
                x0, x1_ = sample_pair(
                    mode=mode,
                    x1=x1,
                    tokens=tokens,
                    vocab=vocab,
                    min_len=min_len,
                    max_len=max_len,
                    background_dataset=bg,
                )
                pairs.append((x0, x1_))

            batch_x0 = [p[0] for p in pairs]
            batch_x1_ = [p[1] for p in pairs]
            batch = collate_variable_length(batch_x0, batch_x1_, tokens)

            x0 = batch["x0"].to(device, non_blocking=True)
            x1 = batch["x1"].to(device, non_blocking=True)
            x0_len = batch["x0_len"].to(device, non_blocking=True)
            x1_len = batch["x1_len"].to(device, non_blocking=True)

            z = align_batch(x0, x1, x0_len, x1_len, tokens=tokens)
            z0 = z["z0"].to(device, non_blocking=True)
            z1 = z["z1"].to(device, non_blocking=True)
            z_mask = z["z_mask"].to(device, non_blocking=True)

            t = torch.rand((z0.shape[0],), device=device)
            zt = sample_zt(z0, z1, t, tokens=tokens)

            x_t, x_mask, _ = build_xt_from_zt(zt, tokens)
            out = model(x_t, t, x_mask)
            # ---- memoryless base: Q_base(t) = Q_edit(t) + gamma(t) Q_mix ----
            out = apply_memoryless_mixing(out, t=t, dna_vocab=vocab, cfg_mem=cfg.get("memoryless"))

            loss = editflows_loss_fig13(
                zt=zt,
                z1=z1,
                z_mask=z_mask,
                model_out=out,
                t=t,
                tokens=tokens,
                dna_vocab=vocab,
            )

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            step += 1

            # ---- save ----
            if is_main_process() and (step % save_every == 0):
                os.makedirs("checkpoints", exist_ok=True)
                ckpt_path = os.path.join("checkpoints", f"editflows_dna_{step:06d}.pt")
                state = (model.module.state_dict() if ddp else model.state_dict())

                save_dict = {
                    "model": state,
                    "config": cfg,
                    "step": step,
                    "optimizer": opt.state_dict(),
                    # RNG는 DDP에서 rank별로 달라서 rank0만 저장하는 형태(비-DDP 재현에 유용)
                    "rng": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    },
                }

                torch.save(save_dict, ckpt_path)
                print(f"[{step}] Checkpoint saved: {ckpt_path}")

            # ---- log train loss ----
            if is_main_process() and (step % 50 == 0):
                wandb.log({"train/loss": float(loss.item())}, step=step)
                print(f"step={step}/{num_steps} loss={loss.item():.4f}")

            # ---- memoryless diagnostics ----  ✅ step%50과 독립
            memlog_every = int(cfg.get("train", {}).get("log_memoryless_every", 0) or 0)
            if is_main_process() and memlog_every > 0 and (step % memlog_every == 0):
                log_memoryless_diagnostics(
                    model=model.module if ddp else model,
                    cfg=cfg,
                    tokens=tokens,
                    device=device,
                    step=step,
                )

            # ---- Evaluation ----
            if is_main_process() and (step % eval_every == 0) and (oracle_wrapper is not None):
                print(f"Running evaluation at step {step}...")

                # 1. Sample Sequences (+ edit stats)
                gen_raw, edit_stats = sample_for_eval(
                    model=model.module if ddp else model,
                    cfg=cfg,
                    tokens=tokens,
                    device=device,
                    n_samples=eval_n_samples,
                )

                # Log Raw Statistics: length
                lens = np.array([len(s) for s in gen_raw], dtype=np.float32)

                # edit 평균 (per-sample counts 평균)
                ins_mean = float(edit_stats["ins"].double().mean().item()) if edit_stats is not None else float("nan")
                del_mean = float(edit_stats["del"].double().mean().item()) if edit_stats is not None else float("nan")
                sub_mean = float(edit_stats["sub"].double().mean().item()) if edit_stats is not None else float("nan")
                tot_mean = float(edit_stats["total"].double().mean().item()) if edit_stats is not None else float("nan")

                wandb.log({
                    "gen_len/raw_mean": float(lens.mean()) if lens.size > 0 else 0.0,
                    "gen_len/raw_std": float(lens.std()) if lens.size > 0 else 0.0,
                    "eval/edits/ins_mean": ins_mean,
                    "eval/edits/del_mean": del_mean,
                    "eval/edits/sub_mean": sub_mean,
                    "eval/edits/total_mean": tot_mean,
                }, step=step)

                # 2. Metric A: Pred-Activity (DRAKES: median over HepG2 activity)
                # Oracles are trained on 200bp, so only oracle input is normalized to 200.
                gen_for_oracle = [normalize_seq_random_pad(s, L=200) for s in gen_raw]

                acts = oracle_wrapper.predict_activity(gen_for_oracle, "eval")
                med_act = float(np.median(acts[:, 0]))  # HepG2 index=0 (DRAKES)

                # 3. Metric B: ATAC-Acc (DRAKES: fraction of HepG2 ATAC prob > 0.5; index=1)
                atac_probs = oracle_wrapper.predict_atac(gen_for_oracle)
                atac_idx = getattr(args, "atac_hepg2_index", 1)
                atac_acc = float((atac_probs[:, atac_idx] > 0.5).mean())

                # 4. Correlation Metrics (variable-length "true" evaluation; no padding)
                #    - 3-mer Corr: Pearson corr between k-mer count vectors (ref scaled by size ratio)
                k3_corr_val = kmer_pearson_corr(gen_raw, reference_top01, k=3, scale_mode="kmers")

                #    - JASPAR Corr: Spearman corr between motif count vectors (grelu scan_sequences, 'jaspar')
                jaspar_corr_val = jaspar_spearman_corr(gen_raw, reference_top01, motif_db="jaspar", scale_mode="kmers")

                # Log Metrics
                wandb.log({
                    "eval/Pred-Activity_median": float(med_act),
                    "eval/ATAC-Acc": float(atac_acc),
                    "eval/3mer_Corr": float(k3_corr_val),
                    "eval/JASPAR_Corr": float(jaspar_corr_val),
                    "eval/App-Log-Lik_median": float("nan")  # Baseline needed
                }, step=step)

                print(
                    f"[Eval] Act:{med_act:.3f}, ATAC:{atac_acc:.3f}, "
                    f"3mer:{k3_corr_val:.3f}, JASPAR:{jaspar_corr_val:.3f} | "
                    f"edits(ins/del/sub/total)={ins_mean:.2f}/{del_mean:.2f}/{sub_mean:.2f}/{tot_mean:.2f}"
                )

            if step >= num_steps:
                break

    # ---- final save ----
    if is_main_process():
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = os.path.join("checkpoints", "editflows_dna_final.pt")
        state = (model.module.state_dict() if ddp else model.state_dict())
        torch.save({
            "model": state,
            "config": cfg,
            "step": step,
            "optimizer": opt.state_dict(),
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }, ckpt_path)
        print("Final checkpoint saved:", ckpt_path)
        wandb.finish()

    ddp_cleanup()


if __name__ == "__main__":
    main()

'''
torchrun --nproc_per_node=3 train.py --config configs/config.yaml

CUDA_VISIBLE_DEVICES=3,4,5 torchrun --nproc_per_node=3 --master_port=29501 train_resume.py \
  --resume_ckpt /mnt/data1/intern/jeongchan/editflows_memoryless/checkpoints/editflows_dna_005000.pt

'''

