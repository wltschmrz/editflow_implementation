# train.py (Updated with DRAKES Standard Evaluation)
import argparse
import os
import random
from typing import List, Optional, Dict

import yaml
import numpy as np
import pandas as pd

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import wandb
wandb.login(key="d0eaf79bc1b9ac8962e21de0a5eebe8c129b2770")

from data.datasets_gosai_csv import GosaiCSVDataset
from data.datasets import TargetDataset, BackgroundDataset, SpecialTokens
from data.coupling import sample_pair
from data.collate import collate_variable_length

from editflows.alignment import align_batch
from editflows.path import sample_zt
from editflows.loss import build_xt_from_zt, editflows_loss_fig13
from models.editflow_transformer import SimpleEditFlowsTransformer

# ---- UPDATED EVAL IMPORTS ----
# 기존의 load_oracles, compute_drakes_metrics 대신 아래 모듈 사용
from eval.oracles import OracleWrapper  # 새로 만든 OracleWrapper
from eval.metrics import calculate_jaspar_corr_biopython, kmer_pearson_corr


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
        # A, C, G, T 중 랜덤 선택
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

    # Valid한 DNA만 필터링
    valid_mask = seqs.apply(lambda s: all(c in "ACGTN" for c in s)).to_numpy()
    
    seqs = seqs[valid_mask].tolist()
    hepg2 = hepg2[valid_mask]

    # Top 0.1% Threshold
    thr = np.quantile(hepg2, 0.999)
    idx = np.where(hepg2 > thr)[0]

    # Reference 서열도 200bp로 정규화해서 반환
    ref = [normalize_seq_random_pad(seqs[i], L=L) for i in idx]
    return ref


@torch.no_grad()
def sample_for_eval(model, cfg: dict, tokens: SpecialTokens, device: str, n_samples: int) -> List[str]:
    from sampling.z_euler_sampler import sample_sequences_zspace

    model.eval()
    seqs = sample_sequences_zspace(
        model=model,
        tokens=tokens,
        device=device,
        n_samples=n_samples,
        sampler_cfg=cfg["eval"]["sampler"],
    )
    # Convert token IDs to Strings
    if len(seqs) > 0 and not isinstance(seqs[0], str):
        lookup = np.array(["A", "C", "G", "T"])
        out = []
        for arr in seqs:
            arr = np.asarray(arr)
            # 0~3 이외의 값(special tokens)이 섞여있다면 걸러내거나 A로 치환
            # 여기서는 clip으로 안전장치
            valid_mask = (arr >= 0) & (arr < 4)
            arr_valid = arr[valid_mask]
            out.append("".join(lookup[arr_valid]))
        seqs = out
    model.train()
    return seqs


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

    # ---- wandb & Oracle Init (rank0 only) ----
    oracle_wrapper = None
    reference_top01 = None
    
    if is_main_process():
        wandb.init(
            project=cfg.get("wandb", {}).get("project", "editflows-dna"),
            name=cfg.get("wandb", {}).get("name", None),
            config=cfg,
        )
        
        only_main_print(f"gosai_csv={args.gosai_csv} seq_col={args.seq_col} expected_len={args.expected_len}")
        only_main_print(f"[DDP={ddp}] rank={rank}/{world} device={device}")

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
    step = 0
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

            if is_main_process() and (step % save_every == 0):
                os.makedirs("checkpoints", exist_ok=True)
                
                ckpt_path = os.path.join("checkpoints", f"editflows_dna_{step:06d}.pt")
                
                state = (model.module.state_dict() if ddp else model.state_dict())
                
                save_dict = {
                    "model": state,
                    "config": cfg,
                    "step": step,
                    "optimizer": opt.state_dict() # 이어하기(Resume)를 위해 옵티마이저도 저장 권장
                }
                
                torch.save(save_dict, ckpt_path)
                print(f"[{step}] Checkpoint saved: {ckpt_path}")

            # ---- log train loss ----
            if is_main_process() and (step % 50 == 0):
                wandb.log({"train/loss": float(loss.item())}, step=step)
                print(f"step={step}/{num_steps} loss={loss.item():.4f}")

            # ---- Evaluation ----
            if is_main_process() and (step % eval_every == 0) and (oracle_wrapper is not None):
                print(f"Running evaluation at step {step}...")
                
                # 1. Sample Sequences
                gen_raw = sample_for_eval(
                    model=model.module if ddp else model,
                    cfg=cfg,
                    tokens=tokens,
                    device=device,
                    n_samples=eval_n_samples,
                )

                # Log Raw Statistics
                lens = np.array([len(s) for s in gen_raw], dtype=np.float32)
                wandb.log({
                    "gen_len/raw_mean": float(lens.mean()) if lens.size > 0 else 0,
                    "gen_len/raw_std": float(lens.std()) if lens.size > 0 else 0,
                }, step=step)

                # 2. Metric A: Pred-Activity (OracleWrapper handles random padding internally)
                # OracleWrapper.predict_* 메소드는 내부에서 _normalize_seq를 호출하여 
                # 짧은 서열은 랜덤 패딩을 붙이고, 긴 서열은 자릅니다.
                acts = oracle_wrapper.predict_activity(gen_raw, "eval")
                med_act = np.median(acts[:, 0]) # HepG2 Median

                # 3. Metric B: ATAC-Acc
                atac_probs = oracle_wrapper.predict_atac(gen_raw)
                # Assuming HepG2 is index 0
                atac_acc = (atac_probs[:, 0] > 0.5).mean() * 100.0

                # 4. Correlation Metrics (Padding Explicitly Needed)
                # Correlation 계산을 위해 생성된 서열들을 200bp로 맞춰줍니다.
                # (OracleWrapper와 동일한 Random Padding 로직 적용)
                gen_padded = [normalize_seq_random_pad(s, L=200) for s in gen_raw]

                # Metric C: 3-mer Correlation
                k3_corr_val = kmer_pearson_corr(gen_padded, reference_top01, k=3)

                # Metric D: JASPAR Correlation (Biopython)
                jaspar_corr_val = calculate_jaspar_corr_biopython(
                    gen_padded, 
                    reference_top01, 
                    args.jaspar_meme
                )

                # Log Metrics
                wandb.log({
                    "eval/Pred-Activity_median": med_act,
                    "eval/ATAC-Acc": atac_acc,
                    "eval/3mer_Corr": k3_corr_val,
                    "eval/JASPAR_Corr": jaspar_corr_val,
                    "eval/App-Log-Lik_median": float("nan") # Baseline needed
                }, step=step)
                
                print(f"[Eval] Act:{med_act:.3f}, ATAC:{atac_acc:.1f}%, 3mer:{k3_corr_val:.3f}, JASPAR:{jaspar_corr_val:.3f}")

            if step >= num_steps:
                break

    # ---- save ----
    if is_main_process():
        os.makedirs("checkpoints", exist_ok=True)
        ckpt_path = os.path.join("checkpoints", "editflows_dna_final.pt")
        state = (model.module.state_dict() if ddp else model.state_dict())
        torch.save({
            "model": state, 
            "config": cfg, 
            "step": step,
            "optimizer": opt.state_dict()
        }, ckpt_path)
        print("Final checkpoint saved:", ckpt_path)
        wandb.finish()

    ddp_cleanup()


if __name__ == "__main__":
    main()