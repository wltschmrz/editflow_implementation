'''
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import pearsonr

from eval.oracles import normalize_len


def count_kmers(seqs: List[str], k: int = 3) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in seqs:
        for i in range(len(s) - k + 1):
            kmer = s[i : i + k]
            counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def kmer_pearson_corr(
    generated: List[str],
    reference: List[str],
    k: int = 3,
) -> float:
    """
    Pearson corr between kmer-count vectors. Matches oracle.py style union-of-keys counting.
    See cal_kmer_corr logic: union set + pearsonr. :contentReference[oaicite:12]{index=12}
    """
    gen_counts = count_kmers(generated, k=k)
    ref_counts = count_kmers(reference, k=k)

    keys = sorted(set(gen_counts.keys()) | set(ref_counts.keys()))
    if len(keys) == 0:
        return float("nan")

    x = np.array([gen_counts.get(key, 0) for key in keys], dtype=np.float64)
    y = np.array([ref_counts.get(key, 0) for key in keys], dtype=np.float64)

    # avoid degenerate corr
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(pearsonr(x, y)[0])


@dataclass
class DRAKESMetricsConfig:
    oracle_seq_len: int = 200
    # In Gosai activity oracle output, index 0 is HepG2 (hepg2,k562,sknsh) :contentReference[oaicite:13]{index=13}
    hepg2_activity_index: int = 0
    # In ATAC oracle output (N,7), we need the HepG2 column index.
    # If you later confirm mapping, set it here. Default=0 (common in some releases but not guaranteed).
    atac_hepg2_index: int = 0
    atac_threshold: float = 0.5


def pred_activity_median(
    seqs: List[str],
    gosai_eval_oracle,
    cfg: DRAKESMetricsConfig,
) -> float:
    seqs = [normalize_len(s, L=cfg.oracle_seq_len) for s in seqs]
    preds = gosai_eval_oracle.predict(seqs, L=cfg.oracle_seq_len)  # (N,3)
    hepg2 = preds[:, cfg.hepg2_activity_index]
    return float(np.median(hepg2))


def atac_acc_percent(
    seqs: List[str],
    atac_oracle,
    cfg: DRAKESMetricsConfig,
) -> float:
    seqs = [normalize_len(s, L=cfg.oracle_seq_len) for s in seqs]
    proba = atac_oracle.predict_proba(seqs, L=cfg.oracle_seq_len)  # (N,7)
    p = proba[:, cfg.atac_hepg2_index]
    acc = (p >= cfg.atac_threshold).mean() * 100.0
    return float(acc)


def jaspar_corr(
    generated: List[str],
    reference: List[str],
    jaspar_meme_path: str,
    cfg: DRAKESMetricsConfig,
) -> float:
    """
    Practical implementation:
    - Load motifs from a MEME-format file (downloaded from JASPAR or prepared offline).
    - Count motif hits per motif in generated & reference.
    - Pearson corr over motif-hit count vectors.
    If jaspar_meme_path is missing, return NaN so you notice.

    DRAKES description: scan generated seqs with JASPAR TF binding profiles and correlate. :contentReference[oaicite:14]{index=14}
    """
    import os
    if jaspar_meme_path is None or (not os.path.exists(jaspar_meme_path)):
        print(f"[Warning] JASPAR path not found: {jaspar_meme_path}")
        return float("nan")    # Minimal, dependency-light motif scanning using gimmemotifs if installed.
    
    try:
        from gimmemotifs.motif import read_motifs
        from gimmemotifs.scanner import Scanner
    except Exception:
        return float("nan")

    generated = [normalize_len(s, L=cfg.oracle_seq_len) for s in generated]
    reference = [normalize_len(s, L=cfg.oracle_seq_len) for s in reference]

    motifs = read_motifs(jaspar_meme_path)
    scanner = Scanner()
    scanner.set_motifs(motifs)

    def scan_counts(seqs: List[str]) -> np.ndarray:
        # counts per motif
        counts = np.zeros((len(motifs),), dtype=np.float64)
        for s in seqs:
            # returns hits per motif; gimmemotifs gives dict with motif ids
            results = scanner.scan(s)
            # results: {motif_id: [(pos, score, strand), ...], ...}
            for mi, m in enumerate(motifs):
                hits = results.get(m.id, [])
                counts[mi] += len(hits)
        return counts

    cg = scan_counts(generated)
    cr = scan_counts(reference)
    if np.std(cg) < 1e-12 or np.std(cr) < 1e-12:
        return 0.0
    return float(pearsonr(cg, cr)[0])


def app_log_lik_median(
    seq_tokens_np: np.ndarray,
    baseline_model,
) -> float:
    """
    DRAKES 'App-Log-Lik (median)' = approximate likelihood under pretrained model (or old model),
    similar to oracle.py cal_avg_likelihood using old_model._forward_pass_diffusion(...). :contentReference[oaicite:15]{index=15}

    Here we keep it generic:
    - baseline_model must expose: forward_pass_diffusion(tokens)->(N,L) log-prob contributions (or similar)
    """
    with np.errstate(all="ignore"):
        ll = baseline_model.forward_pass_diffusion(seq_tokens_np)  # user-defined
    ll_sum = ll.sum(axis=-1)
    return float(np.median(ll_sum))


def compute_drakes_metrics(
    generated_seqs: List[str],
    reference_top01_seqs: List[str],
    gosai_eval_oracle,
    atac_oracle,
    cfg: Optional[DRAKESMetricsConfig] = None,
    jaspar_meme_path: Optional[str] = None,
) -> Dict[str, float]:
    """
    Returns the core DRAKES Table-1 metrics.
    (App-Log-Lik needs pretrained diffusion baseline; not computed here unless you wire it.)
    """
    cfg = cfg or DRAKESMetricsConfig()

    out: Dict[str, float] = {}
    out["Pred-Activity_median"] = pred_activity_median(generated_seqs, gosai_eval_oracle, cfg)
    out["ATAC-Acc_percent"] = atac_acc_percent(generated_seqs, atac_oracle, cfg)
    out["3mer_Corr"] = kmer_pearson_corr(
        [normalize_len(s, cfg.oracle_seq_len) for s in generated_seqs],
        [normalize_len(s, cfg.oracle_seq_len) for s in reference_top01_seqs],
        k=3,
    )
    out["JASPAR_Corr"] = jaspar_corr(
        generated_seqs,
        reference_top01_seqs,
        jaspar_meme_path=jaspar_meme_path,
        cfg=cfg,
    )
    # App-Log-Lik: wire if you have baseline diffusion model.
    out["App-Log-Lik_median"] = float("nan")
    return out

'''

# eval/metrics.py
import numpy as np
from scipy.stats import pearsonr
from typing import List, Dict, Optional
import os
import re

# Biopython (Motif 객체 생성을 위해 필요)
try:
    from Bio import motifs
    from Bio.Seq import Seq
except ImportError:
    print("Please install biopython: pip install biopython")

def parse_meme_text_manually(meme_path: str):
    """
    Biopython의 motifs.parse가 XML 에러를 낼 때 사용하는
    강력한 수동 파서입니다. (MEME version 4 텍스트 포맷용)
    """
    parsed_motifs = []
    
    with open(meme_path, 'r') as f:
        lines = f.readlines()

    current_name = None
    current_matrix = []
    nsites = 20 # Default value if not found
    
    # 정규표현식: 실수 4개가 있는 행 찾기 (0.1 0.9 0.0 0.0)
    matrix_row_pattern = re.compile(r"^\s*(\d*\.?\d+)\s+(\d*\.?\d+)\s+(\d*\.?\d+)\s+(\d*\.?\d+)")

    for line in lines:
        line = line.strip()
        if not line: continue

        # 1. Start of new motif
        if line.startswith("MOTIF"):
            # 이전 모티프 저장
            if current_name and current_matrix:
                m = _create_motif_obj(current_name, current_matrix, nsites)
                if m: parsed_motifs.append(m)
            
            parts = line.split()
            current_name = parts[1] if len(parts) > 1 else "Unknown"
            current_matrix = []
            nsites = 20 # Reset default

        # 2. Header info (nsites 찾기)
        elif line.startswith("letter-probability matrix"):
            # 예: letter-probability matrix: alength= 4 w= 6 nsites= 20 E= 0
            if "nsites=" in line:
                try:
                    nsites_str = line.split("nsites=")[1].split()[0]
                    nsites = float(nsites_str)
                except:
                    nsites = 20
        
        # 3. Matrix rows (A C G T probabilities)
        elif matrix_row_pattern.match(line):
            # 확률값 추출
            probs = [float(x) for x in line.split()]
            if len(probs) == 4:
                current_matrix.append(probs)

    # 마지막 모티프 저장
    if current_name and current_matrix:
        m = _create_motif_obj(current_name, current_matrix, nsites)
        if m: parsed_motifs.append(m)

    return parsed_motifs

def _create_motif_obj(name, matrix_probs, nsites):
    """확률 매트릭스를 Biopython Motif 객체(Counts)로 변환"""
    try:
        # matrix_probs는 list of [pA, pC, pG, pT]
        # Biopython Motif는 Counts를 선호하므로 확률 * nsites로 변환
        counts = {"A": [], "C": [], "G": [], "T": []}
        for row in matrix_probs:
            # row: [pA, pC, pG, pT]
            # nsites를 곱해서 정수로 반올림
            c_row = [round(p * nsites) for p in row]
            counts["A"].append(c_row[0])
            counts["C"].append(c_row[1])
            counts["G"].append(c_row[2])
            counts["T"].append(c_row[3])
        
        m = motifs.Motif(alphabet="ACGT", counts=counts)
        m.name = name
        return m
    except Exception as e:
        print(f"Warning: Failed to create motif {name}: {e}")
        return None

def calculate_jaspar_corr_biopython(
    generated_seqs: List[str], 
    reference_seqs: List[str], 
    meme_path: str
) -> float:
    """
    JASPAR Motif Hit Correlation 계산 (수동 파서 적용됨)
    """
    if not os.path.exists(meme_path):
        print(f"[Warning] MEME file not found: {meme_path}")
        return float("nan")

    # [수정됨] Biopython 내장 파서 대신 수동 파서 사용
    try:
        all_motifs = parse_meme_text_manually(meme_path)
    except Exception as e:
        print(f"[Error] Failed to parse MEME manually: {e}")
        return float("nan")

    if len(all_motifs) == 0:
        print("[Warning] No motifs found in MEME file.")
        return float("nan")

    # --- 여기서부터는 기존 로직과 동일 ---
    def count_hits(seq_list):
        counts = np.zeros(len(all_motifs))
        for seq_str in seq_list:
            # DNA 유효성 검사 및 변환
            # (Biopython Seq 객체는 문자열을 그대로 받음)
            
            # 성능 최적화: Consensus 문자열 카운팅 (DRAKES 근사 방식)
            # PSSM 스캔은 너무 느릴 수 있으므로, Consensus 매칭으로 대체
            for idx, m in enumerate(all_motifs):
                try:
                    consensus = str(m.consensus)
                    counts[idx] += seq_str.count(consensus)
                except:
                    continue
        return counts

    gen_counts = count_hits(generated_seqs)
    ref_counts = count_hits(reference_seqs)

    if np.std(gen_counts) < 1e-9 or np.std(ref_counts) < 1e-9:
        return 0.0
        
    corr, _ = pearsonr(gen_counts, ref_counts)
    return float(corr)

def count_kmers(seqs: List[str], k: int = 3) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in seqs:
        for i in range(len(s) - k + 1):
            kmer = s[i : i + k]
            counts[kmer] = counts.get(kmer, 0) + 1
    return counts

def kmer_pearson_corr(
    generated: List[str],
    reference: List[str],
    k: int = 3,
) -> float:
    """Pearson corr between kmer-count vectors."""
    gen_counts = count_kmers(generated, k=k)
    ref_counts = count_kmers(reference, k=k)

    keys = sorted(set(gen_counts.keys()) | set(ref_counts.keys()))
    if len(keys) == 0:
        return float("nan")

    x = np.array([gen_counts.get(key, 0) for key in keys], dtype=np.float64)
    y = np.array([ref_counts.get(key, 0) for key in keys], dtype=np.float64)
    
    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0

    corr, _ = pearsonr(x, y)
    return float(corr)
'''
# App-Log-Lik용 Stub (사용하지 않음)
@dataclass
class DRAKESMetricsConfig:
    oracle_seq_len: int = 200
    hepg2_activity_index: int = 0
    atac_hepg2_index: int = 0
'''
def compute_drakes_metrics(*args, **kwargs):
    # train.py에서 이 함수를 호출하지 않고 직접 계산하도록 변경했으므로
    # 호환성을 위해 남겨두거나 비워둠.
    pass