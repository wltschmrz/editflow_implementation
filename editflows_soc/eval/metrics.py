from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import pearsonr, spearmanr

# DRAKES eval.ipynb uses grelu's JASPAR motif scanner:
#   from grelu.interpret.motifs import scan_sequences
try:
    from grelu.interpret.motifs import scan_sequences  # type: ignore
except Exception as e:  # pragma: no cover
    scan_sequences = None  # type: ignore


DNA_ALPHABET = set(["A", "C", "G", "T"])


def _clean_seq(s: str) -> str:
    """Keep only A/C/G/T (uppercase)."""
    s = s.upper()
    return "".join([c for c in s if c in DNA_ALPHABET])


def _total_kmer_positions(seqs: Sequence[str], k: int) -> int:
    total = 0
    for s in seqs:
        L = len(s)
        if L >= k:
            total += (L - k + 1)
    return total


def count_kmers(seqs: Sequence[str], k: int = 3) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in seqs:
        s = _clean_seq(s)
        for i in range(len(s) - k + 1):
            kmer = s[i : i + k]
            counts[kmer] = counts.get(kmer, 0) + 1
    return counts


def kmer_pearson_corr(
    generated: Sequence[str],
    reference: Sequence[str],
    k: int = 3,
    *,
    scale: bool = True,
    scale_mode: str = "seqs",
) -> float:
    """
    Match DRAKES eval.ipynb:
      - Count k-mers in reference (top 0.1% activity) and generated.
      - Build vectors over union(kmers).
      - Optionally scale reference counts by size ratio (generated/reference).
        DRAKES scales by number of sequences (n_sp2 / n_sp1).
      - Return Pearson correlation.
    For variable-length sequences, `scale_mode="kmers"` is usually less biased.
    """
    gen_counts = count_kmers(generated, k=k)
    ref_counts = count_kmers(reference, k=k)
    kmer_set = set(gen_counts.keys()) | set(ref_counts.keys())

    if scale and len(reference) > 0:
        if scale_mode == "kmers":
            denom = _total_kmer_positions(reference, k)
            numer = _total_kmer_positions(generated, k)
            factor = (numer / denom) if denom > 0 else 1.0
        else:  # "seqs" (DRAKES default)
            factor = (len(generated) / len(reference))
    else:
        factor = 1.0

    x = np.zeros((len(kmer_set),), dtype=np.float64)  # generated
    y = np.zeros((len(kmer_set),), dtype=np.float64)  # reference (scaled)
    for i, km in enumerate(kmer_set):
        x[i] = float(gen_counts.get(km, 0))
        y[i] = float(ref_counts.get(km, 0)) * factor

    # Handle degenerate case
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    corr, _ = pearsonr(x, y)
    return float(corr)


# eval/metrics.py

import numpy as np
from collections import Counter

def _load_motifs(motif_db: str = "jaspar", meme_path: str | None = None):
    """
    Returns a motif object that can be fed to grelu.interpret.motifs.scan_sequences.
    Priority:
      1) meme_path (local)
      2) motif_db == "jaspar" via grelu.io.motifs.get_jaspar()
    """
    if meme_path is not None:
        # meme 파일을 로드하는 함수 이름은 grelu 버전에 따라 다를 수 있음
        # 가장 흔한 형태들을 순차적으로 시도
        try:
            from grelu.io.motifs import read_meme
            return read_meme(meme_path)
        except Exception:
            pass
        try:
            from grelu.io.motifs import load_meme
            return load_meme(meme_path)
        except Exception:
            pass
        # 여기까지 왔다면 meme loader가 없음 → 사용자에게 명확히 에러
        raise RuntimeError(f"Could not load MEME motifs from {meme_path}. "
                           f"Your grelu version may not expose read_meme/load_meme.")

    if motif_db.lower() == "jaspar":
        from grelu.io.motifs import get_jaspar
        # 기본값: 최신 CORE 컬렉션을 쓰는 경우가 많음 (버전에 따라 args 다를 수 있음)
        # 일단 인자 없이 호출되는 형태를 우선 시도
        try:
            return get_jaspar()
        except TypeError:
            # 혹시 버전이 collection 인자를 요구하면 CORE로
            return get_jaspar(collection="CORE")

    raise ValueError(f"Unknown motif_db={motif_db}. Provide meme_path.")


def jaspar_spearman_corr(
    gen_seqs: list[str],
    ref_seqs: list[str],
    motif_db: str = "jaspar",
    meme_path: str | None = None,
    scale_mode: str = "seqs",   # DRAKES: 시퀀스 수로 정규화
):
    """
    DRAKES-style:
      - scan motifs over sequences
      - count motif hits per motif id
      - normalize (either by #seqs or by total hits)
      - Spearman corr between generated and reference motif-count vectors

    Returns float (nan if undefined).
    """
    from grelu.interpret.motifs import scan_sequences
    from scipy.stats import spearmanr

    motifs = _load_motifs(motif_db=motif_db, meme_path=meme_path)

    gdf = scan_sequences(gen_seqs, motifs)
    rdf = scan_sequences(ref_seqs, motifs)

    # 빈 스캔이면 corr 정의 불가
    if (gdf is None) or (rdf is None) or (len(gdf) == 0) or (len(rdf) == 0):
        return float("nan")

    # gdf/rdf에는 보통 "motif" 컬럼이 있음
    g_counts = Counter(gdf["motif"].astype(str).tolist())
    r_counts = Counter(rdf["motif"].astype(str).tolist())

    # union of motif ids
    keys = sorted(set(g_counts.keys()) | set(r_counts.keys()))
    if len(keys) == 0:
        return float("nan")

    x = np.array([g_counts.get(k, 0) for k in keys], dtype=np.float64)
    y = np.array([r_counts.get(k, 0) for k in keys], dtype=np.float64)

    # scaling (DRAKES는 보통 seqs 기준이 안전)
    if scale_mode == "seqs":
        x = x / max(len(gen_seqs), 1)
        y = y / max(len(ref_seqs), 1)
    elif scale_mode == "hits":
        x = x / max(x.sum(), 1.0)
        y = y / max(y.sum(), 1.0)

    # 상수 벡터면 spearman undefined
    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")

    return float(spearmanr(x, y).correlation)



# --- compatibility config ---
@dataclass
class DRAKESMetricsConfig:
    oracle_seq_len: int = 200
    hepg2_activity_index: int = 0
    atac_hepg2_index: int = 1  # DRAKES eval.ipynb uses [:,1] > 0.5 for HepG2
