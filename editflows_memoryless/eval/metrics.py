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


def jaspar_spearman_corr(
    generated: Sequence[str],
    reference: Sequence[str],
    *,
    motif_db: str = "jaspar",
    scale: bool = True,
    scale_mode: str = "seqs",
) -> float:
    """
    Match DRAKES eval.ipynb:
      - scan_sequences(seqs, 'jaspar') -> DataFrame with 'motif' column
      - motif_count = df['motif'].value_counts()
      - corr(method='spearman') between top_data and generated

    We return Spearman rho between motif-count vectors (aligned on union motifs).
    """
    if scan_sequences is None:
        raise ImportError(
            "grelu is not available; install grelu or switch to your previous JASPAR scanner."
        )

    gen_df = scan_sequences([_clean_seq(s) for s in generated], motif_db)
    ref_df = scan_sequences([_clean_seq(s) for s in reference], motif_db)

    gen_counts = gen_df["motif"].value_counts()
    ref_counts = ref_df["motif"].value_counts()

    motifs = gen_counts.index.union(ref_counts.index)
    x = gen_counts.reindex(motifs, fill_value=0).to_numpy(dtype=np.float64)
    y = ref_counts.reindex(motifs, fill_value=0).to_numpy(dtype=np.float64)

    if scale and len(reference) > 0:
        if scale_mode == "kmers":
            # Motif hits scale roughly with length; approximate with total bp ratio
            denom = sum(len(_clean_seq(s)) for s in reference)
            numer = sum(len(_clean_seq(s)) for s in generated)
            factor = (numer / denom) if denom > 0 else 1.0
        else:
            factor = (len(generated) / len(reference))
        y = y * factor

    if np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rho, _ = spearmanr(x, y)
    return float(rho)


# --- compatibility config ---
@dataclass
class DRAKESMetricsConfig:
    oracle_seq_len: int = 200
    hepg2_activity_index: int = 0
    atac_hepg2_index: int = 1  # DRAKES eval.ipynb uses [:,1] > 0.5 for HepG2
