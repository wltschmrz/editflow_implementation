# data/datasets_gosai_csv.py
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional, List


_DNA2ID = {"A": 0, "C": 1, "G": 2, "T": 3}


def dna_to_ids(seq: str) -> torch.Tensor:
    seq = seq.strip().upper()
    ids = [_DNA2ID[c] for c in seq]  # will raise KeyError if ambiguous
    return torch.tensor(ids, dtype=torch.long)


class GosaiCSVDataset(Dataset):
    """
    DRAKES 스타일: gosai_all.csv에서 seq 컬럼을 읽어 고정 길이(200) 토큰 시퀀스를 반환.
    labels(hepg2,k562,sknsh)는 학습(EditFlows)에는 필요 없어서 기본은 반환 안 함.
    """

    def __init__(
        self,
        csv_path: str,
        seq_col: str = "seq",
        expected_len: int = 200,
        return_labels: bool = False,
        label_cols: Optional[List[str]] = None,
        drop_ambiguous: bool = True,
    ):
        self.csv_path = csv_path
        self.seq_col = seq_col
        self.expected_len = expected_len
        self.return_labels = return_labels
        self.label_cols = label_cols or ["hepg2", "k562", "sknsh"]
        self.drop_ambiguous = drop_ambiguous

        df = pd.read_csv(csv_path)
        if seq_col not in df.columns:
            raise ValueError(f"seq_col='{seq_col}' not in columns={list(df.columns)}")

        seqs = df[seq_col].dropna().astype(str).tolist()

        keep_seqs = []
        keep_labels = []

        # optional labels
        has_labels = all(c in df.columns for c in self.label_cols)
        labels_raw = None
        if self.return_labels:
            if not has_labels:
                raise ValueError(f"return_labels=True but missing label cols: {self.label_cols}")
            labels_raw = df[self.label_cols].values

        for i, s in enumerate(seqs):
            s = s.strip().upper()
            if expected_len is not None and len(s) != expected_len:
                continue

            if drop_ambiguous and any(c not in "ACGT" for c in s):
                continue

            keep_seqs.append(s)
            if self.return_labels:
                keep_labels.append(labels_raw[i])

        self.seqs = keep_seqs
        self.labels = keep_labels if self.return_labels else None

        if len(self.seqs) == 0:
            raise RuntimeError("No sequences left after filtering. Check expected_len / seq_col / ambiguous chars.")

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = dna_to_ids(self.seqs[idx])
        if self.expected_len is not None:
            # sanity
            if x.numel() != self.expected_len:
                raise RuntimeError(f"Bad length at idx={idx}: {x.numel()} (expected {self.expected_len})")

        if not self.return_labels:
            return x  # (200,)
        y = torch.tensor(self.labels[idx], dtype=torch.float32)  # (3,)
        return x, y
