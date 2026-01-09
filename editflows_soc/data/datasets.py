import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

DNA_ALPHABET = {"A": 0, "C": 1, "G": 2, "T": 3}
ID_TO_BASE = {v: k for k, v in DNA_ALPHABET.items()}


def encode_dna(seq: str) -> List[int]:
    seq = seq.strip().upper()
    ids = []
    for ch in seq:
        if ch in DNA_ALPHABET:
            ids.append(DNA_ALPHABET[ch])
        else:
            # drop ambiguous bases by skipping; caller can filter upstream
            # alternatively map 'N' to random base
            continue
    return ids


def decode_dna(ids: List[int]) -> str:
    return "".join(ID_TO_BASE.get(i, "N") for i in ids)


class TargetDataset(Dataset):
    """
    Target dataset providing x1 sequences (enhancer/accessibility training sequences).

    Expected FASTA file: one sequence per record.
    Labels are optional here; training of edit flows itself does not require labels,
    but evaluation/oracle training might.
    """
    def __init__(self, fasta_path: str, max_records: Optional[int] = None):
        self.seqs: List[List[int]] = []
        self._load_fasta(fasta_path, max_records=max_records)

    def _load_fasta(self, fasta_path: str, max_records: Optional[int] = None):
        seq = []
        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if seq:
                        self.seqs.append(encode_dna("".join(seq)))
                        seq = []
                        if max_records is not None and len(self.seqs) >= max_records:
                            break
                else:
                    seq.append(line)
            if seq and (max_records is None or len(self.seqs) < max_records):
                self.seqs.append(encode_dna("".join(seq)))

        # Filter very short sequences (after removing ambiguous)
        self.seqs = [s for s in self.seqs if len(s) > 0]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.seqs[idx], dtype=torch.long)


class BackgroundDataset(Dataset):
    """
    Background dataset for sampling x0 in 'background' mode.

    Two common definitions:
      (1) negatives/low-activity sequences (if you have labels)
      (2) generic genomic regions (e.g., random accessible/inaccessible regions)
    Here we simply load a FASTA and treat it as background pool.
    """
    def __init__(self, fasta_path: str, max_records: Optional[int] = None):
        self.seqs: List[List[int]] = []
        self._load_fasta(fasta_path, max_records=max_records)

    def _load_fasta(self, fasta_path: str, max_records: Optional[int] = None):
        seq = []
        with open(fasta_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if seq:
                        self.seqs.append(encode_dna("".join(seq)))
                        seq = []
                        if max_records is not None and len(self.seqs) >= max_records:
                            break
                else:
                    seq.append(line)
            if seq and (max_records is None or len(self.seqs) < max_records):
                self.seqs.append(encode_dna("".join(seq)))
        self.seqs = [s for s in self.seqs if len(s) > 0]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.tensor(self.seqs[idx], dtype=torch.long)


@dataclass
class SpecialTokens:
    BOS: int
    PAD: int
    EPS0: int
    EPS1: int

    @property
    def vocab_size(self) -> int:
        return max(self.BOS, self.PAD, self.EPS0, self.EPS1) + 1
