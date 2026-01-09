import argparse
from typing import List

import numpy as np
import torch

from data.datasets import decode_dna
from eval.metrics import pred_activity_median, atac_acc_percent, k3_corr
from eval.oracles import BaseOracle


def ids_to_str(seqs_ids: List[torch.Tensor]) -> List[str]:
    return [decode_dna(s.tolist()) for s in seqs_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_pt", type=str, required=True, help="Path to torch file containing generated sequences (list of LongTensor).")
    parser.add_argument("--top_real_fasta", type=str, default=None, help="Optional: fasta of top 0.1%% sequences for 3-mer corr.")
    args = parser.parse_args()

    seqs_ids = torch.load(args.generated_pt)
    seqs = ids_to_str(seqs_ids)

    # TODO: plug in real oracles
    print(f"Loaded {len(seqs)} generated sequences.")
    print("Implement oracle calls in eval/oracles.py and add DRAKES metrics here.")

if __name__ == "__main__":
    main()
