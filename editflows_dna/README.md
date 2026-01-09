# Edit Flows DNA (starter scaffold)

This scaffold is designed to help you adapt:
- DRAKES DNA dataloader logic (ACGT tokens)
- an unofficial Edit Flows-style implementation (alignment + edit ops)

## Quick start (training)
```bash
python train.py --config configs/config.yaml \
  --target_fasta /path/to/<fold>_sequences_train.fa \
  --background_fasta /path/to/background.fa
```

## Notes
- This is a *starter* implementation that mirrors Edit Flows structure:
  alignment z-space (with EPS), mixture path, op/token heads, Euler sampling stub.
- For full DRAKES metrics (JASPAR Corr, App-Log-Lik), see eval/metrics.py TODOs.
