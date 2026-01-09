CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 --master_port=29503 train_soc_doob.py \
  --config configs/config.yaml \
  --base_ckpt /mnt/data1/intern/jeongchan/editflows_memoryless/checkpoints_ver2/editflows_dna_039000.pt \
  --oracle_ckpt /mnt/data1/intern/jeongchan/enhancer_data/data_and_model/mdlm/outputs_gosai/lightning_logs/reward_oracle_ft.ckpt \
  --eval_oracle_ckpt /mnt/data1/intern/jeongchan/enhancer_data/data_and_model/mdlm/outputs_gosai/lightning_logs/reward_oracle_eval.ckpt \
  --atac_ckpt /mnt/data1/intern/jeongchan/enhancer_data/data_and_model/mdlm/gosai_data/binary_atac_cell_lines.ckpt \
  --gosai_csv /mnt/data1/intern/jeongchan/enhancer_data/data_and_model/mdlm/gosai_data/processed_data/gosai_all.csv \
  --jaspar_meme /mnt/data1/intern/jeongchan/enhancer_data/JASPAR2026_CORE_non-redundant_pfms_meme.txt \
  --resume_ckpt /mnt/data1/intern/jeongchan/editflows_soc/soc_checkpoints/doob_soc_001000.pt