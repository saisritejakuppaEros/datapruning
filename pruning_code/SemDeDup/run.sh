python parquet_to_memmap.py \
  --clip_embeds_dir /mnt/data0/teja/data_pruning/output/clip_embeds \
  --output_dir output/semdedup_embeds \
  --dataset_path /datasets/ai-core-object/d-gpu-06097851-2053-4b67-8400-b5d404c04261/teja/internet_dataset/laionasthetic_v2 \
  -j 8



python run_semdedup_local.py   --embeddings_dir output/semdedup_embeds   --eps_list 0.06 0.07   --ncentroids 50000