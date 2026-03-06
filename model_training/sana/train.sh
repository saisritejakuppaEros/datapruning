# python tools/create_wids_metadata.py /data/corerndimage/DiffusionModels/raw_dataset > /data/corerndimage/DiffusionModels/raw_dataset/wids-meta.json
# python tools/create_wids_metadata.py /data/corerndimage/data_pruning/raw_dataset > /data/corerndimage/data_pruning/raw_dataset/wids-meta.json
# cd /data/corerndimage/DiffusionModels/datapruning/model_training/sana && source /data/miniconda3/etc/profile.d/conda.sh && conda activate sana && pip uninstall mmcv -y 2>/dev/null; pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html


export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7

# Train from scratch (600M @ 512px)
torchrun --nproc_per_node=6 train_scripts/train.py \
  --config_path=configs/sana_config/512ms/Sana_600M_img512.yaml \
  --data.data_dir="[/data/corerndimage/data_pruning/raw_dataset]" \
  --train.train_batch_size=48 \
  --report_to=tensorboard \
  --repa.enabled=true