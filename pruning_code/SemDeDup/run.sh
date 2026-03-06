# python parquet_to_memmap.py \
#   --clip_embeds_dir /mnt/data0/teja/data_pruning/output/clip_embeds \
#   --output_dir output/semdedup_embeds \
#   --dataset_path /datasets/ai-core-object/d-gpu-06097851-2053-4b67-8400-b5d404c04261/teja/internet_dataset/laionasthetic_v2 \
#   -j 8



# python run_semdedup_local.py   --embeddings_dir output/semdedup_embeds   --eps_list 0.06 0.07   --ncentroids 50000



# Run from SemDeDup directory (script auto-cds here)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate rnd_pixart conda env (has faiss-gpu for H100)
if command -v conda &>/dev/null; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate rnd_pixart
elif [ -f /data/h100/sfs/Venkat/wire_removal/miniconda38/etc/profile.d/conda.sh ]; then
  source /data/h100/sfs/Venkat/wire_removal/miniconda38/etc/profile.d/conda.sh
  conda activate rnd_pixart
fi

dinoembsdir='/data/corerndimage/data_pruning/dino_embs'
dinoopdir='output/dinoouts/'

# Step 1: Convert parquet to memmap (run once)
# python parquet_to_memmap.py --dino_embeds_dir $dinoembsdir --output_dir $dinoopdir -j 16

# Step 2: Run SemDeDup for ncentroids 100, 1000, 5000 (each in its own folder)
# Create per-ncentroids dirs with symlinks to shared embeddings (avoids copying ~50GB)
for n in 10000; do
  outdir="output/dinoouts_n${n}"
  mkdir -p $outdir
  # Only create symlinks if not already present (preserve existing setup)
  for f in embeddings.npy paths.npy dataset_info.json; do
    [ ! -e "$outdir/$f" ] && ln -sf "$(realpath $dinoopdir/$f)" "$outdir/$f"
  done
  # H100: no pre-built FAISS GPU has sm_90 kernels; use CPU with parallel workers
  # FAISS_CPU_WORKERS=8 uses 8 cores for assignment step (~8x faster)
  # --resume: skips clustering if dist_to_cent/nearest_cent exist; sort/semdedup have per-cluster resume
  # Uses absolute path so it works regardless of cwd
  FAISS_CPU_ONLY=1 FAISS_CPU_WORKERS=8 python run_semdedup_local.py \
    --embeddings_dir "$(realpath $outdir)" \
    --eps_list 0.06 0.07 \
    --ncentroids $n \
    --output_txt "$(realpath $outdir)/kept_paths_eps0.06.txt" \
    --cpu --resume
done

# Outputs:
#   output/dinoouts_n10000/kept_paths_eps0.06.txt
#
# To run ONLY for existing dinoouts_n10000 (no symlink setup, just resume):
#   cd datapruning/pruning_code/SemDeDup
#   FAISS_CPU_ONLY=1 FAISS_CPU_WORKERS=8 python run_semdedup_local.py \
#     --embeddings_dir output/dinoouts_n10000 --eps_list 0.06 0.07 --ncentroids 10000 \
#     --output_txt output/dinoouts_n10000/kept_paths_eps0.06.txt --cpu --resume