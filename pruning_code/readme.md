# Run from SemDeDup directory: cd datapruning/pruning_code/SemDeDup

dinoembsdir='/data/corerndimage/data_pruning/dino_embs'
dinoopdir='output/dinoouts/'

# Step 1: Convert parquet to memmap (run once)
python parquet_to_memmap.py --dino_embeds_dir $dinoembsdir --output_dir $dinoopdir -j 16

# Step 2: Run SemDeDup for ncentroids 100, 1000, 5000 (each in its own folder)
# Create per-ncentroids dirs with symlinks to shared embeddings (avoids copying ~50GB)
for n in 100 1000 5000; do
  outdir="output/dinoouts_n${n}"
  mkdir -p $outdir
  for f in embeddings.npy paths.npy dataset_info.json; do
    ln -sf "$(realpath $dinoopdir/$f)" "$outdir/$f"
  done
  python run_semdedup_local.py --embeddings_dir $outdir --eps_list 0.06 0.07 --ncentroids $n --output_txt $outdir/kept_paths_eps0.06.txt
done

# Outputs:
#   output/dinoouts_n100/kept_paths_eps0.06.txt
#   output/dinoouts_n1000/kept_paths_eps0.06.txt
#   output/dinoouts_n5000/kept_paths_eps0.06.txt