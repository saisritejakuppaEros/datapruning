#!/usr/bin/env bash
# Effective pruning pipeline: CLIP embeddings + dataset -> kept_paths.txt (SemDeDup format)
# Run from repo root: /mnt/data0/teja/data_pruning

set -e

# Config
DATASET_PATH="/datasets/ai-core-object/d-gpu-06097851-2053-4b67-8400-b5d404c04261/teja/internet_dataset/laionasthetic_v2"
CLIP_EMBEDS_DIR="/mnt/data0/teja/data_pruning/output/clip_embeds"
WORK_DIR="/mnt/data0/teja/data_pruning/output/effective_pruning"
NUM_CENTROIDS=100
PRUNE_RATIO=0.8

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SEMDEDUP_DIR="${REPO_ROOT}/codes/SemDeDup"
EFFECTIVE_DIR="${REPO_ROOT}/codes/effective_pruning"
EMBEDDINGS_DIR="${WORK_DIR}/embeds"
RESULTS_DIR="${WORK_DIR}/results"

mkdir -p "${EMBEDDINGS_DIR}"
mkdir -p "${RESULTS_DIR}"

echo "=== Step 0: Parquet to memmap ==="
# python "${SEMDEDUP_DIR}/parquet_to_memmap.py" \
#   --clip_embeds_dir "${CLIP_EMBEDS_DIR}" \
#   --output_dir "${EMBEDDINGS_DIR}" \
#   --dataset_path "${DATASET_PATH}" \
#   -j 8

# echo "=== Step 1: Clustering + sort clusters ==="
# python "${EFFECTIVE_DIR}/run_clustering_only.py" \
#   --embeddings_dir "${EMBEDDINGS_DIR}" \
#   --ncentroids "${NUM_CENTROIDS}"

SORTED_CLUSTERS="${EMBEDDINGS_DIR}/sorted_clusters"
CENTROIDS_FILE="${EMBEDDINGS_DIR}/clustering/kmeans_centroids.npy"
TOTAL_SIZE=$(python3 -c "import json; print(json.load(open('${EMBEDDINGS_DIR}/dataset_info.json'))['dataset_size'])")

echo "=== Step 2: Compute d_intra ==="
python "${EFFECTIVE_DIR}/compute_avg_dist_to_cluster_centroid.py" \
  --num_clusters "${NUM_CENTROIDS}" \
  --sorted_clusters_path "${SORTED_CLUSTERS}" \
  --avg_distance_to_cent_save_path "${RESULTS_DIR}/d_intra.npy"

echo "=== Step 3: Compute d_inter ==="
python "${EFFECTIVE_DIR}/compute_centroid_distances.py" \
  --centroids_file "${CENTROIDS_FILE}" \
  --mean_centroid_distances_save_path "${RESULTS_DIR}/d_inter.npy" \
  --dim 768

echo "=== Step 4: Prune ==="
python "${EFFECTIVE_DIR}/prune.py" \
  --num-centroids "${NUM_CENTROIDS}" \
  --sorted-clusters-path "${SORTED_CLUSTERS}" \
  --mean_centroid_distances_save_path "${RESULTS_DIR}/d_inter.npy" \
  --avg_distance_to_cent_save_path "${RESULTS_DIR}/d_intra.npy" \
  --total-dataset-size "${TOTAL_SIZE}" \
  --prune-ratio "${PRUNE_RATIO}" \
  --save-dir "${RESULTS_DIR}" \
  --output-txt "${RESULTS_DIR}/kept_paths_effective_pruning.txt"

echo "=== Done. Output: ${RESULTS_DIR}/kept_paths_effective_pruning.txt ==="
