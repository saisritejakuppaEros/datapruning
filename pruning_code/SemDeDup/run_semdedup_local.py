"""
Local runner for SemDeDup on CLIP embeddings (no SLURM required).

Runs steps 2-5 of the SemDeDup pipeline: clustering, sort clusters,
SemDeDup, and extract pruned paths.
"""

import os
import sys
import json
import logging
import argparse
from types import SimpleNamespace

# Add SemDeDup root for imports (clustering, semdedup, extract_dedup_data)
_SEMDEDUP_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SEMDEDUP_ROOT)

import numpy as np
import yaml

# Clustering imports (after path setup)
from clustering.clustering import compute_centroids
from clustering.sort_clusters import assign_and_sort_clusters
from clustering.utils import get_logger

# SemDeDup imports
from semdedup import SemDeDupJob
from extract_dedup_data import extract_pruned_data


def load_dataset_info(embeddings_dir: str):
    info_path = os.path.join(embeddings_dir, "dataset_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(
            f"dataset_info.json not found in {embeddings_dir}. "
            "Run parquet_to_memmap.py first."
        )
    with open(info_path, "r") as f:
        return json.load(f)


def run_clustering(embeddings_dir: str, ncentroids: int, seed: int):
    info = load_dataset_info(embeddings_dir)
    dataset_size = info["dataset_size"]
    emb_size = info["emb_size"]

    emb_path = os.path.join(embeddings_dir, "embeddings.npy")
    paths_path = os.path.join(embeddings_dir, "paths.npy")
    save_folder = os.path.join(embeddings_dir, "clustering")
    sorted_clusters_loc = os.path.join(embeddings_dir, "sorted_clusters")

    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(sorted_clusters_loc, exist_ok=True)

    logger = get_logger(
        file_name=os.path.join(save_folder, "compute_centroids.log"),
        level=logging.INFO,
        stdout=True,
    )

    data = np.load(emb_path, mmap_mode="r")
    if data.shape[0] != dataset_size or data.shape[1] != emb_size:
        raise ValueError(
            f"Embedding shape {data.shape} does not match dataset_info "
            f"({dataset_size}, {emb_size})"
        )

    logger.info(f"Running K-means: {dataset_size} points, {emb_size} dims, {ncentroids} centroids")
    compute_centroids(
        data=data,
        ncentroids=ncentroids,
        niter=100,
        seed=seed,
        Kmeans_with_cos_dist=True,
        save_folder=save_folder,
        logger=logger,
        verbose=True,
    )


def run_sort_clusters(embeddings_dir: str, ncentroids: int):
    info = load_dataset_info(embeddings_dir)
    dataset_size = info["dataset_size"]
    emb_size = info["emb_size"]
    path_str_dtype = info.get("path_str_dtype", "S80")

    emb_path = os.path.join(embeddings_dir, "embeddings.npy")
    paths_path = os.path.join(embeddings_dir, "paths.npy")
    save_folder = os.path.join(embeddings_dir, "clustering")
    sorted_clusters_loc = os.path.join(embeddings_dir, "sorted_clusters")

    data = np.load(emb_path, mmap_mode="r")
    paths_list = np.load(paths_path, mmap_mode="r")

    logger = get_logger(
        file_name=os.path.join(save_folder, "sort_clusters.log"),
        level=logging.INFO,
        stdout=True,
    )

    logger.info("Sorting clusters...")
    assign_and_sort_clusters(
        data=data,
        paths_list=paths_list,
        sim_metric="cosine",
        keep_hard=True,
        kmeans_with_cos_dist=True,
        save_folder=save_folder,
        sorted_clusters_file_loc=sorted_clusters_loc,
        cluster_ids=range(ncentroids),
        logger=logger,
    )


def run_semdedup(embeddings_dir: str, eps_list, ncentroids: int, seed: int):
    info = load_dataset_info(embeddings_dir)
    dataset_size = info["dataset_size"]
    emb_size = info["emb_size"]

    emb_path = os.path.join(embeddings_dir, "embeddings.npy")
    save_loc = os.path.join(embeddings_dir, "semdedup")  # semdedup.py appends dataframes/
    sorted_clusters_path = os.path.join(embeddings_dir, "sorted_clusters")

    os.makedirs(os.path.join(save_loc, "dataframes"), exist_ok=True)

    args = SimpleNamespace(
        embs_memory_loc=emb_path,
        dataset_size=dataset_size,
        emd_size=emb_size,
        save_loc=save_loc,
        sorted_clusters_path=sorted_clusters_path,
        eps_list=eps_list,
        which_to_keep="hard",
        seed=seed,
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        num_clusters=ncentroids,
        clusters_per_job=ncentroids,
        largest_cluster_size_to_process=10000000,
    )

    job = SemDeDupJob(args, job_start_cluster=0)
    job._process_shard(0, ncentroids)


def run_extract(
    embeddings_dir: str,
    output_txt_path: str,
    eps: float,
    ncentroids: int,
):
    sorted_clusters_path = os.path.join(embeddings_dir, "sorted_clusters")
    semdedup_pruning_tables_path = os.path.join(
        embeddings_dir, "semdedup", "dataframes"
    )

    extract_pruned_data(
        sorted_clusters_path=sorted_clusters_path,
        semdedup_pruning_tables_path=semdedup_pruning_tables_path,
        eps=eps,
        num_clusters=ncentroids,
        output_txt_path=output_txt_path,
        retreive_kept_samples=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run SemDeDup pipeline locally on CLIP embeddings"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory with embeddings.npy, paths.npy, dataset_info.json (from parquet_to_memmap.py)",
    )
    parser.add_argument(
        "--eps_list",
        type=float,
        nargs="+",
        default=[0.06, 0.07],
        help="Epsilon values for SemDeDup (lower = more aggressive pruning). Default: 0.06 0.07",
    )
    parser.add_argument(
        "--ncentroids",
        type=int,
        default=50000,
        help="Number of K-means clusters. Default: 50000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed. Default: 1234",
    )
    parser.add_argument(
        "--output_txt",
        type=str,
        default=None,
        help="Output file for kept paths (default: <embeddings_dir>/kept_paths_eps<eps>.txt)",
    )
    parser.add_argument(
        "--eps_for_extract",
        type=float,
        default=None,
        help="Which epsilon to use when extracting paths. Default: first value from eps_list",
    )
    parser.add_argument(
        "--skip_clustering",
        action="store_true",
        help="Skip steps 2-3 (clustering, sort) if already run",
    )
    parser.add_argument(
        "--skip_semdedup",
        action="store_true",
        help="Skip step 4 (SemDeDup) if already run",
    )
    parser.add_argument(
        "--extract_only",
        action="store_true",
        help="Only run step 5 (extract pruned paths)",
    )
    args = parser.parse_args()

    embeddings_dir = os.path.abspath(args.embeddings_dir)
    info = load_dataset_info(embeddings_dir)
    ncentroids = min(args.ncentroids, info["dataset_size"])

    if not args.extract_only:
        if not args.skip_clustering:
            print("Step 2: K-means clustering...")
            run_clustering(embeddings_dir, ncentroids, args.seed)
            print("Step 3: Sort clusters...")
            run_sort_clusters(embeddings_dir, ncentroids)
        elif not args.skip_semdedup:
            print("Step 3: Sort clusters...")
            run_sort_clusters(embeddings_dir, ncentroids)

        if not args.skip_semdedup:
            print("Step 4: SemDeDup...")
            run_semdedup(
                embeddings_dir,
                args.eps_list,
                ncentroids,
                args.seed,
            )

    eps_extract = args.eps_for_extract if args.eps_for_extract is not None else args.eps_list[0]
    output_txt = args.output_txt or os.path.join(
        embeddings_dir, f"kept_paths_eps{eps_extract}.txt"
    )

    print(f"Step 5: Extract pruned paths (eps={eps_extract})...")
    run_extract(embeddings_dir, output_txt, eps_extract, ncentroids)
    print(f"Done. Kept paths saved to {output_txt}")


if __name__ == "__main__":
    main()
