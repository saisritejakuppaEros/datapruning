"""
Create a JSON report with image paths, pruned status, and cluster assignments.

Output format:
{
  "metadata": {
    "total_images": N,
    "num_clusters": K,
    "eps_value": 0.06,
    "pruned_count": X,
    "kept_count": Y
  },
  "images": [
    {
      "path": "split_059/00000.tar|000001342.jpg",
      "cluster": 528,
      "pruned": false
    },
    ...
  ]
}
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


def process_cluster(cluster_id, sorted_clusters_dir, semdedup_dir, eps_col):
    """
    Process a single cluster and return pruning status mapping.
    
    Returns:
        dict: Mapping from global_index to pruning status (True if pruned)
    """
    cluster_pruning = {}
    
    # Load sorted cluster file (contains indices and metadata)
    cluster_file = os.path.join(sorted_clusters_dir, f"cluster_{cluster_id}.npy")
    if not os.path.exists(cluster_file):
        return cluster_pruning
        
    cluster_data = np.load(cluster_file, mmap_mode='r')
    # cluster_data shape: (cluster_size, 4)
    # columns: [path, global_index, similarity, cluster_id]
    
    # Load pruning dataframe for this cluster
    pkl_file = os.path.join(semdedup_dir, f"cluster_{cluster_id}.pkl")
    if not os.path.exists(pkl_file):
        return cluster_pruning
        
    with open(pkl_file, 'rb') as f:
        df = pickle.load(f)
    
    # Map pruning status to global indices
    for local_idx in range(len(cluster_data)):
        global_idx = int(cluster_data[local_idx, 1])
        
        # Check if this local index is in the dataframe
        if local_idx < len(df):
            if eps_col in df.columns:
                cluster_pruning[global_idx] = bool(df.iloc[local_idx][eps_col])
    
    return cluster_pruning


def create_pruning_report(embeddings_dir: str, eps: float, output_json: str, num_threads: int = 16):
    """
    Create a JSON report combining cluster assignments and pruning decisions.
    
    Args:
        embeddings_dir: Directory with clustering and semdedup outputs
        eps: Epsilon value to use for pruning status
        output_json: Output JSON file path
        num_threads: Number of threads for parallel processing
    """
    
    print(f"Loading data from {embeddings_dir}...")
    
    # Load paths (image file paths)
    paths_path = os.path.join(embeddings_dir, "paths.npy")
    paths = np.load(paths_path, mmap_mode='r')
    total_images = len(paths)
    print(f"Total images: {total_images}")
    
    # Load cluster assignments
    nearest_cent_path = os.path.join(embeddings_dir, "clustering", "nearest_cent.npy")
    cluster_assignments = np.load(nearest_cent_path, mmap_mode='r')
    num_clusters = int(cluster_assignments.max()) + 1
    print(f"Number of clusters: {num_clusters}")
    
    # Load pruning decisions from semdedup dataframes using multithreading
    print(f"Loading pruning decisions for eps={eps} using {num_threads} threads...")
    semdedup_dir = os.path.join(embeddings_dir, "semdedup", "dataframes")
    sorted_clusters_dir = os.path.join(embeddings_dir, "sorted_clusters")
    eps_col = f"eps={eps}"
    
    # Create a mapping from global index to pruning status
    pruning_status = {}
    
    # Process clusters in parallel
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {
            executor.submit(process_cluster, cluster_id, sorted_clusters_dir, semdedup_dir, eps_col): cluster_id
            for cluster_id in range(num_clusters)
        }
        
        for future in tqdm(as_completed(futures), total=num_clusters, desc="Processing clusters"):
            cluster_pruning = future.result()
            pruning_status.update(cluster_pruning)
    
    # Build the JSON structure
    print("Building JSON structure...")
    images_list = []
    pruned_count = 0
    kept_count = 0
    
    for idx in tqdm(range(total_images), desc="Creating image records"):
        path_bytes = paths[idx]
        # Decode and strip padding
        path_str = path_bytes.decode('utf-8').strip()
        
        is_pruned = pruning_status.get(idx, False)
        cluster_id = int(cluster_assignments[idx])
        
        if is_pruned:
            pruned_count += 1
        else:
            kept_count += 1
        
        images_list.append({
            "path": path_str,
            "cluster": cluster_id,
            "pruned": is_pruned
        })
    
    # Create final report
    report = {
        "metadata": {
            "total_images": total_images,
            "num_clusters": num_clusters,
            "eps_value": eps,
            "pruned_count": pruned_count,
            "kept_count": kept_count
        },
        "images": images_list
    }
    
    # Write to JSON file
    print(f"Writing to {output_json}...")
    with open(output_json, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport created successfully!")
    print(f"Total images: {total_images}")
    print(f"Pruned: {pruned_count} ({100*pruned_count/total_images:.2f}%)")
    print(f"Kept: {kept_count} ({100*kept_count/total_images:.2f}%)")
    print(f"Clusters: {num_clusters}")
    print(f"Output: {output_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Create JSON report with image paths, cluster assignments, and pruning status"
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory with clustering and semdedup outputs (e.g., output/dinoouts_n1000)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=0.06,
        help="Epsilon value to use for pruning status (default: 0.06)",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Output JSON file path (default: <embeddings_dir>/pruning_report_eps<eps>.json)",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=16,
        help="Number of threads for parallel processing (default: 16)",
    )
    
    args = parser.parse_args()
    
    embeddings_dir = os.path.abspath(args.embeddings_dir)
    
    if args.output_json is None:
        output_json = os.path.join(embeddings_dir, f"pruning_report_eps{args.eps}.json")
    else:
        output_json = args.output_json
    
    create_pruning_report(embeddings_dir, args.eps, output_json, args.num_threads)


if __name__ == "__main__":
    main()


# python create_pruning_report.py --embeddings_dir output/dinoouts_n100 --eps 0.06 --output_json output/dinoouts_n100/pruning_report_eps0.06.json --num_threads 16