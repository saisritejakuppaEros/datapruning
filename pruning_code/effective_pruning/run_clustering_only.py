"""
Run only clustering + sort (no SemDeDup, no extract).
Used by run_effective_pruning.sh - does not modify SemDeDup.
"""
import sys
import os

# Add SemDeDup for imports
_SEMDEDUP = os.path.join(os.path.dirname(__file__), "..", "SemDeDup")
sys.path.insert(0, _SEMDEDUP)

from run_semdedup_local import run_clustering, run_sort_clusters, load_dataset_info

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--embeddings_dir", required=True)
    p.add_argument("--ncentroids", type=int, default=10000)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    ncentroids = min(args.ncentroids, load_dataset_info(args.embeddings_dir)["dataset_size"])
    print("Clustering...")
    run_clustering(args.embeddings_dir, ncentroids, args.seed)
    print("Sorting clusters...")
    run_sort_clusters(args.embeddings_dir, ncentroids)
    print("Done (clustering + sort only)")
