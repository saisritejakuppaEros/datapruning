"""
Convert CLIP parquet embeddings (from clip_embeds_gen.py) to SemDeDup memmap format.

Handles incomplete embeddings: filters invalid/NaN, skips missing parquet files,
and reports a summary of excluded rows.
"""

import os
import json
import glob
import logging
import argparse
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

PATH_SEP = "|"
PATH_STR_DTYPE = "S80"
EMB_SIZE = 768


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(__name__)


logger = setup_logging()


def get_parquet_files(clip_embeds_dir: str, limit: int = None):
    """
    Get parquet files to process. Uses source_to_parquet_mapping.json if present;
    otherwise globs *.parquet. Only returns files that exist on disk.
    limit: max number of parquet files to return (for testing).
    """
    mapping_file = os.path.join(clip_embeds_dir, "source_to_parquet_mapping.json")
    if os.path.exists(mapping_file):
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        parquet_files = []
        for rel_tar_path, parquet_name in sorted(mapping.items()):
            parquet_path = os.path.join(clip_embeds_dir, parquet_name)
            if os.path.exists(parquet_path):
                parquet_files.append((rel_tar_path, parquet_path))
            else:
                logger.debug(f"Skipping missing parquet: {parquet_path}")
        if parquet_files:
            if limit:
                parquet_files = parquet_files[:limit]
            return parquet_files
        logger.warning("Mapping found but no parquet files exist; falling back to glob")

    parquet_paths = sorted(glob.glob(os.path.join(clip_embeds_dir, "*.parquet")))
    result = [(None, p) for p in parquet_paths]
    if limit:
        result = result[:limit]
    return result


def load_and_filter_parquet(parquet_path: str, rel_tar_path: str):
    """
    Load parquet and return embeddings, paths, and counts of valid/skipped rows.
    """
    table = pq.read_table(parquet_path)
    n_total = table.num_rows

    embeddings_list = []
    paths_list = []
    skipped_no_emb = 0
    skipped_nan = 0

    has_emb_col = "has_embedding" in table.column_names
    emb_col = table["embedding"]
    path_col = table["path"]
    rel_col = table["rel_tar_path"] if "rel_tar_path" in table.column_names else None

    for i in range(n_total):
        if has_emb_col:
            has_emb = table["has_embedding"][i]
            if hasattr(has_emb, "as_py"):
                has_emb = has_emb.as_py()
            if not has_emb:
                skipped_no_emb += 1
                continue

        emb = emb_col[i]
        emb_arr = np.array(emb, dtype=np.float32)
        if np.any(np.isnan(emb_arr)) or np.any(np.isinf(emb_arr)):
            skipped_nan += 1
            continue

        path_val = path_col[i]
        if hasattr(path_val, "as_py"):
            path_val = path_val.as_py()
        path_val = str(path_val) if path_val is not None else ""

        rel_val = rel_tar_path
        if rel_val is None and rel_col is not None:
            rel_val = rel_col[i]
            if hasattr(rel_val, "as_py"):
                rel_val = rel_val.as_py()
            rel_val = str(rel_val) if rel_val else ""
        if not rel_val:
            rel_val = os.path.basename(parquet_path).replace(".parquet", ".tar").replace("_", "/", 1)

        composite_path = f"{rel_val}{PATH_SEP}{path_val}"
        if len(composite_path.encode("utf-8")) > 80:
            logger.warning(f"Path truncated (>{PATH_STR_DTYPE}): {composite_path[:50]}...")

        embeddings_list.append(emb_arr)
        paths_list.append(composite_path)

    return np.stack(embeddings_list) if embeddings_list else np.empty((0, EMB_SIZE)), paths_list, n_total, skipped_no_emb, skipped_nan


def _load_parquet_item(args):
    """Wrapper for ProcessPoolExecutor: (rel_tar_path, parquet_path) -> result tuple."""
    rel_tar_path, parquet_path = args
    embs, paths, n_total, skip_no_emb, skip_nan = load_and_filter_parquet(
        parquet_path, rel_tar_path
    )
    return embs, paths, n_total, skip_no_emb, skip_nan


def main(
    clip_embeds_dir: str,
    output_dir: str,
    dataset_path: str = None,
    limit: int = None,
    workers: int = 1,
):
    os.makedirs(output_dir, exist_ok=True)

    parquet_items = get_parquet_files(clip_embeds_dir, limit=limit)
    if not parquet_items:
        raise ValueError(f"No parquet files found in {clip_embeds_dir}")

    logger.info(f"Processing {len(parquet_items)} parquet files (workers={workers})")

    all_embeddings = []
    all_paths = []
    total_read = 0
    total_skipped_no_emb = 0
    total_skipped_nan = 0

    if workers <= 1:
        for rel_tar_path, parquet_path in tqdm(parquet_items, desc="Reading parquets"):
            embs, paths, n_total, skip_no_emb, skip_nan = load_and_filter_parquet(
                parquet_path, rel_tar_path
            )
            total_read += n_total
            total_skipped_no_emb += skip_no_emb
            total_skipped_nan += skip_nan
            if embs.size > 0:
                all_embeddings.append(embs)
                all_paths.extend(paths)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for embs, paths, n_total, skip_no_emb, skip_nan in tqdm(
                executor.map(_load_parquet_item, parquet_items, chunksize=1),
                total=len(parquet_items),
                desc="Reading parquets",
            ):
                total_read += n_total
                total_skipped_no_emb += skip_no_emb
                total_skipped_nan += skip_nan
                if embs.size > 0:
                    all_embeddings.append(embs)
                    all_paths.extend(paths)

    if not all_embeddings:
        raise ValueError(
            f"No valid embeddings found. Total rows: {total_read}, "
            f"skipped (no embedding): {total_skipped_no_emb}, skipped (NaN): {total_skipped_nan}"
        )

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    dataset_size = embeddings.shape[0]
    emb_size = embeddings.shape[1]

    logger.info(
        f"Summary: total rows read={total_read}, skipped (no embedding)={total_skipped_no_emb}, "
        f"skipped (NaN)={total_skipped_nan}, final dataset_size={dataset_size}"
    )

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    embeddings = embeddings / norms

    paths_arr = np.array([p.encode("utf-8")[:80].ljust(80) for p in all_paths], dtype=PATH_STR_DTYPE)

    emb_path = os.path.join(output_dir, "embeddings.npy")
    paths_path = os.path.join(output_dir, "paths.npy")
    np.save(emb_path, embeddings)
    np.save(paths_path, paths_arr)

    info = {
        "dataset_size": int(dataset_size),
        "emb_size": int(emb_size),
        "path_str_dtype": PATH_STR_DTYPE,
        "dataset_path": dataset_path or "",
    }
    info_path = os.path.join(output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Saved embeddings to {emb_path} ({dataset_size} x {emb_size})")
    logger.info(f"Saved paths to {paths_path}")
    logger.info(f"Saved dataset_info to {info_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CLIP parquet embeddings to SemDeDup memmap format"
    )
    parser.add_argument(
        "--clip_embeds_dir",
        type=str,
        required=True,
        help="Directory containing clip_embeds parquet files (e.g. output/clip_embeds)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for embeddings.npy, paths.npy, dataset_info.json",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Base path to image dataset (stored in dataset_info for path resolution)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of parquet files to process (for testing)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers for reading parquet files (default: 1)",
    )
    args = parser.parse_args()
    main(
        clip_embeds_dir=args.clip_embeds_dir,
        output_dir=args.output_dir,
        dataset_path=args.dataset_path,
        limit=args.limit,
        workers=args.workers,
    )
