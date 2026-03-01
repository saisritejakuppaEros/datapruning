import os
import csv
import tarfile
import json
import argparse
import io
from PIL import Image
import math
from tqdm import tqdm
import sys

# Increase CSV field size limit for large captions
csv.field_size_limit(sys.maxsize)

def process_csv_dataset(csv_path_input, output_dir, split_index, num_splits, image_root=None, caption_root=None, dataset_type="movie", samples_per_tar=10000):
    """
    Converts a dataset from CSV format to WebDataset tar files.
    
    Args:
        csv_path_input (str): Path to the CSV file or directory of CSVs.
        output_dir (str): Directory to save output tar files.
        split_index (int): Index of the split to process (0-indexed).
        num_splits (int): Total number of splits.
        image_root (str): Optional root directory to prepend to image paths.
        caption_root (str): Optional root directory to prepend to caption paths.
        dataset_type (str): 'movie' or 'stock'.
        samples_per_tar (int): Number of samples per tar file.
    """
    
    # 1. Gather all samples from CSV(s)
    samples = []
    csv_files = []
    
    if os.path.isdir(csv_path_input):
        csv_files = sorted([os.path.join(csv_path_input, f) for f in os.listdir(csv_path_input) if f.endswith(".csv")])
    elif os.path.isfile(csv_path_input):
        csv_files = [csv_path_input]
    else:
        print(f"Error: {csv_path_input} is not a valid file or directory.")
        return

    print(f"Reading {len(csv_files)} CSV file(s)...")
    
    for f_path in csv_files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader, None) # Skip header
                if not header: continue

                for row in reader:
                    if not row: continue
                    
                    if dataset_type == "movie":
                        # CSV format: path, actors, caption, caption_file_path
                        # We use the 'caption' text directly from the CSV (column index 2)
                        if len(row) >= 3:
                            samples.append({
                                "image_path": row[0],
                                "caption": row[2],
                                "original_csv": f_path
                            })
                    elif dataset_type == "stock":
                        # CSV format: image_path, caption_path, type
                        if len(row) >= 2:
                            samples.append({
                                "image_path": row[0],
                                "caption_path": row[1],
                                "original_csv": f_path
                            })
        except Exception as e:
            print(f"Error reading CSV {f_path}: {e}")

    total_samples = len(samples)
    print(f"Total samples found: {total_samples}")

    if total_samples == 0:
        print("No samples found. Exiting.")
        return

    import pandas as pd

    # ... (previous code)

    # 2. Determine Split
    if num_splits > 1:
        samples_per_split = math.ceil(total_samples / num_splits)
        start_idx = split_index * samples_per_split
        end_idx = min(start_idx + samples_per_split, total_samples)
    else:
        start_idx = 0
        end_idx = total_samples
    
    split_samples = samples[start_idx:end_idx]
    print(f"Processing split {split_index}/{num_splits} (Samples {start_idx} to {end_idx}, Count: {len(split_samples)})")

    if not split_samples:
        print("This split is empty or index out of range.")
        return

    # Create split specific output directory
    split_dir_name = f"split_{split_index:03d}"
    split_output_dir = os.path.join(output_dir, split_dir_name)
    os.makedirs(split_output_dir, exist_ok=True)

    # 3. Process Samples and Write to Tar
    num_tars = math.ceil(len(split_samples) / samples_per_tar)
    
    for tar_idx in range(num_tars):
        tar_start = tar_idx * samples_per_tar
        tar_end = min(tar_start + samples_per_tar, len(split_samples))
        chunk_samples = split_samples[tar_start:tar_end]
        
        # Naming: 00000.tar inside split_000/
        tar_basename = f"{tar_idx:05d}"
        tar_filename = f"{tar_basename}.tar"
        tar_path = os.path.join(split_output_dir, tar_filename)
        
        print(f"Writing {tar_path} ({len(chunk_samples)} samples)...")
        
        chunk_metadata_list = []
        success_count = 0
        
        with tarfile.open(tar_path, "w") as tar:
            for i, sample in enumerate(tqdm(chunk_samples, desc=f"Tar {tar_idx:05d}", leave=False)):
                
                # --- Resolve Image Path ---
                img_p = sample["image_path"]
                if image_root:
                    if not os.path.isabs(img_p):
                        img_p = os.path.join(image_root, img_p)
                
                if not os.path.exists(img_p):
                    if i < 5:
                        print(f"Warning: Image not found {img_p}")
                    continue

                # --- Read Image ---
                try:
                    with open(img_p, "rb") as f:
                        img_binary = f.read()
                    
                    with Image.open(io.BytesIO(img_binary)) as img:
                        img.verify()
                        width, height = img.size
                        img_format = img.format.lower() if img.format else "png"
                except Exception as e:
                    print(f"Error reading image {img_p}: {e}")
                    continue
                
                # --- Get Caption ---
                caption = ""
                if dataset_type == "movie":
                    caption = sample.get("caption", "")
                elif dataset_type == "stock":
                    cap_p = sample.get("caption_path", "")
                    if caption_root and not os.path.isabs(cap_p):
                        cap_p = os.path.join(caption_root, cap_p)
                    
                    if os.path.exists(cap_p):
                        try:
                            with open(cap_p, "r", encoding="utf-8") as cf:
                                caption = cf.read().strip()
                        except Exception as e:
                            print(f"Error: {e}")
                    else:
                        pass

                # --- Write to Tar ---
                # Key: {tar_idx:05d}{i:05d} -> inside the tar 
                # e.g. 0000000025 (Tar 0, Sample 25)
                # Ensure unique key across global dataset if possible? 
                # If we use simple counter i, it's unique within tar. 
                key = f"{i:05d}" 
                
                # 1. Image
                ext = os.path.splitext(img_p)[1]
                if not ext: ext = f".{img_format}"
                
                ti_img = tarfile.TarInfo(name=f"{key}{ext}")
                ti_img.size = len(img_binary)
                tar.addfile(ti_img, io.BytesIO(img_binary))
                
                # 2. Caption (.txt)
                cap_data = caption.encode("utf-8")
                ti_cap = tarfile.TarInfo(name=f"{key}.txt")
                ti_cap.size = len(cap_data)
                tar.addfile(ti_cap, io.BytesIO(cap_data))
                
                # 3. Metadata (.json)
                metadata = {
                    "file_name": os.path.basename(img_p),
                    "original_path": img_p,
                    "width": width,
                    "height": height,
                    "caption": caption
                }
                meta_json = json.dumps(metadata, indent=4).encode("utf-8")
                ti_json = tarfile.TarInfo(name=f"{key}.json")
                ti_json.size = len(meta_json)
                tar.addfile(ti_json, io.BytesIO(meta_json))
                
                chunk_metadata_list.append(metadata)
                success_count += 1

        # --- Write Parquet ---
        if chunk_metadata_list:
            df = pd.DataFrame(chunk_metadata_list)
            parquet_filename = f"{tar_basename}.parquet"
            parquet_path = os.path.join(split_output_dir, parquet_filename)
            df.to_parquet(parquet_path, index=False)
            print(f"Generated {parquet_filename}")
            
            # --- Write Stats JSON ---
            stats = {
                "count": success_count,
                "tar_file": tar_filename,
                "parquet_file": parquet_filename
            }
            stats_filename = f"{tar_basename}_stats.json"
            stats_path = os.path.join(split_output_dir, stats_filename)
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=4)
            print(f"Generated {stats_filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV datasets to WebDataset format with splitting.")
    
    parser.add_argument("--csv_path", type=str, required=True, 
                        help="Path to the input CSV file or directory of CSV files.")
    parser.add_argument("--output_dir", type=str, required=True, 
                        help="Directory where the output TAR files will be saved.")
    
    parser.add_argument("--split_index", type=int, default=0, 
                        help="The index of the current split/worker (0 to num_splits-1).")
    parser.add_argument("--num_splits", type=int, default=1, 
                        help="The total number of splits to divide the dataset into.")
    
    parser.add_argument("--dataset_type", type=str, default="movie", choices=["movie", "stock"], 
                        help="Type of dataset: 'movie' (uses caption in CSV) or 'stock' (uses external text files).")
    
    parser.add_argument("--samples_per_tar", type=int, default=10000, 
                        help="Number of samples to pack into each .tar file.")
    
    parser.add_argument("--image_root", type=str, default=None, 
                        help="Root path to prepend to image paths if they are relative.")
    parser.add_argument("--caption_root", type=str, default=None, 
                        help="Root path to prepend to caption paths if they are relative (only for stock).")

    args = parser.parse_args()
    
    process_csv_dataset(
        args.csv_path, 
        args.output_dir, 
        args.split_index, 
        args.num_splits, 
        image_root=args.image_root, 
        caption_root=args.caption_root,
        dataset_type=args.dataset_type,
        samples_per_tar=args.samples_per_tar
    )
