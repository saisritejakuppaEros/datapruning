# Dataset Conversion Tool

This tool converts Movie and Stock datasets defined in CSV files into WebDataset-compliant `.tar` archives.

## Usage

### 1. Movie Dataset Conversion

```bash
python tools/convert_scripts/convert_csv_to_webdataset.py \
  --dataset_type movie \
  --csv_path /mnt/data0/parth/movie_captions/captions_output/ \
  --output_dir /mnt/data0/parth/movie_captions_webdataset/ \
  --num_splits 10 \
  --split_index 0
```

- **`num_splits`**: Divides the total dataset into 10 equal parts.
- **`split_index`**: Processes the first part (0). To process the next part, change this to 1, then 2, etc.

### 2. Stock Dataset Conversion

```bash
python tools/convert_scripts/convert_csv_to_webdataset.py \
  --dataset_type stock \
  --csv_path /mnt/data0/parth/prompts/image_caption_map.csv \
  --output_dir /mnt/data0/stock_webdataset/ \
  --num_splits 5 \
  --split_index 0
```

- **`dataset_type`**: `stock` (reads captions from text files referenced in CSV).
- **`csv_path`**: Path to the stock dataset mapping CSV.
