
import os
from huggingface_hub import snapshot_download

def download_dataset():
    repo_id = "ma-xu/fine-t2i"
    local_dir = "/mnt/data0/parth/fine_t2i_dataset"
    
    print(f"Starting download of {repo_id} to {local_dir}...")
    
    # Ensure directory exists
    os.makedirs(local_dir, exist_ok=True)
    
    # Download with resume capability
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        local_dir_use_symlinks=False, # To have actual files, not symlinks
        resume_download=True,
        max_workers=8 # Adjust based on bandwidth/CPU
    )
    
    print(f"Download complete! Dataset stored in {local_dir}")

if __name__ == "__main__":
    download_dataset()
