from huggingface_hub import snapshot_download
import os

DATASET_REPO = "ESA-PhiLab-Edge/OEOBench-Burnt_Area_Dataset"
OUTPUT_DIR = "/local/s3167445/datasets/OEOBench-Burnt_Area_Dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Downloading dataset: {DATASET_REPO}")

snapshot_download(
    repo_id=DATASET_REPO,
    repo_type="dataset",
    local_dir=OUTPUT_DIR,
    local_dir_use_symlinks=False,  # ensures full copy (good for clusters / archives)
    resume_download=True
)

print("Download complete!")
print(f"Dataset saved to: {OUTPUT_DIR}")
