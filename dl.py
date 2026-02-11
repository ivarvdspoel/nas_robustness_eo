import os
import requests
from huggingface_hub import list_repo_files
from tqdm import tqdm # Run 'pip install tqdm' for a progress bar

# --- CONFIGURATION ---
TOKEN = "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"
REPO_ID = "xiang709/REOBench"
LOCAL_DIR = "/local/s3167445/reobench_data"

def download_full_dataset():
    headers = {"Authorization": f"Bearer {TOKEN}"}
    os.makedirs(LOCAL_DIR, exist_ok=True)

    print(f"📡 Fetching complete file list for {REPO_ID}...")
    try:
        all_files = list_repo_files(repo_id=REPO_ID, repo_type="dataset", token=TOKEN)
        # Filter to keep only the data files
        files_to_download = [f for f in all_files if not f.startswith(".") and f != "README.md"]
        
        print(f"✅ Found {len(files_to_download)} files. Starting batch download...")

        for filename in files_to_download:
            target_path = os.path.join(LOCAL_DIR, filename)
            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            # Skip already downloaded files
            if os.path.exists(target_path):
                # Basic check: if file is 0 bytes, it's a failed download, don't skip
                if os.path.getsize(target_path) > 0:
                    print(f"⏩ Skipping {filename} (already exists)")
                    continue

            url = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{filename}"
            
            # Use stream=True to handle large remote sensing files
            response = requests.get(url, headers=headers, stream=True)
            
            if response.status_code == 200:
                total_size = int(response.headers.get('content-length', 0))
                
                # Progress bar for the current file
                progress = tqdm(total=total_size, unit='iB', unit_scale=True, desc=f"📥 {filename}")
                
                with open(target_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024*1024): # 1MB chunks
                        if chunk:
                            f.write(chunk)
                            progress.update(len(chunk))
                progress.close()
            else:
                print(f"❌ Failed to download {filename}: Status {response.status_code}")

        print(f"\n✨ SUCCESS: Entire dataset processed to {LOCAL_DIR}")

    except Exception as e:
        print(f"❌ Critical error: {e}")

if __name__ == "__main__":
    download_full_dataset()
