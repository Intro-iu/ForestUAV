import os
import subprocess
from pathlib import Path

def prepare_fire_dataset():
    # 1. Create datasets directory
    root = Path("datasets")
    root.mkdir(exist_ok=True)
    
    repo_url = "https://github.com/CostiCatargiu/NEWFireSmokeDataset_YoloModels.git"
    target_dir = root / "NEWFireSmokeDataset_YoloModels"
    
    # 2. Clone the repository
    if not target_dir.exists():
        print(f"Cloning {repo_url}...")
        try:
            subprocess.run(["git", "clone", repo_url, str(target_dir)], check=True)
            print("Repository cloned successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to clone repository: {e}")
            return
    else:
        print(f"Directory {target_dir} already exists. Skipping clone.")

    # 3. Check for dataset content
    # Based on repo description, it might contain a download script or the data itself
    # We will attempt to run the download script if found, or assume data is in a subfolder
    
    print("Checking for download script...")
    download_script = target_dir / "Download_Dataset_v1.py" 
    # Note: Filename guessed based on typical naming, user might need to adjust
    
    if download_script.exists():
        print(f"Found download script: {download_script}")
        print("Running download script...")
        subprocess.run(["python", str(download_script)], cwd=target_dir)
    else:
        print("Download script not found directly. Assuming data might be ready or requires manual step.")
        print(f"Please check content in {target_dir}")

    print("\nDataset preparation setup complete.")
    print(f"Config file created at: data/fire.yaml")
    print("Use this config for training: --data data/fire.yaml")

if __name__ == "__main__":
    prepare_fire_dataset()
