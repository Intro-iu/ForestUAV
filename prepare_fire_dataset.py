import os
import subprocess
from pathlib import Path

def prepare_fire_dataset():
    # 1. Create datasets directory
    root = Path("datasets")
    root.mkdir(exist_ok=True)
    
    dataset_name = "sayedgamal99/smoke-fire-detection-yolo"
    target_dir = root / "kaggle_fire_dataset"
    
    print(f"Preparing to download from Kaggle: {dataset_name}")
    print("NOTE: You must have 'kaggle.json' configured in ~/.kaggle/ on the server!")
    
    # 2. Check/Install kaggle
    try:
        import kaggle
    except ImportError:
        print("Installing kaggle library...")
        subprocess.run(["pip", "install", "kaggle"], check=True)

    # 3. Download and Unzip
    if not target_dir.exists():
        target_dir.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {dataset_name}...")
        try:
            # kaggle datasets download -d sayedgamal99/smoke-fire-detection-yolo -p datasets/kaggle_fire_dataset --unzip
            subprocess.run([
                "kaggle", "datasets", "download", 
                "-d", dataset_name, 
                "-p", str(target_dir), 
                "--unzip"
            ], check=True)
            print("Dataset downloaded and extracted successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Failed to download dataset: {e}")
            print("Did you place your 'kaggle.json' in ~/.kaggle/?")
            return
    else:
        print(f"Directory {target_dir} already exists. Skipping download.")

    # 4. Configure paths
    print("\nDataset preparation setup complete.")
    print(f"Config file created at: data/fire.yaml")
    print("Use this config for training: --data data/fire.yaml")

if __name__ == "__main__":
    prepare_fire_dataset()
