
import torch
import zipfile
from pathlib import Path
import os
import requests
from tqdm import tqdm

def download_coco128(dir='.'):
    # Download COCO128 dataset (128 images)
    dir = Path(dir)
    url = 'https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip'
    f = dir / 'coco128.zip'
    
    if (dir / 'coco128').exists():
        print("coco128 directory already exists.")
        return

    print(f'Downloading {url} to {f}...')
    response = requests.get(url, stream=True)
    total_size_in_bytes = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    
    with open(f, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()

    print(f'Unzipping {f}...')
    with zipfile.ZipFile(f, 'r') as zip_ref:
        zip_ref.extractall(dir)
        
    print('Dataset ready!')

if __name__ == "__main__":
    download_coco128()
