import os
import cv2
import numpy as np

def create_dummy_data():
    base_dir = "data/dummy"
    img_dir = os.path.join(base_dir, "images")
    label_dir = os.path.join(base_dir, "labels")
    
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    # Create valid absolute paths for the text files
    abs_base_dir = os.path.abspath(base_dir)
    abs_img_path = os.path.abspath(img_dir)
    
    train_txt = os.path.join(base_dir, "train.txt")
    val_txt = os.path.join(base_dir, "val.txt")
    test_txt = os.path.join(base_dir, "test.txt")
    
    train_files = []
    
    for i in range(4): # Create 4 dummy images
        img_name = f"dummy_{i}.jpg"
        img_path = os.path.join(img_dir, img_name)
        label_path = os.path.join(label_dir, f"dummy_{i}.txt")
        
        # Create image (black image)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        cv2.imwrite(img_path, img)
        
        # Create label (class 0, center x, center y, width, height)
        with open(label_path, "w") as f:
            f.write("0 0.5 0.5 0.2 0.2\n")
            
        train_files.append(os.path.join(abs_img_path, img_name))
        
    # Write train.txt
    with open(train_txt, "w") as f:
        f.write("\n".join(train_files))
        
    # Copy for val and test
    with open(val_txt, "w") as f:
        f.write("\n".join(train_files))
        
    with open(test_txt, "w") as f:
        f.write("\n".join(train_files))
        
    print("Dummy data created.")

if __name__ == "__main__":
    create_dummy_data()
