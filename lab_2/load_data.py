import kagglehub

# Download latest version
path = kagglehub.dataset_download("zalando-research/fashionmnist")

print("Path to dataset files:", path)

import os
import shutil

# Copy dataset files into lab_2/data/ so the path is local and easy to find
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

for filename in os.listdir(path):
    src = os.path.join(path, filename)
    dst = os.path.join(DATA_DIR, filename)
    if not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"  Copied: {filename}")
    else:
        print(f"  Already exists: {filename}")

print("Local data directory:", DATA_DIR)