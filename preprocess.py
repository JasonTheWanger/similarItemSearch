import shutil
import clip
import torch
from PIL import Image
from tqdm import tqdm
import os
import pandas as pd


device = "mps" if torch.backends.mps.is_available() else "cpu"
csv_path = "shopee-product-matching/train.csv"
image_dir = "shopee-product-matching/train_images"
sample_csv_path = "shopee-product-matching/train_sample.csv"
sample_image_dir = "shopee-product-matching/train_sample_images"

df = pd.read_csv(csv_path)

df_sample = df.sample(n=3000, random_state=42).reset_index(drop=True)

df_sample.to_csv(sample_csv_path, index=False)
print(f"Saved 3000 sample rows to {sample_csv_path}")

os.makedirs(sample_image_dir, exist_ok=True)

for image_name in tqdm(df_sample["image"]):
    src = os.path.join(image_dir, image_name)
    dst = os.path.join(sample_image_dir, image_name)
    shutil.copy(src, dst)

print(f"Copied 3000 images to {sample_image_dir}")