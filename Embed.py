import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip


device = "mps" if torch.backends.mps.is_available() else "cpu"
image_dir = "shopee-product-matching/train_sample_images"
csv_path = "shopee-product-matching/train_sample.csv"
output_features_path = "shopee-product-matching/image_embeddings.npy"
output_title_path = "shopee-product-matching//image_title.npy"
output_ids_path = "shopee-product-matching/image_ids.txt"

model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

df = pd.read_csv(csv_path)
titles = df['title'].tolist()
titles = [title[:75] for title in titles]  # truncate overly long ones
tokenized_titles = clip.tokenize(titles).to(device)

with torch.no_grad():
    title_embeddings = model.encode_text(tokenized_titles)
    title_embeddings = title_embeddings / title_embeddings.norm(dim=-1, keepdim=True)

# Save
np.save(output_title_path, title_embeddings.cpu().numpy())
title_embeddings_np = title_embeddings.cpu().numpy()


image_embeddings = []
image_ids = []

with torch.no_grad():
    for row in tqdm(df.itertuples(), total=len(df)):
        image_path = os.path.join(image_dir, row.image)
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            embedding = model.encode_image(image_tensor)
            embedding /= embedding.norm(dim=-1, keepdim=True)  # normalize
            image_embeddings.append(embedding.cpu().numpy().squeeze())
            image_ids.append(row.posting_id)
        except Exception as e:
            print(f"Error with image {row.image}: {e}")

image_embeddings = np.stack(image_embeddings)
np.save(output_features_path, image_embeddings)

with open(output_ids_path, "w") as f:
    for pid in image_ids:
        f.write(f"{pid}\n")

print(f"Saved {len(image_embeddings)} embeddings to {output_features_path}")

final_embeddings = (image_embeddings + title_embeddings_np) / 2
final_embeddings /= np.linalg.norm(final_embeddings, axis=1, keepdims=True)
np.save("shopee-product-matching/final_embeddings.npy", final_embeddings)
