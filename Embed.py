import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
import clip

# ====== CONFIGURATION ======
device = "mps" if torch.backends.mps.is_available() else "cpu"
image_dir = "shopee-product-matching/train_images"
csv_path = "shopee-product-matching/train.csv"
output_features_path = "shopee-product-matching/v3_full_image_embeddings.npy"
output_title_path = "shopee-product-matching/v3_full_image_title.npy"
output_ids_path = "shopee-product-matching/v3_full_image_ids.txt"
output_final_path = "shopee-product-matching/v3_full_final_embeddings.npy"
batch_size = 256  # Safe for MPS memory

# ====== LOAD MODEL ======
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# ====== LOAD CSV ======
df = pd.read_csv(csv_path)
titles = [title[:75] for title in df['title'].tolist()]  # Truncate to avoid CLIP errors

# ====== TITLE EMBEDDING (BATCHED) ======
title_embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(titles), batch_size), desc="Embedding Titles"):
        batch = titles[i:i+batch_size]
        tokens = clip.tokenize(batch).to(device)
        batch_embed = model.encode_text(tokens)
        batch_embed /= batch_embed.norm(dim=-1, keepdim=True)
        title_embeddings.append(batch_embed.cpu().numpy())

title_embeddings_np = np.vstack(title_embeddings)
np.save(output_title_path, title_embeddings_np)

# ====== IMAGE EMBEDDING ======
image_embeddings = []
image_ids = []


BATCH_SIZE = 32
image_batch = []
id_batch = []

with torch.no_grad():
    for row in tqdm(df.itertuples(), total=len(df), desc="Embedding Images"):
        try:
            image_path = os.path.join(image_dir, row.image)
            image = Image.open(image_path).convert("RGB")
            image_tensor = preprocess(image).unsqueeze(0)
            image_batch.append(image_tensor)
            id_batch.append(row.posting_id)

            if (len(image_batch) == BATCH_SIZE):
                batch_tensor = torch.cat(image_batch).to(device)
                batch_embed =  model.encode_image(batch_tensor)
                batch_embed = batch_embed / batch_embed.norm(dim = -1, keepdim = True)
                image_embeddings.extend(batch_embed.cpu().numpy())
                image_ids.extend(id_batch)
                image_batch = []
                id_batch = []

        except Exception as e:
            print(f"Error with image {row.image}: {e}")

    if image_batch:
        batch_tensor = torch.cat(image_batch).to(device)
        batch_embed =  model.encode_image(batch_tensor)
        batch_embed = batch_embed / batch_embed.norm(dim = -1, keepdim = True)
        image_embeddings.extend(batch_embed.cpu().numpy())
        image_ids.extend(id_batch)

image_embeddings_np = np.stack(image_embeddings)
np.save(output_features_path, image_embeddings_np)

with open(output_ids_path, "w") as f:
    for pid in image_ids:
        f.write(f"{pid}\n")

print(f"✅ Saved {len(image_embeddings_np)} image embeddings to {output_features_path}")

# ====== FINAL MULTIMODAL EMBEDDING ======
final_embeddings = 0.3 * title_embeddings_np + 0.7 * image_embeddings_np
final_embeddings /= np.linalg.norm(final_embeddings, axis=1, keepdims=True)
np.save(output_final_path, final_embeddings)

print(f"✅ Saved final multimodal embeddings to {output_final_path}")
