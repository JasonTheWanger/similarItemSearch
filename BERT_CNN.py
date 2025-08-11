import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch 
import torch.nn as nn
from torchvision import models, transforms
from transformers import BertTokenizer, BertModel

device = torch.device("cpu")

csv_path = "shopee-product-matching/train.csv"
image_dir = "shopee-product-matching/train_images"
output_text_path = "shopee-product-matching/v1_bert_cnn_text.npy"
output_image_path = "shopee-product-matching/v1_bert_cnn_image.npy"
output_combined_path = "shopee-product-matching/v1_bert_cnn_embeddings.npy"

df = pd.read_csv(csv_path)

titles = df['title'].tolist()
image_paths = [os.path.join(image_dir, name) for name in df['image']]

bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
cnn = models.resnet50(pretrained = True)
cnn.fc = nn.Identity()
cnn.eval().to(device)

transform = transforms.Compose([ transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

text_embeddings = []
image_embeddings = []

print("Encoding titles:")
with torch.no_grad():
    for i in tqdm(range(0, len(titles), 32)):
        batch = titles[i:i+32]
        encoded = bert_tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(device)
        output = bert_model(**encoded)
        cls_embed = output.last_hidden_state[:, 0, :]
        text_embeddings.append(cls_embed.cpu().numpy())

text_embeddings = np.vstack(text_embeddings)
np.save(output_text_path, text_embeddings)

print("Encoding images:")
with torch.no_grad():
    for img_path in tqdm(image_paths):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = transform(img).unsqueeze(0).to(device)
            embed = cnn(tensor).squeeze()
            image_embeddings.append(embed.cpu().numpy())
        except Exception as e:
            print(f"Error with image {img_path}: {e}")

image_embeddings = np.vstack(image_embeddings)
np.save(output_image_path, image_embeddings)

text_tensor = torch.tensor(text_embeddings, dtype=torch.float32)
image_tensor = torch.tensor(image_embeddings, dtype=torch.float32)

text_proj = nn.Linear(768, 512).to(device).eval()
image_proj = nn.Linear(2048, 512).to(device).eval()

with torch.no_grad():
    text_out = text_proj(text_tensor.to(device))
    image_out = image_proj(image_tensor.to(device))

    combined = text_out * 0.5 + image_out * 0.5
    combined = combined / combined.norm(dim=1, keepdim=True)

combined_embeddings = combined.cpu().numpy()
np.save(output_combined_path, combined_embeddings)

with open("shopee-product-matching/bert_cnn_ids.txt", "w") as f:
    for pid in df['posting_id']:
        f.write(f"{pid}\n")

print("✅ Embeddings saved")
