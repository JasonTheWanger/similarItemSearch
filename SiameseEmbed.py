from SiameseModel import SiameseNet
import torch
import numpy as np
from sklearn.preprocessing import normalize

device = "mps" if torch.backends.mps.is_available() else "cpu"

original_embeddings = np.load("shopee-product-matching/combined_embeddings.npy").astype('float32')
num = input()
model = SiameseNet()
model.load_state_dict(torch.load(f"siamese_model_epoch{num}.pth"))
model.eval()
model.to(device)

with torch.no_grad():
    tensor_embeddings = torch.tensor(original_embeddings, device=device)
    projected = model.projector(tensor_embeddings).cpu().numpy()
    projected = normalize(projected, axis=1)

np.save("shopee-product-matching/siamese_projected_embeddings.npy", projected)
