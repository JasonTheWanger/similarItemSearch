from OnlinePairDataset import OnlinePairDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import Adam
import torch

from SiameseModel import SiameseNet, ContrastiveLoss

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = SiameseNet().to(device)
criterion = ContrastiveLoss(margin=1.0)
optimizer = Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

dataset = OnlinePairDataset(
    embedding_path="shopee-product-matching/combined_embeddings.npy",
    id_file="shopee-product-matching/bert_cnn_ids.txt",
    train_csv="shopee-product-matching/train.csv",
    # if your OnlinePairDataset uses pos_prob, remove negatives_per_anchor
    seed=42
)

loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

EPOCHS = 15
for epoch in range(EPOCHS):
    model.train()
    total = 0.0

    for emb1, emb2, label in loader:
        
        emb1  = emb1.to(device, dtype=torch.float32)
        emb2  = emb2.to(device, dtype=torch.float32)
        label = label.to(device, dtype=torch.float32)

        z1, z2 = model(emb1, emb2)
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)

        loss = criterion(z1, z2, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()

    torch.save(model.state_dict(), f"siamese_model_epoch{epoch}.pth")
    print(f"Epoch {epoch}: Avg Loss = {total/len(loader):.4f}")
