import numpy as np
import pandas as pd
import faiss

embeddings = np.load("shopee-product-matching/final_embeddings.npy")

with open("shopee-product-matching/image_ids.txt") as f:
    image_ids = [line.strip() for line in f]

embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

K = 50  
distances, indices = index.search(embeddings, K)

predictions = []

for i in range(len(indices)):
    matched = [
        image_ids[j]
        for j, sim in zip(indices[i], distances[i])
        if sim >= 0.7  
    ]

    if image_ids[i] not in matched:
        matched.insert(0, image_ids[i])

    predictions.append(" ".join(matched))

submission_df = pd.DataFrame({
    "posting_id": image_ids,
    "matches": predictions
})

submission_df.to_csv("submission.csv", index=False)
print("✅ Saved submission.csv with", len(submission_df), "rows.")
