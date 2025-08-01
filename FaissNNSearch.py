import numpy as np
import pandas as pd
import faiss

embeddings = np.load("shopee-product-matching/v1_full_final_embeddings.npy")
output_path = "v1_full_submission.csv"

with open("shopee-product-matching/v1_full_image_ids.txt") as f:
    image_ids = [line.strip() for line in f]

embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

K = 50  
distances, indices = index.search(embeddings, K)

predictions = []

for i in range(len(indices)):
    sims = distances[i]
    idxs = indices[i]
    query_id = image_ids[i]

    threshold = sims.mean() + sims.std() * 1.44

    matched = [
        image_ids[j]
        for j, sim in zip(idxs, sims)
        if sim >= threshold
    ]

    if query_id not in matched:
        matched.insert(0, query_id)

    predictions.append(" ".join(matched))

submission_df = pd.DataFrame({
    "posting_id": image_ids,
    "matches": predictions
})

submission_df.to_csv(output_path, index=False)
print("✅ Saved submission.csv with", len(submission_df), "rows using dynamic thresholding.")
