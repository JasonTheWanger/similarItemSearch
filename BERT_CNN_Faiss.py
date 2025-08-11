import numpy as np
import pandas as pd
import faiss

embeddings = np.load("shopee-product-matching/v1_bert_cnn_embeddings.npy")
output_submission_path = "v1_bert_cnn_submission.csv"
with open("shopee-product-matching/bert_cnn_ids.txt") as f:
    posting_ids = [line.strip() for line in f]

embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

K = 50
distances, indices = index.search(embeddings, K)

similarities = distances.flatten()
threshold = similarities.mean() + similarities.std() * 1.5

predictions = []

for i in range(len(indices)):
    matched = [
        posting_ids[j]
        for j, sim in zip(indices[i], distances[i])
        if sim > threshold
        ]
    
    if posting_ids[i] not in matched:
        matched.insert(0, posting_ids[i])
    
    predictions.append(" ".join(matched))

submission_df = pd.DataFrame({
    "posting_id" : posting_ids,
    "matches" : predictions
})

submission_df.to_csv(output_submission_path, index=False)
print("✅ Saved submission file.")
