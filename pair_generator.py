import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# ---------------- Config ----------------
NEGATIVES_PER_ANCHOR = 2
RNG_SEED = 42
# ----------------------------------------

rng = np.random.default_rng(RNG_SEED)

# ---------- Load & fuse embeddings (optional but kept so you still save them) ----------
clip_embeddings = np.load('shopee-product-matching/v1_full_final_embeddings.npy').astype('float32')
bert_cnn_embeddings = np.load('shopee-product-matching/v1_bert_cnn_embeddings.npy').astype('float32')

clip_embeddings = clip_embeddings / np.linalg.norm(clip_embeddings, axis=1, keepdims=True)
bert_cnn_embeddings = bert_cnn_embeddings / np.linalg.norm(bert_cnn_embeddings, axis=1, keepdims=True)

combined_embed = 0.5 * clip_embeddings + 0.5 * bert_cnn_embeddings
combined_embed = combined_embed / np.linalg.norm(combined_embed, axis=1, keepdims=True)
np.save('shopee-product-matching/combined_embeddings.npy', combined_embed)

# ---------- IDs & labels ----------
posting_ids = [line.strip() for line in open('shopee-product-matching/bert_cnn_ids.txt')]
df = pd.read_csv('shopee-product-matching/train.csv')

df['label_group_encoded'] = LabelEncoder().fit_transform(df['label_group'])
label_map = dict(zip(df['posting_id'], df['label_group_encoded']))

# Safety check: ensure ordering aligns (optional)
if len(posting_ids) != combined_embed.shape[0]:
    raise ValueError("posting_ids length does not match embeddings rows.")

# Precompute label arrays / index buckets
labels_array = np.array([label_map[pid] for pid in posting_ids])

from collections import defaultdict
label_to_indices = defaultdict(list)
for idx, pid in enumerate(posting_ids):
    label_to_indices[labels_array[idx]].append(idx)
label_to_indices = {k: np.array(v, dtype=np.int64) for k, v in label_to_indices.items()}

pairs = []

for i, pid in tqdm(enumerate(posting_ids), total=len(posting_ids)):
    anchor_label = labels_array[i]

    # ---------- POSITIVES ----------
    same_idxs = label_to_indices[anchor_label]
    if same_idxs.size > 1:
        pos_idxs = same_idxs[same_idxs != i]
        for j in pos_idxs:
            pairs.append((pid, posting_ids[j], 1))

    # ---------- RANDOM NEGATIVES ----------
    negs_added = 0
    # simple rejection sampling: draw until you get a different label
    while negs_added < NEGATIVES_PER_ANCHOR:
        j = int(rng.integers(0, len(posting_ids)))
        if labels_array[j] != anchor_label:
            pairs.append((pid, posting_ids[j], 0))
            negs_added += 1

# ---------- Save ----------
pair_df = pd.DataFrame(pairs, columns=['id1', 'id2', 'label'])
out_path = 'shopee-product-matching/pairs_random_negative.csv'
pair_df.to_csv(out_path, index=False)
print(f'✅ Saved {len(pair_df)} pairs to {out_path}')
