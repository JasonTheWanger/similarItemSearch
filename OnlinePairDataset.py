import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

class OnlinePairDataset(Dataset):
    def __init__(self, embedding_path, id_file, train_csv,
                 negatives_per_anchor=1, seed=42):
        self.embeddings = np.load(embedding_path).astype("float32")
        self.posting_ids = [line.strip() for line in open(id_file)]
        self.df = pd.read_csv(train_csv)

        self.df["label_group_encoded"] = LabelEncoder().fit_transform(self.df["label_group"])
        self.label_map = dict(zip(self.df["posting_id"], self.df["label_group_encoded"]))
        self.labels = np.array([self.label_map[pid] for pid in self.posting_ids], dtype=np.int64)

        # buckets: label -> indices
        from collections import defaultdict
        buckets = defaultdict(list)
        for i, lab in enumerate(self.labels):
            buckets[lab].append(i)
        self.buckets = {k: np.array(v, dtype=np.int64) for k, v in buckets.items()}

        self.negatives_per_anchor = negatives_per_anchor
        self.rng = np.random.default_rng(seed)

        # safety
        if len(self.posting_ids) != self.embeddings.shape[0]:
            raise ValueError("IDs and embeddings row count mismatch.")

    def __len__(self):
        # one anchor per row; __getitem__ decides pos or neg on the fly
        return len(self.posting_ids)

    def sample_positive(self, i):
        lab = self.labels[i]
        pool = self.buckets[lab]
        if len(pool) <= 1:  # no other positives
            # fallback: return self (label=1, won’t help much but keeps shape)
            return i
        j = i
        # pick a different index with the same label
        while j == i:
            j = int(self.rng.choice(pool))
        return j

    def sample_negative(self, i):
        lab = self.labels[i]
        j = i
        # rejection sample until different label
        while self.labels[j] == lab:
            j = int(self.rng.integers(0, len(self.labels)))
        return j

    def __getitem__(self, i):
        # 50/50 chance to emit a positive or a negative pair
        if self.rng.random() < 0.5:
            j = self.sample_positive(i)
            label = 1.0
        else:
            j = self.sample_negative(i)
            label = 0.0

        return (
            self.embeddings[i],   # emb1
            self.embeddings[j],   # emb2
            label                 # float label for contrastive loss
        )
