import numpy as np
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader

class ShopeePairDataset(Dataset):
    def __init__(self, pair_csv, embedding_path, id_file):
        self.pairs = pd.read_csv(pair_csv)
        self.embedding = np.load(embedding_path).astype('float32')
        self.id2idx = {pid.strip() : i for i, pid in enumerate(open(id_file))}
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        row = self.pairs.iloc[index]
        id1, id2, label = row['id1'], row['id2'], row['label']
        emb1 = self.embedding[self.id2idx[id1]]
        emb2 = self.embedding[self.id2idx[id2]]
        return torch.tensor(emb1), torch.tensor(emb2), torch.tensor(label, dtype = torch.float32)
    
