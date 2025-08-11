# dataset_pk.py
from PIL import Image
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
import random, os

class ShopeeImgDataset(Dataset):
    def __init__(self, df, img_root="shopee-product-matching/train_images", transform=None):
        self.df = df.reset_index(drop=True)
        self.paths = self.df["image"].tolist()
        self.labels = self.df["label_group"].tolist()
        self.transform = transform
        self.root = img_root

    def __len__(self): 
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.paths[i])).convert("RGB")
        if self.transform: 
            img = self.transform(img)
        return img, self.labels[i], i  # (tensor, int label, local index)

class PKSampler(Sampler):
    """Yields indices to form batches with P classes and K samples per class."""
    def __init__(self, labels, P=16, K=4):
        super().__init__(None)
        self.P, self.K = P, K
        self.by_cls = defaultdict(list)
        for idx, c in enumerate(labels): 
            self.by_cls[c].append(idx)
        self.classes = list(self.by_cls.keys())

    def __iter__(self):
        while True:
            chosen = random.sample(self.classes, self.P)
            batch = []
            for c in chosen:
                # choices allows repeat if class has < K images
                batch += random.choices(self.by_cls[c], k=self.K)
            yield from batch

    def __len__(self):
        # Infinite sampler; DataLoader will just take batch_size each step
        return 10**9
