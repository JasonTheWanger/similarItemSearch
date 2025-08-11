# sbert_utils.py
import torch
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_sbert(model_name="all-MiniLM-L6-v2", device="cpu"):
    m = SentenceTransformer(model_name, device=device)
    m.max_seq_length = 64
    return m

@torch.no_grad()
def encode_titles(df: pd.DataFrame, model: SentenceTransformer, batch_size=256, text_col="title"):
    texts = (df[text_col].fillna("").astype(str)).tolist()
    embs = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True, normalize_embeddings=True)
    return embs  # torch.Tensor [N, D], L2-normalized
