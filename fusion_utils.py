# fusion_utils.py
import torch

@torch.no_grad()
def cosine_topk_from_embs(embs, topk=50):
    sims = embs @ embs.T
    sims.fill_diagonal_(0.0)
    vals, idxs = torch.topk(sims, k=min(topk, sims.shape[0]-1), dim=1)
    return vals.cpu(), idxs.cpu()

def fuse_vals_by_image_candidates(vals_img, idxs_img, vals_txt, idxs_txt, w_img=0.8):
    """
    Align text sims to the image candidate list row-wise, then weighted sum.
    vals_*, idxs_*: [N, K] tensors
    """
    N, K = idxs_img.shape
    fused_vals = torch.zeros_like(vals_img)
    for i in range(N):
        # map text neighbors -> score
        tmap = { int(j): float(s) for j, s in zip(idxs_txt[i].tolist(), vals_txt[i].tolist()) }
        aligned = [ tmap.get(int(j), 0.0) for j in idxs_img[i].tolist() ]
        fused_vals[i] = w_img * vals_img[i] + (1 - w_img) * torch.tensor(aligned)
    return fused_vals, idxs_img
