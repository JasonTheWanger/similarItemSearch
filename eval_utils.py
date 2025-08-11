# eval_utils.py
import torch
import numpy as np

def build_true_sets(label_groups):
    # ensure plain ints (not tensors) for hashing/grouping
    label_groups = [int(g) for g in label_groups]
    idx_by_group = {}
    for i, g in enumerate(label_groups):
        idx_by_group.setdefault(g, []).append(i)
    true_sets = []
    for i, g in enumerate(label_groups):
        s = set(idx_by_group[g])
        s.discard(i)
        true_sets.append(s)
    return true_sets

def micro_f1_kaggle(pred_sets, true_sets):
    tp = fp = fn = 0
    for p, t in zip(pred_sets, true_sets):
        tp += len(p & t)
        fp += len(p - t)
        fn += len(t - p)
    return 2 * tp / (2 * tp + fp + fn + 1e-9)

@torch.no_grad()
def cosine_topk(embs, topk=50):
    sims = embs @ embs.T
    sims.fill_diagonal_(0.0)
    vals, idxs = torch.topk(sims, k=min(topk, sims.shape[0]-1), dim=1)
    return vals.cpu(), idxs.cpu()

def sweep_threshold(vals, idxs, true_sets, sweep=np.linspace(0.0, 0.95, 40)):
    best_f1, best_th = -1.0, float(sweep[0])
    for th in sweep:
        pred_sets = [ set(idxs[i, (vals[i] >= th)].tolist()) for i in range(idxs.shape[0]) ]
        f1 = micro_f1_kaggle(pred_sets, true_sets)
        if f1 >= best_f1:            # note: >= so ties pick higher th later in sweep
            best_f1, best_th = float(f1), float(th)
    return best_f1, best_th

def apply_threshold(vals, idxs, th):
    return [ set(idxs[i, (vals[i] >= th)].tolist()) for i in range(idxs.shape[0]) ]
