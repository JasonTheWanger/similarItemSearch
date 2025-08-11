# losses.py
import torch
import torch.nn as nn

class ArcMarginProduct(nn.Module):
    """
    ArcFace margin head:
    logits = s * cos(theta + m) for the target class, cos(theta) for others.
    Inputs must be L2-normalized.
    """
    def __init__(self, emb_dim, num_classes, s=30.0, m=0.2):
        super().__init__()
        self.W = nn.Parameter(torch.randn(num_classes, emb_dim))
        nn.init.xavier_uniform_(self.W)
        self.s, self.m = s, m

    def forward(self, x, labels):
        W = nn.functional.normalize(self.W, dim=1)
        x = nn.functional.normalize(x, dim=1)
        cos = torch.matmul(x, W.t()).clamp(-1 + 1e-7, 1 - 1e-7)
        # add margin to target
        theta = torch.acos(cos)
        target_cos = torch.cos(theta + self.m)
        onehot = torch.zeros_like(cos)
        onehot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = cos * (1 - onehot) + target_cos * onehot
        return logits * self.s
