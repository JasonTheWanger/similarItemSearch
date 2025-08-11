import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNet(nn.Module):
    
    def __init__(self, embedding_dim = 512):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
    
    def forward(self, x1, x2):
        out1 = self.projector(x1)
        out2 = self.projector(x2)
        return out1, out2
    
class ContrastiveLoss(nn.Module):

    def __init__(self, margin = 1):
        super().__init__()
        self.margin = margin
    
    def forward(self, x1, x2, label):
        dists = F.pairwise_distance(x1, x2)
        loss = label * torch.pow(dists, 2) + (1 - label) * torch.pow(torch.clamp(self.margin - dists, min=0.0), 2)
        return loss.mean()
    
