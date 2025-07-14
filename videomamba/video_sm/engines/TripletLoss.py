import torch.nn as nn
from torch.nn import functional as F

class TripletLoss(nn.Module):
    def __init__(self):
        super(TripletLoss, self).__init__()
        self.margin = 0.2

    def forward(self, anchor, positive, negative):
        # pos_dist = (anchor - positive).pow(2).sum(1)
        # neg_dist = (anchor - negative).pow(2).sum(1)
        pos_dist = anchor - positive
        neg_dist = anchor - negative
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()
