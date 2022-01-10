import torch
import torch.nn as nn
import torch.nn.functional as F

class PairwiseDistanceScaler(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.dim = dim
        self.core = nn.Sequential(
            nn.Linear(self.dim * 2, 128),
            nn.BatchNorm1d(128, momentum=1, affine=True),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        #self.alpha = nn.Parameter(torch.Tensor(1))
        #nn.init.constant_(self.alpha, 0)
        #self.beta = nn.Parameter(torch.Tensor(1))
        #nn.init.constant_(self.beta, 0)

    def forward(self, query, supp):
        q, s = query.size(0), supp.size(0)
        matrix = torch.cat([query[:, None, :].repeat(1, s, 1), supp[None,:,:].repeat(q, 1, 1)], dim=2)
        matrix = self.core(matrix.reshape(q * s, -1)).view(q, s)
        return matrix #torch.exp(self.alpha) * matrix + torch.exp(self.beta)