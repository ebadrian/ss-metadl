from typing import Optional
import torch

def gaussian_label_propagate(x_supp: torch.Tensor, x_query: torch.Tensor, label_one_hot_supp: torch.Tensor, distance_scaler: Optional[torch.Tensor] = None, epsilon=0.01):
    num_l = x_supp.size(0)
    num_u = x_query.size(0)
    if distance_scaler is not None:
        x_query /= distance_scaler.unsqueeze(0)
        x_supp /= distance_scaler.unsqueeze(0)
    # w = e ^ (-|~|^2_2)
    Wuall = torch.sum((x_query[:,None,:] - torch.cat([x_supp, x_query])[None,:,:]) ** 2, dim=2)
    # Wuall = torch.exp(-x_query @ torch.cat([x_supp, x_query], dim=0).transpose(0, 1))
    # D = \sum w
    Duall = torch.sum(Wuall, dim=1, keepdim=True)
    # P = D^-1 @ W
    Puall = Wuall / Duall
    # P = P * (1-eps) + eps * U
    Prevised = Puall * (1-epsilon) + epsilon / (num_l + num_u)
    Puu = Prevised[:, num_l:]
    Pul = Prevised[:, :num_l]
    # Fu = (I - Puu)^-1 @ Pul @ Fl
    labelu = torch.inverse((torch.eye(num_u).to(x_supp.device)) - Puu) @ Pul @ label_one_hot_supp.float()
    return labelu