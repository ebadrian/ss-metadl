import torch
import numpy as np
import torch.nn.functional as F
from src.utils.embedding_propagation import embedding_propagation as ep

METRICS = {
    'cosine': lambda gallery, query: 1. - F.cosine_similarity(query[:, None, :], gallery[None, :, :], dim=2),
    'euclidean': lambda gallery, query: ((query[:, None, :] - gallery[None, :, :]) ** 2).sum(2),
    'l1': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=1, dim=2),
    'l2': lambda gallery, query: torch.norm((query[:, None, :] - gallery[None, :, :]), p=2, dim=2),
}

def normalize(emb):
    emb = emb / emb.norm(dim=1)[:,None]
    return emb

def QRreduction(datas):
    # datas: ((shot+query)*way) * dim.
    # return ndatas: ((shot+query)*way) * ((shot+query)*way)
    ndatas = torch.qr(datas.permute(1,0)).R 
    ndatas = ndatas.permute(1,0)
    return ndatas

# def power_transform(emb, beta=0.5):
#     # Power transform
#     emb = torch.pow(emb+1e-6, beta)
#     emb = emb / torch.norm(emb, p=2, dim=1).unsqueeze(1)
#     return emb

def power_transform(emb, beta=0.5, requires_qr=False):
    # Power transform
    emb = torch.pow(emb+1e-6, beta)
    if requires_qr:
        emb = QRreduction(emb)  # TODO: check if QR reduction helps
    emb = emb / torch.norm(emb, p=2, dim=1).unsqueeze(1)
    return emb

def embeding_propagation(query, supp, alpha=0.5, rbf_scale=1, norm_prop=False):
    x = torch.cat([query, supp], dim=0)
    x = ep(x, alpha, rbf_scale, norm_prop)
    return x[:len(query)], x[len(query):]

def shift_compensation(query, supp):
    return query + (supp.mean(dim=0) - query.mean(dim=0))[None,:], supp

def proto_rectification(query, supp, distance='cosine'):
    # we rectify prototype according to the predicted labels
    x = torch.cat([supp, query], dim=0)
    distance = METRICS['cosine'](supp, x)
    predict = torch.argmin(distance, dim=1)
    cos_sim = F.cosine_similarity(x[:, None, :], supp[None, :, :], dim=2)
    cos_sim = 10 * cos_sim
    W = F.softmax(cos_sim,dim=1)
    gallery_list = [(W[predict==i,i].unsqueeze(1)*x[predict==i]).mean(0,keepdim=True) for i in predict.unique()]
    supp = torch.cat(gallery_list,dim=0)
    return query, supp
