import torch
import torch.nn.functional as F
import numpy as np
from src.utils.lshot import bound_update
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from src.utils.embedding_propagation import label_propagation
from src.utils.gaussian_label_propagation import gaussian_label_propagate
from .embedding_propagation import normalize, power_transform

def distance_label_propagation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    n = a.shape[0]
    m = b.shape[0]
    logits = -((a.unsqueeze(1).expand(n, m, -1) -
                b.unsqueeze(0).expand(n, m, -1))**2).sum(dim=2)
    return logits

def make_protomap(support_set, way):
    B, D = support_set.shape
    shot = B // way
    protomap = support_set.reshape(shot, way, D)
    protomap = protomap.mean(dim=0)

    return protomap


def flatten(sets):
    # flatten
    sets = torch.flatten(sets, start_dim=1)
    sets = F.normalize(sets)

    return sets

def add_query(support_set, query_set, prob, way):

    B, D = support_set.shape
    shot = B // way
    per_class = support_set.reshape(shot, way, D)

    protomap = []
    for i in range(way):
        ith_prob = prob[:, i].reshape(prob.size(0), 1)
        ith_map = torch.cat((per_class[:, i], query_set * ith_prob), dim=0)
        ith_map = torch.sum(ith_map, dim=0, keepdim=True) / (ith_prob.sum() + shot)
        protomap.append(ith_map)

    return torch.cat(protomap, dim=0)

def param_free_distance_cal(query, supp):
    query, supp = normalize(query), normalize(supp)
    logit = distance_label_propagation(query, supp)
    return F.softmax(logit, dim=-1)

def mct_label_propagation(query, supp, cal_distance=param_free_distance_cal, shot=1, iters=11):
    prob_list = []
    way = len(supp) // shot
    for iter in range(iters):
        # Make Protomap
        if iter == 0:
            protomap = make_protomap(supp, way)
        else:
            protomap = add_query(supp, query, prob_list[iter-1], way)

        prob = cal_distance(query, protomap)
        prob_list.append(prob)
    return torch.log(prob_list[-1] + 1e-6)

def gaussian_label_propagation(query, supp):
    return gaussian_label_propagate(supp, query, torch.eye(len(supp)).to(supp.device))

def cosine_distance_label_propagation(query, supp):
    distance = F.cosine_similarity(query[:,None,:], supp[None,:,:], dim=2)
    return distance

def create_affinity(X, knn):
    N, D = X.shape
    # print('Compute Affinity ')
    nbrs = NearestNeighbors(n_neighbors=knn).fit(X)
    dist, knnind = nbrs.kneighbors(X)

    row = np.repeat(range(N), knn - 1)
    col = knnind[:, 1:].flatten()
    data = np.ones(X.shape[0] * (knn - 1))
    W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=np.float)
    return W

def lshot_prediction(knn, lmd, X, unary):

    W = create_affinity(X, knn)
    l = bound_update(unary, W, lmd)
    return l

# laplacian
def laplacian_label_propagation(query, supp, knn=5, lmd=0.5):
    subtract = supp[:, None, :] - query[None, :, :]
    distance = (subtract ** 2).sum(2).detach().cpu().numpy()
    unary = distance.transpose()
    out = lshot_prediction(knn, lmd, query.detach().cpu().numpy(), unary)
    return np.eye(len(supp))[out]

def graph_label_propagation(query, supp, alpha=0.5, rbf_scale=1, norm_prop=False):
    way = len(supp)
    x = torch.cat([supp, query])
    lab = torch.cat([torch.arange(way), torch.ones(len(query)) * way]).long().to(supp.device)
    return label_propagation(x, lab, way, alpha=alpha, rbf_scale=rbf_scale, norm_prop=norm_prop, apply_log=True)[way:]

def map_label_propagation(query, supp, lam=10, alpha=0.2, n_epochs=20):
    way = len(supp)
    model = GaussianModel(way, lam, supp.device)
    model.initFromLabelledDatas(supp)
    
    optim = MAP(alpha)

    prob, _ = optim.loop(model, query, n_epochs, None)
    return torch.log(prob + 1e-6)

class GaussianModel():
    def __init__(self, n_ways, lam, device):
        self.n_ways = n_ways
        self.device = device
        self.lam = lam
        
    def clone(self):
        other = GaussianModel(self.n_ways, self.lam, self.device)
        other.mus = self.mus.clone()
        return self

    def to(self, device):
        self.mus = self.mus.to(device)
        
    def initFromLabelledDatas(self, shot_data):  # TODO: support when shot is larger than 1.
        # self.mus = shot_data.detach().clone()
        # self.mus_origin = shot_data.detach().clone()
        self.mus = shot_data
        self.mus_origin = shot_data

    def updateFromEstimate(self, estimate, alpha):   
        
        Dmus = estimate - self.mus
        self.mus = self.mus + alpha * (Dmus)

    def compute_optimal_transport(self, M, r, c, epsilon=1e-6):
        
        r = r.to(self.device)
        c = c.to(self.device)
        # M: queries * supps
        n, m = M.shape
        P = torch.exp(- self.lam * M)
        P = P / P.view(-1).sum()

        u = torch.zeros(n).to(self.device)
        maxiters = 1000
        iters = 1
        # normalize this matrix
        while torch.max(torch.abs(u - P.sum(1))) > epsilon:
            u = P.sum(1)
            P = P * (r / u).view((-1, 1))
            P = P * (c / P.sum(0)).view((1, -1))
            if iters == maxiters:
                break
            iters = iters + 1
        return P, torch.sum(P * M)
    
    def getProbas(self, quer_vec):
        # mus: n_shot * dim
        # quer_vec: n_query * dim
        dist = (quer_vec.unsqueeze(1) - self.mus.unsqueeze(0)).norm(dim=2).pow(2)
        
        n_usamples, n_ways = quer_vec.size(0), 5
        n_queries = n_usamples // n_ways

        r = torch.ones(n_usamples)
        c = torch.ones(n_ways) * n_queries
       
        p_xj_test, _ = self.compute_optimal_transport(dist, r, c, epsilon=1e-6)
        
        return p_xj_test

    def estimateFromMask(self, quer_vec, mask):

        # mask: queries * ways
        # quer_vec: queries * dim
        return ((mask.permute(1, 0) @ quer_vec) + self.mus_origin) / (mask.sum(dim=0).unsqueeze(1) + 1)

class MAP:
    def __init__(self, alpha=None):
        
        self.alpha = alpha
    
    def getAccuracy(self, probas, labels):
        olabels = probas.argmax(dim=1)
        matches = labels.eq(olabels).float()
        acc_test = matches.mean()
        return acc_test
    
    def performEpoch(self, model, quer_vec, labels):
        m_estimates = model.estimateFromMask(quer_vec, self.probas)
               
        # update centroids
        model.updateFromEstimate(m_estimates, self.alpha)

        self.probas = model.getProbas(quer_vec)  # TODO: check if not differentiable
        if labels is not None:
            acc = self.getAccuracy(self.probas, labels)
            return acc
        return 0.

    def loop(self, model, quer_vec, n_epochs=20, labels=None):
        
        self.probas = model.getProbas(quer_vec)
        acc_list = []
        if labels is not None:
            acc_list.append(self.getAccuracy(self.probas, labels))
           
        for epoch in range(1, n_epochs+1):
            acc = self.performEpoch(model, quer_vec, labels)
            if labels is not None:
                acc_list.append(acc)
        
        return self.probas, acc_list