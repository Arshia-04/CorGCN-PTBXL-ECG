"""
Build the base graph A^0 for CorGCN from node feature vectors, since
PTB-XL has no natural graph structure.

Default: k-NN graph in feature (embedding) space, made symmetric.
Optionally combine with a "same patient" graph if patient_id is available
(PTB-XL has repeat visits for some patients).
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
import scipy.sparse as sp


def build_knn_graph(X: np.ndarray, k: int = 10, metric: str = "cosine"):
    """Return a symmetric sparse adjacency matrix (scipy csr, 0/1 entries)
    connecting each node to its k nearest neighbors in feature space."""
    n = X.shape[0]
    nn = NearestNeighbors(n_neighbors=k + 1, metric=metric).fit(X)
    _, indices = nn.kneighbors(X)  # (n, k+1), includes self at col 0

    rows = np.repeat(np.arange(n), k)
    cols = indices[:, 1:].reshape(-1)  # drop self
    data = np.ones(len(rows), dtype=np.float32)

    A = sp.csr_matrix((data, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)  # symmetrize
    A.setdiag(0)
    A.eliminate_zeros()
    return A


def add_patient_edges(A: sp.csr_matrix, patient_ids: np.ndarray):
    """Optionally add edges between recordings belonging to the same
    patient (useful in PTB-XL, which has repeat visits)."""
    A = A.tolil()
    order = np.argsort(patient_ids)
    sorted_pids = patient_ids[order]
    start = 0
    for end in range(1, len(sorted_pids) + 1):
        if end == len(sorted_pids) or sorted_pids[end] != sorted_pids[start]:
            group = order[start:end]
            for i in group:
                for j in group:
                    if i != j:
                        A[i, j] = 1.0
            start = end
    return A.tocsr()


def normalize_adj(A: sp.csr_matrix):
    """Symmetric GCN normalization: D^{-1/2} (A + I) D^{-1/2}."""
    n = A.shape[0]
    A_hat = A + sp.eye(n)
    deg = np.array(A_hat.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5, where=deg > 0)
    d_inv_sqrt[deg == 0] = 0.0
    D_inv_sqrt = sp.diags(d_inv_sqrt)
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def scipy_to_torch_sparse(A: sp.csr_matrix):
    import torch
    A = A.tocoo()
    idx = torch.from_numpy(np.vstack([A.row, A.col])).long()
    vals = torch.from_numpy(A.data).float()
    return torch.sparse_coo_tensor(idx, vals, A.shape).coalesce()
