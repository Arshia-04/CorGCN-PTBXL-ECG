"""
CorGCN implementation (Bei et al., KDD'25), adapted for PTB-XL multi-label
ECG classification.

Pipeline (matches the paper's Sec 4.1 - 4.3):
  1. CorrelationAwareFeatureDecomposition
       - learns per-label prototypes E^l
       - projects transformed node features E^x onto each label's
         direction -> E^proj  (Eq. 2-8)
       - contrastive loss (L_cmi, Eq. 4) + likelihood loss (L_lm, Eq. 5-6)
  2. CorrelationAwareStructureDecomposition
       - builds a label-specific k-NN-like graph per label from E^proj
         aggregated over the base graph's neighborhoods (Eq. 9-13)
  3. CorrelationEnhancedGraphConv
       - intra-label GCN message passing per label-view graph (Eq. 14)
       - inter-label correlation propagation via attention (Eq. 15-16)
  4. Classifier head (Eq. 17-18)

All symbols follow the paper's notation where practical.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# 1. Correlation-Aware Feature Decomposition  (Sec 4.1.1)
# ---------------------------------------------------------------------
class CorrelationAwareFeatureDecomposition(nn.Module):
    def __init__(self, in_dim, d, n_classes, gamma=2.0):
        super().__init__()
        self.K = n_classes
        self.d = d
        self.gamma = gamma
        self.W_t = nn.Linear(in_dim, d, bias=False)          # Eq. 3: E^x = X W_t
        self.label_prototypes = nn.Parameter(torch.randn(n_classes, d) * 0.1)  # E^l, Eq. 2
        # likelihood decoder theta: maps a d-dim vector -> per-class logits
        self.decoder = nn.Linear(d, n_classes)

    def forward(self, X):
        E_x = self.W_t(X)                      # (n, d)          Eq. 3
        E_l = self.label_prototypes             # (K, d)          Eq. 2
        # projection coefficients w_{i,k} = cosine(E_x_i, E_l_k)   Eq. 7
        E_x_n = F.normalize(E_x, dim=-1)
        E_l_n = F.normalize(E_l, dim=-1)
        w = E_x_n @ E_l_n.t()                   # (n, K)
        # E_proj[:,k,:] = w[:,k] * E_x                              Eq. 8
        E_proj = w.unsqueeze(-1) * E_x.unsqueeze(1)   # (n, K, d)
        return E_x, E_l, w, E_proj

    def contrastive_loss(self, E_x, E_l, Y, labeled_mask):
        """L_cmi, Eq. 4. Only computed over labeled nodes."""
        idx = labeled_mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return torch.tensor(0.0, device=E_x.device)
        E_x_l = F.normalize(E_x[idx], dim=-1)          # (m, d)
        E_l_n = F.normalize(E_l, dim=-1)                # (K, d)
        logits = E_x_l @ E_l_n.t()                      # (m, K) = exp inside will be applied by softmax
        y = Y[idx]                                       # (m, K) multi-hot
        # for each sample: log-sum-exp over ALL classes as denominator,
        # numerator averaged over positive (Pos(y_i)) classes -- Eq. 4
        log_denom = torch.logsumexp(logits, dim=1, keepdim=True)  # (m,1)
        log_probs = logits - log_denom                             # (m,K)
        pos_count = y.sum(dim=1, keepdim=True).clamp(min=1.0)
        loss_per_sample = -(y * log_probs).sum(dim=1) / pos_count.squeeze(1)
        return loss_per_sample.mean()

    def likelihood_loss(self, E_x, E_l, Y, labeled_mask):
        """L_lm, Eq. 5-6, with focal-style focusing factor gamma and
        rho_k class-balance weighting."""
        idx = labeled_mask.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            return torch.tensor(0.0, device=E_x.device)
        y = Y[idx]                       # (m, K)
        K = y.shape[1]

        # rho_k, Eq. 5: inverse-sqrt class frequency, normalized
        pos_counts = y.sum(dim=0).clamp(min=1.0)             # (K,)
        inv_sqrt = 1.0 / torch.sqrt(pos_counts)
        rho = inv_sqrt / inv_sqrt.sum()                       # (K,)

        # decode E_x (per node) -> predicted prob per class
        p_hat_x = torch.sigmoid(self.decoder(E_x[idx]))       # (m, K)
        # decode y_i . E^l  (weighted sum of prototypes by true labels) -> per class
        yE = (y.unsqueeze(-1) * E_l.unsqueeze(0)).sum(dim=1)  # (m, d) weighted-avg prototype
        # normalize by number of positive labels to keep scale sane
        yE = yE / y.sum(dim=1, keepdim=True).clamp(min=1.0)
        p_hat_l = torch.sigmoid(self.decoder(yE))              # (m, K)

        def focal_bce(p_hat):
            # p_k(E|theta) = p_hat if y=1 else (1 - p_hat)   (paper's Eq. 6 definition)
            p_true = torch.where(y == 1, p_hat, 1 - p_hat).clamp(1e-6, 1 - 1e-6)
            focal = (1 - p_true).pow(self.gamma) * torch.log(p_true)
            return (rho.unsqueeze(0) * focal).sum(dim=1)

        loss = -0.5 * (focal_bce(p_hat_x) + focal_bce(p_hat_l))
        return loss.mean()


# ---------------------------------------------------------------------
# 2. Correlation-Aware Structure Decomposition (Sec 4.1.2)
# ---------------------------------------------------------------------
def structure_decompose(E_proj, A0_norm_dense, top_lambda: int):
    """
    E_proj: (n, K, d) per-label projected features
    A0_norm_dense: (n, n) dense normalized base adjacency (fine for small/
                   medium n; for large graphs, replace with a sparse /
                   mini-batch k-NN implementation, see notes at bottom
                   of train.py)
    top_lambda: number of edges to keep per node per label view (Eq. 11)

    Returns a list of K dense (n, n) 0/1 adjacency matrices (label-view
    graphs), following Eq. 9-12.
    """
    n, K, d = E_proj.shape
    device = E_proj.device

    # Eq. 9: aggregate neighborhood's projected features (mean over
    # neighbors + self) per label view, using the ORIGINAL topology A0.
    # We use the (row-normalized) A0 (with self loops) as the aggregator.
    deg = A0_norm_dense.sum(dim=1, keepdim=True).clamp(min=1.0)
    agg_w = A0_norm_dense / deg   # simple mean aggregation, Eq. 9's 1/(|N|+1) factor

    label_graphs = []
    for k in range(K):
        Ek = E_proj[:, k, :]                       # (n, d)
        E_sd = agg_w @ Ek                          # (n, d)   Eq. 9
        E_sd_n = F.normalize(E_sd, dim=-1)
        S_k = E_sd_n @ E_sd_n.t()                  # (n, n)   Eq. 10 cosine sim
        S_k.fill_diagonal_(-1e9)                   # exclude self from top-k
        lam = min(top_lambda, n - 1)
        _, top_idx = torch.topk(S_k, k=lam, dim=1)  # Eq. 11 Top-lambda
        Ak = torch.zeros((n, n), device=device)
        Ak.scatter_(1, top_idx, 1.0)
        Ak = torch.maximum(Ak, Ak.t())              # symmetrize
        label_graphs.append(Ak)
    return label_graphs


def gcn_normalize(A: torch.Tensor):
    """Dense symmetric GCN normalization D^-1/2 (A+I) D^-1/2 (Eq. 14)."""
    n = A.shape[0]
    A_hat = A + torch.eye(n, device=A.device)
    deg = A_hat.sum(dim=1)
    d_inv_sqrt = deg.pow(-0.5)
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    D = torch.diag(d_inv_sqrt)
    return D @ A_hat @ D


# ---------------------------------------------------------------------
# 3. Correlation-Enhanced Graph Convolution (Sec 4.2)
# ---------------------------------------------------------------------
class IntraLabelGCNLayer(nn.Module):
    """One GCN layer applied independently within each label-view graph
    (Eq. 14). Weights are shared across views for parameter efficiency
    (can be made per-view if you have enough data)."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, Z, A_norm_list):
        """
        Z: (n, V, in_dim)  -- V = K+1 views (including base graph view 0)
        A_norm_list: list of V normalized (n,n) adjacency tensors
        returns Z_hat: (n, V, out_dim)
        """
        outs = []
        for v, A_norm in enumerate(A_norm_list):
            h = A_norm @ self.W(Z[:, v, :])
            outs.append(F.relu(h))
        return torch.stack(outs, dim=1)


class InterLabelCorrelationPropagation(nn.Module):
    """Attention-based correlation propagation across label views
    (Eq. 15-16)."""

    def __init__(self, dim, n_classes):
        super().__init__()
        self.K = n_classes
        self.W1 = nn.Linear(dim, dim, bias=False)
        self.W2 = nn.Linear(dim, dim, bias=False)
        self.W3 = nn.Linear(dim, dim, bias=False)

    def forward(self, E_l, Z_hat):
        """
        E_l: (K, d) label prototypes
        Z_hat: (n, K, d)  -- the K label-specific views (excludes base
               view 0, which is handled separately / concatenated later)
        returns Z: (n, K, d) correlation-propagated representations
        """
        n, K, d = Z_hat.shape
        Q = self.W1(E_l)                       # (K, d)
        Kmat = self.W2(Z_hat)                  # (n, K, d)
        scores = torch.einsum('kd,nvd->nkv', Q, Kmat) / (d ** 0.5)  # (n,K,K) Eq.15
        Cor = F.softmax(scores, dim=-1)                              # (n,K,K)
        V = self.W3(Z_hat)                                           # (n,K,d)
        Z = torch.einsum('nkv,nvd->nkd', Cor, V)                     # Eq. 16
        return Z, Cor


# ---------------------------------------------------------------------
# 4. Full CorGCN model
# ---------------------------------------------------------------------
class CorGCN(nn.Module):
    def __init__(self, in_dim, d=64, n_classes=5, n_layers=2,
                 top_lambda=10, gamma=2.0):
        super().__init__()
        self.K = n_classes
        self.d = d
        self.top_lambda = top_lambda
        self.n_layers = n_layers

        self.feat_decomp = CorrelationAwareFeatureDecomposition(
            in_dim, d, n_classes, gamma=gamma)

        # V = K+1 views total (view 0 = base graph with correlated but
        # non-decomposed features, views 1..K = label-specific)
        self.gcn_layers = nn.ModuleList(
            [IntraLabelGCNLayer(d, d) for _ in range(n_layers)]
        )
        self.corr_prop = InterLabelCorrelationPropagation(d, n_classes)

        self.classifier = nn.Linear(2 * d, n_classes)   # Eq. 17-18

    def forward(self, X, A0_norm_dense):
        n = X.shape[0]
        device = X.device

        # ---- Step 1: feature decomposition ----
        E_x, E_l, w, E_proj = self.feat_decomp(X)      # E_proj: (n, K, d)

        # ---- Step 2: structure decomposition ----
        label_graphs = structure_decompose(E_proj, A0_norm_dense, self.top_lambda)
        A_norms = [A0_norm_dense] + [gcn_normalize(Ak) for Ak in label_graphs]  # V=K+1

        # initial per-view features Z^(0): view0 = E_x, views 1..K = E_proj[:,k,:]
        Z0 = torch.cat([E_x.unsqueeze(1), E_proj], dim=1)   # (n, K+1, d)

        Z = Z0
        for layer in self.gcn_layers:
            Z_hat = layer(Z, A_norms)                       # intra-label MP, Eq.14
            base_view, label_views = Z_hat[:, 0, :], Z_hat[:, 1:, :]
            corr_views, Cor = self.corr_prop(E_l, label_views)  # Eq. 15-16
            Z = torch.cat([base_view.unsqueeze(1), corr_views], dim=1)

        Z_base = Z[:, 0, :]           # (n, d)
        Z_labels = Z[:, 1:, :]        # (n, K, d)

        # ---- Aggregation & prediction, Eq. 17-18 ----
        sims = torch.einsum('nkd,kd->nk', Z_labels, F.normalize(E_l, dim=-1))  # sim(Z_ik, E_l_k)
        weighted = torch.einsum('nk,nkd->nd', sims, Z_labels)   # sum_k sim*Z_ik
        Z_cls = torch.cat([Z_base, weighted], dim=1)            # (n, 2d)
        logits = self.classifier(Z_cls)
        return logits, E_x, E_l

    def compute_losses(self, X, A0_norm_dense, Y, labeled_mask,
                        alpha=1.0, beta=1.0):
        logits, E_x, E_l = self.forward(X, A0_norm_dense)
        idx = labeled_mask.nonzero(as_tuple=True)[0]

        L_cls = F.binary_cross_entropy_with_logits(logits[idx], Y[idx])
        L_cmi = self.feat_decomp.contrastive_loss(E_x, E_l, Y, labeled_mask)
        L_lm = self.feat_decomp.likelihood_loss(E_x, E_l, Y, labeled_mask)

        # adaptive balancing, as in the paper's Appendix B.3:
        # alpha = |L_cls / (3*L_cmi)|, beta = |L_cls / (3*L_lm)|
        with torch.no_grad():
            alpha_eff = (L_cls / (3 * L_cmi.clamp(min=1e-8))).abs()
            beta_eff = (L_cls / (3 * L_lm.clamp(min=1e-8))).abs()

        total = L_cls + alpha_eff * L_cmi + beta_eff * L_lm
        return total, {
            "cls": L_cls.item(), "cmi": L_cmi.item(), "lm": L_lm.item(),
            "alpha": alpha_eff.item(), "beta": beta_eff.item(),
        }, logits
