"""
Turn raw ECG signals into fixed-length node feature vectors for CorGCN.

CorGCN (like most GNNs) expects a feature matrix X in R^{n x f}. Raw ECG
signals (n_timesteps x 12 leads) are not directly usable, so we provide
two options:

1. `handcrafted_features`: fast, no training required, interpretable.
2. `CNNEncoder` + `train_cnn_encoder`: a small 1D-CNN classifier trained
   on the multi-label task; its penultimate layer is used as the node
   embedding. Generally gives richer / higher-quality features than
   handcrafted ones, at the cost of an extra training step.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------
# 1) Handcrafted features (fast baseline, no training needed)
# ---------------------------------------------------------------------
def handcrafted_features(signals: np.ndarray, sampling_rate: int = 100):
    """
    Parameters
    ----------
    signals : (n_samples, n_timesteps, n_leads)

    Returns
    -------
    X : (n_samples, n_leads * 7) float32 feature matrix
        Per lead: mean, std, min, max, RMS, dominant FFT frequency,
        zero-crossing rate.
    """
    n_samples, n_timesteps, n_leads = signals.shape
    feats = np.zeros((n_samples, n_leads * 7), dtype=np.float32)

    freqs = np.fft.rfftfreq(n_timesteps, d=1.0 / sampling_rate)

    for i in range(n_samples):
        row = []
        for lead in range(n_leads):
            x = signals[i, :, lead]
            mean = x.mean()
            std = x.std() + 1e-8
            xmin, xmax = x.min(), x.max()
            rms = np.sqrt(np.mean(x ** 2))
            spec = np.abs(np.fft.rfft(x))
            dom_freq = freqs[np.argmax(spec[1:]) + 1] if len(spec) > 1 else 0.0
            zero_cross = np.mean(np.diff(np.signbit(x)) != 0)
            row.extend([mean, std, xmin, xmax, rms, dom_freq, zero_cross])
        feats[i] = row

    # standardize columns
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (
        feats.std(axis=0, keepdims=True) + 1e-8
    )
    return feats.astype(np.float32)


# ---------------------------------------------------------------------
# 2) Small 1D-CNN encoder (learned features)
# ---------------------------------------------------------------------
class _ECGDataset(Dataset):
    def __init__(self, signals, labels):
        # signals: (n, T, C) -> torch wants (n, C, T)
        self.X = torch.from_numpy(signals.transpose(0, 2, 1)).float()
        self.Y = torch.from_numpy(labels).float()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class CNNEncoder(nn.Module):
    """Small 1D-CNN. `embed()` returns the penultimate-layer representation
    to be used as node features for CorGCN; `forward()` returns logits for
    pretraining on the multi-label task directly (auxiliary supervision).
    """

    def __init__(self, n_leads=12, n_classes=5, embed_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.embed_head = nn.Linear(128, embed_dim)
        self.classifier = nn.Linear(embed_dim, n_classes)

    def embed(self, x):
        h = self.conv(x).squeeze(-1)          # (batch, 128)
        return self.embed_head(h)             # (batch, embed_dim)

    def forward(self, x):
        e = F.relu(self.embed(x))
        return self.classifier(e)


def train_cnn_encoder(signals, labels, embed_dim=128, epochs=15,
                       batch_size=64, lr=1e-3, device="cpu", verbose=True):
    """Pretrain a CNN classifier on the raw signals with the (masked)
    multi-label task, purely as a feature extractor for CorGCN's input X.
    `labels` should have unknown/test rows zeroed out if you want to avoid
    label leakage into the node-feature extraction step -- or, simpler
    and often fine in practice, just use the training-fold labels here
    and apply the resulting encoder to all nodes (train+val+test) since
    the encoder itself never sees the graph or does transductive learning.
    """
    n_classes = labels.shape[1]
    model = CNNEncoder(n_leads=signals.shape[2], n_classes=n_classes,
                        embed_dim=embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = _ECGDataset(signals, labels)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward()
            opt.step()
            total_loss += loss.item() * xb.size(0)
        if verbose:
            print(f"[CNN encoder] epoch {epoch+1}/{epochs} "
                  f"loss={total_loss/len(ds):.4f}")

    model.eval()
    return model


@torch.no_grad()
def extract_cnn_embeddings(model, signals, device="cpu", batch_size=256):
    model.eval()
    X = torch.from_numpy(signals.transpose(0, 2, 1)).float()
    embeds = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        embeds.append(model.embed(xb).cpu().numpy())
    return np.concatenate(embeds, axis=0).astype(np.float32)
