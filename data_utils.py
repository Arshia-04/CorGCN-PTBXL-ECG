"""
PTB-XL data loading utilities.

Expects the standard PTB-XL directory layout (as downloaded from PhysioNet):

    ptbxl_root/
        ptbxl_database.csv
        scp_statements.csv
        records100/  (or records500/)
            00000/
                00001_lr.dat / .hea   (100Hz version)
            ...

Download from: https://physionet.org/content/ptb-xl/1.0.3/
"""
import os
import ast
import numpy as np
import pandas as pd

try:
    import wfdb
    _HAS_WFDB = True
except ImportError:
    _HAS_WFDB = False


def load_ptbxl_metadata(root: str):
    """Load ptbxl_database.csv and scp_statements.csv, aggregate diagnostic
    superclasses/subclasses for each recording.

    Returns
    -------
    df : pd.DataFrame
        One row per recording, with columns including:
        - ecg_id (index)
        - patient_id
        - strat_fold (1-10, official PTB-XL CV folds)
        - filename_lr / filename_hr (paths to signal files)
        - scp_codes (raw dict of SCP code -> likelihood)
        - superclass_labels (list[str]), subclass_labels (list[str])
    scp_df : pd.DataFrame
        scp_statements.csv indexed by SCP code.
    """
    db_path = os.path.join(root, "ptbxl_database.csv")
    scp_path = os.path.join(root, "scp_statements.csv")

    df = pd.read_csv(db_path, index_col="ecg_id")
    df.scp_codes = df.scp_codes.apply(ast.literal_eval)

    scp_df = pd.read_csv(scp_path, index_col=0)
    scp_df = scp_df[scp_df.diagnostic == 1]

    def agg_superclass(scp_dict):
        labels = set()
        for code in scp_dict.keys():
            if code in scp_df.index:
                sc = scp_df.loc[code].diagnostic_class
                if isinstance(sc, str) and len(sc) > 0:
                    labels.add(sc)
        return sorted(labels)

    def agg_subclass(scp_dict):
        labels = set()
        for code in scp_dict.keys():
            if code in scp_df.index:
                sc = scp_df.loc[code].diagnostic_subclass
                if isinstance(sc, str) and len(sc) > 0:
                    labels.add(sc)
        return sorted(labels)

    df["superclass_labels"] = df.scp_codes.apply(agg_superclass)
    df["subclass_labels"] = df.scp_codes.apply(agg_subclass)
    # drop recordings with no diagnostic label at all
    df = df[df.superclass_labels.apply(len) > 0]
    return df, scp_df


def build_multihot(df: pd.DataFrame, label_col: str):
    """Build a multi-hot label matrix Y (n_samples x n_classes) from a
    column containing lists of label strings.

    Returns Y (np.ndarray, float32), and the sorted list of class names
    (column order of Y).
    """
    all_labels = sorted({l for labels in df[label_col] for l in labels})
    label_to_idx = {l: i for i, l in enumerate(all_labels)}
    Y = np.zeros((len(df), len(all_labels)), dtype=np.float32)
    for i, labels in enumerate(df[label_col].values):
        for l in labels:
            Y[i, label_to_idx[l]] = 1.0
    return Y, all_labels


def load_raw_signals(root: str, df: pd.DataFrame, sampling_rate: int = 100):
    """Load raw ECG signals for every row in df.

    Parameters
    ----------
    sampling_rate : 100 or 500 (Hz). Use 100Hz (records100) for speed.

    Returns
    -------
    signals : np.ndarray, shape (n_samples, n_timesteps, 12)
    """
    if not _HAS_WFDB:
        raise ImportError("pip install wfdb --break-system-packages")

    fname_col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    signals = []
    for fname in df[fname_col].values:
        path = os.path.join(root, fname)
        sig, _ = wfdb.rdsamp(path)  # (n_timesteps, 12)
        signals.append(sig.astype(np.float32))
    return np.stack(signals, axis=0)


def official_splits(df: pd.DataFrame):
    """Return boolean masks for the official PTB-XL train/val/test split
    based on `strat_fold` (folds 1-8 train, 9 val, 10 test), as used in
    the PTB-XL benchmark paper (Strodthoff et al. 2020)."""
    train_mask = df.strat_fold.apply(lambda f: f <= 8).values
    val_mask = (df.strat_fold == 9).values
    test_mask = (df.strat_fold == 10).values
    return train_mask, val_mask, test_mask


# ---------------------------------------------------------------------
# Synthetic data generator -- lets you run/debug the whole pipeline
# without downloading PTB-XL first.
# ---------------------------------------------------------------------
def make_synthetic_ptbxl(n_samples=800, n_timesteps=1000, n_leads=12,
                          n_classes=5, seed=0):
    """Generate a small synthetic multi-label ECG-like dataset that mimics
    PTB-XL's shape, for pipeline testing. Labels are made to be
    correlated (some class pairs co-occur more than chance) so that the
    label-correlation modeling in CorGCN has something real to learn."""
    rng = np.random.default_rng(seed)

    # class "prototype" waveforms (random smooth signals) added to base noise
    t = np.linspace(0, 4 * np.pi, n_timesteps)
    prototypes = np.stack([
        np.sin(t * (k + 1) * 0.5 + rng.uniform(0, np.pi))
        for k in range(n_classes)
    ], axis=0)  # (n_classes, n_timesteps)

    # induce label correlation: class pairs (0,1) and (2,3) co-occur often
    Y = np.zeros((n_samples, n_classes), dtype=np.float32)
    base_p = rng.uniform(0.1, 0.25, size=n_classes)
    for i in range(n_samples):
        y = (rng.random(n_classes) < base_p).astype(np.float32)
        if y[0] == 1.0 and rng.random() < 0.7:
            y[1] = 1.0
        if y[2] == 1.0 and rng.random() < 0.7:
            y[3] = 1.0
        if y.sum() == 0:
            y[rng.integers(0, n_classes)] = 1.0
        Y[i] = y

    signals = np.zeros((n_samples, n_timesteps, n_leads), dtype=np.float32)
    for i in range(n_samples):
        sig = rng.normal(0, 0.3, size=(n_timesteps, n_leads))
        for k in range(n_classes):
            if Y[i, k] == 1.0:
                lead_weights = rng.uniform(0.5, 1.5, size=n_leads)
                sig += np.outer(prototypes[k], lead_weights)
        signals[i] = sig

    class_names = [f"CLASS_{k}" for k in range(n_classes)]
    return signals.astype(np.float32), Y, class_names
