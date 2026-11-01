# Correlation-Aware GNNs for Multi-Label ECG Classification (CorGCN)

An advanced graph-based deep learning pipeline adapting **CorGCN (KDD '25)** for multi-label classification of 12-lead ECG signals using the **PTB-XL dataset**.

## 🚀 Key Features
- **Correlation-Aware Feature & Structure Decomposition:** Mitigates feature and topology ambiguity by dynamically building label-specific k-NN graph views.
- **Correlation-Enhanced Graph Convolution:** Combines intra-label message passing with attention-based inter-label correlation propagation.
- **Robust Clinical Evaluation:** Evaluated across 7 standard multi-label metrics (Macro-AUC, Micro-AUC, LRAP, Macro-AP, Hamming Loss, and Ranking Loss).

## 📊 Results (PTB-XL Test Set)
- **Micro-AUC:** 93.38%
- **Macro-AUC:** 91.90%
- **Macro-AP:** 80.58%

## 📂 Repository Structure
- `gnn-label-correlation.ipynb`: Main execution notebook.
- `model.py`: CorGCN architecture implementation.
- `data_utils.py`: PTB-XL dataset loading and preprocessing utilities.
- `features.py`: Feature extraction pipelines.
- `graph_utils.py`: k-NN graph construction and normalization.
