# Dark Trident GNN

Graph Neural Network classifier for dark trident BSM signal detection in MicroBooNE LArTPC data.

<p align="center">
  <img src="figures/event_display.png" width="50%"/>
</p>

*MicroBooNE NuMI beam-on data event with GNN signal score 1.000. The colour scale represents charge deposition (ADC), with red indicating high charge. The shower-like topology is characteristic of a dark trident candidate.*

## Overview

This repository implements GNN-based binary classifiers to distinguish dark trident signal events (e+e- pairs from dark matter interactions) from Standard Model backgrounds. The work extends the CNN-based analysis of Lepin (2024) by representing LArTPC events as graph structures.

Each event is represented as a graph where nodes are hit pixels extracted from the Y-plane wire image (wire, time, ADC) from the MPID dataset, connected via K-Nearest Neighbour edges (k=8). Empty events (spacepoints = 0) are excluded from training and evaluation for fair comparison across models.

## Models

### Graph Convolutional Network (`train_gnn.py`)
- 4-layer GraphConv with BatchNorm and ReLU
- Global mean + max pooling
- Hidden dims: [32, 64, 128, 256]
- Optimiser: RAdam, lr=5e-4 (selected via LR range test, Smith 2017/2018)
- Early stopping: patience=5
- LR scheduler: ReduceLROnPlateau (factor=0.2, patience=5)

### Graph Transformer Network (`train_gnn_transformer.py`)
- 4-layer TransformerConv with multi-head attention (heads=4)
- Edge features: wire distance, time distance, ADC difference
- Global mean + max pooling
- Hidden dims: [16, 32, 64, 128]
- Optimiser: RAdam, lr=1e-4 (selected via LR range test, Smith 2017/2018)
- Early stopping: patience=5
- LR scheduler: ReduceLROnPlateau (factor=0.2, patience=5)
- Based on Shi et al. (2020) Section 3.1 — attention mechanism only; label propagation and masked prediction not used

## Data

| Dataset | Type | Purpose |
|---------|------|---------|
| MPID training set (62,058 events after empty exclusion) | Simulation | Model training |
| MPID test set (6,807 events after empty exclusion) | Simulation | Performance evaluation (AUC, ROC) |
| Run 1 & Run 3 signal simulation | Simulation | Signal efficiency measurement |
| Run 3 NuMI beam-on data | Real data | Data/MC comparison |

## Pipeline

1. `convert_to_hdf5.py` — Convert LArCV ROOT files to HDF5 spacepoint format
2. `make_graphs.py` — Build KNN graphs (node features only)
3. `make_graphs_edge.py` — Build KNN graphs with edge features (wire/time distance, ADC difference)
4. `lr_finder.py` — Learning rate range test (Smith 2017/2018) for graph-based models
5. `train_gnn.py` — Train Graph Convolutional Network
6. `train_gnn_transformer.py` — Train Graph Transformer Network
7. `inference_gnn.py` — Evaluate Graph Convolutional Network: ROC curve, AUC, score distribution
8. `inference_gnn_transformer.py` — Evaluate Graph Transformer Network: ROC curve, AUC, score distribution, saves test_scores.npy and truth.npy
9. `occlusion_analysis.py` — Node removal study to identify spatially important regions
10. `inference_run3.py` — Apply model to Run 3 NuMI beam-on data
11. `inference_run1_signal.py` / `inference_run3_signal.py` — Signal efficiency measurement

## Results

| Model | AUC | Test Accuracy |
|-------|-----|---------------|
| CNN baseline (Lepin 2024) | 0.9512 | 0.954 |
| Graph Convolutional Network | 0.9740 | tbc |
| Graph Transformer Network | 0.9832 | tbc |

*Results to be updated after retraining with optimal learning rates from LR range test.*

## Interpretability

Occlusion analysis (node removal) reveals the vertex region as the critical spatial determinant for classification — consistent with the e+e- pair topology of the dark trident signal. Further interpretability analysis (t-SNE node embedding, edge feature ablation study) is ongoing.

## Planned

- t-SNE node embedding visualisation by layer
- Edge feature ablation study: Graph Transformer Network with and without edge features
- Graph structure analysis: signal vs background graph properties
- Training size scan: Graph Neural Network vs CNN vs neuromorphic computing data efficiency comparison
- Bootstrapped AUC uncertainty estimation

## Libraries Required

```bash
pip install torch torch_geometric h5py scikit-learn networkx torchinfo
```

## References

- Shi et al. (2020): Masked Label Prediction — TransformerConv (Section 3.1 only)
- Smith (2017/2018): Cyclical Learning Rates / Disciplined approach to neural network hyper-parameters
- Shlomi et al. (2021): Graph Neural Networks in Particle Physics
- Lepin (2024): A Search for Dark Tridents Using the MicroBooNE Detector (MRes thesis, Imperial College London)

## Alternative

`alternative/gnn_example_joe.ipynb` contains the original GNN example notebook by Joe Bateman.