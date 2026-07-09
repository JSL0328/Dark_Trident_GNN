# Dark Trident GNN

Graph Neural Network classifier for dark trident BSM signal detection in MicroBooNE LArTPC data.

## Overview

This repository implements GNN-based binary classifiers to distinguish dark trident signal events (e+e- pairs from dark matter interactions) from Standard Model backgrounds. The work extends the CNN-based analysis of Lepin (2024) by representing LArTPC events as graph structures.

Each event is represented as a graph where nodes are hit pixels extracted from the Y-plane wire image (wire, time, ADC) from the MPID dataset (Simulated, image data only), connected via K-Nearest Neighbour edges.

## Models

### GraphConv GNN (`train_gnn.py`)
- 3-layer GraphConv with BatchNorm and ReLU
- Global mean + max pooling
- Hidden dims: [32, 64, 128, 256]
- Early stopping (patience=10)

### TransformerConv GNN (`train_gnn_transformer.py`)
- 3-layer TransformerConv with multi-head attention (heads=4)
- Edge features: wire distance, time distance, ADC difference
- Global mean + max pooling
- Hidden dims: [32, 64, 128, 256]
- Early stopping (patience=10)

## Data

| Dataset | Type | Purpose |
|---------|------|---------|
| MPID training set (62k events) | Simulation | Model training |
| MPID test set (6.9k events) | Simulation | Performance evaluation (AUC, ROC) |
| run1 signal simulation | Simulation | Signal efficiency measurement |
| run3 NuMI beam-on data | Real data | Data/MC comparison |

## Pipeline

1. `convert_to_hdf5.py` — Convert LArCV ROOT files to HDF5 spacepoint format
2. `make_graphs.py` — Build KNN graphs (node features only)
3. `make_graphs_edge.py` — Build KNN graphs with edge features
4. `train_gnn.py` — Train GraphConv model
5. `train_gnn_transformer.py` — Train TransformerConv model
6. `inference_gnn.py` — Evaluate model: ROC curve, AUC, score distribution

## Results

| Model | AUC |
|-------|-----|
| CNN baseline (Lepin 2024) | 0.9512 |
| GraphConv GNN | 0.9454 |
| TransformerConv GNN | TBD |

## Planned

- Occlusion analysis: node removal study to identify which hits drive classification
- Training size scan: GNN vs CNN data efficiency comparison (Vision Transformer & Neuromorphic network will be discussed as well)

## Libraries Required

```bash
pip install torch torch_geometric h5py scikit-learn networkx torchinfo
```

## Alternative

`alternative/gnn_example_joe.ipynb` contains the original GNN example notebook by Joe Bateman.