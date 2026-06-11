# Dark-Trident-GNN: Dark Matter Interaction Classifier using Graph Neural Networks

Graph Neural Network aimed to discriminate dark trident dark matter interactions with liquid argon from neutrino background interactions. This network repurposes the MPID dataset and extends the DM-CNN approach by representing LArTPC wire images as point clouds and applying graph convolutions.

Each event is represented as a graph where nodes are hit pixels (wire, time, ADC) above an ADC threshold, connected via K-Nearest Neighbor edges.

## Python Libraries

- PyTorch
- torch_geometric
- h5py
- scikit-learn
- networkx
- torchinfo

## Setup

```bash
pip install torch_geometric networkx torchinfo h5py scikit-learn
```

## Data Preparation

Convert LArCV ROOT files to HDF5 spacepoint format:

```bash
python convert_to_hdf5.py
```

Input: LArCV ROOT file (MPID or dark trident simulation)
Output: HDF5 file with (wire, time, ADC) spacepoints per event, with signal/background labels

## Training

Open `gnn_example.ipynb` and set file paths:

```python
training_file = "path/to/MPID_training_set_full.h5"
test_file = "path/to/MPID_test_set_full.h5"
```

Run all cells. GPU recommended (for now, just tested on Google Colab T4 - will be tested in Kerberos including nu overlays).

## Results

| Model | AUC |
|-------|-----|
| DM-CNN | 0.9512 |
| DM-GNN (MPID dataset, 10 epochs) | ~TBD |

## Architecture

3-layer GraphConv network:
- Input: 3 features (wire, time, ADC)
- Hidden layers: [8, 16, 32]
- Output: binary classification (signal/background)
- Loss: BCEWithLogitsLoss
- Optimizer: RAdam
