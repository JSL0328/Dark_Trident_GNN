import numpy as np
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data
import h5py
import os

# Paths
training_file = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/MPID_training_set_full.h5"
test_file     = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/MPID_test_set_full.h5"
output_dir    = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs/"
N_NEIGHBORS   = 8

os.makedirs(output_dir, exist_ok=True)

# Load spacepoints and labels from HDF5
def load_hdf5_data(hdf5_file):
    data_list = []
    with h5py.File(hdf5_file, 'r') as f:
        for event_id in f.keys():
            spacepoints = f[event_id]['spacepoints'][:]  # read full array into memory
            label = f[event_id].attrs['label']
            data_list.append({'spacepoints': spacepoints, 'label': label})
    return data_list

# Convert spacepoints to PyG graph objects using KNN
def make_graphs(data_list, n_neighbors=N_NEIGHBORS):
    graph_list = []
    N_skipped = 0
    for i, event in enumerate(data_list):
        sp    = event['spacepoints']
        label = event['label']
        if len(sp) > n_neighbors + 1:
            A_matrix = kneighbors_graph(sp, n_neighbors=n_neighbors, mode='connectivity', include_self=False).toarray()
        elif len(sp) > 1:
            A_matrix = kneighbors_graph(sp, n_neighbors=min(n_neighbors, len(sp)-1), mode='connectivity', include_self=False).toarray()
        elif len(sp) == 1:
            A_matrix = np.array([[1]])
        else:
            N_skipped += 1
            continue
        edges      = np.array(np.nonzero(A_matrix))
        x          = torch.tensor(sp, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long)
        y          = torch.tensor([label], dtype=torch.float)
        graph_list.append(Data(x=x, edge_index=edge_index, y=y))
        if i % 1000 == 0:
            print(f"Processed {i}/{len(data_list)}")
    print(f"Skipped {N_skipped} events with no spacepoints")
    return graph_list

# Main
print("Loading data...")
train_data = load_hdf5_data(training_file)
test_data  = load_hdf5_data(test_file)
print(f"Training: {len(train_data)} events, Test: {len(test_data)} events")

print("Converting to graphs...")
train_graphs = make_graphs(train_data)
test_graphs  = make_graphs(test_data)

print("Saving...")
torch.save(train_graphs, output_dir + 'train_graphs.pt')
torch.save(test_graphs,  output_dir + 'test_graphs.pt')
print(f"Done. Saved to {output_dir}")
