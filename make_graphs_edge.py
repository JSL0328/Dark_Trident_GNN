import numpy as np
from sklearn.neighbors import kneighbors_graph
import torch
from torch_geometric.data import Data
import h5py
import os

# Paths
training_file = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/MPID_training_set_full.h5"
test_file     = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/MPID_test_set_full.h5"
output_dir    = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs_edge/"
N_NEIGHBORS   = 8

os.makedirs(output_dir, exist_ok=True)

# Load spacepoints and labels from HDF5
def load_hdf5_data(hdf5_file):
    data_list = []
    with h5py.File(hdf5_file, 'r') as f:
        for event_id in f.keys():
            spacepoints = f[event_id]['spacepoints'][:]
            label = f[event_id].attrs['label']
            data_list.append({'spacepoints': spacepoints, 'label': label})
    return data_list

# Convert spacepoints to PyG graph objects with edge features
def make_graphs(data_list, n_neighbors=N_NEIGHBORS):
    graph_list = []
    N_skipped = 0
    for i, event in enumerate(data_list):
        sp    = event['spacepoints']  # (N, 3): wire, time, ADC
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

        edges = np.array(np.nonzero(A_matrix))  # (2, N_edges)

        # Edge features: wire distance, time distance, ADC difference
        src, dst = edges[0], edges[1]
        edge_attr = np.column_stack([
            np.abs(sp[src, 0] - sp[dst, 0]),  # wire distance
            np.abs(sp[src, 1] - sp[dst, 1]),  # time distance
            np.abs(sp[src, 2] - sp[dst, 2])   # ADC difference
        ])

        x          = torch.tensor(sp, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long)
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
        y          = torch.tensor([label], dtype=torch.float)
        graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y))

        if i % 1000 == 0:
            print(f"Processed {i}/{len(data_list)}")
    print(f"Skipped {N_skipped} events with no spacepoints")
    return graph_list

# Main
print("Loading data...")
train_data = load_hdf5_data(training_file)
test_data  = load_hdf5_data(test_file)
print(f"Training: {len(train_data)} events, Test: {len(test_data)} events")

print("Converting training data to graphs...")
train_graphs = make_graphs(train_data)

print("Converting test data to graphs...")
test_graphs  = make_graphs(test_data)

print("Saving...")
# Save in batches of 5000
BATCH_SIZE = 5000
for batch_i, start in enumerate(range(0, len(train_graphs), BATCH_SIZE)):
    chunk = train_graphs[start:start+BATCH_SIZE]
    torch.save(chunk, output_dir + f'train_graphs_{batch_i}.pt')
    print(f"Saved train_graphs_{batch_i}.pt ({len(chunk)} graphs)")

for batch_i, start in enumerate(range(0, len(test_graphs), BATCH_SIZE)):
    chunk = test_graphs[start:start+BATCH_SIZE]
    torch.save(chunk, output_dir + f'test_graphs_{batch_i}.pt')
    print(f"Saved test_graphs_{batch_i}.pt ({len(chunk)} graphs)")

print(f"Done. Saved to {output_dir}")
