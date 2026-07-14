import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import conv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph
import h5py
import os

# Paths
run3_hdf5   = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/run3_NuMI_beamon_full.h5"
weights_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/weights_transformer/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/"

os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Model
class GNNTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, edge_dim, heads=4):
        super(GNNTransformer, self).__init__()
        self.init_batch_norm = BatchNorm(input_dim)
        conv_layers = []
        for hidden_dim in hidden_dims:
            transformer_conv = conv.TransformerConv(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True)
            batch_norm = BatchNorm(hidden_dim * heads)
            activation = nn.ReLU()
            conv_layers.append((transformer_conv, batch_norm, activation))
            input_dim = hidden_dim * heads
            edge_dim  = None
        self.conv_layers = nn.ModuleList([nn.ModuleList(layer) for layer in conv_layers])
        self.output_layer = nn.Linear(input_dim * 2, output_dim)

    def forward(self, data, edges, batch_indices, edge_attr=None, skip_output_activation=False):
        x = self.init_batch_norm(data)
        for i, (transformer_conv, batch_norm, activation) in enumerate(self.conv_layers):
            if i == 0:
                x = transformer_conv(x, edges, edge_attr=edge_attr)
            else:
                x = transformer_conv(x, edges)
            x = batch_norm(x)
            x = activation(x)
        x = torch.cat([global_mean_pool(x, batch_indices), global_max_pool(x, batch_indices)], dim=1)
        x = self.output_layer(x)
        if not skip_output_activation:
            x = torch.sigmoid(x)
        return x

# Load model
input_dim   = 3
edge_dim    = 3
hidden_dims = [32, 64, 128, 256]
heads       = 4
model = GNNTransformer(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1, edge_dim=edge_dim, heads=heads).to(device)
model.load_state_dict(torch.load(weights_dir + 'transformer_model_best.pt', map_location=device, weights_only=False))
model.eval()
print("Model loaded.")

# Load and convert run3 data to graphs
N_NEIGHBORS = 8

def load_hdf5_data(hdf5_file):
    data_list = []
    with h5py.File(hdf5_file, 'r') as f:
        for event_id in f.keys():
            spacepoints = f[event_id]['spacepoints'][:]
            data_list.append({'spacepoints': spacepoints})
    return data_list

def make_graphs(data_list, n_neighbors=N_NEIGHBORS):
    graph_list = []
    N_skipped = 0
    for i, event in enumerate(data_list):
        sp = event['spacepoints']
        if len(sp) > n_neighbors + 1:
            A_matrix = kneighbors_graph(sp, n_neighbors=n_neighbors, mode='connectivity', include_self=False).toarray()
        elif len(sp) > 1:
            A_matrix = kneighbors_graph(sp, n_neighbors=min(n_neighbors, len(sp)-1), mode='connectivity', include_self=False).toarray()
        elif len(sp) == 1:
            A_matrix = np.array([[1]])
        else:
            N_skipped += 1
            continue

        edges = np.array(np.nonzero(A_matrix))
        src, dst = edges[0], edges[1]
        edge_attr = np.column_stack([
            np.abs(sp[src, 0] - sp[dst, 0]),
            np.abs(sp[src, 1] - sp[dst, 1]),
            np.abs(sp[src, 2] - sp[dst, 2])
        ])

        x          = torch.tensor(sp, dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long)
        edge_attr  = torch.tensor(edge_attr, dtype=torch.float)
        graph_list.append(Data(x=x, edge_index=edge_index, edge_attr=edge_attr))

        if i % 1000 == 0:
            print(f"Processed {i}/{len(data_list)}")
    print(f"Skipped {N_skipped} events")
    return graph_list

print("Loading run3 data...")
run3_data = load_hdf5_data(run3_hdf5)
print(f"run3 events: {len(run3_data)}")

print("Building graphs...")
run3_graphs = make_graphs(run3_data)
print(f"run3 graphs: {len(run3_graphs)}")

# Inference
run3_loader = DataLoader(run3_graphs, batch_size=16, shuffle=False)
run3_scores = []

for batch in run3_loader:
    batch = batch.to(device)
    with torch.no_grad():
        outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, skip_output_activation=True)
        scores  = torch.sigmoid(outputs).squeeze()
        run3_scores.extend(scores.cpu().numpy())

run3_scores = np.array(run3_scores)
print(f"Fraction of events with score > 0.5: {(run3_scores > 0.5).mean():.4f}")

# Score distribution
bins = np.linspace(0, 1, 51)
plt.figure(figsize=(8, 6))
plt.hist(run3_scores, bins=bins, alpha=0.7, density=True, label='Run3 NuMI beam-on data')
plt.xlabel('Signal Score')
plt.ylabel('Normalised Counts')
plt.title('Score Distribution - Run3 NuMI Beam-on Data (TransformerConv GNN)')
plt.legend()
plt.savefig(output_dir + 'run3_score_distribution.png')
plt.close()

np.save(output_dir + 'run3_scores.npy', run3_scores)
print(f"Done. Saved to {output_dir}")
