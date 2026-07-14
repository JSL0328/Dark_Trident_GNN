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
import glob
import json

# Paths
run1_dir    = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/run1_signal/"
weights_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/weights_transformer/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/run1/"

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

N_NEIGHBORS = 8
THRESHOLD   = 0.5

def load_and_make_graphs(hdf5_file):
    graph_list = []
    with h5py.File(hdf5_file, 'r') as f:
        for event_id in f.keys():
            sp = f[event_id]['spacepoints'][:]
            if len(sp) > N_NEIGHBORS + 1:
                A_matrix = kneighbors_graph(sp, n_neighbors=N_NEIGHBORS, mode='connectivity', include_self=False).toarray()
            elif len(sp) > 1:
                A_matrix = kneighbors_graph(sp, n_neighbors=min(N_NEIGHBORS, len(sp)-1), mode='connectivity', include_self=False).toarray()
            elif len(sp) == 1:
                A_matrix = np.array([[1]])
            else:
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
    return graph_list

# Process all run1 signal files
h5_files   = sorted(glob.glob(run1_dir + "*.h5"))
results    = {}

print(f"Found {len(h5_files)} signal files")

for h5_file in h5_files:
    filename = os.path.basename(h5_file).replace('.h5', '')
    print(f"\nProcessing {filename}...")

    graphs = load_and_make_graphs(h5_file)
    if len(graphs) == 0:
        print(f"No valid graphs, skipping.")
        continue

    loader = DataLoader(graphs, batch_size=16, shuffle=False)
    scores = []

    for batch in loader:
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, skip_output_activation=True)
            s       = torch.sigmoid(outputs).squeeze()
            scores.extend(s.cpu().numpy())

    scores     = np.array(scores)
    efficiency = (scores > THRESHOLD).mean()
    print(f"Events: {len(scores)}, Efficiency: {efficiency:.4f}")

    # Save scores
    np.save(output_dir + filename + '_scores.npy', scores)

    # Score distribution plot
    bins = np.linspace(0, 1, 51)
    plt.figure(figsize=(8, 6))
    plt.hist(scores, bins=bins, alpha=0.7, density=True)
    plt.xlabel('Signal Score')
    plt.ylabel('Normalised Counts')
    plt.title(f'{filename}\nEfficiency (score>{THRESHOLD}): {efficiency:.3f}')
    plt.savefig(output_dir + filename + '_score_dist.png')
    plt.close()

    results[filename] = {
        'n_events':  len(scores),
        'efficiency': float(efficiency),
        'mean_score': float(scores.mean())
    }

# Save summary
with open(output_dir + 'efficiency_summary.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nEfficiency Summary:")
for name, res in results.items():
    print(f"{name}: efficiency={res['efficiency']:.4f}, n_events={res['n_events']}")

print("\nDone.")
