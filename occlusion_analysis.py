import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import conv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.data import Data
import os

# Paths
graphs_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs_edge/"
weights_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/weights_transformer/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/occlusion/"

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

# Load test graphs
print("Loading test graphs...")
test_sample = []
test_files = sorted([f for f in os.listdir(graphs_dir) if f.startswith("test_graphs_") and f.endswith('.pt')])
for f in test_files:
    chunk = torch.load(os.path.join(graphs_dir, f), weights_only=False)
    test_sample.extend(chunk)
print(f"Test graphs: {len(test_sample)}")

# Get original scores
print("Running baseline inference...")
original_scores = []
for graph in test_sample:
    g = graph.to(device)
    batch_idx = torch.zeros(g.x.shape[0], dtype=torch.long).to(device)
    with torch.no_grad():
        score = model(g.x, g.edge_index, batch_idx, edge_attr=g.edge_attr)
    original_scores.append(score.item())

# Select events
sig_indices = [i for i, g in enumerate(test_sample) if g.y.item() == 1 and original_scores[i] > 0.9][:5]  # true positive (signal)
fp_indices  = [i for i, g in enumerate(test_sample) if g.y.item() == 0 and original_scores[i] > 0.5][:2]  # false positive (background misclassified as signal)
fn_indices  = [i for i, g in enumerate(test_sample) if g.y.item() == 1 and original_scores[i] < 0.5][:2]  # false negative (signal misclassified as background)

selected = ([(i, 'Signal') for i in sig_indices] +
            [(i, 'FalsePositive') for i in fp_indices] +
            [(i, 'FalseNegative') for i in fn_indices])

def run_occlusion(graph, model, device, box_size=50, step=25):
    x = graph.x.cpu().numpy()
    wire = x[:, 0]
    time = x[:, 1]

    wire_min, wire_max = wire.min(), wire.max()
    time_min, time_max = time.min(), time.max()

    wire_steps = np.arange(wire_min, wire_max, step)
    time_steps = np.arange(time_min, time_max, step)

    score_map = np.zeros((len(time_steps), len(wire_steps)))

    for ti, t0 in enumerate(time_steps):
        for wi, w0 in enumerate(wire_steps):
            # Find nodes outside the occlusion box
            mask = ~((wire >= w0) & (wire < w0 + box_size) &
                     (time >= t0) & (time < t0 + box_size))

            if mask.sum() < 2:
                score_map[ti, wi] = 0.0
                continue

            # Build new graph without occluded nodes
            new_x          = graph.x[mask]
            old_to_new     = -np.ones(len(mask), dtype=int)
            old_to_new[mask] = np.arange(mask.sum())

            edges     = graph.edge_index.cpu().numpy()
            edge_mask = mask[edges[0]] & mask[edges[1]]
            new_edges = old_to_new[edges[:, edge_mask]]

            if new_edges.shape[1] == 0:
                score_map[ti, wi] = 0.0
                continue

            new_edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr is not None else None

            new_x          = new_x.to(device)
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).to(device)
            new_batch      = torch.zeros(new_x.shape[0], dtype=torch.long).to(device)
            new_edge_attr  = new_edge_attr.to(device) if new_edge_attr is not None else None

            with torch.no_grad():
                score = model(new_x, new_edge_index, new_batch, edge_attr=new_edge_attr)
            score_map[ti, wi] = score.item()

    return score_map, wire_steps, time_steps

# Run occlusion and plot
for event_idx, label in selected:
    graph = test_sample[event_idx]
    orig_score = original_scores[event_idx]
    print(f"Running occlusion for {label} Event {event_idx} (score={orig_score:.3f})...")

    score_map, wire_steps, time_steps = run_occlusion(graph, model, device, box_size=25, step=10)

    x    = graph.x.cpu().numpy()
    wire = x[:, 0]
    time = x[:, 1]
    adc  = x[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: original event
    sc = axes[0].scatter(wire, time, c=adc, cmap='viridis', s=5)
    plt.colorbar(sc, ax=axes[0], label='ADC')
    axes[0].set_xlabel('Wire')
    axes[0].set_ylabel('Time (tick)')
    axes[0].set_title(f'{label} Event {event_idx} - ADC (score={orig_score:.3f})')

    # Right: score map (lower score = more important region)
    im = axes[1].imshow(score_map, origin='lower', aspect='auto',
                        extent=[wire_steps[0], wire_steps[-1]+25, time_steps[0], time_steps[-1]+25],
                        cmap='RdYlGn', vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[1], label='Score after occlusion')
    axes[1].scatter(wire, time, c='black', s=1, alpha=0.2)
    axes[1].set_xlabel('Wire')
    axes[1].set_ylabel('Time (tick)')
    axes[1].set_title(f'{label} Event {event_idx} - Occlusion Map')

    plt.tight_layout()
    plt.savefig(output_dir + f'occlusion_{label.lower()}_event{event_idx}.png')
    plt.close()
    print(f"Saved occlusion_{label.lower()}_event{event_idx}.png")

print("Done.")