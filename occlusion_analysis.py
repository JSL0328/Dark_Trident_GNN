import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.nn import conv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm

# font settings for plots
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'figure.titlesize': 18
})

graphs_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs_edge_signed/"
weights_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/weights_transformer_signed/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_signed/occlusion/"

os.makedirs(output_dir, exist_ok=True)

BOX_SIZE = 30
STEP     = 10

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

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

input_dim   = 3
edge_dim    = 3
hidden_dims = [16, 32, 64, 128]
heads       = 4
model = GNNTransformer(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1, edge_dim=edge_dim, heads=heads).to(device)
model.load_state_dict(torch.load(weights_dir + 'transformer_model_best.pt', map_location=device, weights_only=False))
model.eval()
print("Model loaded.")

print("Loading test graphs...")
test_sample = []
test_files = sorted([f for f in os.listdir(graphs_dir) if f.startswith("test_graphs_") and f.endswith('.pt')])
for f in test_files:
    chunk = torch.load(os.path.join(graphs_dir, f), weights_only=False)
    test_sample.extend(chunk)
print(f"Test graphs: {len(test_sample)}")

print("Running baseline inference...")
original_logits = []
original_scores = []
for graph in test_sample:
    g = graph.to(device)
    if g.x.shape[0] == 1 and g.x.sum().item() == 0:
        original_logits.append(0.0)
        original_scores.append(0.0)
        continue
    batch_idx = torch.zeros(g.x.shape[0], dtype=torch.long).to(device)
    with torch.no_grad():
        logit = model(g.x, g.edge_index, batch_idx, edge_attr=g.edge_attr, skip_output_activation=True)
        score = torch.sigmoid(logit)
    original_logits.append(logit.item())
    original_scores.append(score.item())

sig_indices = [3435]
fp_indices  = [1534]
fn_indices  = [2692]

node_counts   = np.array([g.x.shape[0] for g in test_sample])
labels_arr    = np.array([g.y.item() for g in test_sample])
tn_candidates = np.where((labels_arr == 0) & (np.array(original_scores) < 0.05) & (node_counts >= 50))[0]
tn_indices    = [tn_candidates[0]]

selected = ([(i, 'Signal (TP)')      for i in sig_indices] +
            [(i, 'False Positive')   for i in fp_indices]  +
            [(i, 'False Negative')   for i in fn_indices]  +
            [(i, 'Background (TN)')  for i in tn_indices])

def run_occlusion(graph, model, device, orig_logit):
    x    = graph.x.cpu().numpy()
    wire = x[:, 0]
    time = x[:, 1]

    wire_steps = np.arange(0, 512, STEP)
    time_steps = np.arange(0, 512, STEP)

    delta_map = np.full((len(time_steps), len(wire_steps)), np.nan)
    hit_map   = np.zeros((len(time_steps), len(wire_steps)), dtype=bool)

    for ti, t0 in enumerate(time_steps):
        for wi, w0 in enumerate(wire_steps):
            mask = ~((wire >= w0) & (wire < w0 + BOX_SIZE) &
                     (time >= t0) & (time < t0 + BOX_SIZE))

            hits_in_box = (~mask).sum()
            if hits_in_box == 0:
                continue

            hit_map[ti, wi] = True

            if mask.sum() < 2:
                delta_map[ti, wi] = orig_logit
                continue

            new_x      = graph.x[mask]
            old_to_new = -np.ones(len(mask), dtype=int)
            old_to_new[mask] = np.arange(mask.sum())

            edges     = graph.edge_index.cpu().numpy()
            edge_mask = mask[edges[0]] & mask[edges[1]]
            new_edges = old_to_new[edges[:, edge_mask]]

            if new_edges.shape[1] == 0:
                delta_map[ti, wi] = orig_logit
                continue

            new_edge_attr = graph.edge_attr[edge_mask] if graph.edge_attr is not None else None
            new_x          = new_x.to(device)
            new_edge_index = torch.tensor(new_edges, dtype=torch.long).to(device)
            new_batch      = torch.zeros(new_x.shape[0], dtype=torch.long).to(device)
            new_edge_attr  = new_edge_attr.to(device) if new_edge_attr is not None else None

            with torch.no_grad():
                logit = model(new_x, new_edge_index, new_batch, edge_attr=new_edge_attr, skip_output_activation=True)
            delta_map[ti, wi] = orig_logit - logit.item()

    return delta_map, hit_map, wire_steps, time_steps

# Run occlusion
results = []
for event_idx, label in selected:
    graph      = test_sample[event_idx]
    orig_logit = original_logits[event_idx]
    orig_score = original_scores[event_idx]
    print(f"Running occlusion for {label} Event {event_idx} (score={orig_score:.3f}, logit={orig_logit:.3f})...")
    delta_map, hit_map, wire_steps, time_steps = run_occlusion(graph, model, device, orig_logit)
    results.append((event_idx, label, graph, orig_score, delta_map, hit_map, wire_steps, time_steps))
    print(f"  Done.")

def plot_occlusion(results, ncols, figsize, filename):
    nrows = (len(results) + ncols - 1) // ncols
    fig, axes_all = plt.subplots(nrows, ncols * 2, figsize=figsize)
    if nrows == 1:
        axes_all = axes_all[np.newaxis, :]

    for plot_idx, (event_idx, label, graph, orig_score, delta_map, hit_map, wire_steps, time_steps) in enumerate(results):
        row = plot_idx // ncols
        col = (plot_idx % ncols) * 2

        x    = graph.x.cpu().numpy()
        wire = x[:, 0]
        time = x[:, 1]
        adc  = x[:, 2]

        extent = [0, 512, 0, 512]
        masked_delta = np.where(hit_map, delta_map, np.nan)

        ax_left  = axes_all[row, col]
        ax_right = axes_all[row, col + 1]

        # Left Plot: Hit Scatter with ADC
        sc = ax_left.scatter(wire, time, c=adc, cmap='viridis', s=6)
        cbar_left = plt.colorbar(sc, ax=ax_left)
        cbar_left.set_label('ADC', fontsize=14)
        cbar_left.ax.tick_params(labelsize=12)

        ax_left.set_xlim(0, 512)
        ax_left.set_ylim(0, 512)
        ax_left.set_xlabel('Wire', fontsize=15)
        ax_left.set_ylabel('Time (tick)', fontsize=15)
        ax_left.set_title(f'{label}\nEvent {event_idx} | Score = {orig_score:.3f}', fontsize=15)
        ax_left.tick_params(labelsize=13)

        # Right Plot: Occlusion Delta Logit Map
        vmax = max(0.01, np.nanmax(np.abs(masked_delta)))
        im = ax_right.imshow(masked_delta, origin='lower', aspect='auto',
                             extent=extent, cmap='RdYlGn_r', vmin=-vmax, vmax=vmax)
        cbar_right = plt.colorbar(im, ax=ax_right)
        cbar_right.set_label(r'$\Delta$ Logit', fontsize=14)
        cbar_right.ax.tick_params(labelsize=12)

        ax_right.scatter(wire, time, c='black', s=2, alpha=0.25)
        ax_right.set_xlim(0, 512)
        ax_right.set_ylim(0, 512)
        ax_right.set_xlabel('Wire', fontsize=15)
        ax_right.set_ylabel('Time (tick)', fontsize=15)
        ax_right.set_title(f'Occlusion Map\n{label}', fontsize=15)
        ax_right.tick_params(labelsize=13)

    plt.suptitle('Occlusion Sensitivity Analysis - Graph Transformer Network', fontsize=18, y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir + filename, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

# 2x4 (발표용)
plot_occlusion(results, ncols=2, figsize=(26, 13), filename='occlusion_2x4.pdf')

# 2x2 occlusion map only (리포트용)
fig, axes = plt.subplots(2, 2, figsize=(15, 13))
axes = axes.flatten()
for plot_idx, (event_idx, label, graph, orig_score, delta_map, hit_map, wire_steps, time_steps) in enumerate(results):
    x    = graph.x.cpu().numpy()
    wire = x[:, 0]
    time = x[:, 1]
    masked_delta = np.where(hit_map, delta_map, np.nan)
    vmax = max(0.01, np.nanmax(np.abs(masked_delta)))
    ax = axes[plot_idx]
    im = ax.imshow(masked_delta, origin='lower', aspect='auto',
                   extent=[0, 512, 0, 512], cmap='RdYlGn_r', vmin=-vmax, vmax=vmax)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r'$\Delta$ Logit', fontsize=14)
    cbar.ax.tick_params(labelsize=12)

    ax.scatter(wire, time, c='black', s=2, alpha=0.25)
    ax.set_xlim(0, 512)
    ax.set_ylim(0, 512)
    ax.set_xlabel('Wire', fontsize=15)
    ax.set_ylabel('Time (tick)', fontsize=15)
    ax.set_title(f'{label} | Event {event_idx} | Score = {orig_score:.3f}', fontsize=15)
    ax.tick_params(labelsize=13)

plt.suptitle('Occlusion Sensitivity Analysis - Graph Transformer Network', fontsize=18, y=0.98)
plt.tight_layout()
plt.savefig(output_dir + 'occlusion_2x2.pdf', bbox_inches='tight')
plt.close()
print("Saved occlusion_2x2.pdf")
print("Done.")