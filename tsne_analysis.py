import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.nn import conv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

# font settings for plots
plt.rcParams.update({
    'font.size': 14,
    'axes.labelsize': 15,
    'axes.titlesize': 16,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

graphs_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs_edge_signed/"
weights_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/weights_transformer_signed/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_signed/"

os.makedirs(output_dir, exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

class GNNTransformerWithEmbeddings(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, edge_dim, heads=4):
        super(GNNTransformerWithEmbeddings, self).__init__()
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

    def forward_with_embeddings(self, data, edges, batch_indices, edge_attr=None):
        x = self.init_batch_norm(data)
        layer_embeddings = []

        for i, (transformer_conv, batch_norm, activation) in enumerate(self.conv_layers):
            if i == 0:
                x = transformer_conv(x, edges, edge_attr=edge_attr)
            else:
                x = transformer_conv(x, edges)
            x = batch_norm(x)
            x = activation(x)
            graph_emb = torch.cat([global_mean_pool(x, batch_indices),
                                   global_max_pool(x, batch_indices)], dim=1)
            layer_embeddings.append(graph_emb.detach().cpu().numpy())

        final_pooled = torch.cat([global_mean_pool(x, batch_indices),
                                  global_max_pool(x, batch_indices)], dim=1)
        scores = torch.sigmoid(self.output_layer(final_pooled)).squeeze(-1)
        return layer_embeddings, scores

# Load the trained model
input_dim, edge_dim, hidden_dims, heads = 3, 3, [16, 32, 64, 128], 4
model = GNNTransformerWithEmbeddings(input_dim=input_dim, hidden_dims=hidden_dims,
                                     output_dim=1, edge_dim=edge_dim, heads=heads).to(device)
model.load_state_dict(torch.load(weights_dir + 'transformer_model_best.pt',
                                  map_location=device, weights_only=False))
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

test_loader = DataLoader(test_sample, batch_size=64, shuffle=False)

# Extract embeddings and scores
n_layers = len(hidden_dims)
all_embeddings = [[] for _ in range(n_layers)]
all_labels, all_scores = [], []

print("Extracting embeddings...")
for batch in test_loader:
    batch = batch.to(device)
    with torch.no_grad():
        layer_embs, scores = model.forward_with_embeddings(
            batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr
        )
        node_counts = torch.bincount(batch.batch, minlength=batch.num_graphs)
        x_sum_per_graph = torch.zeros(batch.num_graphs, device=device)
        x_sum_per_graph.scatter_add_(0, batch.batch, batch.x.sum(dim=1))
        dummy_mask = (node_counts == 1) & (x_sum_per_graph == 0)
        scores[dummy_mask] = 0.0

        for i, emb in enumerate(layer_embs):
            all_embeddings[i].append(emb)
        all_labels.extend(batch.y.cpu().numpy())
        all_scores.extend(scores.cpu().numpy())

all_labels = np.array(all_labels)
all_scores = np.array(all_scores)
for i in range(n_layers):
    all_embeddings[i] = np.vstack(all_embeddings[i])

print(f"Total events: {len(all_labels)}, Signal: {all_labels.sum()}")

# Fixed tracked events (manually selected for analysis)
node_counts_arr = np.array([g.x.shape[0] for g in test_sample])
tn_candidates = np.where((all_labels == 0) & (all_scores < 0.05) & (node_counts_arr >= 50))[0]

tracked = {
    'Signal (TP)':     1400,
    'Background (TN)': int(tn_candidates[0]),
    'FP':              5580,
    'FN':              2692,
}

tracked_colors = {
    'Signal (TP)':     '#D95F02',  # Dark Orange
    'Background (TN)': '#1F78B4',  # Deep Steel Blue
    'FP':              '#E41A1C',  # Crimson Red
    'FN':              '#2CA02C',  # Green
}
tracked_markers = {
    'Signal (TP)':     'o',  # Circle
    'Background (TN)': 's',  # Square
    'FP':              '^',  # Triangle Up
    'FN':              'v',  # Triangle Down
}
print("Tracked events (Fixed):", tracked)

# Sample 2000 events from each class for t-SNE visualisation
np.random.seed(42)
sig_idx = np.where(all_labels == 1)[0]
bkg_idx = np.where(all_labels == 0)[0]
n_sample = min(2000, len(sig_idx), len(bkg_idx))

sig_sample = np.random.choice(sig_idx, n_sample, replace=False)
bkg_sample = np.random.choice(bkg_idx, n_sample, replace=False)

for k, ev in tracked.items():
    if all_labels[ev] == 1 and ev not in sig_sample:
        sig_sample[0] = ev
    elif all_labels[ev] == 0 and ev not in bkg_sample:
        bkg_sample[0] = ev

sample_idx = np.concatenate([sig_sample, bkg_sample])
sample_labels = all_labels[sample_idx]

# Save sampled indices and labels for future reference
np.save(os.path.join(output_dir, 'tsne_sample_idx.npy'), sample_idx)
np.save(os.path.join(output_dir, 'tsne_sample_labels.npy'), sample_labels)
np.save(os.path.join(output_dir, 'tracked_events.npy'), tracked)

# t-SNE analysis plots for each layer
layer_names = [f'Layer {i+1} (dim = {hidden_dims[i]*heads*2})' for i in range(n_layers)]
fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

for i, name in enumerate(layer_names):
    emb_sample = all_embeddings[i][sample_idx]
    sil = silhouette_score(emb_sample, sample_labels)
    print(f"{name} - Silhouette Score: {sil:.4f}")

    print(f"Running t-SNE for {name}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    emb_2d = tsne.fit_transform(emb_sample)

    np.save(os.path.join(output_dir, f'tsne_2d_layer_{i+1}.npy'), emb_2d)

    ax = axes[i]
    ax.scatter(emb_2d[sample_labels==1, 0], emb_2d[sample_labels==1, 1],
               c='#FFA500', s=8, alpha=0.35, label='Signal')
    ax.scatter(emb_2d[sample_labels==0, 0], emb_2d[sample_labels==0, 1],
               c='#4682B4', s=8, alpha=0.35, label='Background')

    for track_label, track_event in tracked.items():
        pos = np.where(sample_idx == track_event)[0]
        if len(pos) > 0:
            ax.scatter(emb_2d[pos, 0], emb_2d[pos, 1],
                       c=tracked_colors[track_label],
                       s=150, marker=tracked_markers[track_label],
                       zorder=6, edgecolors='black', linewidths=1.2,
                       label=f"{track_label} (Evt {track_event})")

    ax.set_title(f'{name}\nSilhouette Score: {sil:.4f}', fontsize=16)
    ax.set_xlabel('t-SNE Dimension 1', fontsize=15)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=15)
    ax.tick_params(labelsize=13)
    ax.legend(fontsize=11, markerscale=1.2, loc='best', framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.4)

plt.suptitle('Layer-wise t-SNE of Graph Transformer Embeddings', fontsize=18, y=0.98)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'tsne_layerwise.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(output_dir, 'tsne_layerwise.pdf'), bbox_inches='tight')
plt.close()

print(f"\nAll tasks completed. Saved tsne_layerwise.png/pdf and 2D arrays to {output_dir}")