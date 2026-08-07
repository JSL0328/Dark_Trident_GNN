import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
from torch_geometric.nn import conv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
import os

# Paths
graphs_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs_edge/"
output_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

print("Loading graphs...")
all_graphs = []
train_files = sorted([f for f in os.listdir(graphs_dir) if f.startswith("train_graphs_") and f.endswith('.pt')])
for f in train_files:
    chunk = torch.load(os.path.join(graphs_dir, f), weights_only=False)
    all_graphs.extend(chunk)

random.seed(42)
random.shuffle(all_graphs)
train_sample, _ = train_test_split(all_graphs, test_size=0.15, random_state=42)
print(f"Train: {len(train_sample)}")

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

class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GNNClassifier, self).__init__()
        self.init_batch_norm = BatchNorm(input_dim)
        conv_layers = []
        for hidden_dim in hidden_dims:
            graph_conv = conv.GraphConv(input_dim, hidden_dim)
            batch_norm = BatchNorm(hidden_dim)
            activation = nn.ReLU()
            conv_layers.append((graph_conv, batch_norm, activation))
            input_dim = hidden_dim
        self.conv_layers = nn.ModuleList([nn.ModuleList(layer) for layer in conv_layers])
        self.output_layer = nn.Linear(input_dim * 2, output_dim)

    def forward(self, data, edges, batch_indices, skip_output_activation=False):
        x = self.init_batch_norm(data)
        for graph_conv, batch_norm, activation in self.conv_layers:
            x = graph_conv(x, edges)
            x = batch_norm(x)
            x = activation(x)
        x = torch.cat([global_mean_pool(x, batch_indices), global_max_pool(x, batch_indices)], dim=1)
        x = self.output_layer(x)
        if not skip_output_activation:
            x = torch.sigmoid(x)
        return x

def run_lr_finder(model, train_sample, model_name, edge_attr=True,
                  start_lr=1e-7, end_lr=1e-1, num_iter=300, batch_size=16):
    print(f"\nRunning LR finder for {model_name}...")
    model = model.to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=start_lr)
    criterion = nn.BCEWithLogitsLoss()

    loader = DataLoader(train_sample[:num_iter * batch_size], batch_size=batch_size, shuffle=True)

    lrs, losses = [], []
    best_loss = float('inf')
    lr        = start_lr
    lr_mult   = (end_lr / start_lr) ** (1 / num_iter)

    model.train()
    for i, batch in enumerate(loader):
        if i >= num_iter:
            break

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        batch = batch.to(device)
        optimizer.zero_grad()

        if edge_attr:
            outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, skip_output_activation=True)
        else:
            outputs = model(batch.x, batch.edge_index, batch.batch, skip_output_activation=True)

        loss = criterion(outputs.squeeze(-1), batch.y.float())

        if torch.isnan(loss):
            print(f"NaN loss at lr={lr:.2e}, stopping.")
            break

        loss.backward()
        optimizer.step()

        lrs.append(lr)
        losses.append(loss.item())
        lr *= lr_mult

        if i % 20 == 0:
            print(f"  iter {i}/{num_iter}, lr={lr:.2e}, loss={loss.item():.4f}")

    # Smooth with moving average (window=10)
    smoothed = np.convolve(losses, np.ones(20)/20, mode='valid')
    smoothed_lrs = lrs[19:]

    return lrs, losses, smoothed_lrs, smoothed

import random
import numpy as np
random.seed(2)
np.random.seed(2)
torch.manual_seed(2)
torch.cuda.manual_seed(2)

# Run for both models
transformer = GNNTransformer(input_dim=3, hidden_dims=[16,32,64,128], output_dim=1, edge_dim=3, heads=4)
gnn         = GNNClassifier(input_dim=3, hidden_dims=[32,64,128,256], output_dim=1)

results = {}
for model, name, use_edge, bs in [(gnn, 'Graph Convolutional Network', False, 32), (transformer, 'Graph Transformer Network', True, 16)]:
    lrs, losses, s_lrs, s_losses = run_lr_finder(model, train_sample, name, edge_attr=use_edge, batch_size=bs, num_iter=300)
    results[name] = {'lrs': lrs, 'losses': losses, 's_lrs': s_lrs, 's_losses': s_losses}

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
optimal_regions = {
    'Graph Convolutional Network': (2e-4, 1.25e-3),
    'Graph Transformer Network':   (4e-5, 2.5e-4)
}
optimal_lr = {
    'Graph Convolutional Network': 5e-4,
    'Graph Transformer Network':   1e-4,
}

for ax, (name, res) in zip(axes, results.items()):
    ax.plot(res['lrs'], res['losses'], alpha=0.3, color='blue', label='Raw loss')
    ax.plot(res['s_lrs'], res['s_losses'], color='red', linewidth=2, label='Smoothed loss')
    x_min, x_max = optimal_regions[name]
    ax.axvspan(x_min, x_max, alpha=0.2, color='green', label='Optimal learning rate region')
    ax.axvline(x=optimal_lr[name], color='green', linestyle='--', label=f'Optimal learning rate: {optimal_lr[name]:.2e}')
    ax.set_xscale('log')
    ax.set_xlabel('Learning Rate')
    ax.set_ylabel('Loss')
    ax.set_title(f'Learning Rate Range Test: {name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir + 'lr_finder_opt.png', dpi=150)
plt.close()
print(f"\nSaved lr_finder_opt.png to {output_dir}")