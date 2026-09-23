import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import os
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch_geometric.nn import conv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
import optuna

# Paths
graphs_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs_edge/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_optuna/"

os.makedirs(output_dir, exist_ok=True)

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
train_sample, val_sample = train_test_split(all_graphs, test_size=0.15, random_state=42)
print(f"Train: {len(train_sample)}, Val: {len(val_sample)}")

class GNNTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, edge_dim, heads=4, dropout=0.0):
        super(GNNTransformer, self).__init__()
        self.init_batch_norm = BatchNorm(input_dim)
        self.dropout = nn.Dropout(dropout)
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
            x = self.dropout(x)
        x = torch.cat([global_mean_pool(x, batch_indices), global_max_pool(x, batch_indices)], dim=1)
        x = self.output_layer(x)
        if not skip_output_activation:
            x = torch.sigmoid(x)
        return x

def objective(trial):
    lr          = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    heads       = trial.suggest_categorical('heads', [2, 4, 8])
    n_layers    = trial.suggest_int('n_layers', 2, 4)
    hidden_dim  = trial.suggest_categorical('hidden_dim', [32, 64, 128, 256])
    dropout     = trial.suggest_float('dropout', 0.0, 0.3)
    hidden_dims = [hidden_dim * (2**i) for i in range(n_layers)]

    model     = GNNTransformer(input_dim=3, hidden_dims=hidden_dims, output_dim=1,
                                edge_dim=3, heads=heads, dropout=dropout).to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    best_val_acc    = 0.0
    best_val_loss   = float('inf')
    epochs_no_improve = 0
    patience        = 10
    smooth_window   = 5
    val_losses      = []

    for epoch in range(100):
        model.train()
        train_loader = DataLoader(train_sample, batch_size=16, shuffle=True)
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, skip_output_activation=True)
            loss    = criterion(outputs.squeeze(-1), batch.y.float())
            loss.backward()
            optimizer.step()

        model.eval()
        epoch_val_loss, epoch_val_acc = 0, 0
        total_val_batches = 0
        val_loader = DataLoader(val_sample, batch_size=16, shuffle=False)
        for batch in val_loader:
            batch = batch.to(device)
            with torch.no_grad():
                outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, skip_output_activation=True)
                scores  = torch.sigmoid(outputs).squeeze(-1)
                loss    = criterion(outputs.squeeze(-1), batch.y.float())
                epoch_val_loss += loss.mean().item()
                epoch_val_acc  += ((scores > 0.5) == batch.y).float().mean().item()
                total_val_batches += 1

        epoch_val_loss /= total_val_batches
        epoch_val_acc  /= total_val_batches
        val_losses.append(epoch_val_loss)

        scheduler.step(epoch_val_loss)

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc

        smoothed_val_loss = np.mean(val_losses[-smooth_window:])
        if smoothed_val_loss < best_val_loss:
            best_val_loss     = smoothed_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

        trial.report(epoch_val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_val_acc

# Run Optuna
study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
study.optimize(objective, n_trials=20)

# Save results
results = {
    'best_params': study.best_params,
    'best_val_accuracy': study.best_value,
    'all_trials': [
        {
            'trial': t.number,
            'params': t.params,
            'val_accuracy': t.value,
            'state': str(t.state)
        }
        for t in study.trials
    ]
}

with open(output_dir + 'optuna_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nBest params: {study.best_params}")
print(f"Best val accuracy: {study.best_value:.4f}")

# Plot
trial_numbers = [t.number for t in study.trials if t.value is not None]
trial_values  = [t.value for t in study.trials if t.value is not None]

plt.figure(figsize=(10, 5))
plt.plot(trial_numbers, trial_values, 'o-', color='blue', markersize=5)
plt.axhline(y=study.best_value, color='red', linestyle='--', label=f'Best: {study.best_value:.4f}')
plt.xlabel('Trial')
plt.ylabel('Val Accuracy')
plt.title('Optuna Trial History (TransformerConv GNN)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir + 'optuna_history.png', dpi=150)
plt.close()

print("Done.")
