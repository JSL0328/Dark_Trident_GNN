import os
import random
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch_geometric.nn import conv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
from torch.optim import RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchinfo import summary
import time

# Paths - NO EDGE FEATURES (graphs/ instead of graphs_edge_signed/)
graphs_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs/"
weights_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/weights_transformer_noedge/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_noedge/"

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(torch.cuda.get_device_name(0))

# Seed
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed(1)

print("Loading all training graphs...")
all_graphs = []
train_files = sorted([f for f in os.listdir(graphs_dir) if f.startswith("train_graphs_") and f.endswith('.pt')])
for f in train_files:
    chunk = torch.load(os.path.join(graphs_dir, f), weights_only=False)
    all_graphs.extend(chunk)
    print(f"Loaded {f}, total so far: {len(all_graphs)}")

print("Loading test graphs...")
test_sample = []
test_files = sorted([f for f in os.listdir(graphs_dir) if f.startswith("test_graphs_") and f.endswith('.pt')])
for f in test_files:
    chunk = torch.load(os.path.join(graphs_dir, f), weights_only=False)
    test_sample.extend(chunk)

random.shuffle(all_graphs)
split = int(0.85 * len(all_graphs))
train_sample = all_graphs[:split]
val_sample   = all_graphs[split:]
print(f"Train: {len(train_sample)}, Val: {len(val_sample)}, Test: {len(test_sample)}")

class GNNTransformerNoEdge(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, heads=4):
        super(GNNTransformerNoEdge, self).__init__()
        self.init_batch_norm = BatchNorm(input_dim)
        conv_layers = []
        for hidden_dim in hidden_dims:
            transformer_conv = conv.TransformerConv(input_dim, hidden_dim, heads=heads, concat=True)
            batch_norm = BatchNorm(hidden_dim * heads)
            activation = nn.ReLU()
            conv_layers.append((transformer_conv, batch_norm, activation))
            input_dim = hidden_dim * heads
        self.conv_layers = nn.ModuleList([nn.ModuleList(layer) for layer in conv_layers])
        self.output_layer = nn.Linear(input_dim * 2, output_dim)

    def forward(self, data, edges, batch_indices, skip_output_activation=False):
        x = self.init_batch_norm(data)
        for transformer_conv, batch_norm, activation in self.conv_layers:
            x = transformer_conv(x, edges)
            x = batch_norm(x)
            x = activation(x)
        x = torch.cat([global_mean_pool(x, batch_indices), global_max_pool(x, batch_indices)], dim=1)
        x = self.output_layer(x)
        if not skip_output_activation:
            x = torch.sigmoid(x)
        return x

# Hyperparameters (same as edge version)
input_dim     = 3
hidden_dims   = [16, 32, 64, 128]
heads         = 4
batch_size    = 16
n_epochs      = 100
learning_rate = 5e-4
patience      = 5

model     = GNNTransformerNoEdge(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1, heads=heads).to(device)
optimizer = RAdam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5)

train_loader = DataLoader(train_sample, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_sample,   batch_size=batch_size, shuffle=False)

# Model summary
sample_batch = next(iter(train_loader)).to(device)
summary(model, input_data=(sample_batch.x, sample_batch.edge_index, sample_batch.batch))

train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
best_val_loss     = float('inf')
epochs_no_improve = 0
stopped_epoch     = n_epochs

start_time = time.time()

for epoch in range(n_epochs):
    epoch_start = time.time()
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, batch.batch, skip_output_activation=True).squeeze(-1)
        loss    = criterion(outputs, batch.y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch.num_graphs
        preds    = (torch.sigmoid(outputs) > 0.5).float()
        correct += (preds == batch.y).sum().item()
        total   += batch.num_graphs

    epoch_train_loss = running_loss / total
    epoch_train_acc  = correct / total
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    model.eval()
    val_running_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            outputs = model(batch.x, batch.edge_index, batch.batch, skip_output_activation=True).squeeze(-1)
            loss    = criterion(outputs, batch.y)
            val_running_loss += loss.item() * batch.num_graphs
            preds        = (torch.sigmoid(outputs) > 0.5).float()
            val_correct += (preds == batch.y).sum().item()
            val_total   += batch.num_graphs

    epoch_val_loss = val_running_loss / val_total
    epoch_val_acc  = val_correct / val_total
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    scheduler.step(epoch_val_loss)
    current_lr  = optimizer.param_groups[0]['lr']
    epoch_time  = time.time() - epoch_start

    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}, "
          f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, LR: {current_lr:.2e}, Time: {epoch_time:.1f}s")

    if epoch_val_loss < best_val_loss:
        best_val_loss     = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), weights_dir + 'transformer_noedge_best.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            stopped_epoch = epoch + 1
            break

total_time = (time.time() - start_time) / 60
print(f"GPU training time: {total_time:.1f} minutes")
print(f"Training complete. Stopped at epoch {stopped_epoch}.")

np.save(output_dir + 'train_losses.npy',     np.array(train_losses))
np.save(output_dir + 'val_losses.npy',       np.array(val_losses))
np.save(output_dir + 'train_accuracies.npy', np.array(train_accuracies))
np.save(output_dir + 'val_accuracies.npy',   np.array(val_accuracies))
