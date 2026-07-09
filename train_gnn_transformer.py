import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import time
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchinfo
from torch_geometric.nn import conv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
import os

# Paths
graphs_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs_edge/"
weights_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/weights_transformer/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/"

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(torch.cuda.get_device_name(0))

# Load all graphs from split files and shuffle
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

random.seed(42)
random.shuffle(all_graphs)

train_sample, val_sample = train_test_split(all_graphs, test_size=0.15, random_state=42)
print(f"Train: {len(train_sample)}, Val: {len(val_sample)}, Test: {len(test_sample)}")

# TransformerConv model with edge features
class GNNTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, edge_dim, heads=4):
        super(GNNTransformer, self).__init__()
        self.init_batch_norm = BatchNorm(input_dim)
        conv_layers = []
        for hidden_dim in hidden_dims:
            # TransformerConv output dim = hidden_dim * heads when concat=True
            transformer_conv = conv.TransformerConv(input_dim, hidden_dim, heads=heads, edge_dim=edge_dim, concat=True)
            batch_norm = BatchNorm(hidden_dim * heads)
            activation = nn.ReLU()
            conv_layers.append((transformer_conv, batch_norm, activation))
            input_dim = hidden_dim * heads  # next layer input
            edge_dim  = None                # edge_dim only for first layer
        self.conv_layers = nn.ModuleList([nn.ModuleList(layer) for layer in conv_layers])
        self.output_layer = nn.Linear(input_dim * 2, output_dim)  # *2 for mean+max

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

# Hyperparameters
input_dim     = 3    # wire, time, ADC
edge_dim      = 3    # wire distance, time distance, ADC difference
hidden_dims   = [32, 64, 128, 256]
heads         = 4
batch_size    = 32
n_epochs      = 100
learning_rate = 1e-3
patience      = 10

model     = GNNTransformer(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1, edge_dim=edge_dim, heads=heads).to(device)
optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

# Print model structure
dummy_x         = torch.randn((10, input_dim)).to(device)
dummy_edge_index= torch.tensor([[0,1,2,3,4,5,6,7,8,9],[1,0,3,2,5,4,7,6,9,8]], dtype=torch.long).to(device)
dummy_edge_attr = torch.randn((10, edge_dim)).to(device)
dummy_batch     = torch.zeros(10, dtype=torch.long).to(device)
print(torchinfo.summary(model, input_data=(dummy_x, dummy_edge_index, dummy_batch, dummy_edge_attr)))

# Training loop with early stopping
train_losses, val_losses         = [], []
train_accuracies, val_accuracies = [], []
best_val_loss                    = float('inf')
epochs_no_improve                = 0
stopped_epoch                    = n_epochs
training_start                   = time.time()

for epoch in range(n_epochs):
    epoch_start = time.time()
    model.train()
    epoch_loss, epoch_acc   = 0, 0
    total_train_batches     = 0
    train_loader = DataLoader(train_sample, batch_size=batch_size, shuffle=True)

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, skip_output_activation=True)
        scores  = torch.sigmoid(outputs).squeeze()
        loss    = criterion(outputs.squeeze(-1), batch.y.float())
        loss.backward()
        optimizer.step()
        epoch_loss          += loss.mean().item()
        epoch_acc           += ((scores > 0.5) == batch.y).float().mean().item()
        total_train_batches += 1

    epoch_loss /= total_train_batches
    epoch_acc  /= total_train_batches
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    model.eval()
    epoch_val_loss, epoch_val_acc = 0, 0
    total_val_batches             = 0
    val_loader = DataLoader(val_sample, batch_size=batch_size, shuffle=False)

    for batch in val_loader:
        batch = batch.to(device)
        with torch.no_grad():
            outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, skip_output_activation=True)
            scores  = torch.sigmoid(outputs).squeeze()
            loss    = criterion(outputs.squeeze(-1), batch.y.float())
            epoch_val_loss      += loss.mean().item()
            epoch_val_acc       += ((scores > 0.5) == batch.y).float().mean().item()
            total_val_batches   += 1

    epoch_val_loss /= total_val_batches
    epoch_val_acc  /= total_val_batches
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}, Time: {epoch_time:.1f}s")

    # Early stopping
    if epoch_val_loss < best_val_loss:
        best_val_loss     = epoch_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), weights_dir + 'transformer_model_best.pt')
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            stopped_epoch = epoch + 1
            break

total_time = time.time() - training_start
print(f"Total training time: {total_time/60:.1f} minutes")
torch.save(model.state_dict(), weights_dir + 'transformer_model_last.pt')
print(f"Weights saved to {weights_dir}")

# Save metrics
np.save(output_dir + 'train_losses.npy',     np.array(train_losses))
np.save(output_dir + 'val_losses.npy',       np.array(val_losses))
np.save(output_dir + 'train_accuracies.npy', np.array(train_accuracies))
np.save(output_dir + 'val_accuracies.npy',   np.array(val_accuracies))

epochs_ran = len(train_losses)
x_epochs   = list(range(1, epochs_ran + 1))

# Loss plot
plt.figure()
plt.plot(x_epochs, train_losses, label='Train Loss', marker='o', markersize=3)
plt.plot(x_epochs, val_losses,   label='Val Loss',   marker='o', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss (TransformerConv)')
plt.xticks(range(0, epochs_ran + 1, 10))
plt.legend()
plt.tight_layout()
plt.savefig(output_dir + 'loss_curve.png')
plt.close()

# Accuracy plot
plt.figure()
plt.plot(x_epochs, train_accuracies, label='Train Accuracy', marker='o', markersize=3)
plt.plot(x_epochs, val_accuracies,   label='Val Accuracy',   marker='o', markersize=3)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy (TransformerConv)')
plt.xticks(range(0, epochs_ran + 1, 10))
plt.legend()
plt.tight_layout()
plt.savefig(output_dir + 'accuracy_curve.png')
plt.close()

print(f"Training complete. Stopped at epoch {stopped_epoch}.")
