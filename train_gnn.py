import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torchinfo
from torch_geometric.nn import conv, global_mean_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
import os

# Paths
graphs_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs/"
weights_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/weights/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/"

os.makedirs(weights_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
if device == 'cuda':
    print(torch.cuda.get_device_name(0))

# Load graphs
print("Loading graphs...")
all_graphs  = torch.load(graphs_dir + 'train_graphs.pt')
test_sample = torch.load(graphs_dir + 'test_graphs.pt')
print(f"Training graphs: {len(all_graphs)}, Test graphs: {len(test_sample)}")

# Move graphs to device
all_graphs  = [data.to(device) for data in all_graphs]
test_sample = [data.to(device) for data in test_sample]

# Train/val split
train_sample, val_sample = train_test_split(all_graphs, test_size=0.15, random_state=42)
print(f"Train: {len(train_sample)}, Val: {len(val_sample)}, Test: {len(test_sample)}")

# GNN model
class GNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(GNNClassifier, self).__init__()
        self.init_batch_norm = BatchNorm(input_dim)
        conv_layers = []
        for hidden_dim in hidden_dims:
            conv_layers.append((conv.GraphConv(input_dim, hidden_dim), BatchNorm(hidden_dim), nn.ReLU()))
            input_dim = hidden_dim
        self.conv_layers = nn.ModuleList([nn.ModuleList(layer) for layer in conv_layers])
        self.output_layer = nn.Linear(input_dim, output_dim)

    def forward(self, data, edges, batch_indices, skip_output_activation=False):
        x = self.init_batch_norm(data)
        for gnn_conv, batch_norm, activation in self.conv_layers:
            x = gnn_conv(x, edges)
            x = batch_norm(x)
            x = activation(x)
        x = global_mean_pool(x, batch_indices)
        x = self.output_layer(x)
        if not skip_output_activation:
            x = torch.sigmoid(x)
        return x

# Hyperparameters
input_dim      = 3   # wire, time, ADC
hidden_dims    = [8, 16, 32]
batch_size     = 32
n_epochs       = 20
learning_rate  = 1e-3

model = GNNClassifier(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1).to(device)
print(torchinfo.summary(model, input_data=(
    torch.randn((10, input_dim)).to(device),
    torch.tensor([[0,1,2,3,4,5,6,7,8,9],[1,0,3,2,5,4,7,6,9,8]], dtype=torch.long).to(device),
    torch.zeros(10, dtype=torch.long).to(device)
)))

optimizer = torch.optim.RAdam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()

train_loader = DataLoader(train_sample, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_sample,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_sample,  batch_size=batch_size, shuffle=False)

# Training loop
train_losses, val_losses         = [], []
train_accuracies, val_accuracies = [], []
model_states                     = []

for epoch in range(n_epochs):
    model.train()
    epoch_loss, epoch_acc = 0, 0
    train_loader = DataLoader(train_sample, batch_size=batch_size, shuffle=True)
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(batch.x, batch.edge_index, batch.batch, skip_output_activation=True)
        scores  = torch.sigmoid(outputs).squeeze()
        loss    = criterion(outputs.squeeze(), batch.y.float())
        loss.backward()
        optimizer.step()
        epoch_loss += loss.mean().item()
        epoch_acc  += ((scores > 0.5) == batch.y).float().mean().item()
    epoch_loss /= len(train_loader)
    epoch_acc  /= len(train_loader)
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    model_states.append(model.state_dict())

    model.eval()
    epoch_val_loss, epoch_val_acc = 0, 0
    for batch in val_loader:
        with torch.no_grad():
            outputs = model(batch.x, batch.edge_index, batch.batch, skip_output_activation=True)
            scores  = torch.sigmoid(outputs).squeeze()
            loss    = criterion(outputs.squeeze(), batch.y.float())
            epoch_val_loss += loss.mean().item()
            epoch_val_acc  += ((scores > 0.5) == batch.y).float().mean().item()
    epoch_val_loss /= len(val_loader)
    epoch_val_acc  /= len(val_loader)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    print(f"Epoch {epoch+1}/{n_epochs} - Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.4f}")

# Save best and last weights
optimal_epoch = np.argmin(val_losses)
print(f"Best model at epoch {optimal_epoch+1}, val loss: {val_losses[optimal_epoch]:.4f}, val acc: {val_accuracies[optimal_epoch]:.4f}")

torch.save(model_states[optimal_epoch], weights_dir + 'gnn_model_best.pt')
torch.save(model_states[-1],            weights_dir + 'gnn_model_last.pt')
print(f"Weights saved to {weights_dir}")

# Save metrics
np.save(output_dir + 'train_losses.npy',      np.array(train_losses))
np.save(output_dir + 'val_losses.npy',        np.array(val_losses))
np.save(output_dir + 'train_accuracies.npy',  np.array(train_accuracies))
np.save(output_dir + 'val_accuracies.npy',    np.array(val_accuracies))

# Loss and accuracy plots
plt.figure()
plt.plot(train_losses,     label='Train Loss')
plt.plot(val_losses,       label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig(output_dir + 'loss_curve.png')
plt.close()

plt.figure()
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(val_accuracies,   label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig(output_dir + 'accuracy_curve.png')
plt.close()

print("Done.")
