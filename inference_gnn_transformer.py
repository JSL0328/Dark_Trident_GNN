import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import conv, global_mean_pool, global_max_pool
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.loader import DataLoader
from sklearn import metrics
import os

# Paths
graphs_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs_edge/"
weights_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/weights_transformer/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/"

os.makedirs(output_dir, exist_ok=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Model (must match train_gnn_transformer.py)
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

# Load model and weights
input_dim   = 3
edge_dim    = 3
hidden_dims = [32, 64, 128, 256]
heads       = 4
model = GNNTransformer(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1, edge_dim=edge_dim, heads=heads).to(device)
model.load_state_dict(torch.load(weights_dir + 'transformer_model_best.pt', map_location=device, weights_only=False))
model.eval()
print("Model loaded.")

# Load test graphs from split files
print("Loading test graphs...")
test_sample = []
test_files = sorted([f for f in os.listdir(graphs_dir) if f.startswith("test_graphs_") and f.endswith('.pt')])
for f in test_files:
    chunk = torch.load(os.path.join(graphs_dir, f), weights_only=False)
    test_sample.extend(chunk)
print(f"Test graphs: {len(test_sample)}")

test_loader = DataLoader(test_sample, batch_size=32, shuffle=False)

# Run inference
test_scores = []
test_flags  = []
for batch in test_loader:
    batch = batch.to(device)
    with torch.no_grad():
        outputs = model(batch.x, batch.edge_index, batch.batch, edge_attr=batch.edge_attr, skip_output_activation=True)
        scores  = torch.sigmoid(outputs).squeeze()
        test_scores.extend(scores.cpu().numpy())
        test_flags.extend(batch.y.cpu().numpy())

test_scores = np.array(test_scores)
test_flags  = np.array(test_flags)

# Accuracy
test_acc = ((test_scores > 0.5) == test_flags).mean()
print(f"Test Accuracy: {test_acc:.4f}")

# ROC and AUC
fpr, tpr, _ = metrics.roc_curve(test_flags, test_scores)
auc_score   = metrics.auc(fpr, tpr)
print(f"AUC: {auc_score:.4f}")

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'TransformerConv GNN (AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Dark Trident TransformerConv GNN')
plt.legend()
plt.savefig(output_dir + 'roc_curve.png')
plt.close()

# Score distribution
bins = np.linspace(0, 1, 51)
for cls, label in [(0, 'Background'), (1, 'Signal')]:
    mask = test_flags == cls
    plt.hist(test_scores[mask], bins=bins, alpha=0.5, density=True, label=label)
plt.xlabel('Signal Score')
plt.ylabel('Normalised Counts')
plt.title('Score Distribution - TransformerConv GNN')
plt.legend()
plt.savefig(output_dir + 'score_distribution.png')
plt.close()

# Confusion matrix
predicted = (test_scores >= 0.5).astype(int)
cm        = metrics.confusion_matrix(test_flags, predicted)
norm_cm   = cm / cm.sum(axis=1, keepdims=True)

plt.figure()
plt.pcolormesh(norm_cm, cmap='Blues', shading='auto')
plt.xticks([0.5, 1.5], ['Background', 'Signal'])
plt.yticks([0.5, 1.5], ['Background', 'Signal'])
for i in range(2):
    for j in range(2):
        plt.text(j+0.5, i+0.5, f'{norm_cm[i,j]:.2f}', ha='center', va='center', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.colorbar()
plt.title('Normalised Confusion Matrix - TransformerConv GNN')
plt.savefig(output_dir + 'confusion_matrix.png')
plt.close()

print(f"All plots saved to {output_dir}")
print("Done.")
