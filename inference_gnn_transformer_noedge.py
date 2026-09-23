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

graphs_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/graphs/"
weights_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/weights_transformer_noedge/"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_noedge/"

os.makedirs(output_dir, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

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

input_dim   = 3
hidden_dims = [16, 32, 64, 128]
heads       = 4
model = GNNTransformerNoEdge(input_dim=input_dim, hidden_dims=hidden_dims, output_dim=1, heads=heads).to(device)
model.load_state_dict(torch.load(weights_dir + 'transformer_noedge_best.pt', map_location=device, weights_only=False))
model.eval()
print("Model loaded.")

print("Loading test graphs...")
test_sample = []
test_files = sorted([f for f in os.listdir(graphs_dir) if f.startswith("test_graphs_") and f.endswith('.pt')])
for f in test_files:
    chunk = torch.load(os.path.join(graphs_dir, f), weights_only=False)
    test_sample.extend(chunk)
print(f"Test graphs: {len(test_sample)}")

test_loader = DataLoader(test_sample, batch_size=16, shuffle=False)

test_scores, test_flags = [], []
for batch in test_loader:
    batch = batch.to(device)
    with torch.no_grad():
        dummy_mask = torch.tensor([
            g.x.shape[0] == 1 and g.x.sum().item() == 0
            for g in batch.to_data_list()
        ]).to(device)

        outputs = model(batch.x, batch.edge_index, batch.batch, skip_output_activation=True)
        scores  = torch.sigmoid(outputs).squeeze(-1)
        scores[dummy_mask] = 0.0

        test_scores.extend(scores.cpu().numpy())
        test_flags.extend(batch.y.cpu().numpy())

test_scores = np.array(test_scores)
test_flags  = np.array(test_flags)

np.save(output_dir + 'test_scores.npy', test_scores)
np.save(output_dir + 'truth.npy', test_flags)
print("Saved test_scores.npy and truth.npy")

test_acc    = ((test_scores > 0.5) == test_flags).mean()
fpr, tpr, _ = metrics.roc_curve(test_flags, test_scores)
auc_score   = metrics.auc(fpr, tpr)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"AUC: {auc_score:.4f}")

np.save(output_dir + 'fpr.npy', fpr)
np.save(output_dir + 'tpr.npy', tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'Graph Transformer (no edge features, AUC = {auc_score:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Graph Transformer without Edge Features')
plt.legend()
plt.savefig(output_dir + 'roc_curve.png')
plt.close()

print(f"All outputs saved to {output_dir}")
print("Done.")
