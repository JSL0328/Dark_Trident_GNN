import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
import pandas as pd

output_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/"

# CNN
cnn_df     = pd.read_csv('/vols/sbn/uboone/ll4420/dark_tridents_wspace/inference/MPID_test_set_full_DM-CNN_scores_excluded_final_9821_steps.csv')
cnn_scores = cnn_df['signal_score'].values
cnn_labels = (cnn_df['run_number'] == 100).astype(int).values
cnn_fpr, cnn_tpr, _ = roc_curve(cnn_labels, cnn_scores)
cnn_auc = auc(cnn_fpr, cnn_tpr)

# GraphConv
gnn_scores = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/test_scores.npy')
gnn_labels = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/truth.npy').astype(int)
gnn_fpr, gnn_tpr, _ = roc_curve(gnn_labels, gnn_scores)
gnn_auc = auc(gnn_fpr, gnn_tpr)

# Graph Transformer
tr_scores = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_signed/test_scores.npy')
tr_labels = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_signed/truth.npy').astype(int)
tr_fpr, tr_tpr, _ = roc_curve(tr_labels, tr_scores)
tr_auc = auc(tr_fpr, tr_tpr)

# Plot
fig, ax = plt.subplots(figsize=(8, 7))

ax.plot(cnn_fpr, cnn_tpr, color='steelblue',  linewidth=2, label=f'Convolutional Neural Network (AUC = {cnn_auc:.4f})')
ax.plot(gnn_fpr, gnn_tpr, color='darkorange',  linewidth=2, label=f'Graph Convolutional Network (AUC = {gnn_auc:.4f})')
ax.plot(tr_fpr,  tr_tpr,  color='forestgreen', linewidth=2, label=f'Graph Transformer Network (AUC = {tr_auc:.4f})')
ax.plot([0, 1],  [0, 1],  color='grey', linestyle='--', linewidth=1, label='Random classifier')

ax.set_xlabel('False Positive Rate (Background Acceptance)', fontsize=13)
ax.set_ylabel('True Positive Rate (Signal Efficiency)', fontsize=13)
ax.set_title('ROC Curve Comparison', fontsize=14)
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(output_dir + 'roc_comparison.pdf', dpi=180)
plt.close()
print(f"Saved roc_comparison.pdf to {output_dir}")
print(f"CNN AUC:         {cnn_auc:.4f}")
print(f"GraphConv AUC:   {gnn_auc:.4f}")
print(f"Transformer AUC: {tr_auc:.4f}")