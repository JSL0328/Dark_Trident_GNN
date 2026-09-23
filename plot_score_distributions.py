import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

output_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/"

# CNN
cnn_df     = pd.read_csv('/vols/sbn/uboone/ll4420/dark_tridents_wspace/inference/MPID_test_set_full_DM-CNN_scores_excluded_final_9821_steps.csv')
cnn_scores = cnn_df['signal_score'].values
cnn_labels = (cnn_df['run_number'] == 100).astype(int).values

# GraphConv
gnn_scores = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/test_scores.npy')
gnn_labels = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/truth.npy').astype(int)

# Transformer
tr_scores = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_signed/test_scores.npy')
tr_labels = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_signed/truth.npy').astype(int)

models = [
    ('Convolutional Neural Network', cnn_scores, cnn_labels),
    ('Graph Convolutional Network',  gnn_scores, gnn_labels),
    ('Graph Transformer Network',    tr_scores,  tr_labels),
]

bins = np.linspace(0, 1, 51)

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for ax, (name, scores, labels) in zip(axes, models):
    sig = scores[labels == 1]
    bkg = scores[labels == 0]
    ax.hist(bkg, bins=bins, alpha=0.6, color='steelblue', label='Background', density=True)
    ax.hist(sig, bins=bins, alpha=0.6, color='darkorange', label='Signal', density=True)
    ax.set_yscale('log')
    ax.set_xlabel('Signal Score', fontsize=15)
    ax.set_ylabel('Normalised Counts', fontsize=15)
    ax.set_title(name, fontsize=16)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir + 'score_distributions.pdf', dpi=150)
plt.close()
print(f"Saved score_distributions.pdf to {output_dir}")