import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score
import os

N_BOOT  = 1000
SEED    = 42
np.random.seed(SEED)

output_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/"
os.makedirs(output_dir, exist_ok=True)

# Load scores and labels
# CNN
cnn_df     = pd.read_csv('/vols/sbn/uboone/ll4420/dark_tridents_wspace/inference/MPID_test_set_full_DM-CNN_scores_excluded_final_9821_steps.csv')
cnn_scores = cnn_df['signal_score'].values
cnn_labels = (cnn_df['run_number'] == 100).astype(int).values

# GraphConv
gnn_scores = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/test_scores.npy')
gnn_labels = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/truth.npy').astype(int)

# Graph Transformer (signed)
tr_scores = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_signed/test_scores.npy')
tr_labels = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_signed/truth.npy').astype(int)

models = [
    ('CNN',               cnn_scores, cnn_labels),
    ('GraphConv',         gnn_scores, gnn_labels),
    ('GraphTransformer',  tr_scores,  tr_labels),
]

results = {}

for name, scores, labels in models:
    print(f"\nBootstrapping {name} (N={N_BOOT})...")
    n = len(scores)

    auc_boot  = []
    acc_boot  = []

    for _ in range(N_BOOT):
        idx = np.random.choice(n, n, replace=True)
        s   = scores[idx]
        l   = labels[idx]
        auc_boot.append(roc_auc_score(l, s))
        acc_boot.append(accuracy_score(l, s > 0.5))

    auc_mean = np.mean(auc_boot)
    auc_std  = np.std(auc_boot)
    acc_mean = np.mean(acc_boot)
    acc_std  = np.std(acc_boot)

    # 95% CI
    auc_ci_lo = np.percentile(auc_boot, 2.5)
    auc_ci_hi = np.percentile(auc_boot, 97.5)

    results[name] = {
        'auc_mean': auc_mean, 'auc_std': auc_std,
        'auc_ci_lo': auc_ci_lo, 'auc_ci_hi': auc_ci_hi,
        'acc_mean': acc_mean, 'acc_std': acc_std,
    }

    print(f"  AUC = {auc_mean:.4f} ± {auc_std:.4f} (95% CI: [{auc_ci_lo:.4f}, {auc_ci_hi:.4f}])")
    print(f"  Accuracy = {acc_mean:.4f} ± {acc_std:.4f}")

# Save results
rows = []
for name, r in results.items():
    rows.append({
        'Model':       name,
        'AUC':         f"{r['auc_mean']:.4f}",
        'AUC_std':     f"{r['auc_std']:.4f}",
        'AUC_CI_lo':   f"{r['auc_ci_lo']:.4f}",
        'AUC_CI_hi':   f"{r['auc_ci_hi']:.4f}",
        'Accuracy':    f"{r['acc_mean']:.4f}",
        'Acc_std':     f"{r['acc_std']:.4f}",
    })

df_out = pd.DataFrame(rows)
df_out.to_csv(output_dir + 'bootstrap_results.csv', index=False)
print(f"\nSaved bootstrap_results.csv to {output_dir}")
print(df_out.to_string(index=False))
