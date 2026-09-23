import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

def compute_auc_variance(labels, scores):
    """Compute AUC and its variance using DeLong method."""
    n1 = int(labels.sum())     # signal
    n0 = int((1-labels).sum()) # background

    sig_scores = scores[labels == 1]
    bkg_scores = scores[labels == 0]

    # Placement values
    V10 = np.array([np.mean(sig > bkg_scores) + 0.5 * np.mean(sig == bkg_scores) for sig in sig_scores])
    V01 = np.array([np.mean(bkg < sig_scores) + 0.5 * np.mean(bkg == sig_scores) for bkg in bkg_scores])

    auc = np.mean(V10)

    # Variance
    var = (np.var(V10) / n1 + np.var(V01) / n0)
    return auc, var, V10, V01

def delong_test(labels, scores1, scores2):
    """DeLong test comparing AUC of two models on the same test set."""
    auc1, var1, V10_1, V01_1 = compute_auc_variance(labels, scores1)
    auc2, var2, V10_2, V01_2 = compute_auc_variance(labels, scores2)

    n1 = int(labels.sum())
    n0 = int((1-labels).sum())

    # Covariance
    cov = (np.cov(V10_1, V10_2)[0,1] / n1 + np.cov(V01_1, V01_2)[0,1] / n0)

    # Test statistic
    z = (auc1 - auc2) / np.sqrt(var1 + var2 - 2 * cov)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return auc1, auc2, z, p_value

# Load scores
cnn_fpr   = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/inference/cnn_fpr.npy')
cnn_tpr   = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/inference/cnn_tpr.npy')

# CNN scores from CSV
import pandas as pd
df = pd.read_csv('/vols/sbn/uboone/ll4420/dark_tridents_wspace/inference/MPID_test_set_full_DM-CNN_scores_excluded_final_9821_steps.csv')
cnn_scores = df['signal_score'].values
cnn_labels = (df['run_number'] == 100).astype(int).values

gnn_scores = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/test_scores_csvordered.npy')
gnn_labels = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/truth_csvordered.npy').astype(int)

tr_scores  = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/test_scores_csvordered.npy')
tr_labels  = np.load('/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/truth_csvordered.npy').astype(int)

print(f"CNN    AUC: {roc_auc_score(cnn_labels, cnn_scores):.4f}, N={len(cnn_scores)}")
print(f"GNN    AUC: {roc_auc_score(gnn_labels, gnn_scores):.4f}, N={len(gnn_scores)}")
print(f"Transformer AUC: {roc_auc_score(tr_labels, tr_scores):.4f}, N={len(tr_scores)}")

print("\nDeLong Tests:")
auc1, auc2, z, p = delong_test(cnn_labels, cnn_scores, tr_scores)
print(f"CNN vs Transformer: AUC={auc1:.4f} vs {auc2:.4f}, z={z:.3f}, p={p:.4f}")

auc1, auc2, z, p = delong_test(cnn_labels, cnn_scores, gnn_scores)
print(f"CNN vs GNN:         AUC={auc1:.4f} vs {auc2:.4f}, z={z:.3f}, p={p:.4f}")

auc1, auc2, z, p = delong_test(tr_labels, tr_scores, gnn_scores)
print(f"Transformer vs GNN: AUC={auc1:.4f} vs {auc2:.4f}, z={z:.3f}, p={p:.4f}")
