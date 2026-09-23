import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import re
import os

output_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/"
os.makedirs(output_dir, exist_ok=True)

# ---------------------------------------------------------
# 1. CNN: parse from raw training log
# ---------------------------------------------------------
cnn_log_path = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/output/outputfile.4694350"

def parse_cnn_log(path):
    with open(path, 'r', errors='ignore') as f:
        text = f.read()

    # Extract (epoch, batch_loss) for training loss
    train_matches = re.findall(r'Train Epoch: (\d+)/\d+.*?Loss: ([\d\.]+)', text)
    # Extract (test_accuracy) entries, in order (this is validation accuracy)
    acc_matches = re.findall(r'Test Accuracy ([\d\.]+)', text)
    # Extract epoch for each "Start eval on test sample" block to align with accuracy
    epoch_for_acc = re.findall(r'Start eval on test sample.*?@epoch\.\.(\d+)', text)
    # Extract validation-style loss following "Start eval on training sample"
    val_loss_matches = re.findall(r'Test Loss ([\d\.]+)', text)
    # Extract train accuracy entries
    train_acc_matches = re.findall(r'Train Accuracy ([\d\.]+)', text)

    train_epochs = np.array([int(e) for e, _ in train_matches])
    train_losses_raw = np.array([float(l) for _, l in train_matches])

    acc_epochs = np.array([int(e) for e in epoch_for_acc])
    acc_values = np.array([float(a) for a in acc_matches])
    train_acc_values = np.array([float(a) for a in train_acc_matches])

    val_loss_values = np.array([float(l) for l in val_loss_matches])
    # val_loss_values, train_acc_values should align 1:1 with acc_epochs
    n = min(len(acc_epochs), len(val_loss_values), len(train_acc_values))
    acc_epochs = acc_epochs[:n]
    acc_values = acc_values[:n]
    val_loss_values = val_loss_values[:n]
    train_acc_values = train_acc_values[:n]

    n_epochs = train_epochs.max() + 1

    epoch_train_loss = []
    epoch_val_loss = []
    epoch_val_acc = []
    epoch_train_acc = []
    for ep in range(n_epochs):
        mask_train = train_epochs == ep
        mask_val = acc_epochs == ep
        epoch_train_loss.append(train_losses_raw[mask_train].mean())
        epoch_val_loss.append(val_loss_values[mask_val].mean())
        epoch_val_acc.append(acc_values[mask_val].mean())
        epoch_train_acc.append(train_acc_values[mask_val].mean())

    return np.array(epoch_train_loss), np.array(epoch_val_loss), np.array(epoch_val_acc), np.array(epoch_train_acc)

cnn_train_loss, cnn_val_loss, cnn_val_acc, cnn_train_acc = parse_cnn_log(cnn_log_path)
cnn_epochs = np.arange(1, len(cnn_train_loss) + 1)
print(f"CNN: total epochs={len(cnn_train_loss)}")
print(f"  Final val_loss={cnn_val_loss[-1]:.4f}, Final val_acc={cnn_val_acc[-1]:.4f}")
best_cnn_epoch = np.argmin(cnn_val_loss) + 1
print(f"  Best epoch={best_cnn_epoch} (val_loss={cnn_val_loss[best_cnn_epoch-1]:.4f})")

plt.figure(figsize=(7, 5))
plt.plot(cnn_epochs, cnn_train_acc, label='Train Accuracy', marker='o')
plt.plot(cnn_epochs, cnn_val_acc, label='Validation Accuracy', marker='o')
plt.axvline(best_cnn_epoch, color='grey', linestyle='--', alpha=0.6, label='Best epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('CNN Training Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir + 'cnn_training_curve.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir + 'cnn_training_curve.pdf', bbox_inches='tight')
plt.close()
print("Saved cnn_training_curve.png/.pdf")

# ---------------------------------------------------------
# 2. GCN: load from npy files
# ---------------------------------------------------------
gcn_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/"
gcn_train_loss = np.load(gcn_dir + 'train_losses.npy')
gcn_val_loss   = np.load(gcn_dir + 'val_losses.npy')
gcn_train_acc  = np.load(gcn_dir + 'train_accuracies.npy')
gcn_val_acc    = np.load(gcn_dir + 'val_accuracies.npy')
gcn_epochs     = np.arange(1, len(gcn_train_loss) + 1)
best_gcn_epoch = np.argmin(gcn_val_loss) + 1
print(f"\nGCN: total epochs={len(gcn_train_loss)}, best epoch={best_gcn_epoch} (val_loss={gcn_val_loss[best_gcn_epoch-1]:.4f})")

plt.figure(figsize=(7, 5))
plt.plot(gcn_epochs, gcn_train_acc, label='Train Accuracy', marker='o')
plt.plot(gcn_epochs, gcn_val_acc, label='Validation Accuracy', marker='o')
plt.axvline(best_gcn_epoch, color='grey', linestyle='--', alpha=0.6, label='Best epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('GCN Training Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(gcn_dir + 'gcn_training_curve.png', dpi=150, bbox_inches='tight')
plt.savefig(gcn_dir + 'gcn_training_curve.pdf', bbox_inches='tight')
plt.close()
print("Saved gcn_training_curve.png/.pdf")

# ---------------------------------------------------------
# 3. Graph Transformer: load from npy files (signed edge version)
# ---------------------------------------------------------
tr_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer_signed/"
if not os.path.exists(tr_dir + 'train_losses.npy'):
    # fallback to non-signed dir if signed npy not saved
    tr_dir = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/"
    print(f"\nWarning: using fallback directory {tr_dir} for Transformer curves")

tr_train_loss = np.load(tr_dir + 'train_losses.npy')
tr_val_loss   = np.load(tr_dir + 'val_losses.npy')
tr_train_acc  = np.load(tr_dir + 'train_accuracies.npy')
tr_val_acc    = np.load(tr_dir + 'val_accuracies.npy')
tr_epochs     = np.arange(1, len(tr_train_loss) + 1)
best_tr_epoch = np.argmin(tr_val_loss) + 1
print(f"\nTransformer: total epochs={len(tr_train_loss)}, best epoch={best_tr_epoch} (val_loss={tr_val_loss[best_tr_epoch-1]:.4f})")

plt.figure(figsize=(7, 5))
plt.plot(tr_epochs, tr_train_acc, label='Train Accuracy', marker='o')
plt.plot(tr_epochs, tr_val_acc, label='Validation Accuracy', marker='o')
plt.axvline(best_tr_epoch, color='grey', linestyle='--', alpha=0.6, label='Best epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Graph Transformer Network Training Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_dir + 'transformer_training_curve.png', dpi=150, bbox_inches='tight')
plt.savefig(output_dir + 'transformer_training_curve.pdf', bbox_inches='tight')
plt.close()
print("Saved transformer_training_curve.png/.pdf")

print("\nAll done.")