import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import h5py
import os

# Paths
run3_hdf5   = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/hdf5/output_hdf5/run3_NuMI_beamon_full.h5"
scores_file = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/run3_scores.npy"
output_dir  = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output_transformer/event_display/"
logo_path   = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-CNN/lib/kalekologo_noshadow_notrack_inverted.png"

os.makedirs(output_dir, exist_ok=True)

scores = np.load(scores_file)
print(f"Total events: {len(scores)}")

top_indices = np.argsort(scores)[::-1][:5]
print(f"Top 5 scores: {scores[top_indices]}")

with h5py.File(run3_hdf5, 'r') as f:
    event_keys = list(f.keys())

    for rank, idx in enumerate(top_indices):
        event_id = event_keys[idx]
        sp     = f[event_id]['spacepoints'][:]
        run    = f[event_id].attrs.get('run', 0)
        subrun = f[event_id].attrs.get('subrun', 0)
        event  = f[event_id].attrs.get('event', 0)
        score  = scores[idx]

        wire   = sp[:, 0]
        time   = sp[:, 1]
        charge = sp[:, 2]

        # Build 512x512 image from spacepoints
        image = np.zeros((512, 512))
        w_idx = np.clip(wire.astype(int), 0, 511)
        t_idx = np.clip(time.astype(int), 0, 511)
        image[t_idx, w_idx] = charge
        image = np.clip(image, 0, 500)

        fig, ax = plt.subplots(1, 1, figsize=(20, 20), dpi=300)
        img = ax.imshow(image, origin="lower", cmap='jet',
                        norm=colors.PowerNorm(gamma=0.35, vmin=0, vmax=500))

        # Colorbar (Louis style)
        newax_cb = fig.add_axes([0.0001, 0.3, 0.2, 0.4], anchor='NE', zorder=1)
        cbar = fig.colorbar(img, ax=newax_cb, shrink=0.7, cmap='jet')
        cbar.set_ticks([0., 500.])
        cbar.ax.set_yticklabels(['Low charge', 'High charge'], color='white', fontsize=50)
        newax_cb.axis('off')

        # Logo (Louis style)
        if os.path.exists(logo_path):
            im_logo = plt.imread(logo_path)
            newax = fig.add_axes([0.15, 0.58, 0.3, 0.3], anchor='NE', zorder=1)
            newax.imshow(im_logo)
            newax.axis('off')

        ax.set_xticks([0, 511])
        ax.set_yticks([511])
        ax.tick_params(axis="y", which='major', direction="out", length=10, width=2.5, pad=10, labelsize=50)
        ax.tick_params(axis="x", which='major', direction="out", length=10, width=2.5, pad=10, labelsize=50)
        ax.set_xlim(0, 511)
        ax.set_ylim(0, 511)
        ax.set_xlabel('Wire Number', size=55, labelpad=1.0)
        ax.set_ylabel('Drift Time', size=55, labelpad=1.0)

        ax.text(20, 80, "MicroBooNE NuMI Data", color="white", fontsize=53)
        ax.text(20, 50, "Signal score: %.3f" % score, color="white", fontsize=53)
        ax.text(20, 20, "Run: {}, Subrun: {}, Event {}".format(run, subrun, event),
                color="white", fontsize=53)

        outfile = output_dir + f'event_display_rank{rank+1}_score{score:.3f}.png'
        plt.savefig(outfile, bbox_inches="tight")
        plt.close()
        print(f"Saved {outfile}")

print("Done.")