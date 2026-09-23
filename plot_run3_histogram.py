import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Directory setup
preselection_dir = "/vols/sbn/uboone/darkTridents/csv/run3/"
inference_dir    = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/inference/"
output_dir       = "/vols/sbn/uboone/ll4420/dark_tridents_wspace/DM-GNN/output/"
os.makedirs(output_dir, exist_ok=True)

# Run 3 NuMI normalization constants
POT_BEAMON     = 5.0e20
POT_NU         = 19.9e20
POT_DIRT       = 10.2e20
HW_BEAMON      = 10385450.0
HW_OFFBEAM     = 34147459.9
PPFX_WEIGHT    = 0.24
DIRT_FACTOR    = 0.35
OFFBEAM_FACTOR = 0.98

scale_nu      = (POT_BEAMON / POT_NU) * PPFX_WEIGHT
scale_dirt    = (POT_BEAMON / POT_DIRT) * DIRT_FACTOR
scale_offbeam = (HW_BEAMON / HW_OFFBEAM) * OFFBEAM_FACTOR

def load_scores(preselect_file, infer_file, is_nu_overlay=False):
    path_presel = os.path.join(preselection_dir, preselect_file)
    path_infer  = os.path.join(inference_dir, infer_file)

    if not os.path.exists(path_presel) or not os.path.exists(path_infer):
        print(f"Warning: Missing {preselect_file} or {infer_file}")
        return np.array([]), np.array([])

    df_presel = pd.read_csv(path_presel)
    df_infer  = pd.read_csv(path_infer)

    id_cols = ['run_number', 'subrun_number', 'event_number']
    for col in id_cols:
        df_presel[col] = df_presel[col].astype(int)
        df_infer[col]  = df_infer[col].astype(int)

    df_merged = pd.merge(df_infer, df_presel, on=id_cols, how='inner')

    # Filter for score >= 0.5 (Signal-rich region)
    df_selected = df_merged[(df_merged['signal_score'] >= 0.5) & (df_merged['signal_score'] <= 1.0)].copy()
    scores = df_selected['signal_score'].values

    # Event weights for nu overlay
    if is_nu_overlay and 'ppfx_cv_good' in df_selected.columns:
        weights = df_selected['ppfx_cv_good'].values * (POT_BEAMON / POT_NU)
    else:
        weights = None

    return scores, weights

# Load datasets
beamon_scores, _  = load_scores('run3_beamon_CNN.csv', 'run3_NuMI_beamon_larcv_cropped_full_set_DM-CNN_scores_run3_beamon.csv')
offbeam_scores, _ = load_scores('run3_offbeam_CNN.csv', 'run3_offbeam_larcv_cropped_DM-CNN_scores_run3_offbeam.csv')
nu_scores, nu_w   = load_scores('run3_nu_overlay_CNN.csv', 'run3_nu_overlay_larcv_cropped_DM-CNN_scores_run3_nu_overlay.csv', is_nu_overlay=True)
dirt_scores, _    = load_scores('run3_dirt_CNN.csv', 'run3_dirt_larcv_cropped_DM-CNN_scores_run3_dirt.csv')
signal_scores, _  = load_scores('run3_dt_ratio_0.6_ma_0.05_eta_CNN.csv', 'run3_dt_ratio_0.6_ma_0.05_eta_larcv_cropped_DM-CNN_scores_run3_signal_ma0p05.csv')

# Event weights
nu_weights = nu_w if nu_w is not None else np.ones(len(nu_scores)) * scale_nu
dirt_weights = np.ones(len(dirt_scores)) * scale_dirt
offbeam_weights = np.ones(len(offbeam_scores)) * scale_offbeam

# 10 Bins in Sigmoid scale: [0.5, 0.55, 0.6, ..., 1.0]
bins = np.linspace(0.5, 1.0, 11)
nbins = len(bins) - 1

bin_centers = 0.5 * (bins[:-1] + bins[1:])
bin_widths  = np.diff(bins)

# Calculate yields
h_nu, _      = np.histogram(nu_scores, bins=bins, weights=nu_weights)
h_nu_w2, _   = np.histogram(nu_scores, bins=bins, weights=nu_weights**2)
h_dirt, _    = np.histogram(dirt_scores, bins=bins, weights=dirt_weights)
h_dirt_w2, _ = np.histogram(dirt_scores, bins=bins, weights=dirt_weights**2)
h_off, _     = np.histogram(offbeam_scores, bins=bins, weights=offbeam_weights)
h_off_w2, _  = np.histogram(offbeam_scores, bins=bins, weights=offbeam_weights**2)

bkg_total = h_nu + h_dirt + h_off
stat_unc_bkg = np.sqrt(h_nu_w2 + h_dirt_w2 + h_off_w2)

# Systematic uncertainties
unc_incryo_sys = h_nu * np.sqrt(0.30**2 + 0.15**2 + 0.15**2 + 0.10**2)
unc_dirt_sys   = h_dirt * 1.00
unc_pot_sys    = (h_nu + h_dirt) * 0.02
tot_sys_unc    = np.sqrt(unc_incryo_sys**2 + unc_dirt_sys**2 + unc_pot_sys**2)
tot_bkg_unc    = np.sqrt(stat_unc_bkg**2 + tot_sys_unc**2)

# Beam-on data
beamon_hist, _ = np.histogram(beamon_scores, bins=bins)
beamon_err     = np.sqrt(beamon_hist)

# Chi-square Calculation
mask = (beamon_err**2 + tot_bkg_unc**2) > 0
chi2 = np.sum(((beamon_hist[mask] - bkg_total[mask])**2) / (beamon_err[mask]**2 + tot_bkg_unc[mask]**2))
ndf = np.sum(mask)
chi2_ndf = chi2 / ndf if ndf > 0 else 0.0

# Statistical deviation (Pull) printout
print("\n" + "="*70)
print(f"{'Bin Range':<15} | {'N_obs':<7} | {'N_bkg ± unc':<18} | {'Combined Unc':<14} | {'Pull (sigma)':<10}")
print("="*70)

for i in range(nbins):
    b_low, b_high = bins[i], bins[i+1]
    n_obs = beamon_hist[i]
    n_bkg = bkg_total[i]
    sig_bkg = tot_bkg_unc[i]
    sig_data = beamon_err[i]

    sig_comb = np.sqrt(sig_data**2 + sig_bkg**2)
    pull = (n_obs - n_bkg) / sig_comb if sig_comb > 0 else 0.0

    print(f"[{b_low:.2f}, {b_high:.2f}]   | {n_obs:<7.1f} | {n_bkg:6.2f} ± {sig_bkg:<8.2f} | {sig_comb:<14.2f} | {pull:+.2f} sigma")

print("="*70 + "\n")

# Figure setup
fig, (ax_main, ax_ratio) = plt.subplots(
    2, 1, figsize=(6.5, 6.8), sharex=True,
    gridspec_kw={'height_ratios': [3.2, 1.1], 'hspace': 0.06}
)

# Main panel: Stacked backgrounds
ax_main.bar(bin_centers, h_nu, width=bin_widths, color='#7B1FA2', edgecolor='black', linewidth=0.5, label='In cryo $\\nu$')
ax_main.bar(bin_centers, h_dirt, width=bin_widths, bottom=h_nu, color='#000080', edgecolor='black', linewidth=0.5, label='Out of cryo $\\nu$')
ax_main.bar(bin_centers, h_off, width=bin_widths, bottom=h_nu + h_dirt, color='#00BFFF', edgecolor='black', linewidth=0.5, label='Beam-off')

# Main panel: Total background uncertainty band
for i in range(nbins):
    ax_main.bar(bin_centers[i], 2 * tot_bkg_unc[i], width=bin_widths[i],
                bottom=bkg_total[i] - tot_bkg_unc[i],
                color='gray', alpha=0.45, hatch='///', edgecolor='dimgray', linewidth=0.5,
                label='Total background uncertainty' if i == 0 else "")

# Main panel: Scaled signal (Step line)
sig_hist, _ = np.histogram(signal_scores, bins=bins)
sig_scale   = (bkg_total.max() * 0.70) / sig_hist.max() if (len(sig_hist) > 0 and sig_hist.max() > 0) else 1.0
sig_step_y  = np.repeat(sig_hist * sig_scale, 2)
sig_step_x  = np.repeat(bins, 2)[1:-1]
ax_main.plot(sig_step_x, sig_step_y, color='red', linewidth=1.8, label=r'Dark trident $\varepsilon = 7 \times 10^{-4}$')

# Main panel: Beam-on data plotting
ax_main.errorbar(bin_centers, beamon_hist, yerr=beamon_err,
                 fmt='ko', markersize=4, label='Beam-on', zorder=5, capsize=2)

# Add headroom for the error bars on top too
main_upper = max((bkg_total + tot_bkg_unc).max(), (beamon_hist + beamon_err).max())
ax_main.set_ylim(0, main_upper * 1.25)
ax_main.set_ylabel('Events', fontsize=12)
ax_main.set_title(r'MicroBooNE NuMI Data Run $3$, $5.0 \times 10^{20}$ POT', fontsize=12)
ax_main.legend(fontsize=9, loc='upper left', framealpha=0.9)

# Chi2 text overlay (Top-Right)
ax_main.text(0.97, 0.95, rf'$\chi^2 / {ndf} = {chi2_ndf:.2f}$',
             transform=ax_main.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='none', pad=3))

ax_main.grid(True, alpha=0.25, linestyle=':')

# Ratio panel: (Data - Pred) / Pred
with np.errstate(divide='ignore', invalid='ignore'):
    fractional_diff = np.where(bkg_total > 0, (beamon_hist - bkg_total) / bkg_total, 0.0)
    data_rel_err    = np.where(bkg_total > 0, beamon_err / bkg_total, 0.0)
    bkg_rel_unc     = np.where(bkg_total > 0, tot_bkg_unc / bkg_total, 0.0)

ax_ratio.axhline(0.0, color='black', linestyle='--', linewidth=1.0)
for i in range(nbins):
    ax_ratio.bar(bin_centers[i], 2 * bkg_rel_unc[i], width=bin_widths[i],
                 bottom=-bkg_rel_unc[i],
                 color='gray', alpha=0.45, hatch='///', edgecolor='dimgray', linewidth=0.5)

ax_ratio.errorbar(bin_centers, fractional_diff, yerr=data_rel_err,
                  fmt='ko', markersize=4, zorder=5, capsize=2)

# Auto-scale the ratio panel so every error bar fits within the plot
max_extent = np.nanmax(np.abs(fractional_diff) + data_rel_err)
max_extent = max(max_extent, np.nanmax(bkg_rel_unc))
y_pad = max_extent * 1.15 if max_extent > 0 else 0.6
ax_ratio.set_ylim(-y_pad, y_pad)

ax_ratio.set_xlim(0.5, 1.0)
ax_ratio.set_xticks(np.arange(0.5, 1.05, 0.1))
ax_ratio.set_xlabel('CNN signal score (Sigmoid, $M_{A\'} = 50$ MeV)', fontsize=12)
ax_ratio.set_ylabel('(Data - Pred)/Pred', fontsize=10)
ax_ratio.grid(True, alpha=0.25, linestyle=':')

# Save output files
plt.tight_layout()
out_pdf = os.path.join(output_dir, 'run3_signal_score_sigmoid_histogram.pdf')
out_png = os.path.join(output_dir, 'run3_signal_score_sigmoid_histogram.png')
plt.savefig(out_pdf, bbox_inches='tight')
plt.savefig(out_png, dpi=200, bbox_inches='tight')
plt.close()

print(f"Saved: {out_pdf}")
print(f"Saved: {out_png}")