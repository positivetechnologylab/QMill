import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 22 
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22

ideal = np.load(os.path.join('ce_high_sim_data_ideal', 'ce_swaptest_high_data_ideal.npz'))
noisy = np.load(os.path.join('ce_high_sim_data',       'ce_swaptest_high_data_sim.npz'))
real  = np.load(os.path.join('ce_high_data',           'ce_swaptest_high_data.npz'))

ce_ideal_raw = ideal['ce']
ce_noisy_raw = noisy['ce']
ce_real_raw  = real['ce']

# ————————————————————————————
# 2. Normalize everything into [0,1]
def normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn)

ce_ideal = normalize(ce_ideal_raw)
ce_noisy = normalize(ce_noisy_raw)
ce_real  = normalize(ce_real_raw)

# ————————————————————————————
# 3. Define the five uniform bins on ideal CE
n_bins    = 5
bin_edges = np.linspace(0.0, 1.0, n_bins+1)
labels    = [f"{bin_edges[i]:.1f}–{bin_edges[i+1]:.1f}" for i in range(n_bins)]

# 4. Digitize ideal CE into those bins
idx_ideal = np.clip(np.digitize(ce_ideal, bin_edges, right=False), 1, n_bins)

# 5. Group noisy & real CE by the idealbin assignments
noisy_by_ideal = [ce_noisy[idx_ideal == i] for i in range(1, n_bins+1)]
real_by_ideal  = [ce_real [idx_ideal == i] for i in range(1, n_bins+1)]

# ————————————————————————————
# 6. Plot paired boxplots
x      = np.arange(1, n_bins+1)
width  = 0.3
gap    = 0.1
offset = width/2 + gap/2

color_noisy = "#4ae6cd"
color_real  = "#1d4670"

fig, ax = plt.subplots(figsize=(10,5))

# Ideal - Noisy
bp1 = ax.boxplot(
    noisy_by_ideal,
    positions = x - offset,
    widths    = width,
    patch_artist=True,
    showfliers=False
)
# Ideal - Real
bp2 = ax.boxplot(
    real_by_ideal,
    positions = x + offset,
    widths    = width,
    patch_artist=True,
    showfliers=False
)

for bp, col in ((bp1, color_noisy), (bp2, color_real)):
    for box in bp['boxes']:
        box.set_facecolor(col)
        box.set_edgecolor('black')
    for whisker in bp['whiskers']:
        whisker.set_color(col)
    for cap in bp['caps']:
        cap.set_color(col)
    for median in bp['medians']:
        median.set_color('black')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_xlabel('Normalized Ideal CE')
ax.set_ylabel('Normalized CE')
ax.set_ylim(0.0, 1.0)
ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
ax.grid(True, alpha=0.3, linestyle='--')
handles = [bp1["boxes"][0], bp2["boxes"][0]]
labels = ['Ideal → Noisy', 'Ideal → Real']

fig.legend(
    handles,
    labels,
    ncol=2,
    edgecolor='black',
    columnspacing=10.48,
    bbox_to_anchor=(0.985, 1.06),
)


plt.tight_layout()
plt.savefig('noisyideal.pdf', bbox_inches='tight')
