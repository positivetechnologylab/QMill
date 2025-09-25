import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 18
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
plt.rcParams['legend.fontsize'] = 18

ansatzes = ['Sixteen', 'Five', 'Custom_One', 'Custom_Two']
ansatz_labels = {'Sixteen': 'A1', 'Five': 'A2', 'Custom_One': 'A3', 'Custom_Two': 'A4'}

arbitrary_dists = ['Uniform', 'Normal', 'Left Weibull', 'Right Weibull']
real_dists = ['MNIST', 'Fashion MNIST', 'CIFAR', 'QCHEM']
sensor_dists = ['Soillow', 'Soilhigh', 'dmlow', 'dmhigh']
all_distributions = arbitrary_dists + real_dists + sensor_dists

display_names = {
    'Uniform': 'Uniform',
    'Normal': 'Normal',
    'Left Weibull': 'L. Weibull',
    'Right Weibull': 'R. Weibull',
    'MNIST': 'MNIST',
    'Fashion MNIST': 'F. MNIST',
    'CIFAR': 'CIFAR',
    'QCHEM': 'QCHEM',
    'Soillow': 'Soil Low',
    'Soilhigh': 'Soil High',
    'dmlow': 'DM Low',
    'dmhigh': 'DM High'
}

# Colors from lightest to darkest (from first script)
bar_colors = ['#ccfdfe', '#4ae6cd', '#1d4670', '#000000']

# --- Computational Logic from Second Script ---

# 2) Helper to draw target samples
# NOTE: This assumes 'dists.py' provides constructors for ALL distributions used.
try:
    from dists import *
    class MockDist: # Placeholder if real constructors aren't in dists.py
        def __init__(self, size): self.size = size
        def createSampleDistributions(self, count): self.samples = np.random.rand(count, self.size) * 0.6 # Example data

    dist_ctors = {
        'Uniform':       lambda: Uniform(100),
        'Normal':        lambda: Normal(100),
        'Left Weibull':  lambda: WeibullLeft(100),
        'Right Weibull': lambda: WeibullRight(100),
        'MNIST':         lambda: MNIST(100), 
        'Fashion MNIST': lambda: FashionMNIST(100), 
        'CIFAR':         lambda: CIFAR(100), 
        'QCHEM':         lambda: QCHEM(100), 
        'Soillow':       lambda: soil(100,'low'), 
        'Soilhigh':      lambda: soil(100,'high'), 
        'dmlow':         lambda: dm(100,'low'), 
        'dmhigh':        lambda: dm(100,'high'), 
    }
except ImportError:
    print("Warning: 'dists.py' not found or incomplete. Using placeholder data for target samples.")
    class MockDist:
        def __init__(self, size): self.size = size
        def createSampleDistributions(self, count): self.samples = np.random.rand(count, self.size) * 0.6
    dist_ctors = {name: lambda: MockDist(100) for name in all_distributions} # Fallback

print("Generating target samples...")
target_samples = {}
for name in all_distributions:
    if name in dist_ctors:
        try:
            ctor = dist_ctors[name]
            d = ctor()
            d.createSampleDistributions(1000) # Generate 1000 target samples
            target_samples[name] = np.array(d.samples).flatten()
            print(f"  Generated target samples for {name}")
        except Exception as e:
            print(f"  Error generating target samples for {name}: {e}. Using random data.")
            target_samples[name] = np.random.rand(1000 * 100) * 0.6 
    else:
         print(f"  Warning: No constructor found for {name}. Using random data.")
         target_samples[name] = np.random.rand(1000 * 100) * 0.6
print("Target sample generation complete.")

# 3) TVD on same bins
def tvd_on_bins(p, q, bins=30, data_range=(0,0.6)):
    """Calculate Total Variation Distance based on histograms."""
    p_hist, edges = np.histogram(p, bins=bins, range=data_range, density=True)
    q_hist, _     = np.histogram(q, bins=edges, density=True)
    widths        = np.diff(edges)
    return 0.5 * np.sum(np.abs(p_hist - q_hist) * widths)

# 4) Collect results by loading .npy and computing histogram TVd
def collect_histogram_tvd(distributions, bins=30, data_range=(0, 0.6)):
    """Collect results by loading .npy files and calculating TVD against target samples."""
    results = {dist: [] for dist in distributions}
    print(f"\nCollecting results for distributions: {', '.join(distributions)}")
    for dist in distributions:
        results[dist] = []
        if dist not in target_samples:
            print(f"  Skipping {dist}: Missing target samples.")
            results[dist] = [np.nan] * len(ansatzes) # Fill with NaN if no target
            continue

        print(f"  Processing {dist}...")
        tgt = target_samples[dist]
        for ans in ansatzes:
            arr_path = Path('Annealing') / ans / dist / '5' / '1' / '1' / f'{ans}_5_1_results.npy'
            if not arr_path.exists():
                print(f"    {ans}: File not found at {arr_path}")
                results[dist].append(np.nan)
            else:
                try:
                    q_generated = np.load(arr_path).flatten()
                    if tgt.size == 0 or q_generated.size == 0:
                         print(f"    {ans}: Empty target ({tgt.size}) or generated ({q_generated.size}) data.")
                         results[dist].append(np.nan)
                         continue

                    tvd = tvd_on_bins(tgt, q_generated, bins=bins, data_range=data_range)
                    results[dist].append(tvd)
                except Exception as e:
                    print(f"    {ans}: Error processing file {arr_path}: {e}")
                    results[dist].append(np.nan)
    print("Result collection complete.")
    return results

def create_bar_plot(results, distributions, include_legend=True, y_max=0.8, title_suffix="Histogram TVD"):
    fig, ax = plt.subplots(figsize=(8, 3.2))

    ax.set_ylim(0, y_max)

    tick_interval = 0.2
    y_ticks = np.arange(0, y_max + tick_interval / 2, tick_interval)

    if np.isclose(y_max % tick_interval, 0) or np.isclose(y_max % tick_interval, tick_interval):
         y_ticks = np.arange(0, y_max + 1e-9, tick_interval)

    ax.set_yticks(y_ticks)
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))

    ax.grid(True, axis='y', linestyle='--', alpha=0.7, zorder=0)
    ax.yaxis.grid(True, which='major')

    x = np.arange(len(distributions))
    width = 0.2
    num_ansatzes = len(ansatzes)

    for i, ansatz in enumerate(ansatzes):
        costs = [results.get(dist, [np.nan]*num_ansatzes)[i] if results.get(dist) and i < len(results.get(dist)) else np.nan for dist in distributions]
        costs = [c if c is not None else np.nan for c in costs]

        bar_pos = x + i*width - (num_ansatzes - 1)*width/2
        ax.bar(bar_pos, costs, width,
               label=ansatz_labels[ansatz],
               color=bar_colors[i % len(bar_colors)], 
               edgecolor='black',
               linewidth=1,
               zorder=3) 

    ax.set_ylabel('TVD')
    ax.set_xticks(x)
    ax.set_xticklabels([display_names.get(dist, dist) for dist in distributions])

    if include_legend:
        ax.legend(loc='upper right',
                  frameon=True,
                  edgecolor='black',
                  fancybox=False,
                  bbox_to_anchor=(.18, 1.0),
                  handlelength=1.5,
                  handletextpad=0.4,
                  labelspacing=0.2)

    plt.tight_layout()

    return fig

os.makedirs('Paper Plots', exist_ok=True)

num_bins = 30
hist_data_range = (0, 0.6)

arbitrary_results = collect_histogram_tvd(arbitrary_dists, bins=num_bins, data_range=hist_data_range)
real_results      = collect_histogram_tvd(real_dists, bins=num_bins, data_range=hist_data_range)
sensor_results    = collect_histogram_tvd(sensor_dists, bins=num_bins, data_range=hist_data_range)

print("\nCreating plots...")

fig1 = create_bar_plot(arbitrary_results, arbitrary_dists,
                       include_legend=True, y_max=0.8)
save_path1 = 'Paper Plots/arbitrary_distributions_comparison.pdf'
plt.savefig(save_path1, bbox_inches='tight', pad_inches=0.05)
plt.close(fig1)
print(f"Saved plot: {save_path1}")

fig2 = create_bar_plot(real_results, real_dists,
                       include_legend=False, y_max=0.8)
save_path2 = 'Paper Plots/real_distributions_comparison.pdf'
plt.savefig(save_path2, bbox_inches='tight', pad_inches=0.05)
plt.close(fig2)
print(f"Saved plot: {save_path2}")

fig3 = create_bar_plot(sensor_results, sensor_dists,
                       include_legend=False, y_max=0.8)
save_path3 = 'Paper Plots/sensor_distributions_comparison.pdf'
plt.savefig(save_path3, bbox_inches='tight', pad_inches=0.05)
plt.close(fig3)
print(f"Saved plot: {save_path3}")

print("\n--- Numerical Results (Histogram TVD) ---")

def print_results(results_dict, dist_list):
    for dist in dist_list:
        print(f"\n{display_names.get(dist, dist)}:")
        if dist in results_dict:
            for i, ansatz in enumerate(ansatzes):
                 cost = results_dict[dist][i] if i < len(results_dict[dist]) else np.nan
                 cost_str = f"{cost:.4f}" if not np.isnan(cost) else "N/A (Missing Data or Error)"
                 print(f"  {ansatz_labels[ansatz]}: {cost_str}")
        else:
            print("  No results collected.")

print("\nArbitrary Distribution Results:")
print_results(arbitrary_results, arbitrary_dists)

print("\nReal Dataset Results:")
print_results(real_results, real_dists)

print("\nSensor Dataset Results:")
print_results(sensor_results, sensor_dists)

print("\n--- Script Finished ---")