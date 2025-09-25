import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os

FONTSZ = 16
plt.rcParams['text.usetex']     = True
plt.rcParams['font.family']     = 'serif'
plt.rcParams['font.size']       = FONTSZ
plt.rcParams['axes.labelsize']  = FONTSZ
plt.rcParams['axes.titlesize']  = FONTSZ
plt.rcParams['xtick.labelsize'] = FONTSZ
plt.rcParams['ytick.labelsize'] = FONTSZ
plt.rcParams['legend.fontsize'] = FONTSZ

from dists import Uniform, Normal, WeibullLeft, WeibullRight
from dists import MNIST, FashionMNIST, CIFAR, QCHEM, soil, dm

DIST_NAMES = [
    'Uniform', 'Normal', 'Left Weibull', 'Right Weibull',
    'MNIST', 'Fashion MNIST', 'CIFAR', 'QCHEM',
    'Soillow', 'Soilhigh', 'dmlow', 'dmhigh'
]

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

ANSATZ_NAMES = ['Sixteen', 'Five', 'Custom_One', 'Custom_Two']
ANSATZ_LABELS = {'Sixteen':'A1','Five':'A2','Custom_One':'A3','Custom_Two':'A4'}
BAR_COLORS   = {'Sixteen':'#ccfdfe','Five':'#4ae6cd','Custom_One':'#1d4670','Custom_Two':'#000000'}

def tvd_on_bins(p, q, bins=20, data_range=(0.0,0.4)):
    """Compute histogram TVD between two sample arrays p, q."""
    p_hist, edges = np.histogram(p, bins=bins, range=data_range, density=True)
    q_hist, _     = np.histogram(q, bins=edges, density=True)
    widths = np.diff(edges)
    return 0.5 * np.sum(np.abs(p_hist - q_hist) * widths)

def collect_results(bins=20):
    """
    For each ansatz and each distribution, load the trained
    CE array and the target CE samples, then compute 20‑bin TVD.
    Returns a dict: results[ansatz] = [tvd for dist1, dist2, ...].
    """
    results = {ans: [] for ans in ANSATZ_NAMES}

    for ans in ANSATZ_NAMES:
        for dist_name in DIST_NAMES:
            arr_path = Path('Annealing')/ans/dist_name/'5'/'1'/'1'/f'{ans}_5_1_results.npy'
            if not arr_path.exists():
                results[ans].append(np.nan)
                continue

            q = np.load(arr_path).flatten()

            dist = dist_ctors[dist_name]()
            dist.createSampleDistributions(1000)
            p = np.array(dist.samples).flatten()

            lo, hi = dist.Range
            tvd = tvd_on_bins(p, q, bins=bins, data_range=(lo, hi))
            results[ans].append(tvd)

    return results

def calculate_metrics(results):
    """Compute mean, median, variance, avg‑rank of each ansatz across all dists."""
    metrics = {}
    A = ANSATZ_NAMES
    for ans in A:
        vals = np.array(results[ans])
        mean   = np.nanmean(vals)
        median = np.nanmedian(vals)
        var    = np.nanvar(vals)

        ranks = []
        for i in range(len(DIST_NAMES)):
            col = [results[a][i] for a in A]
            if np.isnan(results[ans][i]):
                ranks.append(np.nan)
            else:
                ranks.append(sorted(col)[ : ].index(results[ans][i]) + 1)
        avg_rank = np.nanmean(ranks)

        metrics[ans] = {
            'mean_tvd':   mean,
            'median_tvd': median,
            'var_tvd':    var,
            'avg_rank':   avg_rank
        }
    return metrics

def create_comparison_plots(metrics):
    """Draw four side-by-side bars: mean, median, var, avg‑rank."""
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 1.2))
    A = sorted(ANSATZ_NAMES, key=lambda x: ANSATZ_LABELS[x])

    bar_kw = {
        'edgecolor':'black','linewidth':1,'zorder':3,
        'color': [BAR_COLORS[a] for a in A]
    }

    axes[0].bar(A, [metrics[a]['mean_tvd'] for a in A], **bar_kw)
    axes[0].set_ylim(0, 0.22)
    axes[0].set_yticks([0, 0.11, 0.22])
    axes[0].set_title("Mean"); axes[0].grid(True, axis='y', linestyle='--', alpha=0.7, zorder=1)

    axes[1].bar(A, [metrics[a]['median_tvd'] for a in A], **bar_kw)
    axes[1].set_ylim(0, max(0.2, max(metrics[a]['median_tvd'] for a in A)*1.1))
    axes[1].set_yticks([0, 0.1, 0.2])
    axes[1].set_title("Median"); axes[1].grid(True, axis='y', linestyle='--', alpha=0.7, zorder=1)


    axes[2].bar(A, [metrics[a]['var_tvd'] for a in A], **bar_kw)
    axes[2].set_ylim(0, 0.016)
    axes[2].set_yticks([0, 0.008, 0.016])
    axes[2].set_title("Variance"); axes[2].grid(True, axis='y', linestyle='--', alpha=0.7, zorder=1)

    axes[3].bar(A, [metrics[a]['avg_rank'] for a in A], **bar_kw)
    axes[3].set_ylim(0, 4)
    axes[3].set_yticks([0, 2, 4])
    axes[3].set_title("Rank")
    axes[3].yaxis.set_major_formatter(plt.FuncFormatter(lambda x,p: f"{x:.2f}"))
    axes[3].grid(True, axis='y', linestyle='--', alpha=0.7, zorder=1)

    # no x‑ticks
    for ax in axes:
        ax.set_xticks([])

    # shared legend
    legends = [plt.Rectangle((0,0),1,1,facecolor=BAR_COLORS[a],edgecolor='black') for a in A]
    labels  = [ANSATZ_LABELS[a] for a in A]
    fig.legend(legends, labels,
               loc='center', bbox_to_anchor=(0.5,1.25), ncol=len(A),
               handlelength=1.0, handletextpad=0.4,
               frameon=True, edgecolor='black', borderpad=0.3)

    plt.subplots_adjust(wspace=0.67)
    plt.savefig('ansatz_comparison.pdf', bbox_inches='tight')
    plt.close()

def main():
    results = collect_results(bins=20)
    metrics = calculate_metrics(results)
    create_comparison_plots(metrics)

    print("\nDetailed Metrics:")
    for ans, m in metrics.items():
        print(f" {ANSATZ_LABELS[ans]} – mean {m['mean_tvd']:.4f}, "
              f"med {m['median_tvd']:.4f}, var {m['var_tvd']:.5f}, "
              f"rank {m['avg_rank']:.2f}")

if __name__ == "__main__":
    main()
