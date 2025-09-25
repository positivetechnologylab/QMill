import os
import re
import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from circuits import *
from dists import *
from Classification import fullDist, makeFolders
from pathlib import Path     # <— add this line


def check_required_files(filepath):
    """Check if all required .npy files exist for a given filepath."""
    required_files = [
        f'{filepath}_results.npy',
        f'{filepath}_x0_results.npy'
    ]
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Missing files: {missing_files}")
        return False
    return True

def parse_results_file(filepath):
    """Parse a results .txt file and extract relevant information."""
    with open(filepath, 'r') as f:
        content = f.read()

    # Extract key information using regex
    ansatz_match = re.search(r'Ansatz: (\w+)', content)
    dist_match = re.search(r'Dist: (\w+)', content)
    final_cost_match = re.search(r'Final Cost: ([\d.]+)', content)
    training_time_match = re.search(r'Training Time: ([\d.]+)', content)

    if not all([ansatz_match, dist_match, final_cost_match, training_time_match]):
        print(f"Warning: Could not parse all fields from {filepath}")
        return None

    return {
        'ansatz': ansatz_match.group(1),
        'distribution': dist_match.group(1),
        'final_cost': float(final_cost_match.group(1)),
        'training_time': float(training_time_match.group(1)),
        'filepath': filepath
    }


def create_combined_dist_plot(best_results, dist_constructors, ansatz_constructors, bins=20):
    """Create a single figure with subplots for all distributions in  order"""
    # Set up matplotlib parameters for LaTeX formatting
    plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 22
    plt.rcParams['axes.labelsize'] = 22
    plt.rcParams['axes.titlesize'] = 22
    plt.rcParams['xtick.labelsize'] = 22
    plt.rcParams['ytick.labelsize'] = 22
    plt.rcParams['legend.fontsize'] = 22

    ansatz_labels = {
        'Sixteen': 'A1',
        'Five': 'A2',
        'Custom_One': 'A3',
        'Custom_Two': 'A4'
    }

    dist_order = [
        # Top row
        'Uniform', 'Normal', 'Left Weibull', 'Right Weibull',
        # Middle row
        'MNIST', 'Fashion MNIST', 'CIFAR', 'QCHEM',
        #Bottom row
        'Soillow','Soilhigh','dmlow','dmhigh'
    ]

    # mapping for TITLE DISPLAY
    title_name_mapping = {
        'Soillow': 'Soil Low',
        'Soilhigh': 'Soil High',
        'dmlow': 'DM Low',
        'dmhigh': 'DM High'
    }


    fig = plt.figure(figsize=(24, 10.2 * 1.5))
    gs  = gridspec.GridSpec(3, 4, figure=fig)
    gs.update(wspace=0.15, hspace=0.30)

    legend_lines, legend_labels = [], []

    for idx, dist_name in enumerate(dist_order):
        ax = fig.add_subplot(gs[idx//4, idx%4])

        try:
            result      = best_results[dist_name]
            ansatz_name = result['ansatz']

            dist   = dist_constructors[dist_name]()
            ansatz = ansatz_constructors[ansatz_name]()

            fp         = f"Annealing/{ansatz_name}/{dist_name}/5/1/1/{ansatz_name}_5_1"
            trained    = np.load(fp + "_results.npy").flatten()
            initial_x0 = np.load(fp + "_x0_results.npy").flatten()

            dist.createSampleDistributions(1000)
            sample = np.array(dist.samples).flatten()

            mn, mx = sample.min(), sample.max()
            edges = np.linspace(0.0, 0.6, bins+1)

            p_tgt, _ = np.histogram(sample,    bins=edges, density=True)
            p_x0,  _ = np.histogram(initial_x0, bins=edges, density=True)
            p_trn, _ = np.histogram(trained,   bins=edges, density=True)
            centers  = 0.5 * (edges[:-1] + edges[1:])

            l_t = ax.step(centers, p_tgt, where='mid', color='k', linestyle='--', linewidth=2.8)
            l_i = ax.step(centers, p_x0,  where='mid', color='#4ae6cd', linewidth=1.8)
            l_q = ax.step(centers, p_trn, where='mid', color='#1d4670', linewidth=1.8)

            if idx == 0:
                legend_lines  = [l_t[0], l_i[0], l_q[0]]
                legend_labels = ['Target','Initial','QMill']

            ax.grid(True, alpha=0.3, linestyle='--')
            display_name  = title_name_mapping.get(dist_name, dist_name)
            circuit_label = ansatz_labels.get(ansatz_name, ansatz_name)
            ax.set_title(f"{display_name} ({circuit_label})")
            ax.set_xlabel(''); ax.set_ylabel('')
            ax.set_xlim(0.0, 0.6)

        except Exception as e:
            print(f"Plotting Error for {dist_name}: {e}")
            ax.text(0.5,0.5,'Error',transform=ax.transAxes,ha='center',va='center')
            ax.set_xticks([]); ax.set_yticks([])

    fig.legend(legend_lines, legend_labels,
               loc='center right', edgecolor='black', bbox_to_anchor=(0.9,0.55))
    fig.supxlabel('Concentratable Entanglement', y=0.06, fontsize=22)
    fig.supylabel('Density',                x=0.08, fontsize=22)
    plt.tight_layout(rect=[0.09,0.08,0.89,0.97])
    os.makedirs('Results', exist_ok=True)
    plt.savefig('Results/combined_distributions.pdf', bbox_inches='tight', dpi=300)
    plt.close(fig)

def analyze_best_results():
    """
    Recomputes the ‘best’ ansatz by actually minimizing the 20‑bin histogram TVD
    *per distribution*, then re‑draws the 3×4 grid with the corrected A1–A4 titles.
    """
    dist_constructors = {
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
    ansatz_constructors = {
        'Sixteen':    lambda: Sixteen(5,1),
        'Five':       lambda: Five(5,1),
        'Custom_One': lambda: Custom_One(5,1),
        'Custom_Two': lambda: Custom_Two(5,1),
    }

    # Precompute per‑distribution target histograms, edges & widths 
    bins = 20
    target_hists = {}
    edges_map    = {}
    widths_map   = {}

    for dname, ctor in dist_constructors.items():
        dist = ctor()
        dist.createSampleDistributions(1000)
        sample = np.array(dist.samples).flatten()
        edges = np.linspace(dist.Range[0], dist.Range[1], bins+1)
        p_hist, _ = np.histogram(sample, bins=edges, density=True)

        target_hists[dname] = p_hist
        edges_map[dname]    = edges
        widths_map[dname]   = np.diff(edges)

    # for each distribution, pick the ansatz with minimal 20‑bin TVD
    best_results = {}
    for dname, p_hist in target_hists.items():
        edges  = edges_map[dname]
        widths = widths_map[dname]

        best_tvd = float('inf')
        best_ans = None

        for aname, act in ansatz_constructors.items():
            fp = Path('Annealing')/aname/dname/'5'/'1'/'1'/f'{aname}_5_1_results.npy'
            if not fp.exists():
                continue

            q = np.load(fp).flatten()
            q_hist, _ = np.histogram(q, bins=edges, density=True)

            tvd = 0.5 * np.sum(np.abs(p_hist - q_hist) * widths)
            if tvd < best_tvd:
                best_tvd, best_ans = tvd, aname

        best_results[dname] = {
            'ansatz':       best_ans,
            'final_cost':   best_tvd,
            'training_time': None
        }

    create_combined_dist_plot(
        best_results,
        dist_constructors,
        ansatz_constructors,
        bins=bins
    )

    return best_results


if __name__ == "__main__":
    best_results = analyze_best_results()
    print("DONE")