import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 22
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.titlesize'] = 22
plt.rcParams['xtick.labelsize'] = 22
plt.rcParams['ytick.labelsize'] = 22
plt.rcParams['legend.fontsize'] = 22


data = []
current_circuit = None
current_dist = None
ansatz_labels = {'Sixteen': 'A1', 'Five': 'A2', 'Custom_One': 'A3', 'Custom_Two': 'A4'}
bar_colors = {
    'Sixteen': '#ccfdfe',    # A1
    'Five': '#a1ffb7',       # A2
    'Custom_One': '#4ae6cd', # A3
    'Custom_Two': '#033071'  # A4
}

try:
    with open('swapres.txt', 'r') as f:
        for line in f:
            line = line.strip()
            if line.endswith(':') and not line.startswith(' '):
                current_circuit = line.rstrip(':')
            elif ': ' in line and 'ranges' in line:
                current_dist = line.split(': Mean ranges')[0].strip()
            elif ': Mean=' in line:
                parts = line.split(': Mean=')
                if len(parts) == 2:
                    range_str = parts[0].strip()
                    stats = parts[1]
                    try:
                        ce_start, ce_end = map(float, range_str.split('-'))
                        mean = float(stats.split(', Std=')[0])
                        std_part = stats.split(', Std=')[1]
                        std = float(std_part.split(', Pairs=')[0])
                        pairs = int(std_part.split('Pairs=')[1])

                        data.append({
                            'circuit': current_circuit,
                            'distribution': current_dist,
                            'ce_start': ce_start,
                            'ce_end': ce_end,
                            'ce_mid': (ce_start + ce_end) / 2,
                            'mean': mean,
                            'std': std,
                            'pairs': pairs
                        })
                    except ValueError as e:
                        print(f"Skipping line due to parsing error: {line} -> {e}")
                    except IndexError as e:
                         print(f"Skipping line due to index error (likely malformed stats): {line} -> {e}")

except FileNotFoundError:
    print("Error: swapres.txt not found. Please ensure the file exists in the correct directory.")
    exit()

if not data:
    print("Error: No data parsed from swapres.txt. Check the file format and content.")
    exit()

df = pd.DataFrame(data)

plt.figure(figsize=(8, 4))

for circuit in ['Custom_Two', 'Custom_One', 'Five', 'Sixteen']:
    if circuit in df['circuit'].unique():
        mask = df['circuit'] == circuit
        plt.scatter(df[mask]['ce_mid'],
                    df[mask]['mean'],
                    s=df[mask]['pairs']*10 + 100,
                    label=ansatz_labels[circuit],
                    facecolors='none',
                    edgecolors=bar_colors[circuit],
                    linewidths=2
                   )
    else:
        print(f"Warning: Circuit '{circuit}' not found in the data.")

plt.xlabel('Concentratable Entanglement (CE)')
plt.ylabel('SWAP Test Similarity')

plt.legend(bbox_to_anchor=(0.8, .97),
           loc='upper left',
           markerscale=0.6,
           labelspacing=0.2,
           handletextpad=0.2,
           borderpad=0.2,
           borderaxespad=0.1,
           edgecolor='black',
           framealpha=1)

plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
plt.ylim(0.45, 0.7)

plt.tight_layout()
plt.savefig('swap_test_analysis.pdf', bbox_inches='tight')
print("Plot saved as swap_test_analysis.pdf")

print("\nSummary Statistics by Circuit:")
for circuit in sorted(df['circuit'].unique()):
    circuit_data = df[df['circuit'] == circuit]
    if not circuit_data.empty:
        print(f"\n{circuit} ({ansatz_labels.get(circuit, 'N/A')}):")
        print(f"  Average similarity: {circuit_data['mean'].mean():.4f}")
        print(f"  Std Dev of similarity means: {circuit_data['mean'].std():.4f}")
        print(f"  Total pairs tested: {circuit_data['pairs'].sum()}")
        if len(circuit_data) > 1:
             print(f"  CE range covered (midpoints): {circuit_data['ce_mid'].min():.3f} to {circuit_data['ce_mid'].max():.3f}")
        elif len(circuit_data) == 1:
             print(f"  CE midpoint: {circuit_data['ce_mid'].iloc[0]:.3f}")
        else:
             print(f"  No data points found for CE range calculation.")
    else:
        print(f"\n{circuit}: No data found.")