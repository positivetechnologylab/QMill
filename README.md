# QMILL: Synthetic Quantum Dataset Generation Framework

## Overview
**QMill** is a framework for generating **entanglement-rich synthetic quantum datasets**. It addresses the scarcity of real quantum-sensed data by producing low-depth, parameterized ansatz circuits optimized to match user-specified distributions of **Concentratable Entanglement** (CE). QMILL also enforces sample diversity via SWAP-test checks.

This repo contains all code and data needed to reproduce the plots from the related paper: QMILL: Representative Quantum Data Generation for Effective and Efficient Quantum Machine Learning.

---

## Table of Contents

1. [Prerequisites](#prerequisites)  
2. [Installation](#installation)  
3. [Directory Structure](#directory-structure) 
4. [Reproducing Paper Plots](#reproducing-paper-plots)  
5. [Additional Utilities](#additional-utilities)  
---

## Prerequisites
All prerequisites can be found in the requirements.txt file, which should be used to install libraries in a fresh venv.
Note that the qml_stuff folder, where all qml operations take place, has its own requirements.txt; this is because of conflicting Qiskit versions for different functionalities. We recommend using another separate venv to install that set of required libraries since they conflict if you choose to execute code from here.
Example:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

---

## Installation
git clone this repo
cd QMill

---

## Directory Structure


```text
QMill/
├── Annealing/                  # Raw `.npy` outputs from QMill optimizer
├── ce_high_sim_data/           # Noisy-simulation CE `.npz`
├── ce_high_sim_data_ideal/     # Ideal-simulation CE `.npz`
├── ce_high_data/               # Real-hardware CE `.npz`
├── Results/                    # Outputs of `bestdistributions.py`
├── Paper Plots/                # Outputs of `plotfinalcosts.py`
├── swapres.txt                 # SWAP-test raw results
├── circuits.py                 # Ansatz definitions
├── bestdistributions.py        # Fig. 5: combined distributions
├── plotfinalcosts.py           # Figs. 6–8: TVD bar charts
├── swap_plot.py                # Fig. 9: SWAP test vs CE
├── plotrealnoisyideal.py       # Fig. 10: ideal vs noisy vs real CE
├── circuit_comp.py             # Fig. 11: TVD metrics comparison
├── scale_dist.py               # Utility: scale & plot CE distributions
└── …                           # other helper modules and data files
```

---

## Reproducing Paper Plots
All commands assume you are in the repo root and have generated the required raw data (included in the repo for convenience).

**Figure 5: Combined Distribution Comparison**
python3 bestdistributions.py

Outputs:
Results/combined_distributions.pdf

**Figures 6–8: Histogram TVD Comparisons**
python3 plotfinalcosts.py

Outputs:
- Paper Plots/arbitrary_distributions_comparison.pdf
- Paper Plots/real_distributions_comparison.pdf
- Paper Plots/sensor_distributions_comparison.pdf

**Figure 9: SWAP-Test Similarity vs CE**
python3 swap_plot.py

Note: requires swapres.txt at repo root.
Outputs: swap_test_analysis.pdf

**Figure 10: Ideal ∕ Noisy ∕ Real CE Comparison**
python3 plotrealnoisyideal.py

Requires:
- ce_high_sim_data_ideal/ce_swaptest_high_data_ideal.npz
- ce_high_sim_data/ce_swaptest_high_data_sim.npz
- ce_high_data/ce_swaptest_high_data.npz
  
Outputs: noisyideal.pdf

**Figure 11: Ansatz TVD Performance**
python3 circuit_comp.py

Outputs: ansatz_comparison.pdf

---

## Additional Utilities
- scale_dist.py: scale & compare CE histograms
- results.py: quick TVD vs Ansatz plot (test_<dist>.png)

