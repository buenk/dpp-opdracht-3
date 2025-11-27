#!/usr/bin/env python
"""
Script to regenerate graphs from existing analyzed results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime

# Load analyzed results
ANALYZED_RESULTS_FILE = "benchmark_results/analyzed_results_20251126_174011.csv"
OUTPUT_DIR = "benchmark_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def generate_speedup_graphs(df):
    """Generate speedup graphs for all three configurations."""

    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    configs = [
        ('SingleNode', 'Single Node (1-8 processes)', 'TotalProcesses'),
        ('MultiNode-1ppn', 'Multi-Node (1 proc/node)', 'Nodes'),
        ('MultiNode-8ppn', 'Multi-Node (8 proc/node)', 'TotalProcesses'),
    ]

    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))  # 5 problem sizes

    for idx, (config_name, title, x_col) in enumerate(configs):
        ax = axes[idx]
        df_config = df[df['Configuration'] == config_name]

        if df_config.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        # Get x values for ideal line
        x_vals = sorted(df_config[x_col].unique())
        
        # Get actual speedup range for this config
        max_speedup = df_config['Speedup'].max()
        
        # For MultiNode-8ppn, use a reasonable y-axis scale based on actual data
        # Don't scale to number of processes since speedup is < 1
        if config_name == 'MultiNode-8ppn':
            y_max = max(1.5, max_speedup * 1.2)
        else:
            y_max = max(max(x_vals) + 1, max_speedup * 1.2)

        # Plot ideal speedup line (only up to reasonable range)
        if config_name == 'MultiNode-8ppn':
            # For 8ppn, ideal would be relative to 8 processes as baseline
            ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, label='Baseline (1 proc)', alpha=0.7)
        else:
            ax.plot(x_vals, x_vals, 'k--', linewidth=1.5, label='Ideal', alpha=0.7)

        # Plot speedup for each problem size
        for i, N in enumerate(sorted(df_config['N'].unique())):
            df_N = df_config[df_config['N'] == N].sort_values(x_col)
            ax.plot(df_N[x_col], df_N['Speedup'],
                   marker='o', markersize=8, linewidth=2,
                   color=colors[i], label=f'N={N:.0e}')

        ax.set_xlabel(x_col.replace('TotalProcesses', 'Total MPI Processes').replace('Nodes', 'Number of Nodes'))
        ax.set_ylabel('Speedup ($S_p = T_1 / T_p$)')
        ax.set_title(title)
        ax.set_xticks(x_vals)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left' if config_name != 'MultiNode-8ppn' else 'best', fontsize=8)
        ax.set_xlim(min(x_vals) - 0.5, max(x_vals) + 0.5)
        ax.set_ylim(0, y_max)

    plt.suptitle('MPI Wave Simulation - Speedup Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f'speedup_graphs_{TIMESTAMP}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Speedup graphs saved to: {output_path}")
    plt.close()

def generate_efficiency_graphs(df):
    """Generate efficiency graphs for all three configurations."""

    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    configs = [
        ('SingleNode', 'Single Node (1-8 processes)', 'TotalProcesses'),
        ('MultiNode-1ppn', 'Multi-Node (1 proc/node)', 'Nodes'),
        ('MultiNode-8ppn', 'Multi-Node (8 proc/node)', 'TotalProcesses'),
    ]

    colors = plt.cm.viridis(np.linspace(0, 0.9, 5))

    for idx, (config_name, title, x_col) in enumerate(configs):
        ax = axes[idx]
        df_config = df[df['Configuration'] == config_name]

        if df_config.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue

        x_vals = sorted(df_config[x_col].unique())

        # Ideal efficiency line at 1.0
        ax.axhline(y=1.0, color='k', linestyle='--', linewidth=1.5, label='Ideal', alpha=0.7)

        # Plot efficiency for each problem size
        for i, N in enumerate(sorted(df_config['N'].unique())):
            df_N = df_config[df_config['N'] == N].sort_values(x_col)
            ax.plot(df_N[x_col], df_N['Efficiency'],
                   marker='s', markersize=8, linewidth=2,
                   color=colors[i], label=f'N={N:.0e}')

        ax.set_xlabel(x_col.replace('TotalProcesses', 'Total MPI Processes').replace('Nodes', 'Number of Nodes'))
        ax.set_ylabel('Efficiency ($E_p = S_p / P$)')
        ax.set_title(title)
        ax.set_xticks(x_vals)
        ax.set_ylim(0, 1.2)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)

    plt.suptitle('MPI Wave Simulation - Parallel Efficiency Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, f'efficiency_graphs_{TIMESTAMP}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Efficiency graphs saved to: {output_path}")
    plt.close()

def main():
    # Load analyzed results
    if not os.path.exists(ANALYZED_RESULTS_FILE):
        print(f"ERROR: {ANALYZED_RESULTS_FILE} not found!")
        return

    df = pd.read_csv(ANALYZED_RESULTS_FILE)
    print(f"Loaded {len(df)} analyzed results from {ANALYZED_RESULTS_FILE}")

    # Generate graphs
    print("\nGenerating graphs...")
    generate_speedup_graphs(df)
    generate_efficiency_graphs(df)

    print("Done!")

if __name__ == "__main__":
    main()
