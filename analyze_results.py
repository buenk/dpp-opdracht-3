#!/usr/bin/env python
"""
Quick analysis script to re-analyze existing benchmark results.
"""

import pandas as pd
import os
from datetime import datetime

# Load existing raw results
RAW_RESULTS_FILE = "benchmark_results/raw_results_20251126_164420.csv"
OUTPUT_DIR = "benchmark_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

def calculate_speedup(df):
    """Calculate speedup relative to sequential (1 process) execution."""
    results = []

    # Create a lookup for sequential times (SingleNode, 1 process) by N
    sequential_times = {}
    single_node_data = df[df['Configuration'] == 'SingleNode']
    for N in single_node_data['N'].unique():
        seq_row = single_node_data[(single_node_data['N'] == N) & (single_node_data['TotalProcesses'] == 1)]
        if not seq_row.empty:
            sequential_times[N] = seq_row['AvgTime'].iloc[0]

    for config in df['Configuration'].unique():
        df_config = df[df['Configuration'] == config]

        for N in df_config['N'].unique():
            df_N = df_config[df_config['N'] == N].copy()

            # Use global sequential baseline
            if N not in sequential_times:
                print(f"Warning: No sequential time found for N={N}")
                continue

            T1 = sequential_times[N]

            df_N['T1'] = T1
            df_N['Speedup'] = T1 / df_N['AvgTime']
            df_N['Efficiency'] = df_N['Speedup'] / df_N['TotalProcesses']

            results.append(df_N)

    if results:
        return pd.concat(results, ignore_index=True)
    return pd.DataFrame()

def main():
    # Load raw results
    if not os.path.exists(RAW_RESULTS_FILE):
        print(f"ERROR: {RAW_RESULTS_FILE} not found!")
        return

    df = pd.read_csv(RAW_RESULTS_FILE)
    print(f"Loaded {len(df)} results from {RAW_RESULTS_FILE}")

    # Calculate speedup and efficiency
    df_analyzed = calculate_speedup(df)

    if df_analyzed.empty:
        print("ERROR: Could not calculate speedup")
        return

    analyzed_csv_path = os.path.join(OUTPUT_DIR, f'analyzed_results_{TIMESTAMP}.csv')
    df_analyzed.to_csv(analyzed_csv_path, index=False)
    print(f"Analyzed results saved to: {analyzed_csv_path}")

    # Print summary
    print("\nSpeedup Summary:")
    for config in df_analyzed['Configuration'].unique():
        print(f"\n  {config}:")
        df_config = df_analyzed[df_analyzed['Configuration'] == config]
        for N in sorted(df_config['N'].unique()):
            df_N = df_config[df_config['N'] == N]
            max_speedup = df_N['Speedup'].max()
            max_procs = df_N.loc[df_N['Speedup'].idxmax(), 'TotalProcesses']
            print(f"    N={N:.0e}: Max speedup = {max_speedup:.2f}x at {max_procs} processes")

if __name__ == "__main__":
    main()
