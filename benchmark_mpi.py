#!/usr/bin/env python
"""
MPI Performance Benchmark Script for DAS-5

This script runs experiments with different problem sizes and system configurations
to analyze the performance of the MPI wave simulation application.

Configurations tested:
1. Single node, 1-8 MPI processes
2. Multiple nodes (1-8), 1 MPI process per node
3. Multiple nodes (1-8), 8 MPI processes per node

Authors: Generated for DPP Assignment 3
"""

import subprocess
import time
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================

# Problem sizes (number of amplitude points)
N_LIST = [10**3, 10**4, 10**5, 10**6, 10**7]

# Calibrated t_max values targeting ~50s sequential runtime
# These values match the OpenMP benchmark (assignment 1.2) for fair comparison
T_MAP = {
    10**3: 18200000,   # Same as OpenMP benchmark
    10**4: 1820000,    # Same as OpenMP benchmark
    10**5: 182000,     # Same as OpenMP benchmark
    10**6: 18200,      # Same as OpenMP benchmark
    10**7: 1820,       # Same as OpenMP benchmark
}

# Process counts to test (total MPI processes)
PROCESS_COUNTS = [1, 2, 4, 6, 8]

# Node counts for multi-node configurations
NODE_COUNTS = [1, 2, 4, 6, 8]

# Executable path
EXECUTABLE = "./assign3_1"

# Output files
OUTPUT_DIR = "benchmark_results"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Number of repetitions for each test
NUM_REPEATS = 3

# Maximum timeout in seconds (increased for ~50s sequential runtime target)
MAX_TIMEOUT = 840


# =============================================================================
# Helper Functions
# =============================================================================

def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")


def parse_time(output):
    """
    Extract execution time from 'Compute time: X seconds' line in output.
    
    Expected output format:
        Compute time: 0.00227784 seconds
        Took 3.2399 seconds
        Normalized: 3.2399e-06 seconds
    
    We use the compute time (measured with MPI_Wtime inside simulate.c),
    which excludes MPI_Init/Finalize overhead, for accurate speedup calculations.
    Speedup = T_sequential / T_parallel for the SAME problem size.
    """
    for line in output.splitlines():
        # Look for "Compute time: X seconds" (from simulate.c, excludes MPI overhead)
        if "Compute time:" in line and "seconds" in line:
            try:
                # Format: "Compute time: 0.00227784 seconds"
                time_str = line.split('Compute time:')[-1].strip().split('seconds')[0].strip()
                return float(time_str)
            except (ValueError, IndexError):
                continue
    # Fallback to "Took" format if Compute time not found
    for line in output.splitlines():
        if "Took" in line and "seconds" in line and "Normalized" not in line:
            try:
                time_str = line.split('Took')[-1].strip().split('seconds')[0].strip()
                return float(time_str)
            except (ValueError, IndexError):
                continue
    raise ValueError("Could not parse time from output.")


def calculate_timeout(N, T, total_processes):
    """Calculate adaptive timeout based on problem size and process count."""
    # Based on OpenMP benchmark calibration: ~50s sequential for N*T = 1.82e10 work units
    # That's ~2.75e-9 seconds per work unit
    work_units = N * T
    base_time = work_units * 2.75e-9
    
    # Parallel should be faster, but add safety margin for:
    # - Communication overhead
    # - Job scheduling delays on DAS-5
    # - Variance between runs
    expected_time = base_time / max(1, total_processes)
    
    # Add generous buffer: at least 60s, plus 5x expected time, capped at MAX_TIMEOUT
    timeout = max(60, min(expected_time * 5 + 60, MAX_TIMEOUT))
    return timeout


def run_prun_command(nodes, ppn, executable, i_max, t_max, timeout):
    """
    Run the MPI program using prun on DAS-5.
    
    Args:
        nodes: Number of nodes to use
        ppn: Processes per node
        executable: Path to the executable
        i_max: Number of amplitude points
        t_max: Number of time steps
        timeout: Maximum time to wait
    
    Returns:
        Tuple of (stdout, stderr, return_code)
    """
    # Build prun command
    # Format: prun -v -np <nodes> -<ppn> -sge-script $PRUN_ETC/prun-openmpi <exe> <args>
    command = [
        "prun",
        "-v",
        "-np", str(nodes),
        f"-{ppn}",
        "-sge-script", os.environ.get("PRUN_ETC", "/cm/shared/package/prun/etc") + "/prun-openmpi",
        executable,
        str(i_max),
        str(t_max)
    ]
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate(timeout=timeout)
        return stdout, stderr, process.returncode
        
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise


def run_test_case(nodes, ppn, N, T, config_name):
    """
    Run a single test case multiple times and return average time.
    
    Args:
        nodes: Number of nodes
        ppn: Processes per node
        N: Number of amplitude points
        T: Number of time steps
        config_name: Name of the configuration for logging
    
    Returns:
        Average execution time or None if failed
    """
    total_processes = nodes * ppn
    timeout = calculate_timeout(N, T, total_processes)
    
    times = []
    
    for i in range(NUM_REPEATS):
        print(f"  [{config_name}] N={N:.0e}, nodes={nodes}, ppn={ppn}, "
              f"total_procs={total_processes}, repeat {i+1}/{NUM_REPEATS}...")
        
        try:
            stdout, stderr, returncode = run_prun_command(
                nodes, ppn, EXECUTABLE, N, T, timeout
            )
            
            if returncode != 0:
                print(f"    ERROR: Non-zero return code {returncode}")
                print(f"    STDERR: {stderr[:500]}")
                continue
            
            measured_time = parse_time(stdout)
            times.append(measured_time)
            print(f"    Time: {measured_time:.3f}s")
            
        except subprocess.TimeoutExpired:
            print(f"    ERROR: Timeout after {timeout:.0f}s")
            continue
        except ValueError as e:
            print(f"    ERROR: {e}")
            continue
        except Exception as e:
            print(f"    ERROR: Unexpected error: {e}")
            continue
    
    if times:
        avg_time = sum(times) / len(times)
        std_time = np.std(times) if len(times) > 1 else 0
        print(f"  --> Average: {avg_time:.3f}s (std: {std_time:.3f}s)")
        return avg_time, std_time
    
    return None, None


# =============================================================================
# Experiment Configurations
# =============================================================================

def run_single_node_scaling():
    """
    Configuration 1: Single node with varying number of MPI processes (1-8).
    Tests strong scaling on a single machine.
    """
    print("\n" + "="*70)
    print("CONFIGURATION 1: Single Node Scaling (1 node, 1-8 processes)")
    print("="*70)
    
    results = []
    
    for N in N_LIST:
        T = T_MAP[N]
        print(f"\nProblem size N={N:.0e}, T={T}")
        
        for ppn in PROCESS_COUNTS:
            avg_time, std_time = run_test_case(
                nodes=1, ppn=ppn, N=N, T=T,
                config_name="SingleNode"
            )
            
            if avg_time is not None:
                results.append({
                    'Configuration': 'SingleNode',
                    'N': N,
                    'T_max': T,
                    'Nodes': 1,
                    'PPN': ppn,
                    'TotalProcesses': ppn,
                    'AvgTime': avg_time,
                    'StdTime': std_time
                })
    
    return results


def run_multi_node_single_process():
    """
    Configuration 2: Multiple nodes (1-8), each with 1 MPI process.
    Tests distributed memory scaling without intra-node parallelism.
    """
    print("\n" + "="*70)
    print("CONFIGURATION 2: Multi-Node Scaling (1-8 nodes, 1 process/node)")
    print("="*70)
    
    results = []
    
    for N in N_LIST:
        T = T_MAP[N]
        print(f"\nProblem size N={N:.0e}, T={T}")
        
        for nodes in NODE_COUNTS:
            avg_time, std_time = run_test_case(
                nodes=nodes, ppn=1, N=N, T=T,
                config_name="MultiNode-1ppn"
            )
            
            if avg_time is not None:
                results.append({
                    'Configuration': 'MultiNode-1ppn',
                    'N': N,
                    'T_max': T,
                    'Nodes': nodes,
                    'PPN': 1,
                    'TotalProcesses': nodes,
                    'AvgTime': avg_time,
                    'StdTime': std_time
                })
    
    return results


def run_multi_node_full():
    """
    Configuration 3: Multiple nodes (1-8), each with 8 MPI processes.
    Tests full distributed + shared memory scaling.
    """
    print("\n" + "="*70)
    print("CONFIGURATION 3: Multi-Node Full Scaling (1-8 nodes, 8 processes/node)")
    print("="*70)
    
    results = []
    
    for N in N_LIST:
        T = T_MAP[N]
        print(f"\nProblem size N={N:.0e}, T={T}")
        
        for nodes in NODE_COUNTS:
            total_procs = nodes * 8
            avg_time, std_time = run_test_case(
                nodes=nodes, ppn=8, N=N, T=T,
                config_name="MultiNode-8ppn"
            )
            
            if avg_time is not None:
                results.append({
                    'Configuration': 'MultiNode-8ppn',
                    'N': N,
                    'T_max': T,
                    'Nodes': nodes,
                    'PPN': 8,
                    'TotalProcesses': total_procs,
                    'AvgTime': avg_time,
                    'StdTime': std_time
                })
    
    return results


# =============================================================================
# Analysis and Plotting
# =============================================================================

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


def generate_speedup_graphs(df):
    """Generate speedup graphs for all three configurations."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    configs = [
        ('SingleNode', 'Single Node (1-8 processes)', 'TotalProcesses'),
        ('MultiNode-1ppn', 'Multi-Node (1 proc/node)', 'Nodes'),
        ('MultiNode-8ppn', 'Multi-Node (8 proc/node)', 'TotalProcesses'),
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(N_LIST)))
    
    for idx, (config_name, title, x_col) in enumerate(configs):
        ax = axes[idx]
        df_config = df[df['Configuration'] == config_name]
        
        if df_config.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        # Get x values for ideal line
        x_vals = sorted(df_config[x_col].unique())
        
        # Plot ideal speedup line
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
        ax.legend(loc='upper left', fontsize=8)
        ax.set_xlim(0, max(x_vals) + 0.5)
        ax.set_ylim(0, max(x_vals) + 1)
    
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
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(N_LIST)))
    
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


def generate_combined_speedup_graph(df):
    """Generate a single combined speedup graph comparing all configurations."""
    
    plt.figure(figsize=(12, 8))
    
    markers = {'SingleNode': 'o', 'MultiNode-1ppn': 's', 'MultiNode-8ppn': '^'}
    linestyles = {'SingleNode': '-', 'MultiNode-1ppn': '--', 'MultiNode-8ppn': '-.'}
    
    # Use distinct colors for each N value
    colors = plt.cm.tab10(np.linspace(0, 1, len(N_LIST)))
    
    # Plot ideal speedup
    max_procs = df['TotalProcesses'].max()
    plt.plot([1, max_procs], [1, max_procs], 'k:', linewidth=2, label='Ideal Speedup', alpha=0.5)
    
    for config in df['Configuration'].unique():
        df_config = df[df['Configuration'] == config]
        
        for i, N in enumerate(sorted(df_config['N'].unique())):
            df_N = df_config[df_config['N'] == N].sort_values('TotalProcesses')
            
            label = f'{config}, N={N:.0e}'
            plt.plot(df_N['TotalProcesses'], df_N['Speedup'],
                    marker=markers.get(config, 'o'),
                    linestyle=linestyles.get(config, '-'),
                    color=colors[i],
                    markersize=6, linewidth=1.5,
                    label=label, alpha=0.8)
    
    plt.xlabel('Total MPI Processes', fontsize=12)
    plt.ylabel('Speedup ($S_p = T_1 / T_p$)', fontsize=12)
    plt.title('MPI Wave Simulation - Combined Speedup Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left', fontsize=7, ncol=2)
    
    output_path = os.path.join(OUTPUT_DIR, f'speedup_combined_{TIMESTAMP}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Combined speedup graph saved to: {output_path}")
    plt.close()


def generate_execution_time_graphs(df):
    """Generate execution time graphs showing raw performance."""
    
    plt.style.use('default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    configs = [
        ('SingleNode', 'Single Node (1-8 processes)', 'TotalProcesses'),
        ('MultiNode-1ppn', 'Multi-Node (1 proc/node)', 'Nodes'),
        ('MultiNode-8ppn', 'Multi-Node (8 proc/node)', 'TotalProcesses'),
    ]
    
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(N_LIST)))
    
    for idx, (config_name, title, x_col) in enumerate(configs):
        ax = axes[idx]
        df_config = df[df['Configuration'] == config_name]
        
        if df_config.empty:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            continue
        
        x_vals = sorted(df_config[x_col].unique())
        
        for i, N in enumerate(sorted(df_config['N'].unique())):
            df_N = df_config[df_config['N'] == N].sort_values(x_col)
            ax.plot(df_N[x_col], df_N['AvgTime'],
                   marker='o', markersize=8, linewidth=2,
                   color=colors[i], label=f'N={N:.0e}')
        
        ax.set_xlabel(x_col.replace('TotalProcesses', 'Total MPI Processes').replace('Nodes', 'Number of Nodes'))
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(title)
        ax.set_xticks(x_vals)
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('MPI Wave Simulation - Execution Time', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f'execution_time_{TIMESTAMP}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Execution time graphs saved to: {output_path}")
    plt.close()


# =============================================================================
# Main Execution
# =============================================================================

def main():
    """Main function to run all experiments and generate reports."""
    
    print("="*70)
    print("MPI Performance Benchmark for Wave Simulation")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # Check if executable exists
    if not os.path.exists(EXECUTABLE):
        print(f"ERROR: Executable '{EXECUTABLE}' not found!")
        print("Please compile first with: make")
        sys.exit(1)
    
    ensure_output_dir()
    
    # Collect all results
    all_results = []
    
    # Run all three configurations
    print("\nStarting experiments...")
    print(f"Problem sizes: {[f'{n:.0e}' for n in N_LIST]}")
    print(f"Repeats per test: {NUM_REPEATS}")
    
    # Configuration 1: Single node scaling
    results1 = run_single_node_scaling()
    all_results.extend(results1)
    
    # Configuration 2: Multi-node, single process per node
    results2 = run_multi_node_single_process()
    all_results.extend(results2)
    
    # Configuration 3: Multi-node, 8 processes per node
    results3 = run_multi_node_full()
    all_results.extend(results3)
    
    # Create DataFrame and save raw results
    if not all_results:
        print("\nERROR: No successful test results collected!")
        sys.exit(1)
    
    df = pd.DataFrame(all_results)
    
    raw_csv_path = os.path.join(OUTPUT_DIR, f'raw_results_{TIMESTAMP}.csv')
    df.to_csv(raw_csv_path, index=False)
    print(f"\nRaw results saved to: {raw_csv_path}")
    
    # Calculate speedup and efficiency
    df_analyzed = calculate_speedup(df)
    
    if df_analyzed.empty:
        print("ERROR: Could not calculate speedup (missing sequential baseline)")
        sys.exit(1)
    
    analyzed_csv_path = os.path.join(OUTPUT_DIR, f'analyzed_results_{TIMESTAMP}.csv')
    df_analyzed.to_csv(analyzed_csv_path, index=False)
    print(f"Analyzed results saved to: {analyzed_csv_path}")
    
    # Generate graphs
    print("\nGenerating graphs...")
    
    try:
        generate_speedup_graphs(df_analyzed)
        generate_efficiency_graphs(df_analyzed)
        generate_combined_speedup_graph(df_analyzed)
        generate_execution_time_graphs(df)
    except Exception as e:
        print(f"Warning: Could not generate some graphs: {e}")
        print("Results are still saved in CSV format.")
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\nSpeedup Summary:")
    for config in df_analyzed['Configuration'].unique():
        print(f"\n  {config}:")
        df_config = df_analyzed[df_analyzed['Configuration'] == config]
        for N in sorted(df_config['N'].unique()):
            df_N = df_config[df_config['N'] == N]
            max_speedup = df_N['Speedup'].max()
            max_procs = df_N.loc[df_N['Speedup'].idxmax(), 'TotalProcesses']
            print(f"    N={N:.0e}: Max speedup = {max_speedup:.2f}x at {max_procs} processes")
    
    print(f"\n{'='*70}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results directory: {OUTPUT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()

