import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # For better aesthetics
import math

from plotting.utils import *

# --- Configuration ---
#csv_filename = 'results/feedback_latency_impact_16q.csv'
#csv_filename = 'results/software_mitigation_fidelity.csv'
csv_filename = 'results/readout_duration_teleportation_fidelity.csv'
#output_image_filename = 'plotting/benchmark_fidelity.pdf'
#output_image_filename = 'plotting/software_performance.pdf'
output_image_filename = 'plotting/mcm_impact.pdf'


x_axes = [
    'State length (# qubits)',
    'Physical qubits distance',
    'Teleportation steps',
    'Teleportation repetitions'
]

# --- Load Data ---
try:
    df = pd.read_csv(csv_filename)
    
except FileNotFoundError:
    print(f"Error: File '{csv_filename}' not found.")
    exit()

# --- Plotting ---
sns.set_theme(style="whitegrid")
benchmarks = (df['Benchmark'].unique()) # Sort for consistent order
num_benchmarks = len(benchmarks)


def plot_methods_comparison(
    thesis: bool = True,
    csv_filename: str = 'results/software_mitigation_fidelity.csv',
    output_filename: str = 'plotting/software_performance.pdf',
):
    """Raw vs MCMit vs Qiskit M3 across all four benchmarks (1x4)."""
    methods_to_plot = ['Raw', 'MCMit', 'Qiskit M3'] # Order of bars
    method_colors = sns.color_palette("pastel", len(methods_to_plot)) # Example color palette

    df = pd.read_csv(csv_filename)
    benchmarks = df['Benchmark'].unique()

    y_limits = (0, 1.1)

    nrows = 1
    ncols = 4

    # Create figure and axes for subplots
    # Wide-and-short 1x4 row, matching the QEC figures (plot_qec.py)
    THESIS_SIZE = (WIDE_FIGSIZE * 2.8, HEIGHT_FIGSIZE * 1.3)
    NORMAL_SIZE = (WIDE_FIGSIZE, HEIGHT_FIGSIZE)
    figsize = THESIS_SIZE if thesis else NORMAL_SIZE
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes_flat = axes.flatten() # Flatten the 2D array of axes for easy iteration

    # --- Iterate Through Benchmarks and Plot on Subplots ---
    for i, benchmark in enumerate(benchmarks):
        ax = axes_flat[i] # Select the current subplot axis

        # Filter data for the current benchmark
        df_bench = df[df['Benchmark'] == benchmark].sort_values('N')
        df_bench.columns = df_bench.columns.str.strip()
        df_bench['Method'] = df_bench['Method'].astype(str).str.strip()
        df_bench['Benchmark'] = df_bench['Benchmark'].astype(str).str.strip()

        # Get unique N values and indices for filtering
        n_values = sorted(df_bench['N'].unique())
        indices_to_keep = list(range(0, len(n_values), 2))  # Every other index
        n_values = [n_values[i] for i in indices_to_keep]

        # --- Prepare data for grouped bars ---
        plot_data = {}
        for method in methods_to_plot:
            method_data = df_bench[df_bench['Method'] == method].sort_values('N')
            # Filter fidelities to match n_values
            all_fidelities = method_data['Fidelity'].tolist()
            fidelities = [all_fidelities[i] for i in indices_to_keep]

            plot_data[method] = fidelities

        # --- Plotting Setup for subplot ---
        group_width = 0.5
        x = np.arange(len(n_values)) * group_width  # Label locations
        width = group_width / (len(methods_to_plot) + 1)  # Width of the bars
        multiplier = 0

        # --- Create Bars for Each Method on the current axis 'ax' ---
        for j, (method, fidelities) in enumerate(plot_data.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset,
                fidelities,
                width,
                label=method,
                color=method_colors[j],
                hatch=code_hatches[j % len(code_hatches)],
                edgecolor='black')
            #ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)
            multiplier += 1

        # --- Add Labels and Title for the subplot ---
        ax.set_xlabel(x_axes[i])
        ax.set_title(f'({chr(97 + i)}) {benchmark}', fontsize=12, fontweight="bold") # Add (a), (b), etc.
        ax.set_xticks(x + width * (len(methods_to_plot) - 1) / 2, n_values)
        ax.set_ylim(*y_limits)

    axes_flat[0].set_ylabel('Fidelity')

    # Reserve room at top for the "Higher is better" note and at bottom for the legend
    plt.subplots_adjust(top=0.80, bottom=0.30, wspace=0.1)

    plt.text(
        0.5, 0.97,
        "Higher is better ↑",
        transform=fig.transFigure,
        fontsize=14, fontweight="bold", color="blue",
        va="center", ha="center"
    )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.04),
        fontsize=12,
        frameon=False,
        ncol=10,
        columnspacing=0.7
    )

    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"Saved combined plot: {output_filename}")
    plt.close(fig) # Close the figure

    print("\nPlotting complete.")


def plot_latency_impact(
    thesis: bool = True,
    csv_filename: str = 'results/readout_duration_teleportation_fidelity.csv',
    output_filename: str = 'plotting/mcm_impact.pdf',
):
    """Feedback-latency sweep (250ns - 1000ns) across two benchmarks (1x2)."""
    methods_to_plot = ['250ns', '500ns', '750ns', '1000ns']
    method_colors = sns.color_palette("pastel", len(methods_to_plot)) # Example color palette

    df = pd.read_csv(csv_filename)
    benchmarks = df['Benchmark'].unique()

    y_limits = (0.7, 1.01)

    nrows = 1
    ncols = 2

    # Create figure and axes for subplots
    # Wide-and-short 1x2 row (half the width of the 1x4 methods figure)
    THESIS_SIZE = (WIDE_FIGSIZE * 1.4, HEIGHT_FIGSIZE * 1.3)
    NORMAL_SIZE = (WIDE_FIGSIZE, HEIGHT_FIGSIZE)
    figsize = THESIS_SIZE if thesis else NORMAL_SIZE
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=True)
    axes_flat = axes.flatten() # Flatten the 2D array of axes for easy iteration

    # --- Iterate Through Benchmarks and Plot on Subplots ---
    for i, benchmark in enumerate(benchmarks):
        ax = axes_flat[i] # Select the current subplot axis

        # Filter data for the current benchmark
        df_bench = df[df['Benchmark'] == benchmark].sort_values('N')
        df_bench.columns = df_bench.columns.str.strip()
        df_bench['Method'] = df_bench['Method'].astype(str).str.strip()
        df_bench['Benchmark'] = df_bench['Benchmark'].astype(str).str.strip()

        # Get unique N values and indices for filtering
        n_values = sorted(df_bench['N'].unique())
        indices_to_keep = list(range(0, len(n_values), 2))  # Every other index
        n_values = [n_values[i] for i in indices_to_keep]

        # --- Prepare data for grouped bars ---
        plot_data = {}
        for method in methods_to_plot:
            method_data = df_bench[df_bench['Method'] == method].sort_values('N')
            # Filter fidelities to match n_values
            all_fidelities = method_data['Fidelity'].tolist()
            fidelities = [all_fidelities[i] for i in indices_to_keep]

            plot_data[method] = fidelities

        # --- Plotting Setup for subplot ---
        group_width = 0.5
        x = np.arange(len(n_values)) * group_width  # Label locations
        width = group_width / (len(methods_to_plot) + 1)  # Width of the bars
        multiplier = 0

        # --- Create Bars for Each Method on the current axis 'ax' ---
        for j, (method, fidelities) in enumerate(plot_data.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset,
                fidelities,
                width,
                label=method,
                color=method_colors[j],
                hatch=code_hatches[j % len(code_hatches)],
                edgecolor='black')
            #ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)
            multiplier += 1

        # --- Add Labels and Title for the subplot ---
        ax.set_xlabel(x_axes[i])
        ax.set_title(f'({chr(97 + i)}) {benchmark}', fontsize=12, fontweight="bold") # Add (a), (b), etc.
        ax.set_xticks(x + width * (len(methods_to_plot) - 1) / 2, n_values)
        ax.set_ylim(*y_limits)

    axes_flat[0].set_ylabel('Fidelity')

    # Reserve room at top for the "Higher is better" note and at bottom for the legend
    plt.subplots_adjust(top=0.80, bottom=0.30, wspace=0.1)

    plt.text(
        0.5, 0.95,
        "Higher is better ↑",
        transform=fig.transFigure,
        fontsize=14, fontweight="bold", color="blue",
        va="center", ha="center"
    )

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        fontsize=12,
        frameon=False,
        ncol=10,
        columnspacing=0.7
    )

    plt.savefig(output_filename, dpi=600, bbox_inches='tight')
    print(f"Saved combined plot: {output_filename}")
    plt.close(fig) # Close the figure

    print("\nPlotting complete.")

def plot_single(xlabel: str):
    methods_to_plot = ['Qubic', 'MCMit'] # Order of bars
    method_colors = sns.color_palette("pastel", len(methods_to_plot)) # Example color palette
    # Create figure and axes for subplots
    fig, ax = plt.subplots(1, 1, figsize=(COLUMN_FIGSIZE, HEIGHT_FIGSIZE))

    # --- Iterate Through Benchmarks and Plot on Subplots ---
    for i, benchmark in enumerate(benchmarks):
        # Filter data for the current benchmark
        df_bench = df[df['Benchmark'] == benchmark].sort_values('N')

        # Get unique N values
        n_values = sorted(df_bench['N'].unique())
        n_values = n_values[::2]

        # --- Prepare data for grouped bars ---
        plot_data = {}
        for method in methods_to_plot:
            fidelities = df_bench[df_bench['Method'] == method].set_index('N').reindex(n_values)['Fidelity'].fillna(0).tolist()
            plot_data[method] = fidelities

        # --- Plotting Setup for subplot ---
        x = np.arange(len(n_values))  # Label locations
        width = 0.25  # Width of the bars
        multiplier = 0

        # --- Create Bars for Each Method on the current axis 'ax' ---
        for j, (method, fidelities) in enumerate(plot_data.items()):
            offset = width * multiplier
            rects = ax.bar(x + offset, 
                fidelities, 
                width, 
                label=method, 
                color=method_colors[j],
                hatch=code_hatches[j % len(code_hatches)],
                edgecolor='black')
            #ax.bar_label(rects, padding=3, fmt='%.2f', fontsize=8)
            multiplier += 1

        # --- Add Labels and Title for the subplot ---
        #ax.set_ylabel('Fidelity')
        ax.set_xlabel(xlabel)
        ax.set_title("Classical feedback latency impact", fontsize=12, fontweight="bold") # Add (a), (b), etc.
        ax.set_xticks(x + width * (len(methods_to_plot) - 1) / 2, n_values)
        #ax.legend()
        ax.set_ylim(0.5, 1.0)


    #plt.subplots_adjust(left=0.06, right=0.9, bottom=0.25, top=0.75, wspace=0.05)
    plt.ylabel('Fidelity')
    plt.text(
        0.5, 0.95, 
        "Higher is better ↑",
        transform=fig.transFigure,
        fontsize=12, fontweight="bold", color="blue",
        va="center", ha="center"
    )

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="right",
        bbox_to_anchor=(0.5, 0.4),
        fontsize=10,
        #frameon=False,
        ncol=1,
        #columnspacing=0.7
    )
    plt.tight_layout() # Adjust subplot spacing
    plt.savefig("plotting/feedback_impact.pdf", dpi=600, )
    print(f"Saved combined plot: {output_image_filename}")
    plt.close(fig) # Close the figure

    print("\nPlotting complete.")


def plot_feedback_impact_4panel(
    csv_16q: str = 'results/feedback_latency_impact_16q.csv',
    csv_32q: str = 'results/feedback_latency_impact_32q.csv',
    output:  str = 'plotting/feedback_impact.pdf',
):
    """
    1×4 grouped-bar figure comparing Qubic vs MCMit fidelity for:
      (a) Constant-depth GHZ  — 16 qubits  (feedbacklatency.csv)
      (b) Constant-depth GHZ  — 32 qubits  (decoherence_32q.csv)
      (c) Long-range CNOT     — 16 qubits
      (d) Long-range CNOT     — 32 qubits

    Matches the bar/hatch/colour style of plot_multiple().
    Also prints average and maximum MCMit speed-up over Qubic.
    """
    # ── load & clean ──────────────────────────────────────────────────────────
    def load(path):
        d = pd.read_csv(path)
        d.columns = d.columns.str.strip()
        d['Benchmark'] = d['Benchmark'].str.strip()
        d['Method']    = d['Method'].str.strip()
        return d

    try:
        df16 = load(csv_16q)
        df32 = load(csv_32q)
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        return

    methods_to_plot = ['Qubic', 'MCMit']
    method_colors   = sns.color_palette("pastel", len(methods_to_plot))

    PANELS = [
        ('Constant-depth GHZ', df16, '16q'),
        ('Long-range CNOT',    df16, '16q'),        
        ('Constant-depth GHZ', df32, '32q'),
        ('Long-range CNOT',    df32, '32q'),
    ]

    TEXT_WIDTH = 10
    THESIS_SIZE = (TEXT_WIDTH, TEXT_WIDTH * 0.4)
    fig, axes = plt.subplots(2, 2,
                             figsize=THESIS_SIZE,
                             sharey=True, sharex=True)

    # ── draw each panel ───────────────────────────────────────────────────────
    axes = axes.flatten()
    axes[0].set_ylabel('Fidelity')
    axes[2].set_ylabel('Fidelity')
    axes[2].set_xlabel('Number of instances')
    axes[3].set_xlabel('Number of instances')

    for idx, (bench, df, qubit_label) in enumerate(PANELS):
        ax    = axes[idx]
        label = chr(97 + idx)      # 'a', 'b', 'c', 'd'

        sub = df[df['Benchmark'] == bench].sort_values('N')
        n_values = sorted(sub['N'].unique())

        group_width = 0.5
        x     = np.arange(len(n_values)) * group_width
        width = group_width / (len(methods_to_plot) + 1)

        for j, method in enumerate(methods_to_plot):
            fidelities = (
                sub[sub['Method'] == method]
                   .set_index('N')
                   .reindex(n_values)['Fidelity']
                   .fillna(0)
                   .tolist()
            )
            offset = width * j
            ax.bar(x + offset, fidelities, width,
                   label=method,
                   color=method_colors[j],
                   hatch=code_hatches[j % len(code_hatches)],
                   edgecolor='black')

            # Mark near-zero bars with a small black × so they're visible
            ZERO_THRESH = 0.005
            for xi, fid in zip(x + offset - 0.1 + width / 2, fidelities):
                if fid < ZERO_THRESH:
                    ax.plot(xi, 0.02, marker='x', color='black',
                            markersize=5, markeredgewidth=1.5, zorder=5)

        ax.set_title(f'({label}) {bench} ({qubit_label})', fontweight='bold')
        
        ax.set_xticks(x + width * (len(methods_to_plot) - 1) / 2, n_values,
                      fontsize=7, rotation=0)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='y', labelsize=8)
        #ax.set_ylabel('Fidelity', fontsize=10)
        # Legend only on the first panel
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    
    #axes[0].set_xlabel('Number of Instances')

    plt.text(0.5, 0.95, "Higher is better ↑",
             transform=fig.transFigure,
             fontsize=14, fontweight='bold', color='blue',
             va='top', ha='center')

    plt.subplots_adjust(left=0.1, right=0.99, bottom=0.20, top=0.82, wspace=0.08)
    plt.savefig(output, dpi=600, bbox_inches='tight')
    print(f"Saved: {output}")
    plt.close(fig)

    # ── statistics ────────────────────────────────────────────────────────────
    print("\n── MCMit improvement over Qubic ──────────────────────────────────")
    for bench, df, qubit_label in PANELS:
        sub = df[df['Benchmark'] == bench].sort_values('N')
        n_values = sorted(sub['N'].unique())
        ratios = []
        for n in n_values:
            row = sub[sub['N'] == n]
            mcmit = row[row['Method'] == 'MCMit']['Fidelity'].values
            qubic = row[row['Method'] == 'Qubic']['Fidelity'].values
            if len(mcmit) and len(qubic) and qubic[0] > 0:
                ratios.append(mcmit[0] / qubic[0])
        if ratios:
            short = 'GHZ' if 'GHZ' in bench else 'CNOT'
            print(f"  {short} {qubit_label:10s}  "
                  f"avg={sum(ratios)/len(ratios):.3f}x  "
                  f"max={max(ratios):.3f}x  (N={n_values[ratios.index(max(ratios))]})")

plot_methods_comparison()
plot_latency_impact()
#plot_feedback_impact_4panel()



