import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # For better aesthetics
import math

from plotting.utils import *

# --- Configuration ---
#csv_filename = 'results/feedbacklatency.csv'
csv_filename = 'results/dummy.csv'
#output_image_filename = 'benchmark_fidelity.pdf'
output_image_filename = 'software_performance.pdf'


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


def plot_multiple():
    methods_to_plot = ['Raw', 'MCMit', 'Qiskit M3'] # Order of bars
    #methods_to_plot = ['Qubic', 'MCMit']
    #methods_to_plot = ['250ns', '500ns', '750ns', '1000ns']
    method_colors = sns.color_palette("pastel", len(methods_to_plot)) # Example color palette
    
    nrows = 1
    ncols = num_benchmarks

    # Create figure and axes for subplots
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.05 * WIDE_FIGSIZE, HEIGHT_FIGSIZE), sharey=True)
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
        print(benchmark)
        print(len(n_values))
        indices_to_keep = list(range(0, len(n_values), 2))  # Every other index
        n_values = [n_values[i] for i in indices_to_keep]
        print((indices_to_keep))
        # --- Prepare data for grouped bars ---
        plot_data = {}
        for method in methods_to_plot:
            method_data = df_bench[df_bench['Method'] == method].sort_values('N')
            # Filter fidelities to match n_values
            all_fidelities = method_data['Fidelity'].tolist()
            print(method)
            print(len(all_fidelities))
            fidelities = [all_fidelities[i] for i in indices_to_keep]
            
            plot_data[method] = fidelities

        # --- Plotting Setup for subplot ---
        group_width = 0.5
        x = np.arange(len(n_values)) * group_width  # Label locations
        width = group_width / (len(methods_to_plot) + 1)  # Width of the bars
        multiplier = 0

        # --- Create Bars for Each Method on the current axis 'ax' ---
        for j, (method, fidelities) in enumerate(plot_data.items()):
            print(len(fidelities))
            print(len(x))
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
        ax.set_xlabel(x_axes[i])
        #ax.set_xlabel("Number of instances")
        #ax.set_xlabel('Number of teleportation steps')
        ax.set_title(f'({chr(97 + i)}) {benchmark}', fontsize=12, fontweight="bold") # Add (a), (b), etc.
        ax.set_xticks(x + width * (len(methods_to_plot) - 1) / 2, n_values)
        #ax.legend(loc="lower left")
        ax.set_ylim(0, 1.0)

    axes[0].set_ylabel('Fidelity')
    plt.subplots_adjust(left=0.05, right=0.9, bottom=0.25, top=0.75, wspace=0.05)

    plt.text(
        0.5, 0.92, 
        "Higher is better ↑",
        transform=fig.transFigure,
        fontsize=12, fontweight="bold", color="blue",
        va="center", ha="center"
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
       handles, labels,
       loc="right",
       bbox_to_anchor=(1, 0.5),
        fontsize=10,
        #frameon=False,
        ncol=1,
        columnspacing=0.7
    )
    #plt.tight_layout() # Adjust subplot spacing
    plt.savefig(output_image_filename, dpi=600)
    print(f"Saved combined plot: {output_image_filename}")
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
    plt.savefig("feedback_impact.pdf", dpi=600, )
    print(f"Saved combined plot: {output_image_filename}")
    plt.close(fig) # Close the figure

    print("\nPlotting complete.")


#plot_single('State length (# qubits)')
#plot_multiple()


def plot_feedback_impact_4panel(
    csv_16q: str = 'results/feedbacklatency.csv',
    csv_32q: str = 'results/decoherence_32q.csv',
    output:  str = 'feedback_impact.pdf',
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
        ('Constant-depth GHZ', df32, '32q'),
        ('Long-range CNOT',    df16, '16q'),
        ('Long-range CNOT',    df32, '32q'),
    ]

    fig, axes = plt.subplots(1, 4,
                             figsize=(4 * 2.7, HEIGHT_FIGSIZE - 1),
                             sharey=True)

    # ── draw each panel ───────────────────────────────────────────────────────
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

        ax.set_title(f'({label}) {bench} ({qubit_label})',
                     fontsize=9, fontweight='bold')
        ax.set_xlabel('Number of instances', fontsize=8)
        ax.set_xticks(x + width * (len(methods_to_plot) - 1) / 2, n_values,
                      fontsize=7, rotation=0)
        ax.set_ylim(0, 1.0)
        ax.tick_params(axis='y', labelsize=8)
        # Legend only on the first panel
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)

    axes[0].set_ylabel('Fidelity', fontsize=10)


    plt.text(0.5, 1.1, "Higher is better ↑",
             transform=fig.transFigure,
             fontsize=9, fontweight='bold', color='blue',
             va='top', ha='center')

    plt.subplots_adjust(left=0.07, right=0.99, bottom=0.15, top=0.82, wspace=0.08)
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


plot_feedback_impact_4panel()



