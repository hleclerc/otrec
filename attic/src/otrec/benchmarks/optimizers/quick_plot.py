"""Quick plotting of benchmark results."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def create_demo_plot():
    """Create a demo plot based on observed benchmark results."""

    # Data from actual benchmark run
    optimizers = {
        'Gradient Descent': {
            'steps': list(range(1, 41)),
            'losses': [157.66, 153.81, 150.76, 147.77, 144.84, 142.55, 139.17, 135.86, 132.64, 128.97,
                      125.41, 121.46, 117.63, 113.93, 113.47, 109.46, 105.58, 101.04, 97.07, 96.69,
                      92.52, 88.18, 84.05, 79.79, 75.74, 71.62, 67.72, 64.03, 60.06, 56.56, 53.26,
                      50.16, 43.61, 40.91, 38.37, 35.99, 33.76, 25.02, 23.37, 21.84],
            'final_loss': 21.23,
            'time': 29.49,
            'color': 'C0'
        },
        'GD + Line Search': {
            'steps': list(range(1, 36)),
            'losses': [157.66, 141.92, 128.36, 116.09, 104.99, 96.89, 85.88, 76.13, 67.49, 58.64,
                      50.95, 43.39, 36.95, 31.47, 30.84, 25.74, 21.49, 17.24, 14.11, 13.83,
                      11.09, 8.72, 6.86, 5.29, 4.08, 3.09, 2.34, 1.77, 1.29, 0.96,
                      0.72, 0.54, 0.40, 0.40, 0.40],
            'final_loss': 0.40,
            'time': 64.75,
            'color': 'C1'
        },
        'Adam': {
            'steps': list(range(1, 36)),
            'losses': [157.66, 151.61, 146.43, 141.38, 136.47, 132.64, 127.08, 121.71, 116.56, 110.79,
                      105.28, 99.29, 93.62, 88.24, 87.59, 81.92, 76.58, 70.49, 65.34, 64.85,
                      59.61, 54.34, 49.50, 44.70, 40.33, 36.06, 32.20, 28.73, 25.18, 22.22,
                      19.59, 17.25, 15.17, 15.17, 15.17],
            'final_loss': 15.17,
            'time': 17.08,
            'color': 'C2'
        },
        'L-BFGS': {
            'steps': list(range(1, 13)),
            'losses': [157.66, 0.00059858, 0.00007824, 0.00006837, 0.00006837, 0.00006837,
                      0.00006837, 0.00006837, 0.00006837, 0.00006837, 0.00006837, 0.00006837],
            'final_loss': 0.00006837,
            'time': 2.32,
            'color': 'C3'
        }
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Optimizer Comparison: 10k Diracs (100×100 sinogram)', fontsize=14, fontweight='bold')

    # Plot 1: Loss vs Iteration
    for name, data in optimizers.items():
        steps = np.array(data['steps'])
        losses = np.array(data['losses'])
        ax1.semilogy(steps, losses, marker='o', label=name, linewidth=2.5, markersize=4,
                     color=data['color'], alpha=0.8)

    ax1.axhline(157.66, color='gray', linestyle='--', alpha=0.4, linewidth=1, label='Initial loss')
    ax1.set_xlabel('Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    ax1.set_title('Convergence: Loss vs Iteration', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11, loc='upper right')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim(0, 500)

    # Plot 2: Loss vs Time
    for name, data in optimizers.items():
        losses = np.array(data['losses'])
        time_per_step = data['time'] / len(losses)
        time_array = np.cumsum([time_per_step] * len(losses))
        ax2.semilogy(time_array, losses, marker='s', label=name, linewidth=2.5, markersize=4,
                    color=data['color'], alpha=0.8)

    ax2.axhline(157.66, color='gray', linestyle='--', alpha=0.4, linewidth=1, label='Initial loss')
    ax2.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Loss (log scale)', fontsize=12, fontweight='bold')
    ax2.set_title('Convergence: Loss vs Time', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11, loc='upper right')
    ax2.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    output_path = Path(__file__).parent / 'results' / 'convergence_10k_demo.png'
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path}")
    plt.close()

    # Create performance table plot
    fig, ax = plt.subplots(figsize=(12, 6))

    names = list(optimizers.keys())
    iterations = [len(optimizers[n]['losses']) for n in names]
    times = [optimizers[n]['time'] for n in names]
    losses = [optimizers[n]['final_loss'] for n in names]

    x = np.arange(len(names))
    width = 0.25

    ax2_1 = ax.twinx()
    ax2_2 = ax.twinx()
    ax2_2.spines['right'].set_position(('outward', 60))

    bars1 = ax.bar(x - width, iterations, width, label='Iterations', color='C0', alpha=0.7)
    bars2 = ax2_1.bar(x, times, width, label='Time (s)', color='C1', alpha=0.7)
    bars3 = ax2_2.bar(x + width, losses, width, label='Final Loss', color='C2', alpha=0.7)

    ax.set_ylabel('Iterations', fontsize=11, fontweight='bold', color='C0')
    ax2_1.set_ylabel('Time (s)', fontsize=11, fontweight='bold', color='C1')
    ax2_2.set_ylabel('Final Loss', fontsize=11, fontweight='bold', color='C2')
    ax.set_xlabel('Optimizer', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_title('Performance Metrics Summary', fontsize=13, fontweight='bold')

    ax.tick_params(axis='y', labelcolor='C0')
    ax2_1.tick_params(axis='y', labelcolor='C1')
    ax2_2.tick_params(axis='y', labelcolor='C2')

    # Add legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2_1.get_legend_handles_labels()
    lines3, labels3 = ax2_2.get_legend_handles_labels()
    ax.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=11)

    plt.tight_layout()
    output_path2 = Path(__file__).parent / 'results' / 'performance_summary.png'
    output_path2.parent.mkdir(exist_ok=True)
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved to {output_path2}")
    plt.close()

if __name__ == "__main__":
    create_demo_plot()
