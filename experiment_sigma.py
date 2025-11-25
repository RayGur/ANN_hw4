"""
Stage 4: Experiment with different initial sigma (σ₀) values
Test σ₀ = 1.0, 2.0, 3.0 on all three datasets
"""

import numpy as np
import matplotlib.pyplot as plt
from som import SOM
from utils import load_data, plot_data_with_mesh


def train_and_evaluate(
    data, dataset_name, sigma_0, n_x=10, n_y=10, learning_rate=0.5, max_iter=1000
):
    """
    Train SOM with given sigma_0 and return results

    Returns:
        som: Trained SOM object
        errors: List of quantization errors
    """
    print(f"\n  Training with σ₀={sigma_0}...")

    # Initialize SOM
    som = SOM(
        n_x=n_x,
        n_y=n_y,
        input_dim=2,
        learning_rate=learning_rate,
        sigma_0=sigma_0,
        max_iter=max_iter,
    )

    # Scale weights to data range
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    som.weights = som.weights * (data_max - data_min) + data_min

    # Train
    som.train(data, verbose=False)

    # Get results
    errors = som.get_quantization_errors()

    print(f"    Final QE: {errors[-1]:.6f}")

    return som, errors


def plot_sigma_comparison(data, soms, sigma_values, dataset_name, save_path):
    """
    Plot comparison of different sigma values side by side
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for idx, (ax, som, sigma_0) in enumerate(zip(axes, soms, sigma_values)):
        # Plot data points
        ax.scatter(data[:, 0], data[:, 1], c="green", alpha=0.5, s=20, label="Data")

        # Plot weight mesh
        weights = som.get_weights()
        n_x, n_y = som.n_x, som.n_y

        # Horizontal connections (blue)
        for i in range(n_x):
            for j in range(n_y - 1):
                w1 = weights[i, j]
                w2 = weights[i, j + 1]
                ax.plot([w1[0], w2[0]], [w1[1], w2[1]], "b-", linewidth=1, alpha=0.7)

        # Vertical connections (red)
        for i in range(n_x - 1):
            for j in range(n_y):
                w1 = weights[i, j]
                w2 = weights[i + 1, j]
                ax.plot([w1[0], w2[0]], [w1[1], w2[1]], "r-", linewidth=1, alpha=0.7)

        # Plot neurons
        for i in range(n_x):
            for j in range(n_y):
                w = weights[i, j]
                ax.plot(
                    w[0],
                    w[1],
                    "o",
                    color="white",
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=1.5,
                    zorder=5,
                )

        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(f"σ₀ = {sigma_0}", fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(loc="upper left")

    plt.suptitle(
        f"{dataset_name} - Initial Sigma (σ₀) Comparison",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved comparison plot: {save_path}")
    plt.close()


def plot_convergence_comparison(errors_list, sigma_values, dataset_name, save_path):
    """
    Plot convergence curves for different sigma values
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "red", "green"]

    for idx, (errors, sigma_0) in enumerate(zip(errors_list, sigma_values)):
        ax.plot(
            range(len(errors)),
            errors,
            color=colors[idx],
            linewidth=2,
            label=f"σ₀ = {sigma_0} (Final QE: {errors[-1]:.4f})",
        )

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Quantization Error", fontsize=12)
    ax.set_title(
        f"{dataset_name} - Convergence Curves (Different σ₀)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved convergence comparison: {save_path}")
    plt.close()


def main():
    """
    Main experiment: test different sigma values on all datasets
    """
    print("=" * 70)
    print("Stage 4: Initial Sigma (σ₀) Experiments")
    print("=" * 70)

    # Datasets
    datasets = [
        ("ThreeGroups", "data/ThreeGroups.txt"),
        ("TwoCircles", "data/TwoCircles.txt"),
        ("TwoRings", "data/TwoRings.txt"),
    ]

    # Sigma values to test
    sigma_values = [1.0, 2.0, 3.0]

    # Training parameters (fixed)
    n_x = 10
    n_y = 10
    learning_rate = 0.5
    max_iter = 1000

    # Summary results
    summary_results = {}

    # Process each dataset
    for dataset_name, data_path in datasets:
        print(f"\n{'='*70}")
        print(f"Processing {dataset_name}")
        print(f"{'='*70}")

        # Load data
        data = load_data(data_path)
        print(f"Data shape: {data.shape}")

        # Store results for this dataset
        soms = []
        errors_list = []

        # Train with different sigma values
        for sigma_0 in sigma_values:
            som, errors = train_and_evaluate(
                data,
                dataset_name,
                sigma_0,
                n_x=n_x,
                n_y=n_y,
                learning_rate=learning_rate,
                max_iter=max_iter,
            )
            soms.append(som)
            errors_list.append(errors)

        # Save results
        summary_results[dataset_name] = {
            "soms": soms,
            "errors_list": errors_list,
            "final_qes": [errors[-1] for errors in errors_list],
        }

        # Generate comparison plots
        sigma_comparison_path = f"results/{dataset_name.lower()}_sigma_comparison.png"
        plot_sigma_comparison(
            data, soms, sigma_values, dataset_name, sigma_comparison_path
        )

        convergence_comparison_path = (
            f"results/{dataset_name.lower()}_sigma_convergence.png"
        )
        plot_convergence_comparison(
            errors_list, sigma_values, dataset_name, convergence_comparison_path
        )

    # Print summary
    print(f"\n{'='*70}")
    print("Summary: Final Quantization Errors")
    print(f"{'='*70}")
    print(f"{'Dataset':<15} {'σ₀=1.0':<12} {'σ₀=2.0':<12} {'σ₀=3.0':<12}")
    print("-" * 70)

    for dataset_name in ["ThreeGroups", "TwoCircles", "TwoRings"]:
        qes = summary_results[dataset_name]["final_qes"]
        print(f"{dataset_name:<15} {qes[0]:<12.6f} {qes[1]:<12.6f} {qes[2]:<12.6f}")

    print(f"\n{'='*70}")
    print("✓ Stage 4 (Initial Sigma Experiments) Completed!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
