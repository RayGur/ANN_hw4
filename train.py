"""
Main training script for SOM
Stage 3: Complete training flow with visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from som import SOM
from utils import load_data, plot_data, plot_data_with_mesh


def train_som_on_dataset(
    dataset_name,
    data_path,
    n_x=10,
    n_y=10,
    learning_rate=0.5,
    sigma_0=2.0,
    max_iter=1000,
    save_prefix="",
):
    """
    Train SOM on a dataset and generate visualizations

    Args:
        dataset_name: Name of the dataset
        data_path: Path to data file
        n_x, n_y: Mesh dimensions
        learning_rate: Learning rate η
        sigma_0: Initial standard deviation
        max_iter: Maximum iterations
        save_prefix: Prefix for saved files

    Returns:
        som: Trained SOM object
    """
    print(f"\n{'='*70}")
    print(f"Training SOM on {dataset_name}")
    print(f"{'='*70}")

    # Load data
    data = load_data(data_path)
    print(f"Data shape: {data.shape}")
    print(f"X range: [{data[:, 0].min():.3f}, {data[:, 0].max():.3f}]")
    print(f"Y range: [{data[:, 1].min():.3f}, {data[:, 1].max():.3f}]")

    # Initialize SOM
    print(f"\nSOM Configuration:")
    print(f"  Mesh size: {n_x} x {n_y}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Initial σ: {sigma_0}")
    print(f"  Max iterations: {max_iter}")

    som = SOM(
        n_x=n_x,
        n_y=n_y,
        input_dim=2,
        learning_rate=learning_rate,
        sigma_0=sigma_0,
        max_iter=max_iter,
    )

    # Scale initial weights to data range
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)
    som.weights = som.weights * (data_max - data_min) + data_min

    # Visualize initial state
    plot_data_with_mesh(
        data,
        som.weights,
        title=f"{dataset_name} - Initial State",
        save_path=f"results/{save_prefix}_initial.png",
    )

    # Train
    print(f"\nTraining started...")
    som.train(data, verbose=True)

    # Get results
    final_weights = som.get_weights()
    errors = som.get_quantization_errors()

    print(f"\nTraining Results:")
    print(f"  Initial QE: {errors[0]:.6f}")
    print(f"  Final QE: {errors[-1]:.6f}")
    print(f"  Improvement: {(errors[0] - errors[-1]) / errors[0] * 100:.2f}%")

    # Visualize final state
    plot_data_with_mesh(
        data,
        final_weights,
        title=f"{dataset_name} - Final State",
        save_path=f"results/{save_prefix}_final.png",
    )

    # Plot convergence curve
    plt.figure(figsize=(10, 6))
    plt.plot(errors, linewidth=2, color="blue")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Quantization Error", fontsize=12)
    plt.title(f"{dataset_name} - Convergence Curve", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{save_prefix}_convergence.png", dpi=150, bbox_inches="tight")
    print(f"  Saved convergence curve to results/{save_prefix}_convergence.png")
    plt.close()

    # Save weight evolution snapshots
    save_weight_evolution(data, som, dataset_name, save_prefix)

    return som


def save_weight_evolution(data, som, dataset_name, save_prefix):
    """
    Visualize weight evolution at different stages
    """
    # Re-train and capture snapshots
    data_min = data.min(axis=0)
    data_max = data.max(axis=0)

    som_snapshot = SOM(
        n_x=som.n_x,
        n_y=som.n_y,
        input_dim=2,
        learning_rate=som.learning_rate,
        sigma_0=som.sigma_0,
        max_iter=som.max_iter,
    )
    som_snapshot.weights = (
        np.random.rand(som.n_x, som.n_y, 2) * (data_max - data_min) + data_min
    )

    snapshots = [
        0,
        som.max_iter // 4,
        som.max_iter // 2,
        3 * som.max_iter // 4,
        som.max_iter - 1,
    ]
    snapshot_weights = []

    for epoch in range(som.max_iter):
        # Train one epoch
        indices = np.random.permutation(len(data))
        for idx in indices:
            x = data[idx]
            bmu_idx = som_snapshot._find_bmu(x)
            sigma = som_snapshot._calculate_sigma(epoch)
            alpha = som_snapshot._gaussian_neighborhood(bmu_idx, sigma)
            som_snapshot._update_weights(x, bmu_idx, alpha)

        # Capture snapshots
        if epoch in snapshots:
            snapshot_weights.append(som_snapshot.weights.copy())

    # Plot evolution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (epoch, weights) in enumerate(zip(snapshots, snapshot_weights)):
        ax = axes[idx]
        ax.scatter(data[:, 0], data[:, 1], c="green", alpha=0.5, s=20, label="Data")

        # Plot mesh
        n_x, n_y = weights.shape[0], weights.shape[1]
        for i in range(n_x):
            ax.plot(weights[i, :, 0], weights[i, :, 1], "r-", linewidth=1, alpha=0.7)
        for j in range(n_y):
            ax.plot(weights[:, j, 0], weights[:, j, 1], "b-", linewidth=1, alpha=0.7)

        ax.scatter(
            weights[:, :, 0].flatten(),
            weights[:, :, 1].flatten(),
            c="black",
            s=40,
            marker="o",
            zorder=5,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"Epoch {epoch}")
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend()

    # Remove extra subplot
    fig.delaxes(axes[5])

    plt.suptitle(f"{dataset_name} - Weight Evolution", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"results/{save_prefix}_evolution.png", dpi=150, bbox_inches="tight")
    print(f"  Saved weight evolution to results/{save_prefix}_evolution.png")
    plt.close()


def main():
    """
    Main training pipeline
    """
    print("=" * 70)
    print("SOM Training Pipeline - Stage 3")
    print("=" * 70)

    # Training configurations
    datasets = [
        ("ThreeGroups", "data/ThreeGroups.txt"),
        ("TwoCircles", "data/TwoCircles.txt"),
        ("TwoRings", "data/TwoRings.txt"),
    ]

    # Default parameters
    default_config = {
        "n_x": 10,
        "n_y": 10,
        "learning_rate": 0.5,
        "sigma_0": 2.0,
        "max_iter": 1000,
    }

    # Train on all datasets
    results = {}

    for dataset_name, data_path in datasets:
        save_prefix = f"{dataset_name.lower()}_10x10_lr0.5_sig2.0"

        som = train_som_on_dataset(
            dataset_name=dataset_name,
            data_path=data_path,
            save_prefix=save_prefix,
            **default_config,
        )

        results[dataset_name] = {
            "som": som,
            "final_qe": som.get_quantization_errors()[-1],
        }

    # Summary
    print(f"\n{'='*70}")
    print("Training Summary")
    print(f"{'='*70}")
    for name, result in results.items():
        print(f"{name:15s}: Final QE = {result['final_qe']:.6f}")

    print(f"\n{'='*70}")
    print("✓ Stage 3 Completed: Training pipeline with visualization")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
