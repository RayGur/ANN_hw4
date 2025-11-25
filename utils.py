"""
Utility functions for data loading and visualization
"""

import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    """
    Load dataset from text file

    Args:
        filepath: Path to data file

    Returns:
        data: numpy array of shape (n_samples, 2)
    """
    data = np.loadtxt(filepath)
    return data


def plot_data(data, title="Dataset", save_path=None):
    """
    Plot 2D dataset

    Args:
        data: numpy array of shape (n_samples, 2)
        title: Plot title
        save_path: If provided, save figure to this path
    """
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c="green", alpha=0.6, s=30)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.tight_layout()
    plt.show()


def plot_data_with_mesh(data, weights, title="SOM Result", save_path=None):
    """
    Plot dataset with SOM weight mesh overlay

    Args:
        data: numpy array of shape (n_samples, 2)
        weights: numpy array of shape (n_x, n_y, 2)
        title: Plot title
        save_path: If provided, save figure to this path
    """
    plt.figure(figsize=(8, 6))

    # Plot data points
    plt.scatter(data[:, 0], data[:, 1], c="green", alpha=0.6, s=30, label="Data")

    # Plot weight mesh
    n_x, n_y = weights.shape[0], weights.shape[1]

    # Plot horizontal connections
    for i in range(n_x):
        plt.plot(weights[i, :, 0], weights[i, :, 1], "r-", linewidth=1, alpha=0.7)

    # Plot vertical connections
    for j in range(n_y):
        plt.plot(weights[:, j, 0], weights[:, j, 1], "b-", linewidth=1, alpha=0.7)

    # Plot weight nodes
    plt.scatter(
        weights[:, :, 0].flatten(),
        weights[:, :, 1].flatten(),
        c="black",
        s=50,
        marker="o",
        zorder=5,
        label="Neurons",
    )

    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {save_path}")

    plt.tight_layout()
    plt.show()
