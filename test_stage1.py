"""
Test script for Stage 1: Data loading and visualization
"""

import numpy as np
from utils import load_data, plot_data

# Test data loading
print("Loading datasets...")
datasets = {
    "ThreeGroups": "data/ThreeGroups.txt",
    "TwoCircles": "data/TwoCircles.txt",
    "TwoRings": "data/TwoRings.txt",
}

for name, filepath in datasets.items():
    data = load_data(filepath)
    print(f"\n{name}:")
    print(f"  Shape: {data.shape}")
    print(f"  X range: [{data[:, 0].min():.3f}, {data[:, 0].max():.3f}]")
    print(f"  Y range: [{data[:, 1].min():.3f}, {data[:, 1].max():.3f}]")

    # Visualize
    plot_data(data, title=f"{name} Dataset", save_path=f"results/{name}_original.png")

print("\n✓ Stage 1 completed: Data loading and visualization verified")
