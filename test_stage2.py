"""
Test script for Stage 2: SOM core algorithm verification
"""

import numpy as np
import matplotlib.pyplot as plt
from som import SOM
from utils import load_data, plot_data_with_mesh

# Load a simple dataset for testing
print("Loading ThreeGroups dataset...")
data = load_data("data/ThreeGroups.txt")
print(f"Data shape: {data.shape}")

# Normalize data to [0, 1] range for better weight initialization
data_min = data.min(axis=0)
data_max = data.max(axis=0)
data_normalized = (data - data_min) / (data_max - data_min)

print("\n" + "=" * 60)
print("Testing SOM Core Components")
print("=" * 60)

# Initialize SOM
n_x, n_y = 10, 10
som = SOM(n_x=n_x, n_y=n_y, input_dim=2, learning_rate=0.5, sigma_0=2.0, max_iter=100)

print(f"\n1. SOM Initialized:")
print(f"   Mesh size: {n_x} x {n_y}")
print(f"   Weights shape: {som.weights.shape}")
print(f"   Learning rate: {som.learning_rate}")
print(f"   Initial σ: {som.sigma_0}")
print(f"   Max iterations: {som.max_iter}")

# Test BMU finding
print(f"\n2. Testing BMU finding:")
test_point = data_normalized[0]
bmu_idx = som._find_bmu(test_point)
print(f"   Test point: {test_point}")
print(f"   BMU index: {bmu_idx}")
print(f"   BMU weight: {som.weights[bmu_idx[0], bmu_idx[1], :]}")

# Test sigma calculation
print(f"\n3. Testing σ(t) calculation:")
for t in [0, 50, 99]:
    sigma = som._calculate_sigma(t)
    print(f"   t={t:3d}: σ={sigma:.6f}")

# Test Gaussian neighborhood
print(f"\n4. Testing Gaussian neighborhood function:")
sigma = 2.0
alpha = som._gaussian_neighborhood(bmu_idx, sigma)
print(f"   alpha shape: {alpha.shape}")
print(f"   alpha at BMU: {alpha[bmu_idx[0], bmu_idx[1]]:.6f} (should be 1.0)")
print(f"   alpha min: {alpha.min():.6f}")
print(f"   alpha max: {alpha.max():.6f}")

# Visualize neighborhood function
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(alpha, cmap="hot", interpolation="nearest")
plt.colorbar(label="α value")
plt.plot(bmu_idx[1], bmu_idx[0], "b*", markersize=15, label="BMU")
plt.title(f"Gaussian Neighborhood (σ={sigma})")
plt.xlabel("y index")
plt.ylabel("x index")
plt.legend()

plt.subplot(1, 2, 2)
# Plot cross-section through BMU
cross_section = alpha[bmu_idx[0], :]
plt.plot(cross_section, "o-")
plt.axvline(x=bmu_idx[1], color="r", linestyle="--", label="BMU position")
plt.xlabel("y index")
plt.ylabel("α value")
plt.title("Cross-section through BMU")
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig("results/gaussian_neighborhood_test.png", dpi=150, bbox_inches="tight")
print(f"   Saved visualization to results/gaussian_neighborhood_test.png")
plt.close()

# Test quantization error calculation
print(f"\n5. Testing quantization error calculation:")
qe = som._calculate_quantization_error(data_normalized)
print(f"   Initial quantization error: {qe:.6f}")

# Test weight update (single step)
print(f"\n6. Testing weight update:")
old_bmu_weight = som.weights[bmu_idx[0], bmu_idx[1], :].copy()
som._update_weights(test_point, bmu_idx, alpha)
new_bmu_weight = som.weights[bmu_idx[0], bmu_idx[1], :]
print(f"   BMU weight before: {old_bmu_weight}")
print(f"   BMU weight after:  {new_bmu_weight}")
print(f"   Weight change: {np.linalg.norm(new_bmu_weight - old_bmu_weight):.6f}")

print("\n" + "=" * 60)
print("Testing Complete Training Loop (100 epochs)")
print("=" * 60)

# Create new SOM for training test
som_train = SOM(
    n_x=10, n_y=10, input_dim=2, learning_rate=0.5, sigma_0=2.0, max_iter=100
)

# Scale weights to data range for better visualization
som_train.weights = som_train.weights * (data_max - data_min) + data_min

print("\nTraining started...")
som_train.train(data, verbose=True)

# Get final weights and errors
final_weights = som_train.get_weights()
errors = som_train.get_quantization_errors()

print(f"\nTraining results:")
print(f"   Final quantization error: {errors[-1]:.6f}")
print(f"   Error reduction: {(errors[0] - errors[-1]) / errors[0] * 100:.2f}%")

# Plot convergence curve
plt.figure(figsize=(8, 5))
plt.plot(errors, linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("Quantization Error")
plt.title("Convergence Curve")
plt.grid(True, alpha=0.3)
plt.savefig("results/convergence_test.png", dpi=150, bbox_inches="tight")
print(f"   Saved convergence curve to results/convergence_test.png")
plt.close()

# Visualize final result
plot_data_with_mesh(
    data,
    final_weights,
    title="SOM Test Result (100 epochs)",
    save_path="results/som_test_result.png",
)

print("\n" + "=" * 60)
print("✓ Stage 2 completed: All SOM components verified")
print("=" * 60)
