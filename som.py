"""
Self-Organizing Map (SOM) Implementation
"""

import numpy as np


class SOM:
    """
    Self-Organizing Map with Gaussian neighborhood function
    """

    def __init__(
        self, n_x, n_y, input_dim=2, learning_rate=0.5, sigma_0=2.0, max_iter=1000
    ):
        """
        Initialize SOM

        Args:
            n_x: Number of neurons in x-direction
            n_y: Number of neurons in y-direction
            input_dim: Dimension of input data (default: 2)
            learning_rate: Learning rate η
            sigma_0: Initial standard deviation σ₀
            max_iter: Maximum iterations
        """
        self.n_x = n_x
        self.n_y = n_y
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.sigma_0 = sigma_0
        self.max_iter = max_iter

        # Initialize weights randomly in range [0, 1]
        self.weights = np.random.rand(n_x, n_y, input_dim)

        # Store training history
        self.quantization_errors = []

    def _find_bmu(self, x):
        """
        Find Best Matching Unit (BMU) for input vector x

        Args:
            x: Input vector of shape (input_dim,)

        Returns:
            bmu_idx: Tuple (i, j) of BMU position in the mesh
        """
        # Vectorized calculation: (n_x, n_y, input_dim) - (input_dim,) -> (n_x, n_y, input_dim)
        diff = self.weights - x
        # Calculate Euclidean distance: (n_x, n_y)
        distances = np.linalg.norm(diff, axis=2)
        # Find minimum distance position
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)

        return bmu_idx

    def _calculate_sigma(self, t):
        """
        Calculate time-dependent standard deviation σ(t)

        σ(t+1) = σ₀ * e^(-t/MaxIt)

        Args:
            t: Current iteration number

        Returns:
            sigma: Current standard deviation
        """
        sigma = self.sigma_0 * np.exp(-t / self.max_iter)
        return sigma

    def _gaussian_neighborhood(self, bmu_idx, sigma):
        """
        Calculate Gaussian neighborhood function α^(Ω) for all neurons

        Uses grid-space distance (standard SOM approach):
        α^(Ω) = e^(-d_grid² / 2σ(t)²)
        where d_grid is the Euclidean distance in grid coordinates

        Args:
            bmu_idx: Tuple (i, j) of BMU position
            sigma: Current standard deviation

        Returns:
            alpha: Array of shape (n_x, n_y) with neighborhood values
        """
        # Create coordinate grids for all neurons
        i_coords, j_coords = np.meshgrid(
            np.arange(self.n_x), np.arange(self.n_y), indexing="ij"
        )

        # Calculate squared grid distance from BMU to all neurons
        dist_sq = (i_coords - bmu_idx[0]) ** 2 + (j_coords - bmu_idx[1]) ** 2

        # Calculate Gaussian neighborhood
        alpha = np.exp(-dist_sq / (2 * sigma**2))

        return alpha

    def _update_weights(self, x, bmu_idx, alpha):
        """
        Update weights using the two rules:
        Rule 1: w^(v) ← w^(v) + η · (x^(k) - w^(v))  for winner neuron
        Rule 2: w^(Ω) ← w^(Ω) + η · α^(Ω) · (x^(k) - w^(Ω))  for neighbors

        Args:
            x: Input vector
            bmu_idx: BMU position (i, j)
            alpha: Neighborhood function values
        """
        # Vectorized update for all neurons
        # alpha: (n_x, n_y), need to broadcast to (n_x, n_y, input_dim)
        alpha_expanded = alpha[:, :, np.newaxis]  # (n_x, n_y, 1)

        # Calculate update: η * α * (x - w)
        diff = x - self.weights  # (n_x, n_y, input_dim)
        update = self.learning_rate * alpha_expanded * diff

        # Apply update
        self.weights += update

    def _calculate_quantization_error(self, data):
        """
        Calculate average quantization error (distance from data points to their BMU)

        Args:
            data: Training data of shape (n_samples, input_dim)

        Returns:
            error: Average quantization error
        """
        total_error = 0.0

        for x in data:
            bmu_idx = self._find_bmu(x)
            bmu_weight = self.weights[bmu_idx[0], bmu_idx[1], :]
            total_error += np.linalg.norm(x - bmu_weight)

        return total_error / len(data)

    def train(self, data, verbose=True):
        """
        Train the SOM

        Args:
            data: Training data of shape (n_samples, input_dim)
            verbose: If True, print training progress
        """
        n_samples = len(data)

        for epoch in range(self.max_iter):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)

            for idx in indices:
                x = data[idx]

                # Find BMU
                bmu_idx = self._find_bmu(x)

                # Calculate current sigma
                sigma = self._calculate_sigma(epoch)

                # Calculate Gaussian neighborhood
                alpha = self._gaussian_neighborhood(bmu_idx, sigma)

                # Update weights
                self._update_weights(x, bmu_idx, alpha)

            # Calculate and store quantization error
            qe = self._calculate_quantization_error(data)
            self.quantization_errors.append(qe)

            if verbose and (epoch % 100 == 0 or epoch == self.max_iter - 1):
                print(f"Epoch {epoch:4d}/{self.max_iter}, σ={sigma:.4f}, QE={qe:.6f}")

        if verbose:
            print("Training completed!")

    def get_weights(self):
        """
        Get current weight mesh

        Returns:
            weights: Array of shape (n_x, n_y, input_dim)
        """
        return self.weights.copy()

    def get_quantization_errors(self):
        """
        Get training history of quantization errors

        Returns:
            errors: List of quantization errors per epoch
        """
        return self.quantization_errors.copy()
