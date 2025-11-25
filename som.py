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
        # Calculate Euclidean distance to all neurons
        distances = np.zeros((self.n_x, self.n_y))

        for i in range(self.n_x):
            for j in range(self.n_y):
                w = self.weights[i, j, :]
                distances[i, j] = np.linalg.norm(x - w)

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

        α^(Ω) = e^(-||w^(v) - w^(Ω)||² / 2σ(t)²)

        Args:
            bmu_idx: Tuple (i, j) of BMU position
            sigma: Current standard deviation

        Returns:
            alpha: Array of shape (n_x, n_y) with neighborhood values
        """
        alpha = np.zeros((self.n_x, self.n_y))
        bmu_weight = self.weights[bmu_idx[0], bmu_idx[1], :]

        for i in range(self.n_x):
            for j in range(self.n_y):
                w_neighbor = self.weights[i, j, :]

                # Calculate squared distance in weight space
                dist_sq = np.sum((bmu_weight - w_neighbor) ** 2)

                # Gaussian function
                alpha[i, j] = np.exp(-dist_sq / (2 * sigma**2))

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
        for i in range(self.n_x):
            for j in range(self.n_y):
                w = self.weights[i, j, :]

                if (i, j) == bmu_idx:
                    # Rule 1: Update winner neuron
                    self.weights[i, j, :] = w + self.learning_rate * (x - w)
                else:
                    # Rule 2: Update neighbors with Gaussian weighting
                    self.weights[i, j, :] = w + self.learning_rate * alpha[i, j] * (
                        x - w
                    )

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
