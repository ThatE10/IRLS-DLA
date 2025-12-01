import numpy as np
import torch
from typing import Tuple, Optional

class SyntheticDataGenerator:
    def __init__(self, n_features: int = 64,
                 n_components: int = 256,
                 n_samples: int = 2000,
                 sparsity: int = 8,
                 noise_std: float = 0.05,
                 random_state: Optional[int] = None):
        self.n_features = n_features
        self.n_components = n_components
        self.n_samples = n_samples
        self.sparsity = sparsity
        self.noise_std = noise_std
        self.random_state = random_state

    def generate(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate synthetic sparse coding data.

        Returns:
            Y: Noisy observations (n_features, n_samples)
            D_true: True dictionary (n_features, n_components)
            X_true: True sparse codes (n_components, n_samples)
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # True dictionary
        D_true = np.random.randn(self.n_features, self.n_components)
        D_true /= np.linalg.norm(D_true, axis=0, keepdims=True)

        # Sparse codes
        X_true = np.zeros((self.n_components, self.n_samples))
        for i in range(self.n_samples):
            indices = np.random.choice(self.n_components, self.sparsity, replace=False)
            X_true[indices, i] = np.random.randn(self.sparsity)

        # Noisy observations
        Y_clean = D_true @ X_true
        noise = self.noise_std * np.random.randn(self.n_features, self.n_samples)
        Y = Y_clean + noise

        return Y, D_true, X_true

    def generate_torch(self, device='cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate synthetic sparse coding data as PyTorch tensors.
        """
        Y, D, X = self.generate()
        return (torch.as_tensor(Y, dtype=torch.float32, device=device),
                torch.as_tensor(D, dtype=torch.float32, device=device),
                torch.as_tensor(X, dtype=torch.float32, device=device))
