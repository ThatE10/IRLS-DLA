import torch
from typing import Tuple, Optional

class CompressedSensingGenerator:
    def __init__(self,
                 n_features: int = 64,        # corresponds to m (measurements)
                 n_components: int = 256,     # corresponds to N (signal dimension)
                 n_samples: int = 2000,
                 sparsity: int = 8,
                 noise_std: float = 0.05,
                 random_state: Optional[int] = None,
                 device: str = 'cpu'):
        
        self.m = n_features        # measurement dimension Φ is m × N
        self.N = n_components      # signal dimension
        self.n_samples = n_samples
        self.sparsity = sparsity
        self.noise_std = noise_std
        self.random_state = random_state
        self.device = device

    def generate(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate synthetic compressed sensing data.

        Returns:
            Y: Noisy measurements      (m, n_samples)
            Phi: Measurement matrix    (m, N)
            X: True sparse signals     (N, n_samples)
        """
        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Measurement matrix Φ (Gaussian scaled by 1/sqrt(m))
        Phi = torch.randn(self.m, self.N, device=self.device) / torch.sqrt(
            torch.tensor(self.m, dtype=torch.float32, device=self.device)
        )

        # Sparse signals X (N × n_samples)
        X = torch.zeros(self.N, self.n_samples, device=self.device)

        for i in range(self.n_samples):
            # choose exactly 'sparsity' indices — uniform, no replacement
            indices = torch.randperm(self.N, device=self.device)[:self.sparsity]
            X[indices, i] = torch.randn(self.sparsity, device=self.device)

        # Clean measurements
        Y_clean = Phi @ X

        # Additive Gaussian noise
        noise = self.noise_std * torch.randn(self.m, self.n_samples, device=self.device)
        Y = Y_clean + noise

        return Y, Phi, X

    def generate_torch(self, device='cpu') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Provided for symmetry with your original class.
        Ensures tensors are on the correct device.
        """
        Y, Phi, X = self.generate()
        return (
            Y.to(device),
            Phi.to(device),
            X.to(device)
        )
