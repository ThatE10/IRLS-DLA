import numpy as np
import torch
from typing import Optional, Callable, Union

class KSVD:
    """
    Modular K-SVD dictionary learning with swappable sparse coding algorithms.
    
    Usage:
        # Create K-SVD with default OMP sparse coding
        ksvd = KSVD(n_components=128, n_nonzero_coefs=8)
        
        # Swap to use IRLS for sparse coding
        from sparse_coding.irls import IRLS
        irls = IRLS(max_iter=30)
        ksvd.sparse_coding = irls.fit
        
        # Or use FISTA
        from sparse_coding.fista import FISTA
        fista = FISTA(lambda_reg=0.1)
        ksvd.sparse_coding = fista.fit
    """
    
    def __init__(self, n_components: int, n_nonzero_coefs: int = None,
                 max_iter: int = 100, tol: float = 1e-6, verbose: bool = True,
                 device: str = 'cpu'):
        """
        Initialize modular K-SVD.
        
        Args:
            n_components: Number of dictionary atoms
            n_nonzero_coefs: Target sparsity (used by default OMP)
            max_iter: Maximum number of K-SVD iterations
            tol: Convergence tolerance
            verbose: Print progress
            device: 'cpu' or 'cuda'
        """
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.device = device
        self.dictionary_ = None
        self.error_history_ = []
        
        # Default sparse coding method (OMP)
        self._setup_default_sparse_coding()
    
    def _setup_default_sparse_coding(self):
        """Setup default OMP sparse coding."""
        from sparse_coding.omp import OMP
        self._default_omp = OMP(n_nonzero_coefs=self.n_nonzero_coefs)
        self._sparse_coding_fn = self._default_omp.fit
    
    @property
    def sparse_coding(self) -> Callable:
        """Get the current sparse coding function."""
        return self._sparse_coding_fn
    
    @sparse_coding.setter
    def sparse_coding(self, fn: Callable):
        """
        Set a custom sparse coding function.
        
        The function should have signature:
            fn(D: torch.Tensor, Y: torch.Tensor) -> torch.Tensor
        
        Where:
            D: Dictionary (n_features, n_components)
            Y: Signals (n_samples, n_features) for OMP or (n_features, n_samples) for IRLS/FISTA
        
        Returns:
            X: Sparse codes
        """
        self._sparse_coding_fn = fn
    
    def fit(self, Y: Union[np.ndarray, torch.Tensor], 
            init_dict: Optional[Union[np.ndarray, torch.Tensor]] = None) -> 'KSVD':
        """
        Train dictionary on signal set Y.
        
        Args:
            Y: Training signals of shape (n_features, n_samples)
            init_dict: Initial dictionary. If None, randomly initialize
        
        Returns:
            self
        """
        # Convert to torch tensor
        if isinstance(Y, np.ndarray):
            Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        else:
            Y = Y.to(self.device)
        
        n_features, n_samples = Y.shape
        
        # Initialize dictionary
        if init_dict is None:
            self.dictionary_ = torch.randn(n_features, self.n_components, 
                                          dtype=torch.float32, device=self.device)
            self._normalize_dictionary()
        else:
            if isinstance(init_dict, np.ndarray):
                self.dictionary_ = torch.as_tensor(init_dict, dtype=torch.float32, device=self.device)
            else:
                self.dictionary_ = init_dict.to(self.device)
            self._normalize_dictionary()
        
        if self.verbose:
            print(f"K-SVD: Training on {n_samples} signals")
            print(f"Dictionary: {n_features} x {self.n_components}")
            print(f"Device: {self.device}")
        
        # Main K-SVD loop
        for iteration in range(self.max_iter):
            # Stage 1: Sparse Coding (find X given D)
            X = self._compute_sparse_codes(Y)
            
            # Stage 2: Dictionary Update (update D given X)
            self._update_dictionary(Y, X)
            
            # Compute approximation error
            error = torch.norm(Y - self.dictionary_ @ X, 'fro') / torch.sqrt(torch.tensor(Y.numel(), dtype=torch.float32))
            self.error_history_.append(error.item())
            
            if self.verbose and (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1:3d}/{self.max_iter}: RMSE = {error.item():.6f}")
            
            # Check convergence
            if iteration > 0:
                change = abs(self.error_history_[-2] - self.error_history_[-1])
                if change < self.tol:
                    if self.verbose:
                        print(f"Converged at iteration {iteration + 1}")
                    break
        
        if self.verbose:
            print(f"Training complete. Final RMSE: {self.error_history_[-1]:.6f}")
        
        return self
    
    def _compute_sparse_codes(self, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute sparse codes using the assigned sparse coding method.
        
        Args:
            Y: Signal matrix (n_features, n_samples)
        
        Returns:
            X: Sparse codes (n_components, n_samples)
        """
        # Different sparse coding methods expect different input formats:
        # - OMP expects (n_samples, n_features) and returns (n_samples, n_components)
        # - IRLS/FISTA expect (n_features, n_samples) and return (n_components, n_samples)
        
        # Check if this is the default OMP or a custom OMP instance
        is_omp = hasattr(self, '_default_omp') and self._sparse_coding_fn == self._default_omp.fit
        
        if is_omp or 'OMP' in str(type(self._sparse_coding_fn.__self__).__name__):
            # OMP format: (n_samples, n_features) -> (n_samples, n_components)
            X = self._sparse_coding_fn(self.dictionary_, Y.T)
            # Transpose to (n_components, n_samples)
            X = X.T
        else:
            # IRLS/FISTA format: (n_features, n_samples) -> (n_components, n_samples)
            X = self._sparse_coding_fn(self.dictionary_, Y)
        
        return X
    
    def _update_dictionary(self, Y: torch.Tensor, X: torch.Tensor):
        """
        Update dictionary atoms using K-SVD (rank-1 SVD approximation).
        
        Args:
            Y: Signal matrix (n_features, n_samples)
            X: Sparse codes (n_components, n_samples)
        """
        for k in range(self.n_components):
            # Find signals that use this atom
            indices = torch.where(torch.abs(X[k, :]) > 1e-10)[0]
            
            if len(indices) == 0:
                # Atom is unused - reinitialize randomly
                self.dictionary_[:, k] = torch.randn(self.dictionary_.shape[0], device=self.device)
                self.dictionary_[:, k] /= torch.norm(self.dictionary_[:, k])
                continue
            
            # Compute error without this atom's contribution
            E_k = Y[:, indices] - self.dictionary_ @ X[:, indices] + \
                  torch.outer(self.dictionary_[:, k], X[k, indices])
            
            # Update atom and coefficients using SVD (rank-1 approximation)
            U, s, Vt = torch.linalg.svd(E_k, full_matrices=False)
            
            # Update dictionary atom (first left singular vector)
            self.dictionary_[:, k] = U[:, 0]
            
            # Update coefficients (first singular value * first right singular vector)
            X[k, indices] = s[0] * Vt[0, :]
    
    def _normalize_dictionary(self):
        """Normalize all dictionary atoms to unit norm."""
        norms = torch.norm(self.dictionary_, dim=0)
        norms[norms == 0] = 1  # Avoid division by zero
        self.dictionary_ /= norms
    
    def transform(self, Y: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Compute sparse representations for new signals.
        
        Args:
            Y: Signal matrix (n_features, n_samples)
        
        Returns:
            X: Sparse codes (n_components, n_samples)
        """
        if isinstance(Y, np.ndarray):
            Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        else:
            Y = Y.to(self.device)
        
        return self._compute_sparse_codes(Y)
    
    def reconstruct(self, Y: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        """
        Reconstruct signals from their sparse representations.
        
        Args:
            Y: Signal matrix (n_features, n_samples)
        
        Returns:
            Y_reconstructed: Reconstructed signals
        """
        X = self.transform(Y)
        return self.dictionary_ @ X
    
    def get_dictionary(self) -> torch.Tensor:
        """Get the learned dictionary."""
        return self.dictionary_.clone()
    
    def get_dictionary_numpy(self) -> np.ndarray:
        """Get the learned dictionary as numpy array."""
        return self.dictionary_.cpu().numpy()
