import numpy as np
import torch
from typing import Optional, Callable, Union

class ODL:
    """
    Modular Online Dictionary Learning with swappable sparse coding algorithms.
    
    Based on the paper:
    "Online Learning for Matrix Factorization and Sparse Coding" 
    by Mairal et al. (2010)
    
    Usage:
        # Create ODL with default OMP sparse coding
        odl = ODL(n_components=128, n_nonzero_coefs=8, batch_size=32)
        
        # Swap to use IRLS for sparse coding
        from sparse_coding.irls import IRLS
        irls = IRLS(max_iter=30)
        odl.sparse_coding = irls.fit
        
        # Or use FISTA
        from sparse_coding.fista import FISTA
        fista = FISTA(lambda_reg=0.1)
        odl.sparse_coding = fista.fit
    """
    
    def __init__(self, n_components: int, n_nonzero_coefs: int = None,
                 batch_size: int = 32, n_iter: int = 100, 
                 learning_rate: Optional[float] = None,
                 verbose: bool = True, device: str = 'cpu'):
        """
        Initialize modular Online Dictionary Learning.
        
        Args:
            n_components: Number of dictionary atoms
            n_nonzero_coefs: Target sparsity (used by default OMP)
            batch_size: Number of signals per mini-batch
            n_iter: Number of passes over the dataset
            learning_rate: Learning rate for dictionary update. If None, uses 1/(t+1) schedule
            verbose: Print progress
            device: 'cpu' or 'cuda'
        """
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.verbose = verbose
        self.device = device
        self.dictionary_ = None
        self.error_history_ = []
        self.iteration_count_ = 0
        
        # Accumulation matrices for dictionary update
        self.A_ = None  # Accumulates X @ X.T
        self.B_ = None  # Accumulates Y @ X.T
        
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
            init_dict: Optional[Union[np.ndarray, torch.Tensor]] = None) -> 'ODL':
        """
        Train dictionary on signal set Y using online learning.
        
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
        
        # Initialize accumulation matrices
        self.A_ = torch.zeros(self.n_components, self.n_components, 
                             dtype=torch.float32, device=self.device)
        self.B_ = torch.zeros(n_features, self.n_components, 
                             dtype=torch.float32, device=self.device)
        self.iteration_count_ = 0
        
        if self.verbose:
            print(f"ODL: Training on {n_samples} signals")
            print(f"Dictionary: {n_features} x {self.n_components}")
            print(f"Batch size: {self.batch_size}")
            print(f"Device: {self.device}")
        
        # Main ODL loop - iterate over dataset
        for epoch in range(self.n_iter):
            print(f"\n--- Iteration {epoch + 1}/{self.n_iter} ---")
            # Shuffle data for this epoch
            indices = torch.randperm(n_samples)
            epoch_error = 0.0
            n_batches = 0
            
            # Process mini-batches
            for batch_idx in range(0, n_samples, self.batch_size):
                print(f"Batch {batch_idx // self.batch_size + 1}/{n_samples // self.batch_size}")
                batch_indices = indices[batch_idx:batch_idx + self.batch_size]
                Y_batch = Y[:, batch_indices]
                
                # Stage 1: Sparse Coding (find X given D)
                print("  Computing sparse codes...")
                X_batch = self._compute_sparse_codes(Y_batch)
                
                # Stage 2: Online Dictionary Update
                print("  Updating dictionary...")
                self._online_dictionary_update(Y_batch, X_batch)
                
                # Compute batch error
                error = torch.norm(Y_batch - self.dictionary_ @ X_batch, 'fro')
                epoch_error += error.item() ** 2
                n_batches += 1
                
                self.iteration_count_ += 1
            
            # Compute epoch RMSE
            epoch_rmse = np.sqrt(epoch_error / n_samples)
            self.error_history_.append(epoch_rmse)
            
            if self.verbose:
                print(f"Epoch {epoch + 1:3d}/{self.n_iter}: RMSE = {epoch_rmse:.6f}")
        
        if self.verbose:
            print(f"Training complete. Final RMSE: {self.error_history_[-1]:.6f}")
        
        return self
    
    def partial_fit(self, Y: Union[np.ndarray, torch.Tensor]) -> 'ODL':
        """
        Update dictionary with a new batch of data (streaming/online mode).
        
        Args:
            Y: New signals of shape (n_features, n_samples)
        
        Returns:
            self
        """
        # Convert to torch tensor
        if isinstance(Y, np.ndarray):
            Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device)
        else:
            Y = Y.to(self.device)
        
        n_features, n_samples = Y.shape
        
        # Initialize dictionary if first call
        if self.dictionary_ is None:
            self.dictionary_ = torch.randn(n_features, self.n_components, 
                                          dtype=torch.float32, device=self.device)
            self._normalize_dictionary()
            
            self.A_ = torch.zeros(self.n_components, self.n_components, 
                                 dtype=torch.float32, device=self.device)
            self.B_ = torch.zeros(n_features, self.n_components, 
                                 dtype=torch.float32, device=self.device)
            self.iteration_count_ = 0
        
        # Sparse coding
        X = self._compute_sparse_codes(Y)
        
        # Online dictionary update
        self._online_dictionary_update(Y, X)
        
        # Compute error
        error = torch.norm(Y - self.dictionary_ @ X, 'fro') / torch.sqrt(torch.tensor(Y.numel(), dtype=torch.float32))
        self.error_history_.append(error.item())
        
        self.iteration_count_ += 1
        
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
    
    def _online_dictionary_update(self, Y: torch.Tensor, X: torch.Tensor):
        """
        Update dictionary using online/stochastic gradient descent.
        
        Based on block-coordinate descent with accumulation matrices:
        A_t = A_{t-1} + X @ X.T
        B_t = B_{t-1} + Y @ X.T
        
        Then solve: min_D ||Y - D @ X||^2 s.t. ||d_k|| = 1
        
        Args:
            Y: Signal matrix (n_features, n_samples)
            X: Sparse codes (n_components, n_samples)
        """
        # Compute learning rate
        if self.learning_rate is None:
            # Decreasing learning rate schedule
            lr = 1.0 / (self.iteration_count_ + 1.0)
        else:
            lr = self.learning_rate
        
        # Update accumulation matrices with exponential forgetting
        self.A_ = (1 - lr) * self.A_ + lr * (X @ X.T)
        self.B_ = (1 - lr) * self.B_ + lr * (Y @ X.T)
        
        # Update dictionary atoms using block-coordinate descent
        for k in range(self.n_components):
            # Skip if atom is not used
            if self.A_[k, k] < 1e-10:
                continue
            
            # Compute update for atom k
            # d_k = (B[:, k] - D @ A[:, k] + d_k * A[k, k]) / A[k, k]
            u_k = self.B_[:, k] - self.dictionary_ @ self.A_[:, k] + \
                  self.dictionary_[:, k] * self.A_[k, k]
            
            # Normalize
            u_k = u_k / self.A_[k, k]
            norm = torch.norm(u_k)
            
            if norm > 1e-10:
                self.dictionary_[:, k] = u_k / norm
            else:
                # Reinitialize if update is zero
                self.dictionary_[:, k] = torch.randn(self.dictionary_.shape[0], device=self.device)
                self.dictionary_[:, k] /= torch.norm(self.dictionary_[:, k])
    
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
    
    def reset_statistics(self):
        """Reset accumulation matrices (useful for starting fresh)."""
        if self.A_ is not None:
            self.A_.zero_()
        if self.B_ is not None:
            self.B_.zero_()
        self.iteration_count_ = 0
        self.error_history_ = []