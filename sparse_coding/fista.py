import torch

class FISTA:
    def __init__(self, lambda_reg=0.1, max_iter=100, tol=1e-4, L=None):
        self.lambda_reg = lambda_reg
        self.max_iter = max_iter
        self.tol = tol
        self.L = L # Lipschitz constant

    def fit(self, A, y):
        """
        Solve min 0.5 * ||Ax - y||_2^2 + lambda * ||x||_1 using FISTA.
        A: (n_features, n_components) - Dictionary
        y: (n_features, n_samples) - Signals
        """
        if not isinstance(A, torch.Tensor):
            A = torch.as_tensor(A, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y, dtype=torch.float32)

        if y.device != A.device:
            y = y.to(A.device)
            
        self.device = A.device
        
        n_features, n_components = A.shape
        n_samples = y.shape[1]
        
        # Calculate Lipschitz constant if not provided
        if self.L is None:
            # L = max eigenvalue of A^T A
            # Use power iteration or torch.linalg.eigvalsh if A is small enough
            # For speed, let's approximate or compute once.
            # A is (M, N). A^T A is (N, N).
            # If N is large, this is expensive.
            # Power method is better.
            self.L = self._compute_lipschitz(A)
            
        L = self.L
        lambda_val = self.lambda_reg
        
        # Initialize
        x = torch.zeros(n_components, n_samples, device=self.device)
        y_k = x.clone()
        t = 1.0
        
        for k in range(self.max_iter):
            x_old = x.clone()
            
            # Gradient step: y_k - (1/L) * A^T (A y_k - y)
            residual = A @ y_k - y
            grad = A.T @ residual
            z = y_k - (1.0 / L) * grad
            
            # Proximal step (Soft Thresholding)
            x = self._soft_threshold(z, lambda_val / L)
            
            # Check convergence
            if torch.norm(x - x_old) / (torch.norm(x) + 1e-10) < self.tol:
                break
                
            # FISTA update
            t_new = (1.0 + (1.0 + 4.0 * t**2)**0.5) / 2.0
            y_k = x + ((t - 1.0) / t_new) * (x - x_old)
            t = t_new
            
        return x

    def _compute_lipschitz(self, A, max_iter=100):
        # Power iteration to find max eigenvalue of A^T A
        n_components = A.shape[1]
        v = torch.randn(n_components, 1, device=self.device)
        v = v / torch.norm(v)
        
        # Avoid forming A.T @ A explicitly as it can be very large
        
        for _ in range(max_iter):
            # v_new = A.T @ (A @ v)
            Av = A @ v
            v_new = A.T @ Av
            
            v_new = v_new / torch.norm(v_new)
            if torch.norm(v_new - v) < 1e-6:
                v = v_new
                break
            v = v_new
            
        # Rayleigh quotient
        # (v.T @ A.T @ A @ v) / (v.T @ v)
        # = ||Av||^2 / ||v||^2
        Av = A @ v
        eigenvalue = (torch.norm(Av)**2) / (torch.norm(v)**2)
        return eigenvalue.item()

    def _soft_threshold(self, x, alpha):
        return torch.sign(x) * torch.relu(torch.abs(x) - alpha)
