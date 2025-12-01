import torch

class IRLS:
    def __init__(self, max_iter=200, tol=1e-6, p=1.0, epsilon_start=1.0,n_nonzero_coefs=10):
        self.max_iter = max_iter
        self.tol = tol
        self.p = p # Currently implementation focuses on p=1 approximation
        self.epsilon_start = epsilon_start
        self.n_nonzero_coefs = n_nonzero_coefs

    def fit(self, A, y):
        """
        Solve min ||x||_p s.t. Ax = y using IRLS.
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
        
        # Handle batch of signals
        # The original script does:
        # x_opt = torch.linalg.solve(A @ torch.linalg.inv(W) @ A.T, y)
        # x = torch.linalg.inv(W) @ A.T @ x_opt
        # This works for single vector y if W is diagonal.
        # For batch y, W would need to be different for each sample?
        # The original script:
        # y = torch.Tensor(Y.T[0]) -> Single sample.
        # So the original script only handles ONE sample at a time.
        
        # To support batch processing, we need to be careful.
        # If we process samples independently, we can loop or try to batch.
        # Batching matrix inversion might be heavy if batch size is large.
        # But let's try to support batching if possible, or at least loop over samples if not.
        # Given the request "implement IRLS in its own class", I should probably support what the script did but ideally better.
        # The script did single sample.
        # Let's implement for batch if dimensions allow, otherwise iterate.
        
        # If y is (n_features, n_samples)
        #n_features, n_samples = y.shape
        n_components = A.shape[1]
        
        
        # For now, let's loop over samples to ensure correctness matching the script's logic which was per-sample.
        # Optimization to batch can be done if needed, but W changes per sample.
        max_subbatch = 128  # maximum number of columns per sub-batch

        if len(y.shape) > 1 and y.shape[1] > 1:
            n_total = y.shape[1]
            if n_total <= max_subbatch:
                # Small enough, just solve normally
                X = self._solve_batch(
                    A,
                    y,
                    n_iter=self.max_iter,
                    epsilon_init=self.epsilon_start,
                    p=self.p
                )
            else:
                # Split into sub-batches
                X_list = []
                for start in range(0, n_total, max_subbatch):
                    end = min(start + max_subbatch, n_total)
                    X_sub = self._solve_batch(
                        A,
                        y[:, start:end],
                        n_iter=self.max_iter,
                        epsilon_init=self.epsilon_start,
                        p=self.p
                    )
                    X_list.append(X_sub)
                X = torch.cat(X_list, dim=1)
        else:
            X = self._solve_single(
                A,
                y.squeeze(),
                n_iter=self.max_iter,
                epsilon_init=self.epsilon_start,
                p=self.p
            )

        return X


    def _solve_batch(self, A, y, n_iter=200, epsilon_init=1.0, tol=1e-8, p=0.5, 
                 epsilon_decay=0.5, epsilon_min=1e-20, verbose=False):
        """
        Batched IRLS solver for sparse recovery: minimize ||x||_p subject to Ax ≈ y
        
        Solves multiple samples in parallel using vectorized operations.
        Based on: Daubechies, DeVore, Fornasier, Gunturk (2010)
        "Iteratively reweighted least squares minimization for sparse recovery"
        
        Args:
            A: measurement matrix of shape (m, n) where m < n
            y: measurement vectors of shape (m, n_samples)
            n_iter: maximum number of iterations
            epsilon_init: initial regularization parameter
            tol: convergence tolerance (relative change in solution)
            p: p-norm parameter (0 < p < 1 for sparse recovery, typically 0.5 or 0.8)
            epsilon_decay: decay rate for epsilon at each iteration
            epsilon_min: minimum value for epsilon
            verbose: if True, print convergence information
            
        Returns:
            x: recovered sparse solutions of shape (n, n_samples)
        """
        m, n = A.shape
        n_features, n_samples = y.shape  # m should equal n_features
        
        device = A.device
        dtype = A.dtype
        
        # Initialize with least squares solution for each sample
        # Shape: [n, n_samples]
        x = torch.linalg.lstsq(A, y).solution
        
        # Transpose for batched operations: [n_samples, n]
        x = x.T
        
        # Initialize epsilon: [n_samples]
        epsilon = torch.full((n_samples,), epsilon_init, device=device, dtype=dtype)
        
        # Prepare y: [n_samples, m]
        y_batched = y.T
        
        for iteration in range(n_iter):
            # Compute weights: w_i = (|x_i|^2 + epsilon)^(p/2 - 1)
            # Shape: [n_samples, n]
            eps_broadcast = epsilon.unsqueeze(-1)  # [n_samples, 1]
            weights = (x**2 + eps_broadcast) ** (p/2 - 1)
            
            # Compute W^(-2) = 1 / weights^2 with numerical stability
            # Shape: [n_samples, n]
            W_inv_sq = 1.0 / (weights**2 + 1e-12)
            
            # Compute A @ W^(-2) for each sample
            # A_weighted[i, :, :] = A @ diag(W_inv_sq[i, :])
            # Shape: [n_samples, m, n]
            A_weighted = A.unsqueeze(0) * W_inv_sq.unsqueeze(1)
            
            # Compute gram matrix: A @ W^(-2) @ A^T for each sample
            # Shape: [n_samples, m, m]
            gram_matrix = A_weighted @ A.T.unsqueeze(0)
            
            # Add regularization for numerical stability
            # Shape: [n_samples, m, m]
            reg_term = 1e-8 * torch.eye(m, device=device, dtype=dtype).unsqueeze(0)
            gram_matrix = gram_matrix + reg_term
            
            # Solve for Lagrange multipliers: (A W^(-2) A^T) λ = y
            # Shape: [n_samples, m]
            lambda_vals = torch.linalg.solve(gram_matrix, y_batched.unsqueeze(-1)).squeeze(-1)
            
            # Recover solution: x = W^(-2) A^T λ
            # Shape: [n_samples, n]
            x_new = W_inv_sq * (A.T.unsqueeze(0) @ lambda_vals.unsqueeze(-1)).squeeze(-1)
            
            # Check convergence for each sample
            rel_change = torch.norm(x_new - x, dim=1) / (torch.norm(x, dim=1) + 1e-10)
            
            if verbose and iteration % 20 == 0:
                residual = torch.norm(A.unsqueeze(0) @ x_new.unsqueeze(-1) - y_batched.unsqueeze(-1), dim=1).squeeze()
                sparsity_threshold = 1e-3 * torch.max(torch.abs(x_new), dim=1)[0].unsqueeze(-1)
                sparsity = torch.sum(torch.abs(x_new) > sparsity_threshold, dim=1).float()
                
                print(f"Iter {iteration:3d}: epsilon=[{epsilon.min():.2e}, {epsilon.max():.2e}], "
                    f"residual=[{residual.min():.2e}, {residual.max():.2e}], "
                    f"rel_change=[{rel_change.min():.2e}, {rel_change.max():.2e}], "
                    f"sparsity=[{sparsity.min():.1f}, {sparsity.max():.1f}]")
            
            # Check if all samples have converged
            if torch.all(rel_change < tol):
                if verbose:
                    print(f"Converged at iteration {iteration} (max rel_change={rel_change.max():.2e})")
                x_new[x_new.abs() < 1e-4] = 0
                return x_new.T  # [n, n_samples]
            
            # Update for next iteration
            x = x_new
            epsilon = torch.maximum(epsilon * epsilon_decay, 
                                torch.full_like(epsilon, epsilon_min))
            
            if verbose and iteration % 20 == 0:
                sparsity_threshold = 1e-3 * torch.max(torch.abs(x), dim=1)[0].unsqueeze(-1)
                sparsity = torch.sum(torch.abs(x) > sparsity_threshold, dim=1).float()
                print(f"  Updated epsilon range: [{epsilon.min():.2e}, {epsilon.max():.2e}]")
                print(f"  Sparsity range: [{sparsity.min():.1f}, {sparsity.max():.1f}]")
        
        if verbose:
            print(f"Reached maximum iterations ({n_iter})")
        
        # Final sparsification
        x[x.abs() < 1e-4] = 0
        
        # Print final statistics
        if verbose:
            sparsity_threshold = 1e-3 * torch.max(torch.abs(x), dim=1)[0].unsqueeze(-1)
            final_sparsity = torch.sum(torch.abs(x) > sparsity_threshold, dim=1)
            print(f"Final sparsity per sample: {final_sparsity}")
        
        return x.T  # [n, n_samples]

    def _solve_single(self, A, y, n_iter=200, epsilon_init=1.0, tol=1e-8, p=0.5, epsilon_decay=0.5, epsilon_min=1e-20, verbose=False):
        """
        Solve sparse recovery problem using IRLS (Iteratively Reweighted Least Squares).

        Solves: minimize ||x||_p subject to Ax approx y

        Based on: Daubechies, DeVore, Fornasier, Gunturk (2010)
        "Iteratively reweighted least squares minimization for sparse recovery"

        Args:
            A: measurement matrix of shape (m, n) where m < n
            y: measurement vector of shape (m,)
            n_iter: maximum number of iterations
            epsilon_init: initial regularization parameter
            tol: convergence tolerance (relative change in solution)
            p: p-norm parameter (0 < p < 1 for sparse recovery, typically 0.5 or 0.8)
            epsilon_decay: decay rate for epsilon at each iteration
            epsilon_min: minimum value for epsilon
            verbose: if True, print convergence information

        Returns:
            x: recovered sparse solution of shape (n,)
        """

        m, n = A.shape
        print("m", m)
        print("n", n)
        device = A.device
        dtype = A.dtype

        # Initialize with least squares solution
        x = torch.linalg.lstsq(A, y).solution
        epsilon = epsilon_init

        for iteration in range(n_iter):
            # Compute weights: w_i = (|x_i|^2 + epsilon)^(p/2 - 1)
            # For p < 1, this gives larger weights to smaller components
            weights = (x**2 + epsilon) ** (p/2 - 1)
            
            # Solve constrained weighted least squares:
            # minimize ||W x||_2^2 subject to Ax = y
            # 
            # Solution via Lagrange multipliers:
            # x = W^(-2) A^T (A W^(-2) A^T)^(-1) y
            
            W_inv_sq = 1.0 / (weights**2 + 1e-12)  # W^(-2), add small term for stability
            
            # Compute A W^(-2) A^T (m x m matrix)
       

            AW = A @ torch.diag(W_inv_sq.squeeze())  # Broadcasting: (m, n) * (1, n) = (m, n)
            gram_matrix = AW @ A.T  # (m, m)
            
            
            # Add small regularization for numerical stability
            gram_matrix = gram_matrix + 1e-10 * torch.eye(m, device=device, dtype=dtype)
            
            # Solve for Lagrange multipliers: (A W^(-2) A^T) λ = y
            lambda_vec = torch.linalg.solve(gram_matrix, y)
            
            # Recover solution: x = W^(-2) A^T λ
            x_new = W_inv_sq * (A.T @ lambda_vec)
            
            # Check convergence
            rel_change = torch.norm(x_new - x) / (torch.norm(x) + 1e-10)
            
            if verbose and iteration % 20 == 0:
                residual = torch.norm(A @ x_new - y).item()
                sparsity = torch.sum(torch.abs(x_new) > 1e-3 * torch.max(torch.abs(x_new))).item()
                print(f"Iter {iteration:3d}: epsilon={epsilon:.2e}, residual={residual:.2e}, "
                    f"rel_change={rel_change:.2e}, sparsity={sparsity}")
            
            if rel_change < tol:
                if verbose:
                    print(f"Converged at iteration {iteration} (rel_change={rel_change:.2e})")
                return x_new
            
            # Update for next iteration
            x = x_new
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            print("epsilon", epsilon)
            print("sparsity", torch.sum(torch.abs(x_new) > 1e-3 * torch.max(torch.abs(x_new))).item())

        if verbose:
            print(f"Reached maximum iterations ({n_iter})")
        x[x.abs() < 1e-4] = 0
        return x


    
    def _solve_single2(self, A, y, x_solution=None, n_iter=200, epsilon_init=0.001, tol=1e-8):
        """
        IRLS solver for min ||x||_1 s.t. Ax = y
        A : torch.Tensor [m x n]
        y : torch.Tensor [m]
        x_solution : torch.Tensor [n] (optional, ground truth for monitoring)
        """
        n = A.shape[1]
        W = torch.eye(n, device=A.device)  # initial weights
        W_inv = torch.eye(n, device=A.device)  # initial weights inverse
        epsilon = epsilon_init
        print("new iteration")
        x = torch.zeros(n, device=A.device)
        for i in range(n_iter):
            # Solve weighted least squares: x = W^{-1} A^T (A W^{-1} A^T)^{-1} y
            #AW_inv = A @ torch.linalg.inv(W)
            AW_inv = A @ W_inv
            x_opt = torch.linalg.solve(AW_inv @ A.T, y)

            # print out shapes of everything
            print("AW_inv shape: ", AW_inv.shape)
            print("A.T shape: ", A.T.shape)
            print("y shape: ", y.shape)
            print("x_opt shape: ", x_opt.shape)
            
            #x = torch.linalg.inv(W) @ A.T @ x_opt
            x = W_inv @ A.T @ x_opt

            #print out shapes of everything
            print("x shape: ", x.shape)
            print("x_opt shape: ", x_opt.shape)
            print("W_inv shape: ", W_inv.shape)
            print("A.T shape: ", A.T.shape)
            
            # Print diagnostics
            if x_solution is not None:
                print(f"Iter {i+1}: ℓ0 approx: {torch.sum(torch.abs(x)>1e-3):.0f}, "
                    f"||x-x_sol||: {torch.norm(x-x_solution):.4f}")
            
            # Update weights
            W = torch.diag((x**2 + epsilon**2)**-0.5)
            W_inv = torch.diag((x**2 + epsilon**2)**0.5)
            # Geometric reduction of epsilon
            new_eps =  sorted(x.abs().tolist(),reverse=True)[self.n_nonzero_coefs+1]
            print(new_eps)
            epsilon =min(epsilon*.9,new_eps[0]/n)  # avoid zero
            print(epsilon)
            
            # Optional sparsification
            
        x[x.abs() < 1e-5] = 0
        print(x)
        print(self.n_nonzero_coefs)
        print(torch.count_nonzero(x))
        return x


def irls_sparse_recovery(Phi, y, p=0.5, max_iter=1000, epsilon_0=1.0, 
                         epsilon_min=1e-10, tau=0.5, verbose=False):
    """
    Iteratively Reweighted Least Squares (IRLS) for sparse recovery.
    
    Based on: Daubechies, DeVore, Fornasier, Gunturk (2010)
    "Iteratively reweighted least squares minimization for sparse recovery"
    
    Solves: minimize ||x||_p subject to Phi x = y (or approximately Phi x approx y)
    
    Algorithm:
    1. Start with initial solution x^(0)
    2. At iteration n, compute weights: w_i^(n) = (|x_i^(n)|^2 + epsilon_n)^(p/2 - 1)
    3. Solve: x^(n+1) = argmin ||x||_{W^(n)}^2 subject to Phi x = y
       where ||x||_W^2 = sum_i w_i^2 x_i^2
    4. This is equivalent to: minimize ||W^(n) x||_2^2 subject to Phi x = y
    5. Solution via Lagrange multipliers: 
       x^(n+1) = W^(-2) Phi^T (Phi W^(-2) Phi^T)^(-1) y
    
    Args:
        Phi: measurement matrix (m x N)
        y: measurements (m,)
        p: p-norm parameter (0 < p < 1 for sparse recovery, typically 0.5 or 0.8)
        max_iter: maximum number of iterations
        epsilon_0: initial regularization parameter
        epsilon_min: minimum epsilon value
        tau: decay rate for epsilon (epsilon_n = max(tau * epsilon_{n-1}, epsilon_min))
        verbose: print progress
    
    Returns:
        x_final: recovered signal
        history: dictionary with convergence history
    """
    m, N = Phi.shape
    device = Phi.device
    
    # Initialize solution using least squares
    x = torch.linalg.lstsq(Phi, y).solution
    epsilon = epsilon_0
    
    # History tracking
    history = {
        'x_values': [],
        'epsilon_values': [],
        'residuals': [],
        'sparsity': [],
        'objective': []
    }
    print(Phi.shape)
    for iteration in range(max_iter):
        # Save current state
        history['x_values'].append(x.clone())
        history['epsilon_values'].append(epsilon)
        
        # Compute weights: w_i = (|x_i|^2 + epsilon)^(p/2 - 1)
        # For p < 1, this gives higher weights to smaller components
        weights = (x**2 + epsilon) ** (p/2 - 1)
        
        # Solve constrained weighted least squares:
        # minimize ||W x||_2^2 subject to Phi x = y
        # Using Lagrange multipliers, the solution is:
        # x = W^(-2) Phi^T (Phi W^(-2) Phi^T)^(-1) y
        
        W_inv_sq = 1.0 / (weights**2 + 1e-12)  # W^(-2)
        
        # Compute Phi W^(-2) Phi^T
        PhiW = Phi * W_inv_sq.unsqueeze(0)  # Broadcasting
        A = PhiW @ Phi.T  # m x m matrix
        
        # Add small regularization for numerical stability
        A = A + 1e-10 * torch.eye(m, device=device)
        
        # Solve A λ = y for Lagrange multipliers
        lambda_vec = torch.linalg.solve(A, y)
        
        # Recover x = W^(-2) Phi^T λ
        x_new = W_inv_sq * (Phi.T @ lambda_vec)
        
        # Compute residual
        residual = torch.norm(Phi @ x_new - y).item()
        history['residuals'].append(residual)
        
        # Compute objective: sum_i (|x_i|^2 + epsilon)^(p/2)
        objective = torch.sum((x_new**2 + epsilon)**(p/2)).item()
        history['objective'].append(objective)
        
        # Compute sparsity (number of "significant" components)
        threshold = 1e-3 * torch.max(torch.abs(x_new))
        sparsity = torch.sum(torch.abs(x_new) > threshold).item()
        history['sparsity'].append(sparsity)
        
        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration:3d}: epsilon={epsilon:.2e}, "
                  f"residual={residual:.2e}, objective={objective:.3f}, sparsity={sparsity}")
        
        # Check convergence
        rel_change = torch.norm(x_new - x) / (torch.norm(x) + 1e-10)
        if rel_change < 1e-8:
            if verbose:
                print(f"Converged at iteration {iteration} (rel_change={rel_change:.2e})")
            x = x_new
            break
        
        x = x_new
        
        # Update epsilon (decrease over iterations)
        epsilon = max(tau * epsilon, epsilon_min)
    x[x.abs() < 1e-5] = 0
    return x, history