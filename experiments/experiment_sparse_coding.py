import torch
import numpy as np
from sparse_coding.omp import OMP
from sparse_coding.irls import IRLS
from sparse_coding.fista import FISTA
from sklearn.datasets import make_sparse_coded_signal
import time

def test_omp():
    print("Testing OMP...")
    n_samples, n_components, n_features = 100, 100, 50
    n_nonzero_coefs = 10
    
    y, X, w = make_sparse_coded_signal(
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        n_nonzero_coefs=n_nonzero_coefs,
        random_state=42
    )
    
    # X is (n_features, n_components), y is (n_features, n_samples)
    # make_sparse_coded_signal returns y as (n_features, n_samples)
    
    X_torch = torch.as_tensor(X, dtype=torch.float32)
    y_torch = torch.as_tensor(y, dtype=torch.float32)
    
    # Test CPU
    omp = OMP(n_nonzero_coefs=n_nonzero_coefs)
    start = time.time()
    coefs = omp.fit(X_torch, y_torch.T) # OMP expects y as (n_samples, n_features)
    end = time.time()
    print(f"OMP CPU time: {end - start:.4f}s")
    
    # Check reconstruction
    # coefs is (n_samples, n_components)
    # y_pred = (X @ coefs.T).T -> (n_samples, n_features)
    y_pred = (X_torch @ coefs.T).T
    error = torch.norm(y_torch.T - y_pred) / torch.norm(y_torch.T)
    print(f"OMP CPU Reconstruction Error: {error.item():.4f}")
    
    # Test GPU if available
    if torch.cuda.is_available():
        print("Testing OMP on GPU...")
        X_gpu = X_torch.cuda()
        y_gpu = y_torch.cuda()
        omp_gpu = OMP(n_nonzero_coefs=n_nonzero_coefs)
        start = time.time()
        coefs_gpu = omp_gpu.fit(X_gpu, y_gpu.T)
        end = time.time()
        print(f"OMP GPU time: {end - start:.4f}s")
        y_pred_gpu = (X_gpu @ coefs_gpu.T).T
        error_gpu = torch.norm(y_gpu.T - y_pred_gpu) / torch.norm(y_gpu.T)
        print(f"OMP GPU Reconstruction Error: {error_gpu.item():.4f}")

def test_irls():
    print("\nTesting IRLS...")
    # Generate data similar to irls_sparse_regularization.py
    n_features = 64
    n_components = 256
    n_samples = 10 # Reduced for quick test
    sparsity = 8
    noise_std = 0.05
    
    D_true = np.random.randn(n_features, n_components)
    D_true /= np.linalg.norm(D_true, axis=0, keepdims=True)
    
    X_true = np.zeros((n_components, n_samples))
    for i in range(n_samples):
        indices = np.random.choice(n_components, sparsity, replace=False)
        X_true[indices, i] = np.random.randn(sparsity)
        
    Y_clean = D_true @ X_true
    noise = noise_std * np.random.randn(n_features, n_samples)
    Y = Y_clean + noise
    
    A = torch.as_tensor(D_true, dtype=torch.float32)
    y = torch.as_tensor(Y, dtype=torch.float32)
    
    irls = IRLS(max_iter=50, tol=1e-4, epsilon_start=1.0)
    start = time.time()
    X_est = irls.fit(A, y)
    end = time.time()
    print(f"IRLS CPU time: {end - start:.4f}s")
    
    # Check reconstruction
    # X_est is (n_components, n_samples)
    Y_pred = A @ X_est
    error = torch.norm(y - Y_pred) / torch.norm(y)
    print(f"IRLS CPU Reconstruction Error: {error.item():.4f}")
    
    # Check sparsity (approximate)
    sparsity_est = (torch.abs(X_est) > 0.1).sum().float() / n_samples
    print(f"Average Estimated Sparsity: {sparsity_est.item():.2f} (True: {sparsity})")

    if torch.cuda.is_available():
        print("Testing IRLS on GPU...")
        A_gpu = A.cuda()
        y_gpu = y.cuda()
        irls_gpu = IRLS(max_iter=50, tol=1e-4)
        start = time.time()
        X_est_gpu = irls_gpu.fit(A_gpu, y_gpu)
        end = time.time()
        print(f"IRLS GPU time: {end - start:.4f}s")
        Y_pred_gpu = A_gpu @ X_est_gpu
        error_gpu = torch.norm(y_gpu - Y_pred_gpu) / torch.norm(y_gpu)
        print(f"IRLS GPU Reconstruction Error: {error_gpu.item():.4f}")

def test_fista():
    print("\nTesting FISTA...")
    n_samples, n_components, n_features = 100, 100, 50
    # FISTA solves Lasso, so we need sparse data
    y, X, w = make_sparse_coded_signal(
        n_samples=n_samples,
        n_components=n_components,
        n_features=n_features,
        n_nonzero_coefs=10,
        random_state=42
    )
    
    # X is dictionary (n_features, n_components)
    # y is signals (n_features, n_samples)
    
    A = torch.as_tensor(X, dtype=torch.float32)
    y_torch = torch.as_tensor(y, dtype=torch.float32)
    
    fista = FISTA(lambda_reg=0.1, max_iter=100)
    start = time.time()
    X_est = fista.fit(A, y_torch)
    end = time.time()
    print(f"FISTA CPU time: {end - start:.4f}s")
    
    # Check reconstruction
    Y_pred = A @ X_est
    error = torch.norm(y_torch - Y_pred) / torch.norm(y_torch)
    print(f"FISTA CPU Reconstruction Error: {error.item():.4f}")
    
    if torch.cuda.is_available():
        print("Testing FISTA on GPU...")
        A_gpu = A.cuda()
        y_gpu = y_torch.cuda()
        fista_gpu = FISTA(lambda_reg=0.1, max_iter=100)
        start = time.time()
        X_est_gpu = fista_gpu.fit(A_gpu, y_gpu)
        end = time.time()
        print(f"FISTA GPU time: {end - start:.4f}s")
        Y_pred_gpu = A_gpu @ X_est_gpu
        error_gpu = torch.norm(y_gpu - Y_pred_gpu) / torch.norm(y_gpu)
        print(f"FISTA GPU Reconstruction Error: {error_gpu.item():.4f}")

if __name__ == "__main__":
    test_omp()
    test_irls()
    test_fista()
