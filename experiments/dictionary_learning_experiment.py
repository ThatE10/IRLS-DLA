import torch
import numpy as np
import time
from data.synthetic_data import SyntheticDataGenerator
from k_svd import SimpleKSVD, PAKSVD, HAS_CUPY
from sparse_coding.omp import OMP
from sparse_coding.irls import IRLS
from sparse_coding.fista import FISTA

def compute_dictionary_recovery(D_learned, D_true):
    """
    Compute dictionary recovery metric.
    For each learned atom, find the best matching true atom.
    Returns average absolute correlation.
    """
    n_atoms = D_learned.shape[1]
    correlations = []
    
    for i in range(n_atoms):
        # Compute correlation with all true atoms
        corr = np.abs(D_true.T @ D_learned[:, i])
        # Take maximum correlation
        correlations.append(np.max(corr))
    
    return np.mean(correlations)

def run_dictionary_learning_experiment():
    print("=" * 80)
    print("DICTIONARY LEARNING EXPERIMENT")
    print("=" * 80)
    
    # Parameters
    n_features = 64
    n_components = 128
    n_samples = 2000
    sparsity = 8
    noise_std = 0.05
    
    print(f"\nParameters:")
    print(f"  Features: {n_features}")
    print(f"  Dictionary atoms: {n_components}")
    print(f"  Training samples: {n_samples}")
    print(f"  Sparsity: {sparsity}")
    print(f"  Noise std: {noise_std}")
    
    # Generate data
    print("\n" + "-" * 80)
    print("Generating synthetic data...")
    print("-" * 80)
    
    data_gen = SyntheticDataGenerator(
        n_features=n_features,
        n_components=n_components,
        n_samples=n_samples,
        sparsity=sparsity,
        noise_std=noise_std,
        random_state=42
    )
    
    Y, D_true, X_true = data_gen.generate()
    
    print(f"\nData generated:")
    print(f"  Y shape: {Y.shape}")
    print(f"  D_true shape: {D_true.shape}")
    print(f"  X_true shape: {X_true.shape}")
    
    # Compute baseline reconstruction error
    Y_clean = D_true @ X_true
    baseline_rmse = np.sqrt(np.mean((Y - Y_clean) ** 2))
    print(f"  Baseline RMSE (noise level): {baseline_rmse:.6f}")
    
    # Train K-SVD
    print("\n" + "=" * 80)
    print("TRAINING K-SVD")
    print("=" * 80)
    
    ksvd = SimpleKSVD(
        n_components=n_components,
        n_nonzero_coefs=sparsity,
        max_iter=50,
        verbose=True
    )
    
    start = time.time()
    ksvd.fit(Y)
    ksvd_time = time.time() - start
    
    D_learned = ksvd.get_dictionary()
    
    print(f"\nK-SVD training time: {ksvd_time:.2f}s")
    print(f"Final RMSE: {ksvd.error_history_[-1]:.6f}")
    
    # Compute dictionary recovery
    recovery = compute_dictionary_recovery(D_learned, D_true)
    print(f"Dictionary recovery (avg correlation): {recovery:.4f}")
    
    # Test sparse coding methods on learned dictionary
    print("\n" + "=" * 80)
    print("TESTING SPARSE CODING METHODS WITH LEARNED DICTIONARY")
    print("=" * 80)
    
    # Use a subset for testing
    n_test = 100
    Y_test = Y[:, :n_test]
    X_test_true = X_true[:, :n_test]
    
    # Convert to torch for sparse coding methods
    D_torch = torch.as_tensor(D_learned, dtype=torch.float32)
    Y_test_torch = torch.as_tensor(Y_test, dtype=torch.float32)
    
    # Test OMP
    print("\n--- Testing OMP ---")
    omp = OMP(n_nonzero_coefs=sparsity)
    start = time.time()
    X_omp = omp.fit(D_torch, Y_test_torch.T)  # OMP expects (n_samples, n_features)
    omp_time = time.time() - start
    
    Y_recon_omp = (D_torch @ X_omp.T).numpy()
    rmse_omp = np.sqrt(np.mean((Y_test - Y_recon_omp) ** 2))
    sparsity_omp = (torch.abs(X_omp) > 1e-6).sum().item() / n_test
    
    print(f"OMP Time: {omp_time:.4f}s")
    print(f"OMP RMSE: {rmse_omp:.6f}")
    print(f"OMP Avg Sparsity: {sparsity_omp:.2f}")
    
    # Test IRLS
    print("\n--- Testing IRLS ---")
    irls = IRLS(max_iter=30, tol=1e-4)
    start = time.time()
    X_irls = irls.fit(D_torch, Y_test_torch).numpy()
    irls_time = time.time() - start
    
    Y_recon_irls = D_learned @ X_irls
    rmse_irls = np.sqrt(np.mean((Y_test - Y_recon_irls) ** 2))
    sparsity_irls = (np.abs(X_irls) > 1e-6).sum() / n_test
    
    print(f"IRLS Time: {irls_time:.4f}s")
    print(f"IRLS RMSE: {rmse_irls:.6f}")
    print(f"IRLS Avg Sparsity: {sparsity_irls:.2f}")
    
    # Test FISTA
    print("\n--- Testing FISTA ---")
    fista = FISTA(lambda_reg=0.1, max_iter=50)
    start = time.time()
    X_fista = fista.fit(D_torch, Y_test_torch).numpy()
    fista_time = time.time() - start
    
    Y_recon_fista = D_learned @ X_fista
    rmse_fista = np.sqrt(np.mean((Y_test - Y_recon_fista) ** 2))
    sparsity_fista = (np.abs(X_fista) > 1e-6).sum() / n_test
    
    print(f"FISTA Time: {fista_time:.4f}s")
    print(f"FISTA RMSE: {rmse_fista:.6f}")
    print(f"FISTA Avg Sparsity: {sparsity_fista:.2f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nDictionary Learning:")
    print(f"  K-SVD Time: {ksvd_time:.2f}s")
    print(f"  Dictionary Recovery: {recovery:.4f}")
    print(f"  K-SVD Final RMSE: {ksvd.error_history_[-1]:.6f}")
    
    print(f"\nSparse Coding Comparison (on {n_test} test samples):")
    print(f"  {'Method':<10} {'Time (s)':<12} {'RMSE':<12} {'Avg Sparsity':<15}")
    print(f"  {'-'*10} {'-'*12} {'-'*12} {'-'*15}")
    print(f"  {'OMP':<10} {omp_time:<12.4f} {rmse_omp:<12.6f} {sparsity_omp:<15.2f}")
    print(f"  {'IRLS':<10} {irls_time:<12.4f} {rmse_irls:<12.6f} {sparsity_irls:<15.2f}")
    print(f"  {'FISTA':<10} {fista_time:<12.4f} {rmse_fista:<12.6f} {sparsity_fista:<15.2f}")
    print(f"  {'Baseline':<10} {'-':<12} {baseline_rmse:<12.6f} {sparsity:<15}")
    
    # Test with GPU K-SVD if available
    if HAS_CUPY and torch.cuda.is_available():
        print("\n" + "=" * 80)
        print("TESTING GPU K-SVD (PAKSVD)")
        print("=" * 80)
        
        try:
            paksvd = PAKSVD(
                n_components=n_components,
                n_nonzero_coefs=sparsity,
                max_iter=50,
                n_parallel_atoms=32,
                n_update_cycles=1
            )
            
            start = time.time()
            paksvd.fit(Y)
            paksvd_time = time.time() - start
            
            D_learned_gpu = paksvd.get_dictionary()
            recovery_gpu = compute_dictionary_recovery(D_learned_gpu, D_true)
            
            print(f"\nPAKSVD training time: {paksvd_time:.2f}s")
            print(f"PAKSVD Final RMSE: {paksvd.error_history_[-1]:.6f}")
            print(f"PAKSVD Dictionary recovery: {recovery_gpu:.4f}")
            print(f"Speedup vs CPU K-SVD: {ksvd_time / paksvd_time:.2f}x")
            
        except Exception as e:
            print(f"PAKSVD failed: {e}")
    else:
        print("\n" + "=" * 80)
        print("GPU K-SVD (PAKSVD) not available")
        print("=" * 80)
        if not HAS_CUPY:
            print("CuPy not installed")
        if not torch.cuda.is_available():
            print("CUDA not available")

if __name__ == "__main__":
    run_dictionary_learning_experiment()
