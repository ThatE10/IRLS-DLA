import torch
import numpy as np
import time
from basis_pursuit.ksvd import KSVD
from sparse_coding.omp import OMP
from sparse_coding.irls import IRLS
from sparse_coding.fista import FISTA
from data.synthetic_data import SyntheticDataGenerator
from experiments.utils.dictionary_metrics import evaluate_recovery

def test_modular_ksvd():
    print("=" * 80)
    print("TESTING MODULAR K-SVD WITH DIFFERENT SPARSE CODING METHODS")
    print("=" * 80)
    
    # Generate data
    print("\nGenerating synthetic data...")

    data_gen = SyntheticDataGenerator(
        n_features=64,
        n_components=128,
        n_samples=1000,
        sparsity=8,
        noise_std=0.05,
        random_state=42
    )

    Y, D, D_true = data_gen.generate_torch()
    
    print(f"Y shape: {Y.shape}")
    print(f"D_true shape: {D_true.shape}")
    
    # Test 1: K-SVD with OMP (default)
    print("\n" + "=" * 80)
    print("TEST 1: K-SVD with OMP (default)")
    print("=" * 80)
    
    ksvd_omp = KSVD(n_components=128, n_nonzero_coefs=8, max_iter=30, verbose=True)
    start = time.time()
    ksvd_omp.fit(Y)
    time_omp = time.time() - start
    
    print(f"\nK-SVD with OMP completed in {time_omp:.2f}s")
    print(f"Final RMSE: {ksvd_omp.error_history_[-1]:.6f}")
    
    # Test 2: K-SVD with custom OMP
    print("\n" + "=" * 80)
    print("TEST 2: K-SVD with custom OMP instance")
    print("=" * 80)
    
    ksvd_custom_omp = KSVD(n_components=128, max_iter=30, verbose=True)
    omp = OMP(n_nonzero_coefs=8)
    ksvd_custom_omp.sparse_coding = omp.fit
    
    start = time.time()
    ksvd_custom_omp.fit(Y)
    time_custom_omp = time.time() - start
    
    print(f"\nK-SVD with custom OMP completed in {time_custom_omp:.2f}s")
    print(f"Final RMSE: {ksvd_custom_omp.error_history_[-1]:.6f}")
    
    # Test 3: K-SVD with IRLS
    print("\n" + "=" * 80)
    print("TEST 3: K-SVD with IRLS")
    print("=" * 80)
    
    ksvd_irls = KSVD(n_components=128, max_iter=30, verbose=True)
    irls = IRLS(max_iter=20, tol=1e-6)
    ksvd_irls.sparse_coding = irls.fit
    
    start = time.time()
    ksvd_irls.fit(Y)
    time_irls = time.time() - start
    
    print(f"\nK-SVD with IRLS completed in {time_irls:.2f}s")
    print(f"Final RMSE: {ksvd_irls.error_history_[-1]:.6f}")
    
    # Test 4: K-SVD with FISTA
    print("\n" + "=" * 80)
    print("TEST 4: K-SVD with FISTA")
    print("=" * 80)
    
    ksvd_fista = KSVD(n_components=128, max_iter=30, verbose=True)
    fista = FISTA(lambda_reg=0.1, max_iter=50)
    ksvd_fista.sparse_coding = fista.fit
    
    start = time.time()
    ksvd_fista.fit(Y)
    time_fista = time.time() - start
    
    print(f"\nK-SVD with FISTA completed in {time_fista:.2f}s")
    print(f"Final RMSE: {ksvd_fista.error_history_[-1]:.6f}")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Method':<20} {'Time (s)':<12} {'Final RMSE':<12}")
    print(f"{'-'*20} {'-'*12} {'-'*12}")
    print(f"{'K-SVD + OMP':<20} {time_omp:<12.2f} {ksvd_omp.error_history_[-1]:<12.6f}")
    print(f"{'K-SVD + OMP (custom)':<20} {time_custom_omp:<12.2f} {ksvd_custom_omp.error_history_[-1]:<12.6f}")
    print(f"{'K-SVD + IRLS':<20} {time_irls:<12.2f} {ksvd_irls.error_history_[-1]:<12.6f}")
    print(f"{'K-SVD + FISTA':<20} {time_fista:<12.2f} {ksvd_fista.error_history_[-1]:<12.6f}")
    
    # Test reconstruction
    print("\n" + "=" * 80)
    print("TESTING RECONSTRUCTION")
    print("=" * 80)
    
    Y_test = Y[:, :100]
    
    for name, ksvd_model in [("OMP", ksvd_omp), ("IRLS", ksvd_irls), ("FISTA", ksvd_fista)]:
        Y_recon = ksvd_model.reconstruct(Y_test)
        rmse = torch.sqrt(torch.mean((torch.as_tensor(Y_test) - Y_recon) ** 2)).item()
        print(f"{name:<10} Reconstruction RMSE: {rmse:.6f}")
    
    from experiments.utils.dictionary_metrics import evaluate_recovery


    # Evaluate recovery
    print("\n" + "=" * 80)
    print("EVALUATING RECOVERY")
    print("=" * 80)
    metrics = evaluate_recovery(D_true, ksvd_omp.get_dictionary().T, 
                            show_matching=True, n_show=15)
    metrics = evaluate_recovery(D_true, ksvd_irls.get_dictionary().T, 
                            show_matching=True, n_show=15)
    metrics = evaluate_recovery(D_true, ksvd_fista.get_dictionary().T, 
                            show_matching=True, n_show=15)
if __name__ == "__main__":
    test_modular_ksvd()
