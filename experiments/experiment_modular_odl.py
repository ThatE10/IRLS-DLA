import sys, os
# Ensure the project root is in sys.path for imports when script is run from the experiments folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import time
from basis_pursuit.odl import ODL
from basis_pursuit.ksvd import KSVD
from sparse_coding.omp import OMP
from sparse_coding.irls import IRLS
from sparse_coding.fista import FISTA
from data.synthetic_data import SyntheticDataGenerator


def test_modular_odl():
    print("=" * 80)
    print("TESTING MODULAR ONLINE DICTIONARY LEARNING (ODL)")
    print("=" * 80)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:1")
        print(f"Using device: {device}")
    
    # Generate data
    print("\nGenerating synthetic data...")
    data_gen = SyntheticDataGenerator(
        n_features=64,
        n_components=128,
        n_samples=1000,  # Match K-SVD baseline
        sparsity=8,
        noise_std=0.05,
        random_state=42
    )
    print("=" * 80)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        print("CUDA not available. Running on CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:1")
        print(f"Using device: {device}")
    
    # Generate large dataset
    print("\nGenerating large synthetic data (20,000 samples)...")
    data_gen = SyntheticDataGenerator(
        n_features=64,
        n_components=128,
        n_samples=20000,
        sparsity=8,
        noise_std=0.05,
        random_state=42
    )
    
    Y_large, D_true, _ = data_gen.generate_torch(device=device)
    print(f"Y shape: {Y_large.shape}")
    
    # Test ODL
    print("\nTraining ODL...")
    odl = ODL(n_components=128, n_nonzero_coefs=8, batch_size=128, n_iter=10, verbose=False, device=str(device))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    odl.fit(Y_large)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_odl = time.time() - start
    
    print(f"ODL completed in {time_odl:.2f}s")
    print(f"Final RMSE: {odl.error_history_[-1]:.6f}")
    
    # Test K-SVD (only on subset for fairness)
    print("\nTraining K-SVD (on full dataset)...")
    ksvd = KSVD(n_components=128, n_nonzero_coefs=8, max_iter=10, verbose=False, device=str(device))
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    ksvd.fit(Y_large)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    time_ksvd = time.time() - start
    
    print(f"K-SVD completed in {time_ksvd:.2f}s")
    print(f"Final RMSE: {ksvd.error_history_[-1]:.6f}")
    
    print(f"\nSpeedup: ODL is {time_ksvd/time_odl:.2f}x faster than K-SVD")
    print(f"RMSE difference: {abs(odl.error_history_[-1] - ksvd.error_history_[-1]):.6f}")
    
    # Evaluate recovery
    from experiments.utils.dictionary_metrics import evaluate_recovery
    
    print("\n--- ODL Recovery ---")
    evaluate_recovery(D_true, odl.get_dictionary(), show_matching=False)
    
    print("\n--- K-SVD Recovery ---")
    evaluate_recovery(D_true, ksvd.get_dictionary(), show_matching=False)


if __name__ == "__main__":
    # Run main ODL tests
    test_modular_odl()
    
    # Run scalability test
    test_scalability()