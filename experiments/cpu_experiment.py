import torch
import time
from sparse_coding.omp import OMP
from sparse_coding.irls import IRLS
from sparse_coding.fista import FISTA
from data.synthetic_data import SyntheticDataGenerator
from data.data_generator2 import CompressedSensingGenerator
import numpy as np
def run_experiment():
    # Parameters - reduced for CPU
    N = 80    # Features (reduced from 3000)
    K = 200   # Components (reduced from 128000)
    n_samples = 1
    sparsity = 15
    
    print(f"Running CPU Experiment with N={N}, K={K}, Samples={n_samples}, Sparsity={sparsity}")
    print("Note: Parameters reduced for CPU execution. For full GPU experiment (N=3000, K=128000), use gpu_experiment.py on a CUDA-enabled system.\n")
    
    device = torch.device("cpu")
    
    print("Generating data...")
    data_gen = CompressedSensingGenerator(
        n_features=N,
        n_components=K,
        sparsity=sparsity,
        n_samples=n_samples,
        noise_std=1.0,
        random_state=42
    )

    
    Y, D, X_true = data_gen.generate_torch(device=device)
    
    print("Data generated.")
    print(f"Y shape: {Y.shape}")
    print(f"D shape: {D.shape}")
    print(f"X_true shape: {X_true.shape}")
    
    # Run OMP
    print("\n--- Running OMP ---")
    omp = OMP(n_nonzero_coefs=sparsity)
    start = time.time()
    coefs_omp = omp.fit(D, Y.T)
    end = time.time()
    print(f"OMP Time: {end - start:.4f}s")
    
    Y_pred_omp = (D @ coefs_omp.T)
    #check sparsity
    sparsity_omp = torch.count_nonzero(coefs_omp)
    error_omp = torch.norm(Y - Y_pred_omp) / torch.norm(Y)
    print(f"OMP Reconstruction Error: {error_omp.item():.4f}")
    print(f"OMP Sparsity: {sparsity_omp.item():.4f}")


   
    
    # Run IRLS algorithm
    print("\n" + "="*60)
    print("Running IRLS Algorithm")
    print("="*60)

    # Run IRLS
    print("\n--- Running IRLS ---")
    irls = IRLS(max_iter=20, tol=1e-4, n_nonzero_coefs=sparsity)
    start = time.time()
    coefs_irls = irls.fit(D, Y)
    end = time.time()
    print(f"IRLS Time: {end - start:.4f}s")
    
    Y_pred_irls = D @ coefs_irls
    #check sparsity
    sparsity_irls = torch.count_nonzero(coefs_irls)
    error_irls = torch.norm(Y - Y_pred_irls) / torch.norm(Y)
    print(f"IRLS Reconstruction Error: {error_irls.item():.4f}")
    print(f"IRLS Sparsity: {sparsity_irls.item():.4f}")
    
    # Plot results
    if hasattr(irls, 'history'):
        print("Plotting IRLS results...")
        from experiments.utils.plotting import plot_results
        plot_results(X_true, coefs_irls, irls.history, output_dir='plots')
    else:
        print("IRLS history not available.")
    
    # Run FISTA
    print("\n--- Running FISTA ---")
    fista = FISTA(lambda_reg=0.1, max_iter=50)
    start = time.time()
    coefs_fista = fista.fit(D, Y)
    end = time.time()
    print(f"FISTA Time: {end - start:.4f}s")
    
    Y_pred_fista = D @ coefs_fista
    #check sparsity
    sparsity_fista = torch.count_nonzero(coefs_fista)
    error_fista = torch.norm(Y - Y_pred_fista) / torch.norm(Y)
    print(f"FISTA Reconstruction Error: {error_fista.item():.4f}")
    print(f"FISTA Sparsity: {sparsity_fista.item():.4f}")
    
    print("\n=== Summary ===")
    print(f"OMP:   Time={end - start:.4f}s, Error={error_omp.item():.4f}")
    print(f"IRLS:  Time={end - start:.4f}s, Error={error_irls.item():.4f}")
    print(f"FISTA: Time={end - start:.4f}s, Error={error_fista.item():.4f}")

    # Plot Comparison
    print("\nPlotting comparison of all methods...")
    from experiments.utils.plotting import plot_comparison
    methods = {
        'OMP': coefs_omp,
        'IRLS': coefs_irls,
        'FISTA': coefs_fista
    }
    plot_comparison(X_true, methods, output_dir='plots')

if __name__ == "__main__":
    run_experiment()
