import torch
import time
from sparse_coding.omp import OMP
from sparse_coding.irls import IRLS
from sparse_coding.fista import FISTA
from data.synthetic_data import SyntheticDataGenerator

def run_experiment():
    # Parameters
    N = 3000   # Features
    K = 10000 # Components (Atoms)
    n_samples = 10 # Number of signals to process (keep small for speed)
    sparsity = 50
    
    print(f"Running GPU Experiment with N={N}, K={K}, Samples={n_samples}, Sparsity={sparsity}")
    
    if not torch.cuda.is_available():
        print("CUDA not available. Aborting GPU experiment.")
        return

    device = torch.device("cuda:1")
    
    print("Generating data on GPU...")
    # Use SyntheticDataGenerator
    data_gen = SyntheticDataGenerator(
        n_features=N,
        n_components=K,
        n_samples=n_samples,
        sparsity=sparsity,
        noise_std=0.2,
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
    torch.cuda.synchronize()
    start = time.time()
    # OMP expects Y as (n_samples, N)
    coefs_omp = omp.fit(D, Y.T)
    torch.cuda.synchronize()
    end = time.time()
    print(f"OMP Time: {end - start:.4f}s")
    
    # Reconstruction error
    Y_pred_omp = (D @ coefs_omp.T)
    error_omp = torch.norm(Y - Y_pred_omp) / torch.norm(Y)
    print(f"OMP Reconstruction Error: {error_omp.item():.4f}")
    
    # Run IRLS
    print("\n--- Running IRLS ---")
    irls = IRLS(max_iter=20, tol=1e-4) # Reduced iterations for speed
    torch.cuda.synchronize()
    start = time.time()
    coefs_irls = irls.fit(D, Y)
    torch.cuda.synchronize()
    end = time.time()
    print(f"IRLS Time: {end - start:.4f}s")
    
    Y_pred_irls = D @ coefs_irls
    error_irls = torch.norm(Y - Y_pred_irls) / torch.norm(Y)
    print(f"IRLS Reconstruction Error: {error_irls.item():.4f}")
    
    # Run FISTA
    print("\n--- Running FISTA ---")
    fista = FISTA(lambda_reg=0.1, max_iter=50)
    torch.cuda.synchronize()
    start = time.time()
    coefs_fista = fista.fit(D, Y)
    torch.cuda.synchronize()
    end = time.time()
    print(f"FISTA Time: {end - start:.4f}s")
    
    Y_pred_fista = D @ coefs_fista
    error_fista = torch.norm(Y - Y_pred_fista) / torch.norm(Y)
    print(f"FISTA Reconstruction Error: {error_fista.item():.4f}")
    
    # Calculate and display sparsity of solutions
    print("\n--- Sparsity Analysis ---")
    
    # OMP sparsity
    nonzero_omp = (coefs_omp.abs() > 1e-6).sum().item()
    total_elements_omp = coefs_omp.numel()
    sparsity_pct_omp = (nonzero_omp / total_elements_omp) * 100
    print(f"OMP: {nonzero_omp}/{total_elements_omp} non-zero elements ({sparsity_pct_omp:.2f}%)")
    
    # IRLS sparsity
    nonzero_irls = (coefs_irls.abs() > 1e-6).sum().item()
    total_elements_irls = coefs_irls.numel()
    sparsity_pct_irls = (nonzero_irls / total_elements_irls) * 100
    print(f"IRLS: {nonzero_irls}/{total_elements_irls} non-zero elements ({sparsity_pct_irls:.2f}%)")
    
    # FISTA sparsity
    nonzero_fista = (coefs_fista.abs() > 1e-6).sum().item()
    total_elements_fista = coefs_fista.numel()
    sparsity_pct_fista = (nonzero_fista / total_elements_fista) * 100
    print(f"FISTA: {nonzero_fista}/{total_elements_fista} non-zero elements ({sparsity_pct_fista:.2f}%)")
    
    # True sparsity for comparison
    nonzero_true = (X_true.abs() > 1e-6).sum().item()
    total_elements_true = X_true.numel()
    sparsity_pct_true = (nonzero_true / total_elements_true) * 100
    print(f"True X: {nonzero_true}/{total_elements_true} non-zero elements ({sparsity_pct_true:.2f}%)")

if __name__ == "__main__":
    run_experiment()

