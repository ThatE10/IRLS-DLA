import torch
import numpy as np
import matplotlib.pyplot as plt
from basis_pursuit.ksvd import KSVD
from sparse_coding.omp import OMP
from sparse_coding.irls import IRLS
from sparse_coding.fista import FISTA
from data.synthetic_data import SyntheticDataGenerator

def compute_recovery_metrics(X_true, X_recovered):
    """
    Compute sparse code recovery metrics.
    
    Returns:
        support_recovery: Percentage of correctly identified support
        correlation: Average correlation between true and recovered codes
        rmse: Root mean squared error
    """
    # Support recovery (which atoms are used)
    support_true = (np.abs(X_true) > 1e-6)
    support_recovered = (np.abs(X_recovered) > 1e-6)
    
    # True positives, false positives, false negatives
    tp = np.sum(support_true & support_recovered)
    fp = np.sum(~support_true & support_recovered)
    fn = np.sum(support_true & ~support_recovered)
    
    # Precision and recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Correlation
    correlations = []
    for i in range(X_true.shape[1]):
        if np.linalg.norm(X_true[:, i]) > 1e-10 and np.linalg.norm(X_recovered[:, i]) > 1e-10:
            corr = np.abs(np.dot(X_true[:, i], X_recovered[:, i])) / \
                   (np.linalg.norm(X_true[:, i]) * np.linalg.norm(X_recovered[:, i]))
            correlations.append(corr)
    
    avg_correlation = np.mean(correlations) if correlations else 0
    
    # RMSE
    rmse = np.sqrt(np.mean((X_true - X_recovered) ** 2))
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'correlation': avg_correlation,
        'rmse': rmse,
        'avg_sparsity_true': np.mean(np.sum(support_true, axis=0)),
        'avg_sparsity_recovered': np.mean(np.sum(support_recovered, axis=0))
    }

def analyze_sparse_recovery():
    print("=" * 80)
    print("SPARSE RECOVERY ANALYSIS WITH MODULAR K-SVD")
    print("=" * 80)
    
    # Generate data
    print("\nGenerating synthetic data...")
    n_features = 64
    n_components = 128
    n_samples = 500
    sparsity = 8
    noise_std = 0.05
    
    data_gen = SyntheticDataGenerator(
        n_features=n_features,
        n_components=n_components,
        n_samples=n_samples,
        sparsity=sparsity,
        noise_std=noise_std,
        random_state=42
    )
    
    Y, D_true, X_true = data_gen.generate()
    
    print(f"Data shape: Y={Y.shape}, D_true={D_true.shape}, X_true={X_true.shape}")
    print(f"True sparsity: {sparsity}")
    print(f"Noise std: {noise_std}")
    
    # Test different K-SVD configurations
    methods = {
        'K-SVD + OMP': (OMP(n_nonzero_coefs=sparsity), 30),
        'K-SVD + IRLS': (IRLS(max_iter=20, tol=1e-4), 20),
        'K-SVD + FISTA': (FISTA(lambda_reg=0.05, max_iter=50), 30)
    }
    
    results = {}
    
    for method_name, (sparse_coder, max_iter) in methods.items():
        print("\n" + "=" * 80)
        print(f"Testing: {method_name}")
        print("=" * 80)
        
        # Train K-SVD
        ksvd = KSVD(n_components=n_components, n_nonzero_coefs=sparsity, 
                    max_iter=max_iter, verbose=True)
        ksvd.sparse_coding = sparse_coder.fit
        ksvd.fit(Y)
        
        # Get learned dictionary
        D_learned = ksvd.get_dictionary_numpy()
        
        # Compute sparse codes for all training data
        X_recovered = ksvd.transform(Y).cpu().numpy()
        
        # Compute recovery metrics
        metrics = compute_recovery_metrics(X_true, X_recovered)
        
        # Compute dictionary recovery
        dict_recovery = compute_dictionary_recovery(D_learned, D_true)
        
        results[method_name] = {
            'metrics': metrics,
            'dict_recovery': dict_recovery,
            'final_rmse': ksvd.error_history_[-1],
            'D_learned': D_learned,
            'X_recovered': X_recovered
        }
        
        print(f"\nRecovery Metrics:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  Code RMSE: {metrics['rmse']:.6f}")
        print(f"  Avg Sparsity (True): {metrics['avg_sparsity_true']:.2f}")
        print(f"  Avg Sparsity (Recovered): {metrics['avg_sparsity_recovered']:.2f}")
        print(f"  Dictionary Recovery: {dict_recovery:.4f}")
        print(f"  Reconstruction RMSE: {ksvd.error_history_[-1]:.6f}")
    
    # Summary comparison
    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON")
    print("=" * 80)
    print(f"\n{'Method':<20} {'F1-Score':<12} {'Correlation':<12} {'Dict Recovery':<15} {'Recon RMSE':<12}")
    print(f"{'-'*20} {'-'*12} {'-'*12} {'-'*15} {'-'*12}")
    
    for method_name, result in results.items():
        m = result['metrics']
        print(f"{method_name:<20} {m['f1_score']:<12.4f} {m['correlation']:<12.4f} "
              f"{result['dict_recovery']:<15.4f} {result['final_rmse']:<12.6f}")
    
    # Visualize sparsity patterns for first few samples
    print("\n" + "=" * 80)
    print("VISUALIZING SPARSITY PATTERNS")
    print("=" * 80)
    
    n_samples_to_plot = 5
    fig, axes = plt.subplots(len(methods) + 1, n_samples_to_plot, 
                             figsize=(15, 3 * (len(methods) + 1)))
    
    # Plot true sparse codes
    for i in range(n_samples_to_plot):
        axes[0, i].stem(X_true[:, i], basefmt=' ')
        axes[0, i].set_title(f'True Code {i+1}')
        axes[0, i].set_ylim([-3, 3])
        if i == 0:
            axes[0, i].set_ylabel('True')
    
    # Plot recovered codes for each method
    for method_idx, (method_name, result) in enumerate(results.items(), 1):
        X_rec = result['X_recovered']
        for i in range(n_samples_to_plot):
            axes[method_idx, i].stem(X_rec[:, i], basefmt=' ')
            axes[method_idx, i].set_title(f'{method_name.split("+")[1].strip()} {i+1}')
            axes[method_idx, i].set_ylim([-3, 3])
            if i == 0:
                axes[method_idx, i].set_ylabel(method_name.split('+')[1].strip())
    
    plt.tight_layout()
    plt.savefig('sparse_recovery_patterns.png', dpi=150, bbox_inches='tight')
    print(f"\nSparsity pattern visualization saved to: sparse_recovery_patterns.png")
    
    return results

def compute_dictionary_recovery(D_learned, D_true):
    """Compute dictionary recovery metric."""
    n_atoms = D_learned.shape[1]
    correlations = []
    
    for i in range(n_atoms):
        corr = np.abs(D_true.T @ D_learned[:, i])
        correlations.append(np.max(corr))
    
    return np.mean(correlations)

if __name__ == "__main__":
    results = analyze_sparse_recovery()
