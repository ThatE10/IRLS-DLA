import torch
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_results(x_true, x_recovered, history, output_dir='plots'):
    """Plot the true signal, recovered signal, and convergence history."""
    # Ensure inputs are on CPU and detached
    if isinstance(x_true, torch.Tensor):
        x_true = x_true.detach().cpu()
    if isinstance(x_recovered, torch.Tensor):
        x_recovered = x_recovered.detach().cpu()
        
    # Handle batch dimension if present (take first sample)
    if x_true.dim() > 1:
        x_true = x_true.squeeze()
        if x_true.dim() > 1:
            x_true = x_true[0]
            
    if x_recovered.dim() > 1:
        x_recovered = x_recovered.squeeze()
        if x_recovered.dim() > 1:
            x_recovered = x_recovered[0]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Plot 1: True signal
    ax = axes[0, 0]
    ax.stem(x_true.numpy(), basefmt=' ', linefmt='C0-', markerfmt='C0o')
    ax.set_title('True Sparse Signal', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    # Handle case where signal is all zeros or constant
    if x_true.min().item() != x_true.max().item():
        ax.set_ylim([1.1*x_true.min().item(), 1.1*x_true.max().item()])
    
    # Plot 2: Recovered signal
    ax = axes[0, 1]
    ax.stem(x_recovered.numpy(), basefmt=' ', linefmt='C1-', markerfmt='C1o')
    ax.set_title('Recovered Signal (IRLS)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Comparison (overlay)
    ax = axes[0, 2]
    indices = np.arange(len(x_true))
    markerline, stemlines, baseline = ax.stem(indices, x_true.numpy(), 
                                               basefmt=' ', linefmt='C0-', markerfmt='C0o', 
                                               label='True')
    stemlines.set_alpha(0.5)
    markerline, stemlines, baseline = ax.stem(indices + 0.3, x_recovered.numpy(), 
                                               basefmt=' ', linefmt='C1-', markerfmt='C1o',
                                               label='Recovered')
    stemlines.set_alpha(0.5)
    ax.set_title('Overlay: True vs Recovered', fontsize=12, fontweight='bold')
    ax.set_xlabel('Index')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Residual convergence
    ax = axes[1, 0]
    if 'residuals' in history and len(history['residuals']) > 0:
        ax.semilogy(history['residuals'])
        ax.set_title('Residual Convergence', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('||Φx - y||₂')
        ax.grid(True, alpha=0.3)
    
    # Plot 5: Objective function
    ax = axes[1, 1]
    if 'objective' in history and len(history['objective']) > 0:
        ax.plot(history['objective'])
        ax.set_title('Objective Function (Weighted p-norm)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Objective')
        ax.grid(True, alpha=0.3)
    
    # Plot 6: Recovery error
    ax = axes[1, 2]
    if 'x_values' in history and len(history['x_values']) > 0:
        # Calculate errors for each step
        errors = []
        for x in history['x_values']:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu()
                if x.dim() > 1: x = x.squeeze()
                if x.dim() > 1: x = x[0]
            errors.append(torch.norm(x - x_true).item())
            
        ax.semilogy(errors)
        ax.set_title('Recovery Error', fontsize=12, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('||x - x_true||₂')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'irls_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved as '{save_path}'")
    
    # Print final statistics
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"True sparsity:      {torch.sum(x_true != 0).item()}")
    if 'sparsity' in history and len(history['sparsity']) > 0:
        print(f"Recovered sparsity: {history['sparsity'][-1]}")
    
    rec_error = torch.norm(x_recovered - x_true).item()
    print(f"Recovery error:     {rec_error:.6e}")
    
    true_norm = torch.norm(x_true).item()
    if true_norm > 0:
        print(f"Relative error:     {(rec_error / true_norm):.6e}")
    
    if 'residuals' in history and len(history['residuals']) > 0:
        print(f"Final residual:     {history['residuals'][-1]:.6e}")
    
    # Check support recovery
    true_supp = torch.abs(x_true) > 1e-10
    rec_supp = torch.abs(x_recovered) > 1e-3 * torch.max(torch.abs(x_recovered))
    support_overlap = torch.sum(true_supp & rec_supp).item()
    print(f"Support overlap:    {support_overlap}/{torch.sum(true_supp).item()}")

def plot_comparison(x_true, methods_dict, output_dir='plots'):
    """
    Plot comparison of multiple sparse recovery methods.
    
    Args:
        x_true: True sparse signal (torch.Tensor)
        methods_dict: Dictionary mapping method names to recovered signals (torch.Tensor)
                      e.g., {'OMP': x_omp, 'IRLS': x_irls, 'FISTA': x_fista}
        output_dir: Directory to save the plot
    """
    # Ensure x_true is on CPU and detached
    if isinstance(x_true, torch.Tensor):
        x_true = x_true.detach().cpu()
    if x_true.dim() > 1:
        x_true = x_true.squeeze()
        if x_true.dim() > 1: x_true = x_true[0]
        
    n_methods = len(methods_dict)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save true signal separately
    fig_true = plt.figure(figsize=(10, 4))
    plt.stem(x_true.numpy(), basefmt=' ', linefmt='C0-', markerfmt='C0o')
    plt.title('True Sparse Signal', fontsize=14, fontweight='bold')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    if x_true.min().item() != x_true.max().item():
        plt.ylim([1.1*x_true.min().item(), 1.1*x_true.max().item()])
    plt.tight_layout()
    true_path = os.path.join(output_dir, 'true_signal.png')
    plt.savefig(true_path, dpi=150, bbox_inches='tight')
    plt.close(fig_true)
    print(f"Saved: {true_path}")
    
    indices = np.arange(len(x_true))
    
    # Save each method's plots separately
    for name, x_rec in methods_dict.items():
        # Ensure x_rec is on CPU and detached
        if isinstance(x_rec, torch.Tensor):
            x_rec = x_rec.detach().cpu()
        if x_rec.dim() > 1:
            x_rec = x_rec.squeeze()
            if x_rec.dim() > 1: x_rec = x_rec[0]
        
        # Save recovered signal
        fig_rec = plt.figure(figsize=(10, 4))
        plt.stem(x_rec.numpy(), basefmt=' ', linefmt='C1-', markerfmt='C1o')
        plt.title(f'{name}: Recovered Signal', fontsize=12, fontweight='bold')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        rec_path = os.path.join(output_dir, f'{name.lower()}_recovered.png')
        plt.savefig(rec_path, dpi=150, bbox_inches='tight')
        plt.close(fig_rec)
        print(f"Saved: {rec_path}")
        
        # Save overlay
        fig_over = plt.figure(figsize=(10, 4))
        markerline, stemlines, baseline = plt.stem(indices, x_true.numpy(), 
                                                   basefmt=' ', linefmt='C0-', markerfmt='C0o', 
                                                   label='True')
        stemlines.set_alpha(0.5)
        markerline, stemlines, baseline = plt.stem(indices + 0.3, x_rec.numpy(), 
                                                   basefmt=' ', linefmt='C1-', markerfmt='C1o',
                                                   label=f'{name}')
        stemlines.set_alpha(0.5)
        plt.title(f'{name}: Overlay vs True', fontsize=12, fontweight='bold')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        over_path = os.path.join(output_dir, f'{name.lower()}_overlay.png')
        plt.savefig(over_path, dpi=150, bbox_inches='tight')
        plt.close(fig_over)
        print(f"Saved: {over_path}")
    
    # Also create the combined plot
    fig = plt.figure(figsize=(16, 4 * (n_methods + 1)))
    gs = fig.add_gridspec(n_methods + 1, 2)
    
    # Plot True Signal
    ax_true = fig.add_subplot(gs[0, :])
    ax_true.stem(x_true.numpy(), basefmt=' ', linefmt='C0-', markerfmt='C0o')
    ax_true.set_title('True Sparse Signal', fontsize=14, fontweight='bold')
    ax_true.set_xlabel('Index')
    ax_true.set_ylabel('Value')
    ax_true.grid(True, alpha=0.3)
    if x_true.min().item() != x_true.max().item():
        ax_true.set_ylim([1.1*x_true.min().item(), 1.1*x_true.max().item()])
        
    for i, (name, x_rec) in enumerate(methods_dict.items()):
        # Ensure x_rec is on CPU and detached
        if isinstance(x_rec, torch.Tensor):
            x_rec = x_rec.detach().cpu()
        if x_rec.dim() > 1:
            x_rec = x_rec.squeeze()
            if x_rec.dim() > 1: x_rec = x_rec[0]
            
        row = i + 1
        
        # Plot Recovered Signal
        ax_rec = fig.add_subplot(gs[row, 0])
        ax_rec.stem(x_rec.numpy(), basefmt=' ', linefmt='C1-', markerfmt='C1o')
        ax_rec.set_title(f'{name}: Recovered Signal', fontsize=12, fontweight='bold')
        ax_rec.set_xlabel('Index')
        ax_rec.set_ylabel('Value')
        ax_rec.grid(True, alpha=0.3)
        
        # Plot Overlay
        ax_over = fig.add_subplot(gs[row, 1])
        markerline, stemlines, baseline = ax_over.stem(indices, x_true.numpy(), 
                                                   basefmt=' ', linefmt='C0-', markerfmt='C0o', 
                                                   label='True')
        stemlines.set_alpha(0.5)
        markerline, stemlines, baseline = ax_over.stem(indices + 0.3, x_rec.numpy(), 
                                                   basefmt=' ', linefmt='C1-', markerfmt='C1o',
                                                   label=f'{name}')
        stemlines.set_alpha(0.5)
        ax_over.set_title(f'{name}: Overlay vs True', fontsize=12, fontweight='bold')
        ax_over.set_xlabel('Index')
        ax_over.set_ylabel('Value')
        ax_over.legend()
        ax_over.grid(True, alpha=0.3)
        
    plt.tight_layout()
    save_path = os.path.join(output_dir, 'comparison_results.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {save_path}")
