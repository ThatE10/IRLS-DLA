# visualize_patches.py

import sys
import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================================
# CONFIGURATION (Mirroring the structure for compatibility)
# ============================================================================
# Note: In a real scenario, you might only need the checkpoint path and output config
CONFIG = {
    # Super-resolution settings
    'scale_factor': 4,
    'hr_patch_size': 32,
    'lr_patch_size': 8,
    
    # Device
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint and return dictionaries and configuration."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get dictionaries (ensure they are on CPU and converted to numpy)
    D_lr = checkpoint['D_lr'].cpu().numpy()
    D_hr = checkpoint['D_hr'].cpu().numpy()
    
    # Use config from checkpoint, but update device if necessary
    config = checkpoint['config']
    config['device'] = device
    
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"LR Dictionary shape: {D_lr.shape}")
    print(f"HR Dictionary shape: {D_hr.shape}")
    
    return D_lr, D_hr, config

def create_patch_grid(D_matrix, patch_size, n_cols=32, save_path=None):
    """
    Reshapes dictionary atoms (columns of D_matrix) into 2D patches and 
    arranges them in a grid visualization.
    
    Args:
        D_matrix (np.ndarray): Dictionary matrix (patch_dim x num_atoms)
        patch_size (int): Side length of the square patch (e.g., 8 or 32)
        n_cols (int): Number of patches per row in the visualization grid.
        save_path (str | None): Path to save the image.
    
    Returns:
        np.ndarray: The visualization image.
    """
    patch_dim, num_atoms = D_matrix.shape
    
    if patch_dim != patch_size * patch_size:
        raise ValueError(f"Patch dimension mismatch: {patch_dim} != {patch_size}x{patch_size}")
    
    # Normalize each atom independently to [0, 1] for visualization
    # This ensures high-contrast, visible features for all atoms
    D_norm = D_matrix.copy()
    for i in range(num_atoms):
        atom = D_norm[:, i]
        # Min-max normalization for visualization contrast
        atom_min = atom.min()
        atom_max = atom.max()
        if atom_max - atom_min > 1e-8:
            D_norm[:, i] = (atom - atom_min) / (atom_max - atom_min)
        else:
            D_norm[:, i] = 0.5 # Neutral gray if atom is constant
    
    # Reshape all atoms to (patch_size, patch_size, num_atoms)
    patches = D_norm.T.reshape(num_atoms, patch_size, patch_size)
    
    # Calculate grid dimensions
    n_rows = (num_atoms + n_cols - 1) // n_cols
    
    # Create the visualization canvas
    border_size = 1
    total_w = n_cols * patch_size + (n_cols + 1) * border_size
    total_h = n_rows * patch_size + (n_rows + 1) * border_size
    
    # Initialize with a gray background
    grid_img = np.ones((total_h, total_w), dtype=np.float32) * 0.8
    
    patch_idx = 0
    for i in range(n_rows):
        for j in range(n_cols):
            if patch_idx >= num_atoms:
                break
            
            # Calculate coordinates for the current patch
            start_row = i * patch_size + (i + 1) * border_size
            end_row = start_row + patch_size
            start_col = j * patch_size + (j + 1) * border_size
            end_col = start_col + patch_size
            
            # Place the patch
            grid_img[start_row:end_row, start_col:end_col] = patches[patch_idx, :, :]
            
            patch_idx += 1
    
    # Convert to 8-bit image (0-255)
    grid_img_8bit = (grid_img * 255).astype(np.uint8)
    
    if save_path:
        cv2.imwrite(str(save_path), grid_img_8bit)
        print(f"Saved visualization to {save_path}")
    
    return grid_img_8bit

def visualize_dictionaries(checkpoint_path, device, n_cols=32):
    """Load checkpoint and visualize both LR and HR dictionaries."""
    
    D_lr, D_hr, config = load_checkpoint(checkpoint_path, device)
    
    lr_size = config['lr_patch_size']
    hr_size = config['hr_patch_size']
    save_dir = Path('dictionary_visualizations')
    save_dir.mkdir(exist_ok=True)
    
    # --- Visualize LR Dictionary ---
    print("\nVisualizing LR dictionary...")
    lr_viz_path = save_dir / f'D_LR_{lr_size}x{lr_size}_{D_lr.shape[1]}atoms.png'
    create_patch_grid(D_lr, lr_size, n_cols=n_cols, save_path=lr_viz_path)
    
    # --- Visualize HR Dictionary ---
    print("Visualizing HR dictionary...")
    hr_viz_path = save_dir / f'D_HR_{hr_size}x{hr_size}_{D_hr.shape[1]}atoms.png'
    create_patch_grid(D_hr, hr_size, n_cols=n_cols, save_path=hr_viz_path)
    
    # Display using matplotlib (optional, for non-headless environments)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(18, 9))
        
        axes[0].imshow(cv2.cvtColor(cv2.imread(str(lr_viz_path)), cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"LR Dictionary ($D_{{LR}}$) - {lr_size}x{lr_size} Atoms")
        axes[0].axis('off')

        axes[1].imshow(cv2.cvtColor(cv2.imread(str(hr_viz_path)), cv2.COLOR_BGR2RGB))
        axes[1].set_title(f"HR Dictionary ($D_{{HR}}$) - {hr_size}x{hr_size} Atoms")
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Matplotlib display failed (perhaps running in a headless environment): {e}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Example usage: 
    # python visualize_patches.py <path_to_checkpoint.pt>
    
    

    # The checkpoint path is passed as the first argument
    CHECKPOINT_FILE = "C:/Users/enguye17/Desktop/Projects/IRLS-DLA/checkpoints/checkpoint_epoch_001.pt"
    
    # Determine the device to load the checkpoint onto
    device = CONFIG['device']
    
    # Number of columns for the visualization grid. Set to sqrt(K) for a square grid if possible.
    # The default 32 works well for 512 atoms (32x16).
    N_VIZ_COLS = 32 

    visualize_dictionaries(CHECKPOINT_FILE, device, N_VIZ_COLS)