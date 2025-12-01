import sys, os
# Ensure the project root is in sys.path for imports when script is run from the experiments folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
torch.set_default_dtype(torch.float64)
print(f"New default dtype: {torch.get_default_dtype()}")
import numpy as np
import time
import cv2
from pathlib import Path
from basis_pursuit.odl import ODL
from basis_pursuit.ksvd import KSVD
from sparse_coding.omp import OMP
from sparse_coding.irls import IRLS
from sparse_coding.fista import FISTA


# ============================================================================
# CONFIGURATION
# ============================================================================
CONFIG = {
    # Dataset
    'dataset_path': r'C:\Users\enguye17\Downloads\BSDS300-images\BSDS300\images\train',
    'num_training_images': 100,  # Number of images to use for training

    'regularization': 1e-5,  # Lambda for G_reg = G + λI
    'save_every_epoch': True,
    'use_cholesky': False,

    
    # Super-resolution settings
    'scale_factor': 4,  # 4x upscaling
    'hr_patch_size': 32,  # High-res patch size
    'lr_patch_size': 8,   # Low-res patch size (hr_patch_size / scale_factor)
    
    # Patch extraction
    'stride': 4,  # Stride for patch extraction (overlap = patch_size - stride)
    'max_patches_per_image': 1000,  # Limit patches per image
    
    # Dictionary learning
    'n_components': 512,  # Dictionary size (number of atoms)
    'sparsity': 5,  # Number of non-zero coefficients
    
    # ODL parameters
    'batch_size': 2048,
    'n_iter': 10,
    'odl_verbose': True,
    
    # Sparse coding method: 'omp', 'irls', or 'fista'
    'sparse_coding_method': 'irls',
    
    # Method-specific parameters
    'irls_max_iter': 20,
    'irls_tol': 1e-6,
    'fista_lambda': 0.1,
    'fista_max_iter': 50,
    
    # Device
    'device': 'cuda:0' if torch.cuda.is_available() else 'cpu',
    
    # Model saving
    'save_dir': 'checkpoints',
    'save_every_epoch': True,  # Save after each epoch/iteration
    'save_final_only': False,  # Only save final model
    
    # Testing
    'test_image_idx': 0,  # Which training image to use for testing
}


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_training_images(dataset_path, num_images=None):
    """Load training images from Berkeley dataset."""
    dataset_path = Path(dataset_path)
    image_files = sorted(list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png')))
    
    if num_images is not None:
        image_files = image_files[:num_images]
    
    images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            # Convert BGR to RGB and to grayscale for simplicity
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            images.append(img)
    
    print(f"Loaded {len(images)} images")
    return images


def extract_patches(image, patch_size, stride):
    """Extract overlapping patches from an image."""
    h, w = image.shape
    patches = []
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
    
    return np.array(patches)


def create_lr_hr_patch_pairs(images, config):
    """Create low-res and high-res patch pairs for training."""
    hr_patches = []
    lr_patches = []
    
    scale = config['scale_factor']
    hr_size = config['hr_patch_size']
    lr_size = config['lr_patch_size']
    stride = config['stride']
    max_patches = config['max_patches_per_image']
    
    print(f"\nExtracting patches (HR: {hr_size}x{hr_size}, LR: {lr_size}x{lr_size}, stride: {stride})...")
    
    for idx, img in enumerate(images):
        # Extract HR patches
        patches = extract_patches(img, hr_size, stride)
        
        # Limit number of patches per image
        if len(patches) > max_patches:
            indices = np.random.choice(len(patches), max_patches, replace=False)
            patches = patches[indices]
        
        hr_patches.append(patches)
        
        # Create corresponding LR patches by downsampling
        lr_img = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale), 
                           interpolation=cv2.INTER_CUBIC)
        lr_patch_array = extract_patches(lr_img, lr_size, stride//scale)
        
        # Match the number of patches
        if len(lr_patch_array) > len(patches):
            lr_patch_array = lr_patch_array[:len(patches)]
        
        lr_patches.append(lr_patch_array)
        
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(images)} images")
    
    # Concatenate all patches
    hr_patches = np.vstack(hr_patches).T  # Shape: (hr_size^2, n_patches)
    lr_patches = np.vstack(lr_patches).T  # Shape: (lr_size^2, n_patches)
    
    print(f"Extracted {hr_patches.shape[1]} patch pairs")
    print(f"HR patches shape: {hr_patches.shape}")
    print(f"LR patches shape: {lr_patches.shape}")
    
    # Normalize patches (zero mean, unit variance)
    hr_mean = np.mean(hr_patches, axis=0, keepdims=True)
    hr_std = np.std(hr_patches, axis=0, keepdims=True) + 1e-8
    hr_patches = (hr_patches - hr_mean) / hr_std
    
    lr_mean = np.mean(lr_patches, axis=0, keepdims=True)
    lr_std = np.std(lr_patches, axis=0, keepdims=True) + 1e-8
    lr_patches = (lr_patches - lr_mean) / lr_std
    
    return lr_patches, hr_patches, (lr_mean, lr_std, hr_mean, hr_std)


# ============================================================================
# MODEL SAVING AND LOADING
# ============================================================================

def save_checkpoint(odl_lr, D_hr, epoch, config, norm_params, save_dir):
    """Save model checkpoint."""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'D_lr': odl_lr.get_dictionary().cpu(),
        'D_hr': D_hr.cpu() if torch.is_tensor(D_hr) else torch.from_numpy(D_hr),
        'lr_mean': norm_params[0],
        'lr_std': norm_params[1],
        'hr_mean': norm_params[2],
        'hr_std': norm_params[3],
        'config': config,
        'error_history_lr': odl_lr.error_history_,
    }
    
    # Save epoch-specific checkpoint
    epoch_path = save_path / f'checkpoint_epoch_{epoch:03d}.pt'
    torch.save(checkpoint, epoch_path)
    print(f"Saved checkpoint: {epoch_path}")
    
    # Also save as latest
    latest_path = save_path / 'checkpoint_latest.pt'
    torch.save(checkpoint, latest_path)
    
    return epoch_path


def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return checkpoint


# ============================================================================
# DICTIONARY LEARNING
# ============================================================================

def get_sparse_coder(config):
    """Get sparse coding method based on configuration."""
    method = config['sparse_coding_method']
    
    if method == 'omp':
        return OMP(n_nonzero_coefs=config['sparsity'])
    elif method == 'irls':
        return IRLS(max_iter=config['irls_max_iter'], tol=config['irls_tol'])
    elif method == 'fista':
        return FISTA(lambda_reg=config['fista_lambda'], max_iter=config['fista_max_iter'])
    else:
        raise ValueError(f"Unknown sparse coding method: {method}")

def train_coupled_dictionaries(lr_patches, hr_patches, config):
    """
    Train coupled LR and HR dictionaries using proper least squares updates.
    
    Mathematical Formulation:
    -------------------------
    Minimize: ||Y_lr - D_lr·A||² + ||Y_hr - D_hr·A||² + λ||A||₀
    
    Where:
    - Y_lr: (d_lr × N) low-res patches
    - Y_hr: (d_hr × N) high-res patches  
    - D_lr: (d_lr × K) LR dictionary
    - D_hr: (d_hr × K) HR dictionary
    - A:    (K × N) sparse coefficient matrix
    - K: number of atoms, N: number of patches
    
    Update Rules:
    - D_lr: Online dictionary learning (K-SVD/MOD-like)
    - A: Sparse coding (OMP/LASSO/etc.) on LR patches
    - D_hr: Closed-form least squares: D_hr = Y_hr·A^T·(A·A^T + λI)^(-1)
    """
    device = torch.device(config['device'])
    print(f"\nUsing device: {device}")
    print(f"Sparse coding method: {config['sparse_coding_method'].upper()}")
    
    # Convert to torch tensors and transpose to (features × samples)
    lr_patches_torch = torch.from_numpy(lr_patches).float().to(device)
    hr_patches_torch = torch.from_numpy(hr_patches).float().to(device)
    
    d_lr, N = lr_patches_torch.shape
    d_hr = hr_patches_torch.shape[0]
    K = config['n_components']
    
    print(f"Patch dimensions: LR={d_lr}, HR={d_hr}, N={N}, K={K}")
    
    # Create save directory
    save_dir = Path(config['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("TRAINING COUPLED LR-HR DICTIONARIES")
    print("="*80)
    
    # ===================================================================
    # INITIALIZATION
    # ===================================================================
    
    # Initialize Online Dictionary Learning for LR
    odl_lr = ODL(
        n_components=K,
        n_nonzero_coefs=config['sparsity'],
        batch_size=config['batch_size'],
        n_iter=1,  # We'll loop manually
        verbose=config['odl_verbose'],
        device=str(device)
    )
    
    # Set custom sparse coder if not using default OMP
    if config['sparse_coding_method'] != 'omp':
        sparse_coder = get_sparse_coder(config)
        odl_lr.sparse_coding = sparse_coder.fit
    
    # Initialize D_lr using random selection of patches (normalized)
    indices = torch.randperm(N)[:K]
    D_lr_init = lr_patches_torch[:, indices].clone()
    D_lr_init = D_lr_init / (torch.norm(D_lr_init, dim=0, keepdim=True) + 1e-8)
    odl_lr.dictionary_ = D_lr_init
    
    # Initialize D_hr similarly
    D_hr = hr_patches_torch[:, indices].clone()
    D_hr = D_hr / (torch.norm(D_hr, dim=0, keepdim=True) + 1e-8)
    
    # Get regularization parameter
    lambda_reg = config.get('regularization', 1e-5)
    
    # ===================================================================
    # TRAINING LOOP
    # ===================================================================
    start = time.time()
    
    for epoch in range(config['n_iter']):
        print(f"\n--- Epoch {epoch + 1}/{config['n_iter']} ---")
        
        # -------------------------------------------------------------------
        # STEP 1: Update D_lr using online dictionary learning
        # -------------------------------------------------------------------
        odl_lr.n_iter = 1
        print("  Updating LR dictionary...")
        if epoch == 0:
            odl_lr.fit(lr_patches_torch)
        else:
            odl_lr.partial_fit(lr_patches_torch)
        
        odl_lr.n_batches_seen_ = (epoch + 1) * (N // config['batch_size'] + 1)
        
        # Get current LR dictionary (d_lr × K)
        print("  Updating LR dictionary...")
        D_lr = odl_lr.get_dictionary()
        
        # -------------------------------------------------------------------
        # STEP 2: Compute sparse codes A using D_lr
        # Solve: min_A ||Y_lr - D_lr·A||² + λ||A||₀
        # -------------------------------------------------------------------
        print("  Computing shared sparse codes...")
        if config['sparse_coding_method'] == 'omp':
            sparse_coder_temp = OMP(n_nonzero_coefs=config['sparsity'])
        else:
            sparse_coder_temp = get_sparse_coder(config)
        
        # Sparse coders typically return (N × K), but we need (K × N)
        A = sparse_coder_temp.fit(lr_patches_torch, D_lr)
        
        # Ensure A is a torch tensor on the correct device
        if not torch.is_tensor(A):
            A = torch.from_numpy(A).float().to(device)
        print("  Computing shared sparse codes...")
        # CRITICAL: Transpose if needed to get (K × N) shape
        if A.shape[0] == N and A.shape[1] == K:
            # Sparse coder returned (N × K), transpose to (K × N)
            A = A.t()
            print(f"  Transposed sparse codes from ({N} × {K}) to ({K} × {N})")
        elif A.shape[0] != K or A.shape[1] != N:
            raise ValueError(f"Unexpected sparse code shape: {A.shape}, expected ({K} × {N})")
        
        print(f"  Sparse code shape: {A.shape} (should be {K} × {N})")
        
        # -------------------------------------------------------------------
        # STEP 3: Update D_hr using least squares
        # Solve: D_hr* = argmin_D ||Y_hr - D·A||²
        # Solution: D_hr = Y_hr·A^T·(A·A^T + λI)^(-1)
        # -------------------------------------------------------------------
        print("  Updating HR dictionary via least squares...")
        
        # Compute Gram matrix: G = A·A^T (K × K)
        # This is the auto-correlation of sparse codes
        G = torch.mm(A, A.t())  # (K × N) @ (N × K) = (K × K)
        
        # Add regularization: G_reg = A·A^T + λI
        G_reg = G + lambda_reg * torch.eye(K, device=device, dtype=torch.float32)
        
        # Compute cross-correlation: C = Y_hr·A^T (d_hr × K)
        C = torch.mm(hr_patches_torch, A.t())  # (d_hr × N) @ (N × K) = (d_hr × K)
        
        # Solve the linear system for D_hr
        # We have: D_hr = C·G_reg^(-1)
        # Rearranging: G_reg·D_hr^T = C^T (since we solve column-wise)
        # Dimensions: (K×K)·(K×d_hr) = (K×d_hr)
        use_cholesky = config.get('use_cholesky', True)
        
        if use_cholesky:
            try:
                # Cholesky decomposition: G_reg = L·L^T (most stable for SPD matrices)
                # G_reg: (K × K), L: (K × K) lower triangular
                L = torch.linalg.cholesky(G_reg)
                
                # Solve L·X = C^T for X (forward substitution)
                # L: (K × K), C^T: (K × d_hr), X: (K × d_hr)
                X = torch.linalg.solve_triangular(L, C.t(), upper=False)
                
                # Solve L^T·D_hr^T = X for D_hr^T (backward substitution)
                # L^T: (K × K), X: (K × d_hr), D_hr^T: (K × d_hr)
                D_hr_T = torch.linalg.solve_triangular(L.t(), X, upper=True)
                
                # Transpose to get D_hr: (d_hr × K)
                D_hr = D_hr_T.t()
                
            except RuntimeError as e:
                print(f"  Warning: Cholesky failed ({e}), falling back to direct solve")
                use_cholesky = False
        
        if not use_cholesky:
            try:
                # Direct solve using LU decomposition
                # Solve: G_reg·D_hr^T = C^T
                D_hr_T = torch.linalg.solve(G_reg, C.t())
                D_hr = D_hr_T.t()
            except RuntimeError as e:
                print(f"  Warning: Direct solve failed ({e}), using pseudoinverse")
                # Pseudoinverse (most robust but slowest)
                # D_hr = C·G_reg^(-1)
                G_inv = torch.linalg.pinv(G_reg)
                D_hr = torch.mm(C, G_inv)
        
        # Normalize dictionary atoms to unit norm (prevents scaling ambiguity)
        D_hr = D_hr / (torch.norm(D_hr, dim=0, keepdim=True) + 1e-8)
        
        # Also ensure D_lr is normalized (for consistency)
        D_lr_normalized = D_lr / (torch.norm(D_lr, dim=0, keepdim=True) + 1e-8)
        odl_lr.dictionary_ = D_lr_normalized
        
        # -------------------------------------------------------------------
        # STEP 4: Compute reconstruction errors
        # -------------------------------------------------------------------
        # LR reconstruction error: ||Y_lr - D_lr·A||²
        lr_recon = torch.mm(D_lr_normalized, A)
        lr_rmse = torch.sqrt(torch.mean((lr_patches_torch - lr_recon) ** 2)).item()
        
        # HR reconstruction error: ||Y_hr - D_hr·A||²
        hr_recon = torch.mm(D_hr, A)
        hr_rmse = torch.sqrt(torch.mean((hr_patches_torch - hr_recon) ** 2)).item()
        
        # Sparsity statistics
        sparsity_ratio = (A != 0).sum().item() / A.numel()
        avg_nonzero = (A != 0).sum(dim=0).float().mean().item()
        
        print(f"  LR RMSE: {lr_rmse:.6f} | HR RMSE: {hr_rmse:.6f}")
        print(f"  Sparsity: {sparsity_ratio:.2%} | Avg non-zero: {avg_nonzero:.1f}/{K}")
        
        # -------------------------------------------------------------------
        # STEP 5: Save checkpoint
        # -------------------------------------------------------------------
        if config.get('save_every_epoch', False) or (epoch == config['n_iter'] - 1):
            lr_patches_cpu = lr_patches_torch.cpu()
            hr_patches_cpu = hr_patches_torch.cpu()
            
            # Compute normalization parameters (per-feature mean/std)
            norm_params_full = (
                torch.mean(lr_patches_cpu, dim=1, keepdim=True).numpy(),
                (torch.std(lr_patches_cpu, dim=1, keepdim=True) + 1e-8).numpy(),
                torch.mean(hr_patches_cpu, dim=1, keepdim=True).numpy(),
                (torch.std(hr_patches_cpu, dim=1, keepdim=True) + 1e-8).numpy()
            )
            save_checkpoint(odl_lr, D_hr, epoch + 1, config, norm_params_full, save_dir)
    
    # ===================================================================
    # TRAINING COMPLETE
    # ===================================================================
    time_total = time.time() - start
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETED in {time_total:.2f}s")
    print(f"{'='*80}")
    print(f"Final LR RMSE: {lr_rmse:.6f}")
    print(f"Final HR RMSE: {hr_rmse:.6f}")
    print(f"Dictionary size: {K} atoms")
    print(f"Sparsity level: {config['sparsity']} non-zero coefficients")
    print(f"LR dict shape: {D_lr.shape}, HR dict shape: {D_hr.shape}")
    
    # Return normalization parameters
    lr_patches_cpu = lr_patches_torch.cpu()
    hr_patches_cpu = hr_patches_torch.cpu()
    
    norm_params_full = (
        torch.mean(lr_patches_cpu, dim=1, keepdim=True).numpy(),
        (torch.std(lr_patches_cpu, dim=1, keepdim=True) + 1e-8).numpy(),
        torch.mean(hr_patches_cpu, dim=1, keepdim=True).numpy(),
        (torch.std(hr_patches_cpu, dim=1, keepdim=True) + 1e-8).numpy()
    )
    
    return odl_lr, D_hr, norm_params_full

# ============================================================================
# SUPER-RESOLUTION
# ============================================================================

def super_resolve_image(lr_image, odl_lr, D_hr, config, norm_params):
    """Super-resolve WITHOUT normalization (if dictionaries were trained wrong)."""
    device = torch.device(config['device'])
    scale = config['scale_factor']
    lr_size = config['lr_patch_size']
    hr_size = config['hr_patch_size']
    stride = lr_size // 2
    
    h, w = lr_image.shape
    lr_patches = extract_patches(lr_image, lr_size, stride).T
    lr_patches_torch = torch.from_numpy(lr_patches).float().to(device)
    
    # NO NORMALIZATION - work directly in pixel space
    if config['sparse_coding_method'] == 'omp':
        sparse_coder = OMP(n_nonzero_coefs=config['sparsity'])
    else:
        sparse_coder = get_sparse_coder(config)
    
    D_lr = odl_lr.get_dictionary()
    X_sparse = sparse_coder.fit(lr_patches_torch, D_lr)
    
    if not torch.is_tensor(X_sparse):
        X_sparse = torch.from_numpy(X_sparse).float().to(device)
    
    K = D_lr.shape[1]
    if X_sparse.shape[0] != K:
        X_sparse = X_sparse.t()
    
    # Reconstruct HR
    hr_patches = torch.mm(D_hr, X_sparse).cpu().numpy()
    
    # Assemble image
    hr_h, hr_w = h * scale, w * scale
    hr_image = np.zeros((hr_h, hr_w), dtype=np.float32)
    counts = np.zeros((hr_h, hr_w), dtype=np.float32)
    
    patch_idx = 0
    for i in range(0, h - lr_size + 1, stride):
        for j in range(0, w - lr_size + 1, stride):
            if patch_idx >= hr_patches.shape[1]:
                break
            hr_i, hr_j = i * scale, j * scale
            patch = hr_patches[:, patch_idx].reshape(hr_size, hr_size)
            hr_image[hr_i:hr_i+hr_size, hr_j:hr_j+hr_size] += patch
            counts[hr_i:hr_i+hr_size, hr_j:hr_j+hr_size] += 1
            patch_idx += 1
    
    hr_image = hr_image / (counts + 1e-8)
    hr_image = np.clip(hr_image, 0, 255).astype(np.uint8)
    
    return hr_image


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("IMAGE SUPER-RESOLUTION WITH DICTIONARY LEARNING")
    print("="*80)
    print(f"\nConfiguration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Load training images
    print("\n" + "="*80)
    print("LOADING TRAINING DATA")
    print("="*80)
    images = load_training_images(CONFIG['dataset_path'], CONFIG['num_training_images'])
    
    # Create LR-HR patch pairs
    lr_patches, hr_patches, norm_params = create_lr_hr_patch_pairs(images, CONFIG)
    
    # Train coupled dictionaries
    print("\n" + "="*80)
    print("TRAINING COUPLED DICTIONARIES")
    print("="*80)
    odl_lr, D_hr, norm_params_full = train_coupled_dictionaries(lr_patches, hr_patches, CONFIG)
    
    # Test super-resolution
    print("\n" + "="*80)
    print("TESTING SUPER-RESOLUTION")
    print("="*80)
    
    test_img = images[CONFIG['test_image_idx']]
    print(f"Original image shape: {test_img.shape}")
    
    # Create LR version
    lr_test = cv2.resize(test_img, 
                        (test_img.shape[1]//CONFIG['scale_factor'], 
                         test_img.shape[0]//CONFIG['scale_factor']),
                        interpolation=cv2.INTER_CUBIC)
    print(f"LR image shape: {lr_test.shape}")
    
    # Super-resolve
    start = time.time()
    sr_img = super_resolve_image(lr_test, odl_lr, D_hr, CONFIG, norm_params_full)
    sr_time = time.time() - start
    print(f"Super-resolution completed in {sr_time:.2f}s")
    print(f"SR image shape: {sr_img.shape}")
    
    # Bicubic baseline
    bicubic = cv2.resize(lr_test, (sr_img.shape[1], sr_img.shape[0]),
                        interpolation=cv2.INTER_CUBIC)
    
    # Compute metrics (on valid region)
    min_h = min(test_img.shape[0], sr_img.shape[0], bicubic.shape[0])
    min_w = min(test_img.shape[1], sr_img.shape[1], bicubic.shape[1])
    
    original_crop = test_img[:min_h, :min_w]
    sr_crop = sr_img[:min_h, :min_w]
    bicubic_crop = bicubic[:min_h, :min_w]
    
    psnr_sr = cv2.PSNR(original_crop, sr_crop)
    psnr_bicubic = cv2.PSNR(original_crop, bicubic_crop)
    
    print(f"\nPSNR (Dictionary Learning): {psnr_sr:.2f} dB")
    print(f"PSNR (Bicubic):            {psnr_bicubic:.2f} dB")
    print(f"Improvement:               {psnr_sr - psnr_bicubic:.2f} dB")
    
    # Save results
    output_dir = Path('superres_results')
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / 'original.png'), original_crop)
    cv2.imwrite(str(output_dir / 'lr_input.png'), lr_test)
    cv2.imwrite(str(output_dir / 'sr_output.png'), sr_crop)
    cv2.imwrite(str(output_dir / 'bicubic.png'), bicubic_crop)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Create comparison image
    comparison = np.hstack([
        cv2.resize(lr_test, (min_w, min_h), interpolation=cv2.INTER_NEAREST),
        bicubic_crop,
        sr_crop,
        original_crop
    ])
    cv2.imwrite(str(output_dir / 'comparison.png'), comparison)
    print(f"Comparison saved (LR | Bicubic | Dict Learning | Original)")
    
    print(f"\nAll checkpoints saved to {CONFIG['save_dir']}/")


def test_from_checkpoint(checkpoint_path, test_image_path):
    """Test super-resolution using a saved checkpoint."""
    print("="*80)
    print("TESTING FROM CHECKPOINT")
    print("="*80)
    
    # Load checkpoint
    device = torch.device(CONFIG['device'])
    checkpoint = load_checkpoint(checkpoint_path, device)
    
    # Reconstruct ODL object
    odl_lr = ODL(
        n_components=checkpoint['config']['n_components'],
        n_nonzero_coefs=checkpoint['config']['sparsity'],
        batch_size=checkpoint['config']['batch_size'],
        n_iter=1,
        verbose=False,
        device=str(device)
    )
    
    # Set the dictionary
    odl_lr.D_ = checkpoint['D_lr'].to(device)
    odl_lr.error_history_ = checkpoint['error_history_lr']
    
    # Get HR dictionary
    D_hr = checkpoint['D_hr'].to(device)
    
    # Get normalization params
    norm_params = (
        checkpoint['lr_mean'],
        checkpoint['lr_std'],
        checkpoint['hr_mean'],
        checkpoint['hr_std']
    )
    
    # Load and prepare test image
    test_img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Create LR version
    scale = checkpoint['config']['scale_factor']
    lr_test = cv2.resize(test_img, 
                        (test_img.shape[1]//scale, test_img.shape[0]//scale),
                        interpolation=cv2.INTER_CUBIC)
    
    # Super-resolve
    sr_img = super_resolve_image(lr_test, odl_lr, D_hr, checkpoint['config'], norm_params)
    
    # Save results
    output_dir = Path('superres_test_results')
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / 'sr_from_checkpoint.png'), sr_img)
    print(f"Result saved to {output_dir}/sr_from_checkpoint.png")


if __name__ == "__main__":
    main()