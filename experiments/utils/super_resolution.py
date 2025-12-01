import sys, os
import torch
import numpy as np
import cv2
from pathlib import Path

# --- ASSUMED IMPORTS FROM YOUR PROJECT STRUCTURE ---
# Restoring custom sparse coding algorithms as requested
from basis_pursuit.odl import ODL
from sparse_coding.omp import OMP
from sparse_coding.irls import IRLS
from sparse_coding.fista import FISTA
# --------------------------------------------------

# Set default dtype for consistency with training
torch.set_default_dtype(torch.float64)


# ============================================================================
# HELPER FUNCTIONS (Minimal necessary for inference)
# ============================================================================

def print_stats(name, data):
    """Helper to print min/max/mean/std of a tensor or array."""
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()
    
    if data.size == 0:
        print(f"  [DEBUG] {name}: EMPTY")
        return

    print(f"  [DEBUG] {name}: Shape={data.shape} | "
          f"Min={data.min():.4f}, Max={data.max():.4f}, "
          f"Mean={data.mean():.4f}, Std={data.std():.4f}")

def get_sparse_coder(config):
    """Get sparse coding method based on configuration (using custom implementations)."""
    method = config['sparse_coding_method']
    
    if method == 'omp':
        return OMP(n_nonzero_coefs=config['sparsity'])
    elif method == 'irls':
        # Assuming these parameters exist in your config
        return IRLS(max_iter=config.get('irls_max_iter', 20), tol=config.get('irls_tol', 1e-4))
    elif method == 'fista':
        # Assuming these parameters exist in your config
        return FISTA(lambda_reg=config.get('fista_lambda', 0.1), max_iter=config.get('fista_max_iter', 100))
    else:
        raise ValueError(f"Unknown sparse coding method: {method}")

def load_checkpoint(checkpoint_path, device):
    """Load model checkpoint and return dictionaries, config, and normalization params."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    # Ensure the config dictionary includes the actual device used
    config['device'] = str(device) 

    # Get dictionaries (keep as Torch Tensors on the target device)
    D_lr = checkpoint['D_lr'].to(device)
    D_hr = checkpoint['D_hr'].to(device)
    
    # Get normalization params (np arrays)
    norm_params = (
        checkpoint['lr_mean'],
        checkpoint['lr_std'],
        checkpoint['hr_mean'],
        checkpoint['hr_std']
    )

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    print(f"LR Dictionary shape: {D_lr.shape}")
    print(f"HR Dictionary shape: {D_hr.shape}")
    
    # Debug normalization params
    print_stats("Loaded LR Mean", norm_params[0])
    print_stats("Loaded LR Std", norm_params[1])
    
    return D_lr, D_hr, config, norm_params

def extract_patches(image, patch_size, stride):
    """Extract overlapping patches from an image (used for LR input)."""
    h, w = image.shape
    patches = []
    
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            patch = image[i:i+patch_size, j:j+patch_size]
            patches.append(patch.flatten())
    
    # Return as (N_patches, d)
    return np.array(patches)

def reconstruct_image(patches_denormalized, lr_image_shape, hr_patch_size, lr_patch_size, scale_factor, hr_reconstruction_stride):
    """
    Reconstructs the final HR image from overlapping, denormalized HR patches 
    using the overlap-add (averaging) method.
    """
    print("\n  [DEBUG] Starting Image Reconstruction Loop...")

    # 1. Determine the size of the final HR canvas
    hr_h = lr_image_shape[0] * scale_factor
    hr_w = lr_image_shape[1] * scale_factor
    
    final_hr_image = np.zeros((hr_h, hr_w), dtype=np.float64)
    patch_counter = np.zeros((hr_h, hr_w), dtype=np.float64)
    
    # 2. Determine patch loop dimensions based on the LR image
    
    # Reshape patches for iteration: (N_patches, patch_size, patch_size)
    patches_reshaped = patches_denormalized.T.reshape(-1, hr_patch_size, hr_patch_size)
    print_stats("Patches Reshaped for Reconstruction", patches_reshaped)

    k = 0 # Patch index counter
    # Loop over the HR canvas using the correctly scaled stride
    for i in range(0, hr_h - hr_patch_size + 1, hr_reconstruction_stride):
        for j in range(0, hr_w - hr_patch_size + 1, hr_reconstruction_stride):
            if k >= patches_reshaped.shape[0]:
                break 

            patch = patches_reshaped[k]
            
            # Add patch contribution
            final_hr_image[i:i+hr_patch_size, j:j+hr_patch_size] += patch
            patch_counter[i:i+hr_patch_size, j:j+hr_patch_size] += 1.0
            
            k += 1

    print(f"  [DEBUG] Patches processed: {k}")
    print_stats("Accumulated Image (Pre-division)", final_hr_image)
    print_stats("Patch Counter", patch_counter)

    # 3. Normalize by the counter to average overlapping regions
    final_hr_image = np.divide(final_hr_image, patch_counter, 
                                out=np.zeros_like(final_hr_image), 
                                where=patch_counter!=0)
    
    print_stats("Final HR Image (Pre-clip)", final_hr_image)

    # 4. Clip and convert to uint8 (0-255)
    final_hr_image = np.clip(final_hr_image, 0, 255)
    
    return final_hr_image.astype(np.uint8)

# ============================================================================
# CORE SUPER-RESOLUTION FUNCTION
# ============================================================================

def super_resolve_image(lr_image, odl_lr, D_hr, config, norm_params):
    """
    Performs Super-Resolution on a single LR image using trained dictionaries.
    """
    device = D_hr.device
    print("\n--- Starting Super-Resolution ---")
    
    # --- Step 1: Prepare LR Patches ---
    lr_size = config['lr_patch_size']
    hr_size = config['hr_patch_size']
    scale_factor = config['scale_factor']
    
    # FIX: Enforcing non-overlapping patches (no blending/stitching) as requested.
    lr_extraction_stride = lr_size 
    hr_reconstruction_stride = hr_size
    
    print(f"  [DEBUG] FORCING NO BLENDING: LR Extraction Stride (S_LR): {lr_extraction_stride}")
    print(f"  [DEBUG] FORCING NO BLENDING: HR Reconstruction Stride (S_HR): {hr_reconstruction_stride}")

    # Extract patches from the LR image
    lr_patches_np = extract_patches(lr_image, lr_size, lr_extraction_stride).T # (d_lr × N)
    
    # Compute per-patch mean and std
    patch_mean = np.mean(lr_patches_np, axis=0, keepdims=True) # Shape (1, N)
    patch_std = np.std(lr_patches_np, axis=0, keepdims=True) 
    patch_std = np.clip(patch_std, 1e-3, None) # Clip small variance to avoid division by zero/massive scaling
    
    # Normalize LR patches
    lr_patches_norm = (lr_patches_np - patch_mean) / patch_std
    lr_patches_torch = torch.from_numpy(lr_patches_norm).float().to(device)
    print_stats("Normalized LR Patches", lr_patches_torch)
    
    # --- Step 2: Sparse Coding (Using Custom Implementation) ---
    print(f"  Sparse coding {lr_patches_torch.shape[1]} LR patches using custom {config['sparse_coding_method']}...")
    
    sparse_coder = get_sparse_coder(config)
    
    # Compute sparse codes A (K × N)
    # The custom sparse coder is expected to handle the input and return (K x N)
    A = sparse_coder.fit(lr_patches_torch, odl_lr.get_dictionary())
    
    # Ensure A is a torch tensor on the correct device and shape (K x N)
    if not torch.is_tensor(A):
        A = torch.from_numpy(A).float().to(device)
    if A.shape[0] == lr_patches_torch.shape[1] and A.shape[1] == config['n_components']:
        A = A.t() # Transpose if it came back as (N x K)
        
    print_stats("Sparse Codes A", A)
    sparsity = (A != 0).float().mean().item()
    print(f"  [DEBUG] Sparsity Level: {sparsity * 100:.2f}%")
    
    # --- Step 3: HR Reconstruction (Normalized) ---
    print("  Reconstructing normalized HR patches...")
    # Y_hr_recon_norm = D_hr · A
    hr_patches_recon_norm = torch.mm(D_hr, A) 
    
    # --- Step 4: De-normalization ---
    print("  De-normalizing and combining patches...")
    
    hr_patches_recon_denorm_np = hr_patches_recon_norm.cpu().numpy()
    
    # Apply LR patch statistics
    hr_patches_recon_denorm_np = hr_patches_recon_denorm_np * patch_std
    hr_patches_recon_denorm_np = hr_patches_recon_denorm_np + patch_mean
    
    # --- Step 5: Final Image Assembly ---
    hr_image = reconstruct_image(
        hr_patches_recon_denorm_np, 
        lr_image.shape, 
        hr_size, 
        lr_size,
        scale_factor,
        hr_reconstruction_stride # Pass the correct stride
    )
    
    return hr_image

# ============================================================================
# ENTRY POINT FOR IMAGE GENERATION
# ============================================================================

def generate_super_resolved_image(checkpoint_path, test_image_path):
    """
    Loads the model from a checkpoint and performs super-resolution.
    """
    
    # --- Setup ---
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    D_lr, D_hr, config, norm_params = load_checkpoint(checkpoint_path, device)
    
    odl_lr = ODL(
        n_components=config['n_components'],
        n_nonzero_coefs=config['sparsity'],
        batch_size=config['batch_size'],
        n_iter=1,
        verbose=False,
        device=str(device)
    )
    odl_lr.D_ = D_lr
    odl_lr.dictionary_ = D_lr
    
    # --- Data Preparation ---
    test_img = cv2.imread(str(test_image_path), cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        raise FileNotFoundError(f"Could not load test image from {test_image_path}")
    
    scale = config['scale_factor']
    lr_test = cv2.resize(test_img, 
                         (test_img.shape[1]//scale, test_img.shape[0]//scale),
                         interpolation=cv2.INTER_CUBIC)
    
    print(f"Original shape: {test_img.shape}, LR input shape: {lr_test.shape}")
    
    # --- Super-Resolution ---
    start = cv2.getTickCount()
    sr_img = super_resolve_image(lr_test, odl_lr, D_hr, config, norm_params)
    sr_time = (cv2.getTickCount() - start) / cv2.getTickFrequency()
    print(f"Super-resolution completed in {sr_time:.2f}s")
    
    # --- Post-processing ---
    bicubic = cv2.resize(lr_test, (sr_img.shape[1], sr_img.shape[0]),
                         interpolation=cv2.INTER_CUBIC)
    
    min_h = min(test_img.shape[0], sr_img.shape[0], bicubic.shape[0])
    min_w = min(test_img.shape[1], sr_img.shape[1], bicubic.shape[1])
    
    original_crop = test_img[:min_h, :min_w]
    sr_crop = sr_img[:min_h, :min_w]
    bicubic_crop = bicubic[:min_h, :min_w]
    
    psnr_sr = cv2.PSNR(original_crop, sr_crop)
    psnr_bicubic = cv2.PSNR(original_crop, bicubic_crop)
    
    print(f"\nPSNR (Dictionary Learning): {psnr_sr:.2f} dB")
    print(f"PSNR (Bicubic):             {psnr_bicubic:.2f} dB")
    print(f"Improvement:                {psnr_sr - psnr_bicubic:.2f} dB")
    
    return sr_crop, bicubic_crop, original_crop, config

def test_generation():
    """
    Example usage for the super-resolution function. 
    You MUST update these paths to match your system.
    """
    # NOTE: YOU MUST UPDATE THESE PATHS!
    CHECKPOINT_PATH = "C:/Users/enguye17/Desktop/Projects/IRLS-DLA/checkpoints/checkpoint_epoch_005.pt"
    # Use any image from your training set for validation
    TEST_IMAGE_PATH = r'C:\Users\enguye17\Downloads\BSDS300-images\BSDS300\images\train\100075.jpg' 

    if not Path(CHECKPOINT_PATH).exists() or not Path(TEST_IMAGE_PATH).exists():
        print("\n" + "="*80)
        print("ERROR: Checkpoint or Test Image Path Not Found!")
        print("Please edit the CHECKPOINT_PATH and TEST_IMAGE_PATH variables in test_generation()")
        print("to point to your actual files before running.")
        print("="*80)
        return

    print("="*80)
    print("STARTING IMAGE GENERATION TEST")
    print("="*80)
    
    sr_img, bicubic_img, original_img, config = generate_super_resolved_image(CHECKPOINT_PATH, TEST_IMAGE_PATH)
    
    # Save results
    output_dir = Path('sr_inference_results')
    output_dir.mkdir(exist_ok=True)
    
    cv2.imwrite(str(output_dir / 'sr_output.png'), sr_img)
    cv2.imwrite(str(output_dir / 'bicubic.png'), bicubic_img)
    cv2.imwrite(str(output_dir / 'original.png'), original_img)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Create comparison image
    comparison = np.hstack([
        cv2.resize(cv2.resize(original_img, 
                              (original_img.shape[1] // config['scale_factor'], original_img.shape[0] // config['scale_factor']),
                              interpolation=cv2.INTER_CUBIC), (original_img.shape[1], original_img.shape[0]),
                          interpolation=cv2.INTER_NEAREST), # LR (nearest neighbor for visibility)
        bicubic_img,
        sr_img,
        original_img
    ])
    cv2.imwrite(str(output_dir / 'comparison_final.png'), comparison)
    print(f"Comparison saved (LR | Bicubic | Dict Learning | Original)")

if __name__ == "__main__":
    test_generation()