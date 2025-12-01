import sys, os
import torch
import numpy as np
import time
import cv2
import random
from pathlib import Path
from skimage.color import rgb2gray
from skimage.util import view_as_windows
from typing import Tuple, Optional, List

# --- NOTE: Placeholder/Mock Imports for the ODL Framework ---
# WARNING: This script depends on custom modules (basis_pursuit.odl, sparse_coding.omp)
# that must be available in your project structure or Python path. 
# If they are missing, the Dummy classes below will be used, and training will be skipped.
try:
    # Attempt to use the user's provided project structure
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from basis_pursuit.odl import ODL
    from sparse_coding.omp import OMP
    from sparse_coding.irls import IRLS
    from sparse_coding.fista import FISTA
    # Dummy imports for utility functions not critical for the main logic
    from experiments.utils.dictionary_metrics import evaluate_recovery 
    REAL_FRAMEWORK_LOADED = True
except ImportError:
    print("\n--- WARNING: Missing custom ODL/OMP framework modules. Using dummy classes. ---")
    print("Training/reconstruction will be simulated with placeholder functions.")
    # Define minimal dummy classes for demonstration purposes
    class DummyODL:
        def __init__(self, n_components, **kwargs):
            self.n_components = n_components
            self.sparse_coding = None
            self.error_history_ = [0.1]
            self.D_ = torch.randn(64, n_components) # Example dictionary
        def fit(self, Y): 
            print("Dummy ODL.fit called. Dictionary not trained.")
        def reconstruct(self, Y_test):
            # Mock reconstruction: simply return the test data
            return Y_test 
        def get_dictionary(self): return self.D_
        
    class DummyOMP:
        def __init__(self, n_nonzero_coefs, **kwargs): pass
        def fit(self, Y, D): return torch.randn(D.shape[1], Y.shape[1])
        
    ODL = DummyODL
    OMP = DummyOMP
    def evaluate_recovery(*args, **kwargs): pass
    REAL_FRAMEWORK_LOADED = False
    # -----------------------------------------------------------------


# ====================================================================
## 1. Image Data Handling Utilities
# ====================================================================

def load_and_resize_images(dataset_path: str, num_images: int = 10, target_size: Tuple[int, int] = (256, 256)) -> List[np.ndarray]:
    """Load random images from dataset and resize them to consistent size."""
    
    dataset_dir = Path(dataset_path)
    
    if not dataset_dir.exists():
        raise ValueError(f"Dataset path does not exist: {dataset_path}")
    
    # Get all image files
    image_files = list(dataset_dir.glob("*.jpg")) + list(dataset_dir.glob("*.png")) + list(dataset_dir.glob("*.jpeg"))
    
    if len(image_files) < num_images:
        raise ValueError(f"Not enough images in dataset. Found {len(image_files)}, need {num_images}")
    
    # Randomly select images
    selected_files = random.sample(image_files, num_images)
    
    images = []
    for img_path in selected_files:
        # Load in grayscale
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Warning: Could not load {img_path}, skipping...")
            continue
        
        # Resize to target size
        img_resized = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
        
        # Normalize to [0, 1]
        img_normalized = img_resized.astype(np.float32) / 255.0
        images.append(img_normalized)
    
    print(f"Loaded {len(images)} images from {dataset_path}")
    return images


def image_to_patches(image: np.ndarray, patch_size: int, stride: int) -> Tuple[torch.Tensor, Tuple[int, int]]:
    """Converts a 2D image array into a matrix of patches."""
    
    # Check if the image has color channels and convert to grayscale if needed
    if image.ndim == 3:
        image = rgb2gray(image)
    
    # Ensure image is float and normalized (0.0 to 1.0)
    image = image.astype(np.float32)

    # Extract overlapping patches
    patches_blocks = view_as_windows(
        image, 
        (patch_size, patch_size), 
        step=stride
    )
    
    # Store the dimensions for later reassembly
    dims = patches_blocks.shape[:2]

    # Flatten patches: (N_patches, patch_size * patch_size)
    patches = patches_blocks.reshape(-1, patch_size * patch_size)
    
    # Transpose to (D x N) for Sparse Coding (D=features, N=samples)
    patches_tensor = torch.tensor(patches.T, dtype=torch.float32)
    
    return patches_tensor, dims


def add_simulated_text_noise(image: np.ndarray, num_words: int = 15) -> np.ndarray:
    """Adds multiple synthetic text words scattered across an image using OpenCV (cv2)."""
    
    # Input image is expected to be normalized float32 (0.0-1.0) grayscale.
    
    # Convert image to 8-bit integer (0-255) for OpenCV, handling grayscale or color
    if image.ndim == 2:
        img_uint8 = (image * 255).astype(np.uint8)
        # Convert to 3-channel BGR for consistent text color, then back to 1-channel later
        img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    else:
        # Assuming normalized RGB, convert to BGR uint8
        img_bgr = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    
    # List of words to randomly place
    words = ["NOISE", "TEXT", "REMOVED", "CLEAN", "IMAGE", "PATCH", "SPARSE", "DICT", "LEARN", "CODE"]
    
    # Define text parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.1
    color = (255, 255, 255)  # White color in BGR
    thickness = 1
    
    height, width = img_bgr.shape[:2]
    
    # Add multiple words at random positions
    for _ in range(num_words):
        text = random.choice(words)
        # Random position, avoiding edges
        x = random.randint(10, max(10, width - 100))
        y = random.randint(30, max(30, height - 10))
        org = (x, y)
        
        # Apply the text
        cv2.putText(img_bgr, text, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
    # Convert back to original format (grayscale float 0-1)
    if image.ndim == 2:
        img_gray_result = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return img_gray_result.astype(np.float32) / 255.0
    else:
        # If input was color, return color (normalized)
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def reconstruct_image_from_patches(Y_recon: torch.Tensor, original_shape: Tuple[int, int], 
                                   patch_dims: Tuple[int, int], patch_size: int, stride: int) -> np.ndarray:
    """Reassembles the image from reconstructed patches using averaging (overlap-add method)."""
    
    # Convert reconstructed patches back to numpy array, shaped (N_samples, D)
    Y_recon_np = Y_recon.cpu().numpy().T 
    # Reshape to (N_rows_of_patches, N_cols_of_patches, patch_size, patch_size)
    Y_recon_patches = Y_recon_np.reshape(*patch_dims, patch_size, patch_size)
    
    # Initialize output image and counter array
    reconstructed_image = np.zeros(original_shape, dtype=np.float32)
    patch_counter = np.zeros(original_shape, dtype=np.float32)
    
    for row in range(patch_dims[0]):
        for col in range(patch_dims[1]):
            # Determine image coordinates for the current patch
            r_start, c_start = row * stride, col * stride
            r_end, c_end = r_start + patch_size, c_start + patch_size
            
            # Add patch contribution and increment counter
            reconstructed_image[r_start:r_end, c_start:c_end] += Y_recon_patches[row, col]
            patch_counter[r_start:r_end, c_start:c_end] += 1.0
            
    # Normalize by the counter to average overlapping regions
    # This ensures areas covered by multiple patches are correctly weighted.
    reconstructed_image = np.divide(reconstructed_image, patch_counter, 
                                    out=np.zeros_like(reconstructed_image), 
                                    where=patch_counter!=0)
                                    
    return reconstructed_image


# ====================================================================
## 2. Main Sparse Coding Training and Denoising
# ====================================================================

def run_image_denoising_experiment(dataset_path: str):
    """
    Main function to train ODL/OMP on clean patches from multiple images and denoise a corrupted test image.
    """
    print("=" * 80)
    print("SPARSE CODING FOR IMAGE TEXT REMOVAL (ODL + OMP)")
    print("=" * 80)
    
    # --- Setup ---
    PATCH_SIZE = 16
    STRIDE = 4 # Overlapping patches
    N_COMPONENTS = 2048 # Dictionary size
    SPARSITY_K = 4    # K in OMP (n_nonzero_coefs)
    ODL_ITER = 10
    BATCH_SIZE = 128
    TARGET_SIZE = (256, 256)  # Resize all images to this size
    NUM_TRAIN_IMAGES = 100
    NUM_TEXT_WORDS = 15
    SPARSE_RECOVERY_ALGO = "omp" # "irls" or "omp" or "fista"
    
    # Check GPU availability
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # --- Load and Prepare Data ---
    print(f"\nLoading {NUM_TRAIN_IMAGES} training images from: {dataset_path}")
    
    try:
        # Load training images (these will be kept clean)
        train_images = load_and_resize_images(dataset_path, NUM_TRAIN_IMAGES + 1, TARGET_SIZE)
        
        if len(train_images) < NUM_TRAIN_IMAGES + 1:
            print(f"ERROR: Could not load enough images. Only got {len(train_images)}")
            return
        
        # Split: first NUM_TRAIN_IMAGES for training, last one for testing
        training_set = train_images[:NUM_TRAIN_IMAGES]
        test_image = train_images[NUM_TRAIN_IMAGES]
        
        print(f"Training on {len(training_set)} images, testing on 1 image")
        print(f"All images resized to: {TARGET_SIZE}")
        
    except Exception as e:
        print(f"\nERROR loading images: {e}")
        return
    
    # 1. Training Data (Y_train): Patches from MULTIPLE CLEAN images
    print("\nExtracting CLEAN training patches from all training images...")
    all_train_patches = []
    
    for idx, img in enumerate(training_set):
        patches, _ = image_to_patches(img, PATCH_SIZE, STRIDE)
        all_train_patches.append(patches)
        print(f"  Image {idx+1}: {patches.shape[1]} patches extracted")
    
    # Concatenate all patches into one large training set
    Y_train_clean = torch.cat(all_train_patches, dim=1).to(device)
    n_features, n_samples = Y_train_clean.shape
    print(f"\nTotal training patches: {n_samples} samples of size {n_features}")

    # 2. Test Data (Y_test): Patches from a TEXT-CORRUPTED image
    print(f"\nGenerating TEXT-CORRUPTED test image with {NUM_TEXT_WORDS} words...")
    img_corrupted = add_simulated_text_noise(test_image, num_words=NUM_TEXT_WORDS)
    Y_test_corrupted, patch_dims = image_to_patches(img_corrupted, PATCH_SIZE, STRIDE)
    Y_test_corrupted = Y_test_corrupted.to(device)
    
    # Save the original test image and corrupted version for comparison
    cv2.imwrite("test_original.png", (test_image * 255).astype(np.uint8))
    cv2.imwrite("test_corrupted_input.png", (img_corrupted * 255).astype(np.uint8))
    print("Saved 'test_original.png' and 'test_corrupted_input.png'")


    # --- Dictionary Training (ODL + OMP) ---
    print("\n" + "=" * 80)
    print(f"STARTING ODL TRAINING (D: {N_COMPONENTS} atoms, OMP K: {SPARSITY_K})")
    print("=" * 80)
    
    odl_model = ODL(
        n_components=N_COMPONENTS, 
        batch_size=BATCH_SIZE,
        n_iter=ODL_ITER,
        verbose=True,
        device=str(device)
    )
    if SPARSE_RECOVERY_ALGO == "omp":
        omp_solver = OMP(n_nonzero_coefs=SPARSITY_K)
        odl_model.sparse_coding = omp_solver.fit
    elif SPARSE_RECOVERY_ALGO == "irls":
        irls_solver = IRLS(max_iter=200, tol=1e-6)
        odl_model.sparse_coding = irls_solver.fit
    elif SPARSE_RECOVERY_ALGO == "fista":
        fista_solver = FISTA()
        odl_model.sparse_coding = fista_solver.fit
    
    start = time.time()
    # Train on the clean patches from multiple images!
    odl_model.fit(Y_train_clean) 
    time_train = time.time() - start
    
    print(f"\nODL + OMP Training completed in {time_train:.2f}s")
    if odl_model.error_history_:
         print(f"Final Training RMSE: {odl_model.error_history_[-1]:.6f}")

    # --- Text Removal / Reconstruction ---
    print("\n" + "=" * 80)
    print("PERFORMING TEXT REMOVAL RECONSTRUCTION")
    print("=" * 80)
    
    # The reconstruction uses the learned dictionary D to sparsely code the 
    # corrupted patches Y_test_corrupted. The text/noise is filtered out 
    # because D was only trained on clean image features.
    Y_recon_patches = odl_model.reconstruct(Y_test_corrupted)
    
    # 3. Reassemble the image from the reconstructed patches
    img_recon = reconstruct_image_from_patches(
        Y_recon_patches, 
        TARGET_SIZE, 
        patch_dims, 
        PATCH_SIZE, 
        STRIDE
    )
    
    print(f"Image reconstructed to shape: {img_recon.shape}")
    
    # Save the result
    cv2.imwrite("test_text_removed_output.png", (img_recon * 255).astype(np.uint8))
    print("Saved 'test_text_removed_output.png'.")
    
    # --- Evaluation ---
    if REAL_FRAMEWORK_LOADED:
        # Calculate reconstruction RMSE vs. the clean test patches
        test_patches_clean, _ = image_to_patches(test_image, PATCH_SIZE, STRIDE)
        test_patches_clean = test_patches_clean.to(device)
        
        rmse_test = torch.sqrt(torch.mean((test_patches_clean - Y_recon_patches) ** 2)).item()
        print(f"\nReconstruction (Denoising) Patch RMSE vs Clean: {rmse_test:.6f}")
        
        print("\n--- Dictionary Recovery Check (Self-Comparison) ---")
        evaluate_recovery(odl_model.get_dictionary(), odl_model.get_dictionary(), 
                          show_matching=False, n_show=5)


if __name__ == "__main__":
    # --- Usage ---
    # Specify the path to your BSD300 training images directory
    # Example: r'C:\Users\enguye17\Downloads\BSDS300-images\BSDS300\images\train'
    
    DATASET_PATH = r'C:\Users\enguye17\Downloads\BSDS300-images\BSDS300\images\train'
    
    run_image_denoising_experiment(DATASET_PATH)