# IRLS-DLA: Sparse Coding and Dictionary Learning

> **Undergraduate Mathematics Thesis at University of North Carolina at Charlotte**  
> **Primarily Advised by:** Christian Kuemmerle  
> **Advised by:** Dr. Shaozhong Deng  
> **Reader:** Xinje Li

This repository contains implementations of algorithms for Sparse Coding and Dictionary Learning, specifically Iteratively Reweighted Least Squares (IRLS) and K-SVD.

## Files

- `irls_sparse_regularization.py`: Implements IRLS for sparse regularization. It generates synthetic data and attempts to solve the sparse coding problem using an IRLS approach.
- `k_svd.py`: Implements the K-SVD algorithm for dictionary learning. It includes:
    - `PAKSVD`: A Parallel Approximate K-SVD implementation using `cupy` for GPU acceleration.
    - `SimpleKSVD`: A CPU-based K-SVD implementation using `numpy`.

## Experiments

The `experiments/` directory contains various scripts to test and validate the algorithms:

- **`cpu_experiment.py`**: Runs a comparative analysis of OMP, IRLS, and FISTA sparse coding algorithms on synthetic data using the CPU.
- **`gpu_experiment.py`**: Performs the same comparative analysis as `cpu_experiment.py` but utilizes GPU acceleration (via PyTorch/CuPy) for larger scale problems.
- **`dictionary_learning_experiment.py`**: Trains a dictionary using K-SVD and evaluates its performance by testing how well OMP, IRLS, and FISTA can sparsely encode data using the learned dictionary.
- **`super_resolution.py`**: Implements Image Super-Resolution using Sparse Representation. This experiment replicates the approach described by Yang et al. (2010), training coupled dictionaries for low-resolution and high-resolution image patches to upscale images.
- **`experiment_modular_ksvd.py`**: Tests the modular K-SVD implementation, verifying that different sparse coding backends (OMP, IRLS, FISTA) can be swapped into the K-SVD algorithm.

## Prerequisites

- Python 3.8+
- [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (Required for `cupy` GPU acceleration)

## Setup

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone <repository_url>
    cd IRLS-DLA
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment**:
    - Windows:
        ```powershell
        .\venv\Scripts\Activate
        ```
    - Linux/macOS:
        ```bash
        source venv/bin/activate
        ```

4.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    
5.  **Optional: Install CuPy for GPU support**:
    `cupy` is required for the `PAKSVD` class in `k_svd.py`. Install the version matching your CUDA toolkit (e.g., for CUDA 12.x):
    ```bash
    pip install cupy-cuda12x
    ```
    > **Note**: If you do not have a GPU, you can still use the `SimpleKSVD` class in `k_svd.py` (which uses `numpy`).

## Usage

### IRLS Sparse Regularization
Run the script to generate data and perform IRLS:
```bash
python irls_sparse_regularization.py
```

### K-SVD Dictionary Learning
Run the script to demonstrate K-SVD (both Simple and Parallel versions):
```bash
python k_svd.py
```

### Sparse Coding Algorithms
The repository now includes implementations of OMP, IRLS, and FISTA in the `sparse_coding` directory.

#### OMP (Orthogonal Matching Pursuit)
```python
from sparse_coding.omp import OMP
omp = OMP(n_nonzero_coefs=10)
coefs = omp.fit(dictionary, signals)
```

#### IRLS (Iteratively Reweighted Least Squares)
```python
from sparse_coding.irls import IRLS
irls = IRLS(max_iter=50)
coefs = irls.fit(dictionary, signals)
```

#### FISTA (Fast Iterative Shrinkage-Thresholding Algorithm)
```python
from sparse_coding.fista import FISTA
fista = FISTA(lambda_reg=0.1)
coefs = fista.fit(dictionary, signals)
```

## References

1.  **Berkeley Everyday Image Dataset**:  
    D. Martin, C. Fowlkes, D. Tal, and J. Malik. A Database of Human Segmented Natural Images and its Application to Evaluating Segmentation Algorithms and Measuring Ecological Statistics. *Proc. 8th Int'l Conf. Computer Vision*, vol. 2, pp. 416-423, July 2001.

2.  **Image Super-Resolution**:  
    J. Yang, J. Wright, T. S. Huang and Y. Ma, "Image Super-Resolution Via Sparse Representation," in *IEEE Transactions on Image Processing*, vol. 19, no. 11, pp. 2861-2873, Nov. 2010. [PDF](https://www.columbia.edu/~jw2966/papers/YWHM10-TIP.pdf)
