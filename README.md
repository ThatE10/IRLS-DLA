# IRLS-DLA: Sparse Coding and Dictionary Learning

This repository contains implementations of algorithms for Sparse Coding and Dictionary Learning, specifically Iteratively Reweighted Least Squares (IRLS) and K-SVD.

## Files

- `irls_sparse_regularization.py`: Implements IRLS for sparse regularization. It generates synthetic data and attempts to solve the sparse coding problem using an IRLS approach.
- `k_svd.py`: Implements the K-SVD algorithm for dictionary learning. It includes:
    - `PAKSVD`: A Parallel Approximate K-SVD implementation using `cupy` for GPU acceleration.
    - `SimpleKSVD`: A CPU-based K-SVD implementation using `numpy`.

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

