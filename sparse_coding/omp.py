import torch
import numpy as np
from contextlib import contextmanager
from timeit import default_timer

class OMP:
    def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=False, normalize=False, precompute=True, alg='naive'):
        self.n_nonzero_coefs = n_nonzero_coefs
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.precompute = precompute
        self.alg = alg

    def fit(self, X, y):
        """
        Fit the model using X as the dictionary and y as the signals.
        X: (n_samples, n_features) or (n_features, n_components) - Dictionary
           Note: The original code seems to treat X as (n_features, n_components) usually denoted as D.
           Let's assume X is the dictionary D (n_features, n_atoms) and y is the signal (n_features, n_samples).
           Wait, scikit-learn OMP fit(X, y) takes X as (n_samples, n_features) and y as (n_samples, n_targets).
           But in sparse coding D is usually (n_features, n_atoms) and y is (n_features, n_samples).
           Let's look at quickstart.py usage:
           run_omp(X, y, ...)
           In quickstart.py:
           y, X, w = make_sparse_coded_signal(...)
           y is (n_samples, n_features) (transposed later to (n_features, n_samples)?)
           Actually make_sparse_coded_signal returns y (n_features, n_samples), X (n_features, n_components), w (n_components, n_samples).
           In quickstart.py:
           y = y.T -> (n_samples, n_features)
           X is (n_features, n_components)
           run_omp(X, y, ...)
           Inside run_omp:
           X = torch.as_tensor(X)
           y = torch.as_tensor(y)
           
           If we want to follow sklearn API:
           fit(X, y) where X is (n_samples, n_features) and y is (n_samples, n_targets).
           But here X is the dictionary.
           Sklearn OMP:
           fit(X, y): X is data (n_samples, n_features), y is targets (n_samples,).
           This solves y = Xw.
           
           In sparse coding/dictionary learning:
           y = Dx
           We want to find x given y and D.
           
           The quickstart.py `run_omp` takes X (Dictionary) and y (Signals).
           Let's stick to the quickstart.py convention but maybe rename arguments for clarity if needed.
           Actually, `run_omp` in `quickstart.py` seems to treat X as the dictionary (M x N) and y as (B x M) or something?
           
           Let's check `omp_naive` in `quickstart.py`:
           X: MxN array
           y: BxN array ?? No, wait.
           
           Line 209: "Given X as an MxN array and y as an BxN array, do omp to approximately solve Xb=y"
           Wait, if X is MxN, and we solve Xb=y, b must be Nx?.
           If y is BxN? That implies batch processing?
           
           Let's look at dimensions in `quickstart.py` example:
           n_components (atoms) = 100, n_features (dim) = 100.
           y, X, w = make_sparse_coded_signal(...)
           y: (n_features, n_samples)
           X: (n_features, n_components)
           
           y = y.T -> (n_samples, n_features)
           
           run_omp(X, y, ...)
           X: (n_features, n_components)
           y: (n_samples, n_features)
           
           Inside omp_naive:
           XT = X.contiguous().t() -> (n_components, n_features)
           y is (n_samples, n_features)
           
           projections = XT @ r[:, :, None]
           (n_components, n_features) @ (n_samples, n_features, 1) -> (n_samples, n_components, 1)
           
           So it seems X is indeed the dictionary (n_features, n_components).
           And y is the batch of signals (n_samples, n_features).
           
           I will keep this convention.
        """
        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)

        # Handle device placement - move to same device as X if not already
        if y.device != X.device:
            y = y.to(X.device)

        self.device = X.device
        
        return self._run_omp(X, y)

    def _run_omp(self, X, y):
        n_nonzero_coefs = self.n_nonzero_coefs
        tol = self.tol
        precompute = self.precompute
        normalize = self.normalize
        fit_intercept = self.fit_intercept
        alg = self.alg

        if fit_intercept or normalize:
            X = X.clone()
            # assert not isinstance(precompute, torch.Tensor), "If user pre-computes XTX they can also pre-normalize X" \
            #                                                  " as well, so normalize and fit_intercept must be set false."

        if fit_intercept:
            X = X - X.mean(0)
            y = y - y.mean(1)[:, None]

        if normalize is True:
            normalize = (X * X).sum(0).sqrt()
            X /= normalize[None, :]

        if precompute is True or alg == 'v0':
            precompute = X.T @ X

        if alg == 'naive':
            sets, solutions, lengths = self._omp_naive(X, y, n_nonzero_coefs=n_nonzero_coefs, XTX=precompute, tol=tol)
        elif alg == 'v0':
            sets, solutions, lengths = self._omp_v0(X, y, n_nonzero_coefs=n_nonzero_coefs, XTX=precompute, tol=tol)
        else:
            raise ValueError(f"Unknown algorithm {alg}")

        solutions = solutions.squeeze(-1)
        if normalize is not False:
            solutions /= normalize[sets]

        xests = y.new_zeros(y.shape[0], X.shape[1])
        if lengths is None:
            xests[torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)[:, None], sets] = solutions
        else:
            for i in range(y.shape[0]):
                xests[i, sets[i, :lengths[i]]] = solutions[i, :lengths[i]]

        return xests

    def _batch_mm(self, matrix, matrix_batch, return_contiguous=True):
        batch_size = matrix_batch.shape[0]
        vectors = matrix_batch.transpose(1, 2).transpose(0, 1).reshape(matrix.shape[1], -1)
        
        if return_contiguous:
            result = torch.empty(batch_size, matrix.shape[0], matrix_batch.shape[2], device=matrix.device, dtype=matrix.dtype)
            # This part in original code used numpy, adapting to pure torch if possible or keep logic
            # The original code mixed numpy and torch in batch_mm? 
            # "matrix" param in original was numpy array in on_cpu branch.
            # Let's assume we want full torch implementation for GPU support.
            # If on CPU, we can still use torch.
            
            # Actually, let's look at the original batch_mm. It used np.matmul.
            # We should stick to torch operations.
            
            # (m, n) @ (n, b*k) -> (m, b*k)
            res = matrix @ vectors
            res = res.reshape(matrix.shape[0], batch_size, -1).transpose(0, 1)
            return res
        else:
            return (matrix @ vectors).reshape(matrix.shape[0], batch_size, -1).transpose(0, 1)

    def _innerp(self, x, y=None, out=None):
        if y is None:
            y = x
        if out is not None:
            out = out[:, None, None]
        return torch.matmul(x[..., None, :], y[..., :, None], out=out)[..., 0, 0]

    def _cholesky_solve(self, ATA, ATy):
        if ATA.dtype == torch.half or ATy.dtype == torch.half:
            return ATy.to(torch.float).cholesky_solve(torch.linalg.cholesky(ATA.to(torch.float))).to(ATy.dtype)
        return ATy.cholesky_solve(torch.linalg.cholesky(ATA)).to(ATy.dtype)

    def _omp_naive(self, X, y, n_nonzero_coefs, tol=None, XTX=None):
        on_cpu = not (y.is_cuda or y.dtype == torch.half)
        
        XT = X.contiguous().t()
        y = y.contiguous()
        r = y.clone()

        sets = y.new_zeros((n_nonzero_coefs, y.shape[0]), dtype=torch.long).t()
        if tol:
            result_sets = sets.new_zeros(y.shape[0], n_nonzero_coefs)
            result_lengths = sets.new_zeros(y.shape[0])
            result_solutions = y.new_zeros((y.shape[0], n_nonzero_coefs, 1))
            original_indices = torch.arange(y.shape[0], dtype=sets.dtype, device=sets.device)

        ATs = y.new_zeros(r.shape[0], n_nonzero_coefs, X.shape[0])
        ATys = y.new_zeros(r.shape[0], n_nonzero_coefs, 1)
        ATAs = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device)[None].repeat(r.shape[0], 1, 1)
        
        # CPU optimization from original code omitted for simplicity/uniformity unless critical
        # The original code had specific CPU optimizations using numpy/cython in some places or just different torch paths.
        # I will stick to the torch path (the "else" block of on_cpu checks in original where possible, or adapt)
        # Actually, the original code had a big "if on_cpu" block.
        # For simplicity and GPU focus, I will use the torch implementation.
        # If the user wants CPU speedup specifically, we might need to bring back the numpy/cython parts, 
        # but the request emphasizes GPU.
        
        solutions = y.new_zeros((r.shape[0], 0))

        for k in range(n_nonzero_coefs + bool(tol)):
            if tol:
                problems_done = self._innerp(r) <= tol
                if k == n_nonzero_coefs:
                    problems_done[:] = True

                if problems_done.any():
                    remaining = ~problems_done
                    orig_idxs = original_indices[problems_done]
                    result_sets[orig_idxs, :k] = sets[problems_done, :k]
                    result_solutions[orig_idxs, :k] = solutions[problems_done]
                    result_lengths[orig_idxs] = k
                    original_indices = original_indices[remaining]

                    ATs = ATs[remaining]
                    ATys = ATys[remaining]
                    ATAs = ATAs[remaining]
                    sets = sets[remaining]
                    y = y[remaining]
                    r = r[remaining]
                    if problems_done.all():
                        return result_sets, result_solutions, result_lengths

            # Projections
            # Using pure torch for both CPU and GPU for consistency
            projections = XT @ r[:, :, None]
            sets[:, k] = projections.abs().sum(-1).argmax(-1)

            # Update AT
            AT = ATs[:, :k + 1, :]
            updateA = XT[sets[:, k], :]
            AT[:, k, :] = updateA

            # Update ATy
            ATy = ATys[:, :k + 1]
            self._innerp(updateA, y, out=ATy[:, k, 0])

            # Update ATA
            ATA = ATAs[:, :k + 1, :k + 1]
            if XTX is not None:
                ATA[:, k, :k + 1] = XTX[sets[:, k, None], sets[:, :k + 1]]
            else:
                torch.bmm(AT[:, :k + 1, :], updateA[:, :, None], out=ATA[:, k, :k + 1, None])

            # Solve ATAx = ATy
            ATA[:, :k, k] = ATA[:, k, :k]
            solutions = self._cholesky_solve(ATA, ATy)

            # Update residual
            torch.baddbmm(y[:, :, None], AT.permute(0, 2, 1), solutions, beta=-1, out=r[:, :, None])

        return sets, solutions, None

    def _omp_v0(self, X, y, XTX, n_nonzero_coefs=None, tol=None, inverse_cholesky=True):
        B = y.shape[0]
        normr2 = self._innerp(y)
        projections = (X.transpose(1, 0) @ y[:, :, None]).squeeze(-1)
        sets = y.new_zeros(n_nonzero_coefs, B, dtype=torch.int64)

        if inverse_cholesky:
            F = torch.eye(n_nonzero_coefs, dtype=y.dtype, device=y.device).repeat(B, 1, 1)
            a_F = y.new_zeros(n_nonzero_coefs, B, 1)

        D_mybest = y.new_empty(B, n_nonzero_coefs, XTX.shape[0])
        temp_F_k_k = y.new_ones((B, 1))

        if tol:
            result_lengths = sets.new_zeros(y.shape[0])
            result_solutions = y.new_zeros((y.shape[0], n_nonzero_coefs, 1))
            finished_problems = sets.new_zeros(y.shape[0], dtype=torch.bool)

        for k in range(n_nonzero_coefs + bool(tol)):
            if tol:
                problems_done = normr2 <= tol
                if k == n_nonzero_coefs:
                    problems_done[:] = True

                if problems_done.any():
                    new_problems_done = problems_done & ~finished_problems
                    finished_problems.logical_or_(problems_done)
                    result_lengths[new_problems_done] = k
                    if inverse_cholesky:
                        result_solutions[new_problems_done, :k] = F[new_problems_done, :k, :k].permute(0, 2, 1) @ a_F[:k, new_problems_done].permute(1, 0, 2)
                    else:
                        assert False, "inverse_cholesky=False with tol != None is not handled yet"
                    if problems_done.all():
                        return sets.t(), result_solutions, result_lengths

            sets[k] = projections.abs().argmax(1)
            torch.gather(XTX, 0, sets[k, :, None].expand(-1, XTX.shape[1]), out=D_mybest[:, k, :])
            
            if k:
                D_mybest_maxindices = D_mybest.permute(0, 2, 1)[torch.arange(D_mybest.shape[0], dtype=sets.dtype, device=sets.device), sets[k], :k]
                torch.rsqrt(1 - self._innerp(D_mybest_maxindices), out=temp_F_k_k[:, 0])
                D_mybest_maxindices *= -temp_F_k_k
                D_mybest[:, k, :] *= temp_F_k_k
                D_mybest[:, k, :, None].baddbmm_(D_mybest[:, :k, :].permute(0, 2, 1), D_mybest_maxindices[:, :, None])

            temp_a_F = temp_F_k_k * torch.gather(projections, 1, sets[k, :, None])
            normr2 -= (temp_a_F * temp_a_F).squeeze(-1)
            projections -= temp_a_F * D_mybest[:, k, :]
            
            if inverse_cholesky:
                a_F[k] = temp_a_F
                if k:
                    torch.bmm(D_mybest_maxindices[:, None, :], F[:, :k, :], out=F[:, k, None, :])
                    F[:, k, k] = temp_F_k_k[..., 0]
        else:
            if inverse_cholesky:
                solutions = F.permute(0, 2, 1) @ a_F.squeeze(-1).transpose(1, 0)[:, :, None]
            else:
                AT = X.T[sets.T]
                solutions = self._cholesky_solve(AT @ AT.permute(0, 2, 1), AT @ y.T[:, :, None])

        return sets.t(), solutions, None
