# Newton Methods for Computing C-Eigenvalues and M-Eigenvalues of Tensors

MATLAB source code accompanying the paper:

> Jun He and Shi-Liang Wu. *Newton Methods for Computing C-Eigenvalues and M-Eigenvalues of Tensors.*

## Overview

This repository implements Newton-type methods for computing:

- **C-eigenvalues** of piezoelectric-type tensors (3rd-order, `n×n×n`, symmetric in last two indices)
- **M-eigenvalues** of fourth-order partially symmetric tensors (`m×n×m×n`)

The methods eliminate the eigenvalue parameter via the Rayleigh quotient, add a normalization regularization, and solve the resulting nonlinear system by Newton's method with per-step normalization. Under nondegeneracy conditions, the convergence is locally quadratic.

## Requirements

- MATLAB R2018b or later
- No toolboxes or external dependencies

## File Structure

### Core algorithms

| File | Description |
|------|-------------|
| `NewtonC.m` | Newton method for C-eigenvalues of piezoelectric-type tensors (Algorithm 1 in the paper) |
| `NewtonM.m` | Newton method for M-eigenvalues of partially symmetric tensors (Algorithm 3). Supports `mode='plain'` (default) and `mode='safeguard'` (eigenvalue-regularized Jacobian + Armijo line search fallback) |
| `svd_init.m` | SVD-based initialization for M-eigenvalues (unfold + shift + smallest eigenvector + leading singular pair) |

### Tensor data

| File | Description |
|------|-------------|
| `make_piezo_tensor.m` | Creates 3×3×3 piezoelectric tensors for 8 real materials: VFeSb, SiO₂, Cr₂AgBiO₈, RbTaO₃, NaBiS₂, LiBiB₂O₅, KBi₂F₇, BaNiO₃ (data from Chen–Jakli–Qi, J. Phys. A 51, 2018) |
| `make_psym_tensor.m` | Creates 3×3×3×3 partially symmetric tensors: `tetragonal` (elasticity), `random_wang` (Wang–Qi–Zhang 2009), `dolomite` (CaMg(CO₃)₂) |

### Comparison methods

| File | Description |
|------|-------------|
| `SPM_C.m` | Shifted Power Method for largest C-eigenvalue (Zhao 2023) |
| `SIPM_M.m` | Shifted Inverse Power Method for smallest M-eigenvalue (Zhao–Liu–Sang 2024) |

### Experiment scripts

| File | Reproduces |
|------|-----------|
| `run_all_experiments.m` | All main experiments: 8 piezoelectric C-eigenvalue tables, tetragonal/random/dolomite M-eigenvalues, high-dimensional scaling, convergence history |
| `lam8.m` | Complete enumeration of tetragonal M-eigenvalues (multi-start, Ns=400) |
| `enum_bani100.m` | Complete enumeration of BaNiO₃ C-eigenvalues (multi-start, Ns=100) |
| `highdim_v4.m` | High-dimensional random tensors (m,n up to 20×20): SCF init + Newton refine vs SIPM |

## Quick Start

### 1. Run all experiments

```matlab
cd MATLAB_code
run_all_experiments
```

This reproduces every table in the numerical section. Results are saved to `results_all.mat` and printed to the console. Runtime ≈ 5–15 minutes on a standard desktop.

### 2. Single C-eigenvalue computation

```matlab
A = make_piezo_tensor('VFeSb');          % 3×3×3 piezoelectric tensor
c = 1;                                     % regularization parameter
x0 = randn(3,1); x0 = x0/norm(x0);        % random unit start
y0 = randn(3,1); y0 = y0/norm(y0);
[lam, x, y, iter, res_hist] = NewtonC(A, c, x0, y0, 1e-8, 200);
```

### 3. Single M-eigenvalue computation

```matlab
A = make_psym_tensor('tetragonal');        % 3×3×3×3 elasticity tensor
c = 1;
[x0, y0] = svd_init(A, 'smallest');        % SVD initialization
[lam, x, y, iter, res_hist] = NewtonM(A, c, x0, y0, 1e-8, 200);
```

### 4. Compute all C-eigenvalues (multi-start)

```matlab
A = make_piezo_tensor('BaNiO3');
Ns = 100;                                   % number of random starts
c = 1; eps_tol = 1e-8; Kmax = 200;
all_eig = [];
rng(42);
for s = 1:Ns
    xs = randn(3,1); xs = xs/norm(xs);
    ys = randn(3,1); ys = ys/norm(ys);
    [lam_s,~,~,~,~] = NewtonC(A, c, xs, ys, eps_tol, Kmax);
    if all(abs(abs(lam_s) - abs(all_eig)) >= 1e-4)
        all_eig = [all_eig, lam_s];         % cluster new eigenvalues
    end
end
```

## Parameters

| Parameter | Paper notation | Typical value |
|-----------|---------------|---------------|
| `c` | Regularization parameter | 1 (must avoid c = 2λ*) |
| `eps_tol` | Convergence tolerance | 1e-8 |
| `Kmax` | Maximum Newton iterations | 200 |
| `Ns` | Number of multi-start samples | 50–400 |

## Citation

If you use this code, please cite:

```bibtex
@article{HeWu2026Newton,
  author  = {He, Jun and Wu, Shi-Liang},
  title   = {Newton Methods for Computing C-Eigenvalues and M-Eigenvalues of Tensors},
  journal = {SIAM Journal on Scientific Computing},
  year    = {2026},
  note    = {Submitted}
}
```

## License

This code is provided for academic research purposes. Please contact the authors for commercial use.


