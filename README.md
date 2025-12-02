# PyTikhonov

PyTikhonov is a pure-Python toolkit for **Tikhonov regularization** of linear inverse problems, with a focus on **regularization parameter selection** and **GSVD-based diagnostics**.

It is primarily intended for small to moderate-scale problems (e.g., $N$ up to a few thousand) where matrix-like access to $A$ and $L$ is available. For larger problems, it provides a `ProjectedTikhonovFamily` interface designed to be combined with iterative methods (e.g., Krylov / model reduction bases).

---

## Installation

    pip install pytikhonov

---

## Features

- Core object: `TikhonovFamily(A, L, b, d, ...)` representing the entire family $$x_\lambda = \arg\min_x \|Ax - b\|_2^2 + \lambda \|Lx - d\|_2^2$$ for $\lambda > 0$.

- GSVD-based implementation (via [`easygsvd`](https://github.com/jlindbloom/easygsvd)):
  - Fast evaluation of $x_\lambda$ for many $\lambda$.
  - Efficient computation of $\|Ax - b\|_2^2$, $\|Lx - d\|_2^2$, and their derivatives.
  - Support for both $\lambda$ and reciprocal parameterization $\beta = 1/\lambda$.

- Diagnostic tools:
  - Picard plot (discrete Picard condition).
  - L-curve (and curvature-based “L-corner”).
  - Monitoring function and degrees of freedom.

- Regularization parameter selection:
  - L-corner heuristic.
  - Discrepancy principle (DP).
  - Generalized cross validation (GCV).
  - Convenience function to compare all methods on a given problem.
  - Randomization experiments to study robustness of parameter choices.

- Performing projections:
  - `ProjectedTikhonovFamily` for reduced problems on subspaces $\underline{x} + \mathrm{col}(V)$.
  - Designed to plug into Krylov / model-reduction pipelines (e.g., GKB, Arnoldi) for large-scale problems.

---

## Minimal example

    import numpy as np
    from pytikhonov import TikhonovFamily, lcorner

    # Example problem (A, L, b, d as dense arrays)
    A = ...
    L = ...
    b = ...
    d = np.zeros(L.shape[0])

    # Build the Tikhonov family
    tf = TikhonovFamily(A, L, b, d)

    # Select lambda via the L-corner heuristic
    lcorner_data = lcorner(tf)
    lambdah_star = lcorner_data["lambdah"]
    x_star = lcorner_data["x_lambdah"]

For full details, mathematical background, and additional examples (Picard plots, monitoring function, IRLS, projected problems, etc.), see the documentation PDF included in the repository.
