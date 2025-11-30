import numpy as np


def golub_kahan(A, b, k, reorth="none", tol=0.0):
    """
    Golub–Kahan bidiagonalization for a LinearOperator A (m×n) and vector b ∈ R^m.

    Parameters
    ----------
    A : scipy.sparse.linalg.LinearOperator
        Must support A.matvec(x) and A.rmatvec(y).
    b : ndarray, shape (m,)
        Nonzero starting vector; u1 = b/||b||.
    k : int
        Number of steps to attempt.
    reorth : {"none", "mgs", "mgs2", True, False}, default "none"
        Reorthogonalization mode for (v_j, u_{j+1}):
          - "none" or False : no reorthogonalization
          - "mgs"  or True  : single-pass Modified Gram–Schmidt
          - "mgs2"          : double-pass Modified Gram–Schmidt
    tol : float, default 0.0
        Treat norms ≤ tol as breakdown. If 0, uses sqrt(machine_eps).

    Returns
    -------
    U : ndarray, shape (m, ell+1)
    V : ndarray, shape (n, ell)
    B : ndarray, shape (ell+1, ell)
    alphas : ndarray, shape (ell,)
    betas  : ndarray, shape (ell,)
    u0_norm : float
    """
    # normalize reorth flag (backward-compatible with bools)
    if reorth is True:
        reorth = "mgs"
    if reorth is False:
        reorth = "none"
    if reorth not in {"none", "mgs", "mgs2"}:
        raise ValueError("reorth must be one of {'none','mgs','mgs2', True, False}.")

    m, n = A.shape
    b = np.asarray(b).reshape(-1)
    if b.size != m:
        raise ValueError("b has incompatible length with A.")

    eps = np.finfo(float).eps
    if tol <= 0:
        tol = np.sqrt(eps)

    u0_norm = np.linalg.norm(b)
    if u0_norm == 0:
        raise ValueError("b must be nonzero.")

    # Allocate
    U = np.zeros((m, k + 1))
    V = np.zeros((n, k))
    alphas = np.zeros(k)
    betas = np.zeros(k)

    U[:, 0] = b / u0_norm
    v_prev = np.zeros(n)

    def _orth(vec, Q, passes=1):
        """Orthogonalize 'vec' against columns of Q using MGS, 'passes' times."""
        if Q.size == 0:
            return vec
        for _ in range(passes):
            for j in range(Q.shape[1]):
                vec -= np.dot(Q[:, j], vec) * Q[:, j]
        return vec

    passes = 0 if reorth == "none" else (1 if reorth == "mgs" else 2)

    ell = 0
    for j in range(k):
        # r = A^T u_j - beta_{j-1} v_{j-1}
        r = A.rmatvec(U[:, j]) - (betas[j - 1] * v_prev if j > 0 else 0.0)
        if passes:
            r = _orth(r, V[:, :j], passes)
        alpha = np.linalg.norm(r)
        alphas[j] = alpha

        if alpha <= tol:
            ell = j
            break

        v = r / alpha
        V[:, j] = v

        # p = A v_j - alpha_j u_j
        p = A.matvec(v) - alpha * U[:, j]
        if passes:
            p = _orth(p, U[:, :j+1], passes)
        beta = np.linalg.norm(p)
        betas[j] = beta

        if beta <= tol:
            ell = j + 1
            v_prev = v
            break

        U[:, j + 1] = p / beta
        v_prev = v
        ell = j + 1

    # Trim and build B
    U = U[:, :ell + 1]
    V = V[:, :ell]
    alphas = alphas[:ell]
    betas = betas[:ell]

    B = np.zeros((ell + 1, ell))
    for j in range(ell):
        B[j, j] = alphas[j]
        if j < ell - 1 or (j == ell - 1 and U.shape[1] == ell + 1):
            B[j + 1, j] = betas[j]

    return U, V, B, alphas, betas, u0_norm











