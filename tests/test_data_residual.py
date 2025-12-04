# tests/test_data_residual.py
import numpy as np

from pytikhonov import ProjectedTikhonovFamily, TikhonovFamily


def test_data_residual_matches_direct_solution():
    rng = np.random.default_rng(123)
    repeats = 25

    def make_matrices():
        # Choose ranks so that ker(A) ∩ ker(L) = {0} (stacked matrix full column rank)
        mL = int(rng.integers(25, 101))  # rows of L between 25 and 100
        while True:
            rA = rng.integers(1, 51)
            rL = rng.integers(1, min(51, mL + 1))  # rank cannot exceed row count
            if rA + rL >= 50:
                break

        # Orthonormal basis for R^50
        Q, _ = np.linalg.qr(rng.standard_normal((50, 50)))
        A_basis = Q[:, :rA]

        needed = 50 - rA
        complement = list(range(rA, 50))
        needed_idx = complement[:needed]
        extra = rL - needed
        if extra > 0:
            pool = np.arange(50)
            available = pool[np.isin(pool, needed_idx, invert=True)]
            extra_idx = rng.choice(available, size=extra, replace=False).tolist()
            L_idx = needed_idx + extra_idx
        else:
            L_idx = needed_idx
        L_basis = Q[:, L_idx]

        A = rng.standard_normal((100, rA)) @ A_basis.T
        L = rng.standard_normal((mL, rL)) @ L_basis.T

        # Sanity checks: desired ranks and trivial common kernel
        assert np.linalg.matrix_rank(A) == rA
        assert np.linalg.matrix_rank(L) == rL
        assert np.linalg.matrix_rank(np.vstack([A, L])) == 50
        return A, L

    def direct_residual(A, L, b, d, lam_val):
        A_tikh = np.vstack([A, np.sqrt(lam_val) * L])
        b_tikh = np.concatenate([b, np.sqrt(lam_val) * d])
        x_direct, *_ = np.linalg.lstsq(A_tikh, b_tikh, rcond=None)
        return A @ x_direct - b

    for _ in range(repeats):
        lam = 10 ** rng.uniform(-4.0, 4.0)
        lam_vec = np.array([lam, lam * 10.0, lam * 0.1])

        A, L = make_matrices()
        xtrue = rng.standard_normal(50)
        btrue = A @ xtrue
        d = rng.standard_normal(L.shape[0])

        noise_std = 0.03 * np.linalg.norm(btrue) / np.sqrt(btrue.size)
        b = btrue + rng.normal(scale=noise_std, size=btrue.shape)

        tf = TikhonovFamily(A, L, b, d=d)

        # Scalar λ
        residual_tf = tf.data_residual(lam)
        residual_direct = direct_residual(A, L, b, d, lam)
        x_tf = tf.solve(lam)

        assert np.allclose(residual_tf, residual_direct, atol=1e-8)
        assert np.allclose(residual_tf, A @ x_tf - b, atol=1e-8)

        # Scalar β = 1 / λ (reciprocate=True)
        beta = 1.0 / lam
        residual_beta_scalar = tf.data_residual(beta, reciprocate=True)
        assert np.allclose(residual_beta_scalar, residual_tf, atol=1e-8)

        # Batched λ
        residual_tf_vec = tf.data_residual(lam_vec)
        residual_direct_vec = np.column_stack([direct_residual(A, L, b, d, lv) for lv in lam_vec])
        x_tf_vec = tf.solve(lam_vec)

        assert np.allclose(residual_tf_vec, residual_direct_vec, atol=1e-8)
        assert np.allclose(residual_tf_vec, (A @ x_tf_vec) - b[:, None], atol=1e-8)

        # Reciprocal parameterization
        beta_vec = 1.0 / lam_vec
        residual_beta = tf.data_residual(beta_vec, reciprocate=True)
        assert np.allclose(residual_beta, residual_tf_vec, atol=1e-8)


def test_projected_data_residual_matches_direct_solution():
    rng = np.random.default_rng(321)
    repeats = 25

    def make_matrices():
        # Randomize dimensions, mirroring other projected tests
        mL = int(rng.integers(25, 101))  # rows of L
        k = int(rng.integers(25, 76))    # columns of V (subspace dimension)

        # A with 100 rows, 50 cols (full column rank w.p.1 for Gaussian)
        A = rng.standard_normal((100, 50))

        # L with random rank; enforce stacked full-column rank of [A; L]
        while True:
            L = rng.standard_normal((mL, 50))
            if np.linalg.matrix_rank(np.vstack([A, L])) == 50:
                break

        # Orthonormal V in R^{50 x k}
        Q, _ = np.linalg.qr(rng.standard_normal((50, k)))
        V = Q[:, :k]

        return A, L, V

    def direct_projected_residual(AV, LV, A, V, b, d, x_under, lam_val):
        A_tikh = np.vstack([AV, np.sqrt(lam_val) * LV])
        b_tikh = np.concatenate([b, np.sqrt(lam_val) * d])
        z_direct, *_ = np.linalg.lstsq(A_tikh, b_tikh, rcond=None)
        x_direct = x_under + V @ z_direct
        return A @ x_direct - b, z_direct, x_direct

    for _ in range(repeats):
        lam = 10 ** rng.uniform(-4.0, 4.0)
        lam_vec = np.array([lam, lam * 10.0, lam * 0.1])

        A, L, V = make_matrices()
        AV = A @ V
        LV = L @ V

        xtrue = rng.standard_normal(50)
        btrue = A @ xtrue
        d = rng.standard_normal(L.shape[0])

        noise_std = 0.03 * np.linalg.norm(btrue) / np.sqrt(btrue.size)
        b = btrue + rng.normal(scale=noise_std, size=btrue.shape)

        x_under = np.zeros(A.shape[1])
        b_under = b - A @ x_under
        d_under = d - L @ x_under

        tf_proj = ProjectedTikhonovFamily(A, L, V, b, d=d, x_under=x_under, b_under=b_under, d_under=d_under)

        # Scalar λ
        residual_proj = tf_proj.data_residual(lam)
        (z_tf, x_tf) = tf_proj.solve(lam)
        residual_direct, z_direct, x_direct = direct_projected_residual(AV, LV, A, V, b, d, x_under, lam)

        assert np.allclose(residual_proj, residual_direct, atol=1e-8)
        assert np.allclose(residual_proj, A @ x_tf - b, atol=1e-8)
        assert np.allclose(z_tf, z_direct, atol=1e-8)
        assert np.allclose(x_tf, x_direct, atol=1e-8)

        # Scalar β = 1 / λ
        beta = 1.0 / lam
        residual_beta_scalar = tf_proj.data_residual(beta, reciprocate=True)
        assert np.allclose(residual_beta_scalar, residual_proj, atol=1e-8)

        # Batched λ
        residual_proj_vec = tf_proj.data_residual(lam_vec)
        residual_direct_vec = np.column_stack(
            [direct_projected_residual(AV, LV, A, V, b, d, x_under, lv)[0] for lv in lam_vec]
        )
        z_tf_vec, x_tf_vec = tf_proj.solve(lam_vec)

        assert np.allclose(residual_proj_vec, residual_direct_vec, atol=1e-8)
        assert np.allclose(residual_proj_vec, (A @ x_tf_vec) - b[:, None], atol=1e-8)

        # Batched β
        beta_vec = 1.0 / lam_vec
        residual_beta_vec = tf_proj.data_residual(beta_vec, reciprocate=True)
        assert np.allclose(residual_beta_vec, residual_proj_vec, atol=1e-8)
