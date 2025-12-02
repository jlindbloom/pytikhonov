# tests/test_projected_tikhonov_family.py
import numpy as np

from pytikhonov import ProjectedTikhonovFamily, TikhonovFamily


def test_projected_tikhonov_matches_direct():
    rng = np.random.default_rng(321)
    repeats = 25

    def make_matrices():
        # Randomize dimensions
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

    for _ in range(repeats):
        lam = 10 ** rng.uniform(-4.0, 4.0)
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
        z_tf, x_tf = tf_proj.solve(lam)

        # Direct projected Tikhonov: solve over z with AV, LV
        A_tikh = np.vstack([AV, np.sqrt(lam) * LV])
        b_tikh = np.concatenate([b, np.sqrt(lam) * d])
        z_direct, *_ = np.linalg.lstsq(A_tikh, b_tikh, rcond=None)
        x_direct = V @ z_direct

        assert np.allclose(z_tf, z_direct, atol=1e-8)
        assert np.allclose(x_tf, x_direct, atol=1e-8)

        # Validate objectives
        direct_data_fidelity = np.linalg.norm(A @ x_direct - b) ** 2
        direct_reg_term = np.linalg.norm(L @ x_direct - d) ** 2

        assert np.allclose(tf_proj.data_fidelity(lam), direct_data_fidelity, atol=1e-8)
        assert np.allclose(tf_proj.regularization_term(lam), direct_reg_term, atol=1e-8)

        # Derivatives on the reduced system (AV, LV)
        AtA = AV.T @ AV
        LtL = LV.T @ LV
        Atb = AV.T @ b

        def df_derivatives(lam_val):
            S = AtA + lam_val * LtL
            rhs = Atb + lam_val * (LV.T @ d)
            Sprime = LtL
            rhs_prime = LV.T @ d

            x = np.linalg.solve(S, rhs)
            r = AV @ x - b

            x1 = np.linalg.solve(S, rhs_prime - Sprime @ x)
            r1 = AV @ x1
            f1 = 2.0 * r.T @ r1

            x2 = np.linalg.solve(S, -2.0 * (Sprime @ x1))
            r2 = AV @ x2
            f2 = 2.0 * (np.linalg.norm(r1) ** 2 + r.T @ r2)

            x3 = np.linalg.solve(S, -3.0 * (Sprime @ x2))
            r3 = AV @ x3
            f3 = 2.0 * (3.0 * (r1.T @ r2) + r.T @ r3)
            return float(f1), float(f2), float(f3)

        def reg_derivatives(lam_val):
            S = AtA + lam_val * LtL
            rhs = Atb + lam_val * (LV.T @ d)
            Sprime = LtL
            rhs_prime = LV.T @ d

            x = np.linalg.solve(S, rhs)
            y = LV @ x - d

            x1 = np.linalg.solve(S, rhs_prime - Sprime @ x)
            y1 = LV @ x1
            reg1 = 2.0 * y.T @ y1

            x2 = np.linalg.solve(S, -2.0 * (Sprime @ x1))
            y2 = LV @ x2
            reg2 = 2.0 * (np.linalg.norm(y1) ** 2 + y.T @ y2)

            x3 = np.linalg.solve(S, -3.0 * (Sprime @ x2))
            y3 = LV @ x3
            reg3 = 2.0 * (3.0 * (y1.T @ y2) + (y.T @ y3))
            return float(reg1), float(reg2), float(reg3)

        f1, f2, f3 = df_derivatives(lam)
        assert np.allclose(tf_proj.data_fidelity_derivative(lam, order=1, reciprocate=False), f1, atol=1e-8)
        assert np.allclose(tf_proj.data_fidelity_derivative(lam, order=2, reciprocate=False), f2, atol=1e-8)
        assert np.allclose(tf_proj.data_fidelity_derivative(lam, order=3, reciprocate=False), f3, atol=1e-6)

        r1, r2, r3 = reg_derivatives(lam)
        assert np.allclose(tf_proj.regularization_term_derivative(lam, order=1, reciprocate=False), r1, atol=1e-8)
        assert np.allclose(tf_proj.regularization_term_derivative(lam, order=2, reciprocate=False), r2, atol=1e-8)
        assert np.allclose(tf_proj.regularization_term_derivative(lam, order=3, reciprocate=False), r3, atol=1e-6)

        # Reciprocal parameterization: lam = 1 / beta
        beta = 1.0 / lam
        lam_prime = -1.0 / (beta**2)
        lam_double = 2.0 / (beta**3)
        lam_triple = -6.0 / (beta**4)

        g1 = f1 * lam_prime
        g2 = f2 * (lam_prime**2) + f1 * lam_double
        g3 = f3 * (lam_prime**3) + 3 * f2 * lam_prime * lam_double + f1 * lam_triple

        assert np.allclose(tf_proj.data_fidelity_derivative(beta, order=1, reciprocate=True), g1, atol=1e-8)
        assert np.allclose(tf_proj.data_fidelity_derivative(beta, order=2, reciprocate=True), g2, atol=1e-8)
        assert np.allclose(tf_proj.data_fidelity_derivative(beta, order=3, reciprocate=True), g3, atol=1e-6)

        g1_reg = r1 * lam_prime
        g2_reg = r2 * (lam_prime**2) + r1 * lam_double
        g3_reg = r3 * (lam_prime**3) + 3 * r2 * lam_prime * lam_double + r1 * lam_triple

        assert np.allclose(tf_proj.regularization_term_derivative(beta, order=1, reciprocate=True), g1_reg, atol=1e-8)
        assert np.allclose(tf_proj.regularization_term_derivative(beta, order=2, reciprocate=True), g2_reg, atol=1e-8)
        assert np.allclose(tf_proj.regularization_term_derivative(beta, order=3, reciprocate=True), g3_reg, atol=1e-6)


def test_projected_matches_full_when_V_is_identity():
    rng = np.random.default_rng(456)
    repeats = 25

    for _ in range(repeats):
        A = rng.standard_normal((100, 50))

        # L rows between 25 and 100; ensure stacked full-column rank
        while True:
            mL = int(rng.integers(25, 101))
            L = rng.standard_normal((mL, 50))
            if np.linalg.matrix_rank(np.vstack([A, L])) == 50:
                break

        xtrue = rng.standard_normal(50)
        btrue = A @ xtrue
        d = rng.standard_normal(L.shape[0])
        noise_std = 0.03 * np.linalg.norm(btrue) / np.sqrt(btrue.size)
        b = btrue + rng.normal(scale=noise_std, size=btrue.shape)

        V = np.eye(A.shape[1])  # identity basis
        lam = 10 ** rng.uniform(-4.0, 4.0)

        tf = TikhonovFamily(A, L, b, d=d)
        x_tf = tf.solve(lam)

        tf_proj = ProjectedTikhonovFamily(A, L, V, b, d=d, x_under=np.zeros(A.shape[1]), b_under=b, d_under=d)
        z_proj, x_proj = tf_proj.solve(lam)

        assert np.allclose(x_proj, x_tf, atol=1e-8)
        assert np.allclose(z_proj, x_tf, atol=1e-8)
