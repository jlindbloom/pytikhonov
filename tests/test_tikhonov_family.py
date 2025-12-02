# tests/test_tikhonov_family.py
import numpy as np

from pytikhonov import TikhonovFamily



def test_solve_matches_closed_form_with_identity():
    A = np.eye(3)
    L = np.eye(3)
    b = np.array([1.0, -2.0, 4.0])
    lam = 0.5

    tf = TikhonovFamily(A, L, b)
    x = tf.solve(lam)

    expected = b / (1.0 + lam)
    assert np.allclose(x, expected, atol=1e-12)

    # Batched λ should broadcast and match the same closed form per λ
    lam_vec = np.array([0.1, 1.0])
    x_batch = tf.solve(lam_vec)
    expected_batch = b[:, None] / (1.0 + lam_vec)
    assert np.allclose(x_batch, expected_batch, atol=1e-12)




def test_solve_matches_direct_least_squares_randomized():
    rng = np.random.default_rng(123)
    repeats = 10

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

    for _ in range(repeats):
        lam = 10 ** rng.uniform(-4.0, 4.0)
        A, L = make_matrices()
        xtrue = rng.standard_normal(50)
        btrue = A @ xtrue
        d = rng.standard_normal(L.shape[0])

        noise_std = 0.03 * np.linalg.norm(btrue) / np.sqrt(btrue.size)
        b = btrue + rng.normal(scale=noise_std, size=btrue.shape)

        tf = TikhonovFamily(A, L, b, d=d)
        x_tf = tf.solve(lam)

        # Direct Tikhonov via stacked least squares: minimizes ||A x - b||^2 + lam ||L x||^2
        A_tikh = np.vstack([A, np.sqrt(lam) * L])
        b_tikh = np.concatenate([b, np.sqrt(lam) * d])
        x_direct, *_ = np.linalg.lstsq(A_tikh, b_tikh, rcond=None)

        assert np.allclose(x_tf, x_direct, atol=1e-8)

        # Validate data fidelity and regularization terms against direct solution
        direct_data_fidelity = np.linalg.norm(A @ x_direct - b) ** 2
        direct_reg_term = np.linalg.norm(L @ x_direct - d) ** 2

        assert np.allclose(tf.data_fidelity(lam), direct_data_fidelity, atol=1e-8)
        assert np.allclose(tf.regularization_term(lam), direct_reg_term, atol=1e-8)

        # Validate GCV objective directly from components
        gamma = tf.gamma_check
        bperp_sq = np.linalg.norm(tf.b - (tf.Uhat @ tf.Uhattb)) ** 2
        num = bperp_sq + np.sum(((lam / (gamma**2 + lam)) ** 2) * ((tf.U2tb - tf.V2td * gamma) ** 2))
        den = ((1.0 / tf.M) * (tf.M - tf.n_L - np.sum((gamma**2) / (gamma**2 + lam)))) ** 2
        gcv_direct = num / den

        assert np.allclose(tf.gcv_objective(lam), gcv_direct, atol=1e-10)

        # Validate data_fidelity_derivative (orders 1 and 2) against matrix-calculus expressions
        AtA = A.T @ A
        LtL = L.T @ L
        Atb = A.T @ b

        def matrix_calculus_derivatives(lam_val):
            S = AtA + lam_val * LtL
            rhs = Atb + lam_val * (L.T @ d)
            Sprime = LtL
            rhs_prime = L.T @ d

            x = np.linalg.solve(S, rhs)
            r = A @ x - b

            x1 = np.linalg.solve(S, rhs_prime - Sprime @ x)
            r1 = A @ x1
            f1 = 2.0 * r.T @ r1

            x2 = np.linalg.solve(S, -2.0 * (Sprime @ x1))
            r2 = A @ x2
            f2 = 2.0 * (np.linalg.norm(r1) ** 2 + r.T @ r2)

            x3 = np.linalg.solve(S, -3.0 * (Sprime @ x2))
            r3 = A @ x3
            f3 = 2.0 * (3.0 * (r1.T @ r2) + r.T @ r3)
            return float(f1), float(f2), float(f3)

        f1_direct, f2_direct, f3_direct = matrix_calculus_derivatives(lam)
        assert np.allclose(tf.data_fidelity_derivative(lam, order=1, reciprocate=False), f1_direct, atol=1e-8)
        assert np.allclose(tf.data_fidelity_derivative(lam, order=2, reciprocate=False), f2_direct, atol=1e-8)
        assert np.allclose(tf.data_fidelity_derivative(lam, order=3, reciprocate=False), f3_direct, atol=1e-6)

        # Validate regularization_term_derivative (orders 1–3) using analogous formulas
        def regularization_derivatives(lam_val):
            S = AtA + lam_val * LtL
            rhs = Atb + lam_val * (L.T @ d)
            Sprime = LtL
            rhs_prime = L.T @ d

            x = np.linalg.solve(S, rhs)
            y = L @ x - d  # regularization residual

            x1 = np.linalg.solve(S, rhs_prime - Sprime @ x)
            y1 = L @ x1
            reg1 = 2.0 * y.T @ y1

            x2 = np.linalg.solve(S, -2.0 * (Sprime @ x1))
            y2 = L @ x2
            reg2 = 2.0 * (np.linalg.norm(y1) ** 2 + y.T @ y2)

            x3 = np.linalg.solve(S, -3.0 * (Sprime @ x2))
            y3 = L @ x3
            reg3 = 2.0 * (3.0 * (y1.T @ y2) + (y.T @ y3))

            return float(reg1), float(reg2), float(reg3)

        reg1_direct, reg2_direct, reg3_direct = regularization_derivatives(lam)
        assert np.allclose(tf.regularization_term_derivative(lam, order=1, reciprocate=False), reg1_direct, atol=1e-8)
        assert np.allclose(tf.regularization_term_derivative(lam, order=2, reciprocate=False), reg2_direct, atol=1e-8)
        assert np.allclose(tf.regularization_term_derivative(lam, order=3, reciprocate=False), reg3_direct, atol=1e-6)

        # Reciprocal parameterization: lam = 1 / beta
        beta = 1.0 / lam

        # Chain-rule conversion for derivatives of g(beta) = f(1/beta)
        lam_prime = -1.0 / (beta**2)
        lam_double = 2.0 / (beta**3)
        lam_triple = -6.0 / (beta**4)

        g1 = f1_direct * lam_prime
        g2 = f2_direct * (lam_prime**2) + f1_direct * lam_double
        g3 = f3_direct * (lam_prime**3) + 3 * f2_direct * lam_prime * lam_double + f1_direct * lam_triple

        assert np.allclose(tf.data_fidelity_derivative(beta, order=1, reciprocate=True), g1, atol=1e-8)
        assert np.allclose(tf.data_fidelity_derivative(beta, order=2, reciprocate=True), g2, atol=1e-8)
        assert np.allclose(tf.data_fidelity_derivative(beta, order=3, reciprocate=True), g3, atol=1e-6)

        # Regularization derivatives under reciprocal parameterization
        g1_reg = reg1_direct * lam_prime
        g2_reg = reg2_direct * (lam_prime**2) + reg1_direct * lam_double
        g3_reg = reg3_direct * (lam_prime**3) + 3 * reg2_direct * lam_prime * lam_double + reg1_direct * lam_triple

        assert np.allclose(tf.regularization_term_derivative(beta, order=1, reciprocate=True), g1_reg, atol=1e-8)
        assert np.allclose(tf.regularization_term_derivative(beta, order=2, reciprocate=True), g2_reg, atol=1e-8)
        assert np.allclose(tf.regularization_term_derivative(beta, order=3, reciprocate=True), g3_reg, atol=1e-6)
