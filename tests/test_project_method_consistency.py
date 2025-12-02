import numpy as np

from scipy.sparse.linalg import aslinearoperator

from pytikhonov.tikhonov_family import TikhonovFamily
from pytikhonov.projected_tikhonov import ProjectedTikhonovFamily


def test_project_method_matches_direct_construction():
    rng = np.random.default_rng(0)

    def make_matrices():
        # Random dimensions with trivial common kernel (stacked full column rank)
        N = 20
        mA = int(rng.integers(N + 3, N + 15))
        while True:
            rA = int(rng.integers(1, N + 1))
            rL = int(rng.integers(1, N + 1))
            if rA + rL >= N:
                break
        mL = int(rng.integers(max(rL, N // 2), N + 10))

        Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
        A_basis = Q[:, :rA]

        needed = N - rA
        complement = list(range(rA, N))
        needed_idx = complement[:needed]
        extra = rL - needed
        if extra > 0:
            pool = np.arange(N)
            available = pool[np.isin(pool, needed_idx, invert=True)]
            extra_idx = rng.choice(available, size=extra, replace=False).tolist()
            L_idx = needed_idx + extra_idx
        else:
            L_idx = needed_idx
        L_basis = Q[:, L_idx]

        A = rng.standard_normal((mA, rA)) @ A_basis.T
        L = rng.standard_normal((mL, rL)) @ L_basis.T

        assert np.linalg.matrix_rank(A) == rA
        assert np.linalg.matrix_rank(L) == rL
        assert np.linalg.matrix_rank(np.vstack([A, L])) == N
        return A, L

    lambdas = [0.1, 1.0, 10.0]
    lambdas_vec = np.asarray(lambdas)

    for _ in range(5):
        A, L = make_matrices()
        b = rng.standard_normal(A.shape[0])
        d = rng.standard_normal(L.shape[0])

        tf = TikhonovFamily(A, L, b, d=d)

        # Random subspace dimension between 1 and N-1
        k = int(rng.integers(1, tf.N))
        V = rng.standard_normal((tf.N, k))
        x_under = rng.standard_normal(tf.N)

        # Direct construction of the projected family
        ptf_direct = ProjectedTikhonovFamily(aslinearoperator(A), aslinearoperator(L), V, b, d=d, x_under=x_under)
        # Via the TikhonovFamily helper
        ptf_method = tf.project(V, x_under)

        # Check a few lambda values (scalar and vector)
        for lam in lambdas:
            z_direct, x_direct = ptf_direct.solve(lam)
            z_method, x_method = ptf_method.solve(lam)
            np.testing.assert_allclose(z_direct, z_method, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(x_direct, x_method, rtol=1e-12, atol=1e-12)

        # Also verify the batched solve matches
        z_direct_vec, x_direct_vec = ptf_direct.solve(lambdas_vec)
        z_method_vec, x_method_vec = ptf_method.solve(lambdas_vec)
        np.testing.assert_allclose(z_direct_vec, z_method_vec, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(x_direct_vec, x_method_vec, rtol=1e-12, atol=1e-12)
