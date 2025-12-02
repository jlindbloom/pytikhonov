import numpy as np

from pytikhonov.tikhonov_family import TikhonovFamily


def test_gcv_matches_direct_trace():
    rng = np.random.default_rng(2024)

    def make_matrices():
        # Random dimensions and ranks with trivial common kernel
        N = 100
        mA = int(rng.integers(N + 5, N + 31))  # rows of A
        while True:
            rA = int(rng.integers(1, N + 1))
            rL = int(rng.integers(1, N + 1))
            if rA + rL >= N:
                break
        mL = int(rng.integers(max(rL, N // 2), max(rL, N) + 11))

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

    lambdas = np.logspace(-7, 7, num=25)
    
    repeats = 25

    for _ in range(repeats):
        A, L = make_matrices()
        b = rng.standard_normal(A.shape[0])
        d = rng.standard_normal(L.shape[0])

        tf = TikhonovFamily(A, L, b, d=d)

        # GCV from the implementation
        gcv_tf = tf.gcv(lambdas)

        # Direct GCV via the definition: ||A x_λ - b||^2 / (trace(I - A(AᵀA + λLᵀL)⁻¹Aᵀ))^2
        AtA = A.T @ A
        LtL = L.T @ L
        num = tf.data_fidelity(lambdas)

        traces = []
        for lam in lambdas:
            K = AtA + lam * LtL
            AKinvAT = A @ np.linalg.solve(K, A.T)
            traces.append(np.trace(np.eye(A.shape[0]) - AKinvAT))
        traces = np.asarray(traces)

        gcv_direct = num / (traces**2)

        np.testing.assert_allclose(gcv_tf, gcv_direct, rtol=1e-8)
