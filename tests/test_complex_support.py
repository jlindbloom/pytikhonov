import numpy as np

from pytikhonov import TikhonovFamily
from pytikhonov.util import adjoint


def test_complex_residuals_and_gradient():
    rng = np.random.default_rng(2024)

    m, n, k = 12, 8, 10
    A = rng.standard_normal((m, n)) + 1j * rng.standard_normal((m, n))
    L = rng.standard_normal((k, n)) + 1j * rng.standard_normal((k, n))

    x_true = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    d = rng.standard_normal(k) + 1j * rng.standard_normal(k)

    b_true = A @ x_true
    noise = 0.01 * (rng.standard_normal(m) + 1j * rng.standard_normal(m))
    b = b_true + noise

    noise_var = np.real(np.vdot(noise, noise)) / noise.size
    tf = TikhonovFamily(A, L, b, d=d, btrue=b_true, noise_var=noise_var)

    lam = 1e-2

    # Residual norm consistency
    r = tf.data_residual(lam)
    r_norm_sq = np.real(np.vdot(r, r))
    assert np.allclose(tf.data_fidelity(lam), r_norm_sq, atol=1e-10)

    # Regularization norm consistency
    x = tf.solve(lam)
    y = L @ x - d
    y_norm_sq = np.real(np.vdot(y, y))
    assert np.allclose(tf.regularization_term(lam), y_norm_sq, atol=1e-10)

    # Stationarity with Hermitian gradients
    grad = adjoint(A) @ (A @ x - b) + lam * (adjoint(L) @ (L @ x - d))
    assert np.linalg.norm(grad) <= 1e-8 * (1.0 + np.linalg.norm(b))

    # Squared terms are real and nonnegative
    assert np.all(np.real(tf.squared_term) >= -1e-12)
    assert np.allclose(np.imag(tf.squared_term), 0.0, atol=1e-12)
