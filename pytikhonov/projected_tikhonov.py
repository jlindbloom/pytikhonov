import numpy as np
from easygsvd import gsvd
from scipy.linalg import qr, qr_insert
from easygsvd import gsvd as gsvd_func

from .tikhonov_family import TikhonovFamily



class ProjectedTikhonovFamily:
    r"""Represents the projection of a Tikhonov problem onto an affine subspace.

        x_lambda = \arg\min_{x \in x_{\text{under}} + \operatorname{col}(V)} \| A x - b \|_2^2 + \lambda \| L x - d \|_2^2.

    """
    def __init__(self, A, L, V, b, d=None, x_under=None, b_under=None, d_under=None, AV=None, LV=None, btrue=None, noise_var=None):
        
        self.A = A
        self.L = L
        self.M = self.A.shape[0]
        self.N = self.A.shape[1]
        self.K = self.L.shape[0]
        self.noise_var = noise_var
        self.V = V
        self.b = b
        if d is None: d = np.zeros(self.L.shape[0])
        self.d = d
        if x_under is None: x_under = np.zeros(self.A.shape[1])
        self.x_under = x_under
        if b_under is None: b_under = self.b - ( self.A.matvec( self.x_under ) )
        if d_under is None: d_under = self.d - ( self.L.matvec( self.x_under ) )
        self.b_under = b_under
        self.d_under = d_under
        self.btrue = btrue

        # If AV and LV not passed, computed
        if AV is None:
            self.AV = A @ V
        else:
            self.AV = AV
        
        if LV is None:
            self.LV = L @ V
        else:
            self.LV = LV

        # Compute GSVD and projected Tikhonov family
        self.gsvd = gsvd_func(self.AV, self.LV, full_matrices=False)
        self._tf = TikhonovFamily(self.AV, self.LV, self.b_under, self.d_under, gsvd=self.gsvd, btrue=self.btrue, noise_var=self.noise_var)

        # Bind some quantities
        self.b_hat_perp_norm_squared = self._tf.b_hat_perp_norm_squared
        self.d_hat_perp_norm_squared = self._tf.d_hat_perp_norm_squared
        self.squared_term = self._tf.squared_term
        self.squared_term_rev = self._tf.squared_term_rev
        self.gamma_check = self.gsvd.gamma_check



    def solve(self, lambdah, reciprocate=False):
        """Solves the projected problem for fixed lambdah.
        """

        z_lambdah = self._tf.solve(lambdah, reciprocate=reciprocate)
        # Broadcast x_under when solving for multiple lambdas at once
        if np.ndim(z_lambdah) == 1:
            x_lambdah = self.x_under + (self.V @ z_lambdah)
        else:
            x_lambdah = self.x_under[:, None] + (self.V @ z_lambdah)

        return z_lambdah, x_lambdah



    def lcurve(self, regparams, f=None, g=None, reciprocate=False, expectation=False):
        r"""Evaluates the projected L-curve data ( 0.5*log( || A x_{\lambda} - b ||_2^2 ), 0.5*log( || L x_{\lambda} - d ||_2 ) ) for given lambdahs.
        """

        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."

        if f is None: f = 0.0
        if g is None: g = 0.0

        rho_hat = np.log(self._tf.data_fidelity(regparams, reciprocate=reciprocate, expectation=expectation) + f  )
        eta_hat = np.log(self._tf.regularization_term(regparams, reciprocate=reciprocate, expectation=expectation) + g)

        return rho_hat/2.0, eta_hat/2.0
    

    
    def lcurve_curvature(self, regparams, f=None, g=None):
        """Evaluates the curvature of the projected L-curve for given lambdahs.
        """

        return self._tf.lcurve_curvature(regparams, f=f, g=g)
    


    def lcurve_data(self, regparams, f=None, g=None, reciprocate=False, expectation=False):
        """Computes data about the projected L-curve.
        """
        return self._tf.lcurve_data(regparams, f=f, g=g, reciprocate=reciprocate, expectation=expectation)



    def data_fidelity(self, regparams, reciprocate=False, expectation=False):
        r"""Evaluates the full-scale data fidelity || A x_{\lambda} - b ||_2^2 for given lambdahs.
        """

        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."

        return self._tf.data_fidelity(regparams, reciprocate=reciprocate, expectation=expectation)



    def regularization_term(self, regparams, reciprocate=False, expectation=False):
        r"""Evaluates the full-scale regularization penalty || L x_{\lambda} - d ||_2^2 for given lambdahs.
        """
        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."

        return self._tf.regularization_term(regparams, reciprocate=reciprocate, expectation=expectation)
        
    

    def data_fidelity_derivative(self, regparams, order=1, reciprocate=False, expectation=False):
        r"""Evaluates the derivative of the full-scale data fidelity || A x_{\lambda} - b ||_2^2 for given lambdahs.
        """

        return self._tf.data_fidelity_derivative(regparams, order=order, reciprocate=reciprocate, expectation=expectation)



    def regularization_term_derivative(self, regparams, order=1, reciprocate=False, expectation=False):
        r"""Evaluates the derivative of the full-scale regularization penalty || L x_{\lambda} - d ||_2^2 for given lambdahs.
        """

        return self._tf.regularization_term_derivative(regparams, order=order, reciprocate=reciprocate, expectation=expectation)



    def gcv(self, regparams):
        """Evaluates the GCV functional for given lambdahs.
        """

        return self._tf.gcv(regparams)








