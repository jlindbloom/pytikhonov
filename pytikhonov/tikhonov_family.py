import numpy as np
import math
from scipy.sparse.linalg import aslinearoperator

from easygsvd import gsvd as gsvd_func


class TikhonovFamily:
    """Represents the family of Tikhonov solutions for the given problem.

    btrue: the noise-free part of b.
    noise_var: the variance of the noise in b = btrue + e.
    """

    def __init__(self, A, L, b, d=None, gsvd=None, btrue=None, noise_var=None):

        # Bind
        self.A = A
        self.L = L
        assert A.shape[1] == L.shape[1], "A and L must have same number of columns!"
        self.N = A.shape[1]
        self.M = A.shape[0]
        self.K = L.shape[0]
        assert len(b) == A.shape[0], "b is incompatible with A!"
        self.b = b
        if d is None:
            self.d = np.zeros(L.shape[0])
        else:
            assert len(d) == self.K, "d is incompatible with L!"
            self.d = d

        # Get GSVD if not available already
        if gsvd is None:
            self.gsvd = gsvd_func(A, L)
        else:
            self.gsvd = gsvd

        # GSVD quantities
        self.Uhat = self.gsvd.Uhat
        self.Vhat = self.gsvd.Vhat
        self.U1 = self.gsvd.U1
        self.U2 = self.gsvd.U2
        self.X1 = self.gsvd.X1
        self.X2 = self.gsvd.X2
        self.X3 = self.gsvd.X3
        self.V2 = self.gsvd.V2
        self.V3 = self.gsvd.V3
        self.c_check = self.gsvd.c_check
        self.s_check = self.gsvd.s_check
        self.gamma_check = self.gsvd.gamma_check
        self.n_L = self.gsvd.n_L
        self.n_A = self.gsvd.n_A
        self.r_A = self.gsvd.r_A
        self.r_L = self.gsvd.r_L
        self.r_cap = self.N - self.n_A - self.n_L

        # Other things we can compute once and save for later
        self.U1tb = self.U1.T @ self.b
        self.X1U1tb = self.X1 @ self.U1tb
        self.V3td = self.V3.T @ self.d
        self.X3V3td = self.X3 @ self.V3td
        self.U2tb = self.U2.T @ self.b
        self.V2td = self.V2.T @ self.d
        self.Uhattb = self.Uhat.T @ self.b
        self.Vhattd = self.Vhat.T @ self.d
        self.UhatUhattb = self.Uhat @ self.Uhattb
        self.U2U2tb = self.U2 @ self.U2tb
        self.UperpUperptb = self.UhatUhattb - self.b
        self.b_hat_perp_norm_squared = np.linalg.norm(self.b - self.UhatUhattb)**2
        self.d_hat_perp_norm_squared = np.linalg.norm(self.d - (self.Vhat @ self.Vhattd))**2
        self.squared_term = (self.U2tb - self.gamma_check * self.V2td)**2
        self.squared_term_rev = ((1.0/self.gamma_check)*self.U2tb - self.V2td)**2

        # If we pass btrue and noise_var?
        self.btrue = btrue
        if self.btrue is None:
            self.e = None
            self.noise_var = noise_var
        else:
            self.e = self.b - self.btrue
            if noise_var is not None:
                self.noise_var = noise_var
            else:
                self.noise_var = np.var(self.e)

        # Other
        if self.btrue is not None:
            self.Uhattbtrue = self.Uhat.T @ self.btrue
            self.expected_b_hat_perp_norm_squared = ( np.linalg.norm(self.btrue - (self.Uhat @ self.Uhattbtrue))**2 ) + self.noise_var*(self.M - self.gsvd.r_A)
            self.U2tbtrue = self.U2.T @ self.btrue
            self.expected_squared_term = ( (self.U2tbtrue - self.gamma_check * self.V2td)**2 ) + self.noise_var
            self.X1U1tbtrue = self.X1 @ ( self.U1.T @ self.btrue )



    def solve(self, regparam, reciprocate=False, expectation=False):
        """
        Evaluate the Tikhonov solution x_λ via the GSVD expression.
        """

        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."

        # Common base term
        if not expectation:
            base = self.X1U1tb + self.X3V3td
        else:
            base = self.X1U1tbtrue + self.X3V3td

        if reciprocate:
            lam = 1.0/np.asarray(regparam)
        else:
            lam = np.asarray(regparam)

        # if just one
        if lam.ndim == 0:
            lam = lam.item()
            denom = self.gamma_check**2 + lam

            result = base.copy()
            result += self.X2 @ ((lam / (self.s_check * denom)) * self.V2td)
            if not expectation:
                result += self.X2 @ ((self.gamma_check / (self.s_check * denom)) * self.U2tb)
            else:
                result += self.X2 @ ((self.gamma_check / (self.s_check * denom)) * self.U2tbtrue)
            return result  # 1D

        # if many
        lam = lam.astype(getattr(self.gamma_check, "dtype", float), copy=False)
        denom = (self.gamma_check[:, None] ** 2) + lam[None, :]
        s = self.s_check[:, None]                                    
        coeffV = (lam[None, :] / (s * denom)) * self.V2td[:, None] 
        if not expectation:
            coeffU = (self.gamma_check[:, None] / (s * denom)) * self.U2tb[:, None] 
        else:
            coeffU = (self.gamma_check[:, None] / (s * denom)) * self.U2tbtrue[:, None] 
        batched = self.X2 @ (coeffV + coeffU)    
        result = base[:, None] + batched
        return result



    def data_fidelity(self, regparam, reciprocate=False, expectation=False):
        """
        Evaluate the data fidelity term ||A x_λ - b||_2^2 via the GSVD expression.
        """

        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."

        # first term
        if not expectation:
            base = self.b_hat_perp_norm_squared
        else:
            base = self.expected_b_hat_perp_norm_squared
        
        # handle regparam
        rp = np.asarray(regparam)
        if reciprocate:
            lambdah = 1.0 / rp
        else:
            lambdah = rp

        # scalar case
        if np.ndim(lambdah) == 0:
            lam = float(lambdah)
            denom = self.gamma_check**2 + lam
            if not expectation:
                inner = ( (lam / denom)**2 )*( ( self.U2tb - self.gamma_check*self.V2td )**2 )
            else:
                inner = ( (lam / denom)**2 )*( ( ( self.U2tbtrue - self.gamma_check*self.V2td )**2 ) + self.noise_var )
            return base + np.sum(inner)
            

        # batched case
        lam = np.asarray(lambdah, dtype=getattr(self.gamma_check, "dtype", float))
        denom = (self.gamma_check[:, None]**2) + lam[None, :]
        if not expectation:
            inner = ( (lam[None,:] / denom)**2 )*(( self.U2tb[:,None] - self.gamma_check[:, None]*self.V2td[:,None] )**2)
        else:
            inner = ( (lam[None,:] / denom)**2 )*( (( self.U2tbtrue[:,None] - self.gamma_check[:, None]*self.V2td[:,None] )**2) + self.noise_var )

        vals = base + np.sum(inner, axis=0)
        
        return vals
    


    def regularization_term(self, regparam, reciprocate=False, expectation=False):
        """
        Evaluate the prior term ||L x_λ - d||_2^2 via the GSVD expression.
        """

        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."

        # Constant term (independent of λ): sum over the "K" part
        base = self.d_hat_perp_norm_squared
        
        # handle regparam
        rp = np.asarray(regparam)
        lambdah = (1.0 / rp) if reciprocate else rp

        # scalar case
        if np.ndim(lambdah) == 0:
            lam = float(lambdah)
            denom = self.gamma_check**2 + lam
            if not expectation:
                inner = (( (self.gamma_check**2)/denom )**2)*( ( ( self.U2tb - self.gamma_check*self.V2td )**2 ) / (self.gamma_check**2) )
            else:
                inner = (( (self.gamma_check**2)/denom )**2)*( ( ( ( self.U2tbtrue - self.gamma_check*self.V2td )**2 ) + self.noise_var )/ (self.gamma_check**2) )
            
            return base + np.sum(inner)

        # batched case
        lam = np.asarray(lambdah, dtype=getattr(self.gamma_check, "dtype", float)) 
        denom = (self.gamma_check[:, None]**2) + lam[None, :] 

        if not expectation:
            inner = ( ( (self.gamma_check[:,None]**2)/ denom )**2)*( ( ( self.U2tb[:,None] - self.gamma_check[:,None]*self.V2td[:,None] )**2 ) / (self.gamma_check[:,None]**2)  )
        else:
            inner = ( ( (self.gamma_check[:,None]**2)/ denom )**2)*( ( ( self.U2tbtrue[:,None] - self.gamma_check[:,None]*self.V2td[:,None] )**2 + self.noise_var ) / (self.gamma_check[:,None]**2)  )

        vals = base + np.sum(inner, axis=0)

        return vals
    



    def lcurve(self, regparams, f=None, g=None, reciprocate=False, expectation=False):
        r"""Evaluates the L-curve data ( 0.5*log( || A x_{\lambda} - b ||_2^2 ), 0.5*log( || L x_{\lambda} - d ||_2 ) ) for given lambdahs.
        """

        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."

        if f is None: f = 0.0
        if g is None: g = 0.0

        rho_hat = np.log(self.data_fidelity(regparams, reciprocate=reciprocate, expectation=expectation) + f  )
        eta_hat = np.log(self.regularization_term(regparams, reciprocate=reciprocate, expectation=expectation) + g)

        return rho_hat/2.0, eta_hat/2.0
    

    
    def lcurve_curvature(self, regparams, f=None, g=None, reciprocate=False, expectation=False):
        """Evaluates the curvature of the L-curve 
        """

        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."
        assert not reciprocate, "cannot use reciprocate here"

        rho_hat, eta_hat = self.lcurve(regparams, f=f, g=g, reciprocate=False, expectation=expectation)
        rho_hat *= 2.0 # need to multiply by 2 to get rho_hat from lcurve
        eta_hat *= 2.0 # need to multiply by 2 to get eta_hat from lcurve
        rho = np.exp(rho_hat)
        eta = np.exp(eta_hat)

        rho_prime = self.data_fidelity_derivative(regparams, order=1, reciprocate=False, expectation=expectation)
        rho_prime_prime = self.data_fidelity_derivative(regparams, order=2, reciprocate=False, expectation=expectation)
        eta_prime = self.regularization_term_derivative(regparams, order=1, reciprocate=False, expectation=expectation)
        eta_prime_prime = self.regularization_term_derivative(regparams, order=2, reciprocate=False, expectation=expectation)
        
        rho_hat_prime = rho_prime/rho
        eta_hat_prime = eta_prime/eta

        rho_hat_prime_prime = ((rho*rho_prime_prime) - (rho_prime**2))/(rho**2)
        eta_hat_prime_prime = ((eta*eta_prime_prime) - (eta_prime**2))/(eta**2)

        curvature = 2*( (rho_hat_prime*eta_hat_prime_prime) - (rho_hat_prime_prime*eta_hat_prime) )/( np.power(  (rho_hat_prime)**2 + (eta_hat_prime)**2  , 1.5 ) )
        
        return curvature
    
    

    def lcurve_data(self, regparams, f=None, g=None, reciprocate=False, expectation=False):
        """Evaluates the points, slope, second derivative, and curvature of the L-curve at the regparams.
        """

        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."
        assert not reciprocate, "cannot use reciprocate here"

        if f is None: f = 0.0
        if g is None: g = 0.0

        rho = self.data_fidelity(regparams, reciprocate=reciprocate, expectation=expectation) + f 
        eta = self.regularization_term(regparams, reciprocate=reciprocate, expectation=expectation) + g
        rho_hat = np.log(rho)
        eta_hat = np.log(eta)

        rho_prime = self.data_fidelity_derivative(regparams, order=1, reciprocate=False, expectation=expectation)
        rho_prime_prime = self.data_fidelity_derivative(regparams, order=2, reciprocate=False, expectation=expectation)
        eta_prime = self.regularization_term_derivative(regparams, order=1, reciprocate=False, expectation=expectation)
        eta_prime_prime = self.regularization_term_derivative(regparams, order=2, reciprocate=False, expectation=expectation)
        
        rho_hat_prime = rho_prime/rho
        eta_hat_prime = eta_prime/eta

        rho_hat_prime_prime = ((rho*rho_prime_prime) - (rho_prime**2))/(rho**2)
        eta_hat_prime_prime = ((eta*eta_prime_prime) - (eta_prime**2))/(eta**2)

        #log_curvature = safe_log(2*( (rho_hat_prime*eta_hat_prime_prime) - (rho_hat_prime_prime*eta_hat_prime) )) - 1.5*safe_log( (rho_hat_prime)**2 + (eta_hat_prime)**2 )
        slope = eta_hat_prime/rho_hat_prime
        second_deriv = 2*( (eta_hat_prime_prime*rho_hat_prime) - (eta_hat_prime*rho_hat_prime_prime) )/np.power(rho_hat_prime, 3)
        curvature = 2*( (rho_hat_prime*eta_hat_prime_prime) - (rho_hat_prime_prime*eta_hat_prime) )/( np.power(  (rho_hat_prime)**2 + (eta_hat_prime)**2  , 1.5 ) )
    
        ldata = {
            "rho_hat": rho_hat,
            "eta_hat": eta_hat,
            "eta": eta,
            "rho": rho,
            "slope": slope,
            "second_deriv": second_deriv,
            "curvature": curvature,
        }        

        return ldata



    def gcv_objective(self, regparams, reciprocate=False):
        """Evaluates the GCV objective function.
        """

        lambdah = regparams

        bperp_norm_squared = np.linalg.norm(self.b - (self.Uhat @ self.Uhattb))**2
        numerator = bperp_norm_squared + ((( lambdah/(self.gamma_check**2 + lambdah) )**2)*((self.U2tb - self.V2td*self.gamma_check)**2)).sum()
        denominator = ( (1.0/self.M)*(  self.M - self.n_L - ((self.gamma_check**2)/(self.gamma_check**2 + lambdah)).sum()  ) )**2
        #denominator = ( (1.0/self.M)*(  self.M - self.n_L - ((self.gamma_check**2)/(self.gamma_check**2 + lambdah)).sum()  ) )**2


        return numerator/denominator



    def dp_functional(self, regparams, tr_noise_cov, tau=1.01, reciprocate=False):

        phi = self.data_fidelity(regparams, reciprocate=reciprocate) - ( (tau**2)*(tr_noise_cov) )
        
        return phi

    
    
    def data_fidelity_derivative(self, regparam, order=1, reciprocate=False, expectation=False):
        """
        Evaluates the order-th derivative of the data fidelity term ||A x_λ - b||_2^2 via the GSVD expression.
        """

        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."

        if order < 0 or int(order) != order:
            raise ValueError("`order` must be a non-negative integer.")

        if not reciprocate:
            lambdah = np.asarray(regparam)
            gamma = self.gamma_check
            gamma2 = gamma**2
            if not expectation:
                squared_term = self.squared_term
            else:
                squared_term = self.expected_squared_term
            coef = ((-1)**order) * math.factorial(order)

            if np.ndim(lambdah) == 0:
                lam = float(lambdah)
                denom = gamma2 + lam
                num = coef * gamma2 * (gamma2 * (order - 1.0) - 2.0 * lam)
                return float(np.sum((num / (denom**(order + 2))) * squared_term))

            lam = np.asarray(lambdah, dtype=gamma2.dtype)
            denom = gamma2[:, None] + lam[None, :]
            num = coef * gamma2[:, None] * (gamma2[:, None] * (order - 1.0) - 2.0 * lam[None, :])
            return np.sum((num / (denom**(order + 2))) * squared_term[:, None], axis=0)
        
        else: 
            beta = np.asarray(regparam)
            gamma = self.gamma_check
            gamma2 = gamma**2
            if not expectation:
                squared_term = self.squared_term
            else:
                squared_term = self.expected_squared_term
            coef = ((-1)**order) * math.factorial(order+1)

            if np.ndim(beta) == 0:
                beta = float(beta)
                denom = 1 + (beta*gamma2)
                num = coef*np.power(gamma2, order)
                return float(np.sum((num / (denom**(order + 2))) * squared_term))

            denom =  1.0 + beta[None,:]*gamma2[:, None]
            num = coef * np.power(gamma2[:, None], order)
            return np.sum((num / (denom**(order + 2))) * squared_term[:, None], axis=0)



    def regularization_term_derivative(self, regparam, order=1, reciprocate=False, expectation=False):
        """
        Evaluates the order-th derivative of the data fidelity term ||L x_λ - d||_2^2 via the GSVD expression.
        """

        assert not ( (self.btrue is None) and expectation ), "Must pass btrue to TikhonovFamily to compute expectation."

        if order < 0 or int(order) != order:
            raise ValueError("`order` must be a non-negative integer.")

        if not reciprocate:
            lambdah = np.asarray(regparam)
            gamma = self.gamma_check          
            gamma2 = gamma**2               
            if not expectation:
                squared_term = self.squared_term
            else:
                squared_term = self.expected_squared_term
            coef = ((-1)**order) * math.factorial(order + 1)

            # scalar case
            if np.ndim(lambdah) == 0:
                lam = float(lambdah)
                denom = gamma2 + lam                     
                value = np.sum((coef * gamma2 / (denom**(order + 2))) * squared_term)
                return float(value)

            # batched case
            lam = np.asarray(lambdah, dtype=gamma2.dtype)  
            denom = gamma2[:, None] + lam[None, :]  
            vals = np.sum((coef * gamma2[:, None] / (denom**(order + 2))) * squared_term[:, None], axis=0) 
            return vals
        
        else:
            beta = np.asarray(regparam)
            gamma = self.gamma_check          
            gamma2 = gamma**2               
            if not expectation:
                squared_term = self.squared_term
            else:
                squared_term = self.expected_squared_term
            coef = ((-1)**order) * math.factorial(order)

            # scalar case
            if np.ndim(beta) == 0:
                beta = float(beta)
                denom = 1.0 + (beta*gamma2)
                value = np.sum((coef * ( np.power(gamma2, order-1)*( order - 1 - (2*beta*gamma2)   ) ) / (denom**(order + 2))) * squared_term)
                return float(value)

            # batched beta
            denom = 1.0 + gamma2[:, None] * beta[None, :]        # (m, K)
            numer = (np.power(gamma2[:, None], order - 1) *
                    ((order - 1) - 2.0 * gamma2[:, None] * beta[None, :]))              # (m, K)
            vals = np.sum((coef * numer / (denom**(order + 2))) * squared_term[:, None], axis=0)
            
            return vals



    def T(self, regparam, reciprocate=False):
        """
        Degrees of freedom:
            T(λ) = tr(I_M - A (A^T A + λ L^T L)^(-1) A^T)
                = M - n_L - sum_i γ_i^2 / (γ_i^2 + λ)
        """
        rp = np.asarray(regparam)
        lam = (1.0 / rp) if reciprocate else rp

        gamma2 = self.gamma_check**2  # shape (m_mid,)

        # Scalar λ
        if np.ndim(lam) == 0:
            lam = float(lam)
            denom = gamma2 + lam
            s = float(np.sum(gamma2 / denom))
            return float(self.M - self.r_A + self.r_cap - s)

        # Batched λ (vectorized)
        lam = np.asarray(lam, dtype=gamma2.dtype)           # <-- no 'copy='
        denom = gamma2[:, None] + lam[None, :]              # (m_mid, K)
        s = np.sum(gamma2[:, None] / denom, axis=0)         # (K,)
        return self.M - self.r_A + self.r_cap - s
    


    def V(self, regparam, reciprocate=False):
        """
        ν(λ) = ||A x_λ − b||_2^2 / T(λ), with the unsquared-λ convention.
        Accepts a scalar or an array of λ (or β if reciprocate=True).
        Returns a scalar or a 1-D ndarray of the same length as regparam.
        """
        num = self.data_fidelity(regparam, reciprocate=reciprocate)
        den = self.T(regparam, reciprocate=reciprocate)

        # T(λ) > 0 for all λ ≥ 0 under the GSVD setup, so a plain division is fine.
        # Use np.divide for robustness if you prefer:
        # return np.divide(num, den, out=np.full_like(np.asarray(den, dtype=float), np.inf), where=den!=0)
        return num / den



    def gcv(self, regparam, reciprocate=False):
        """
        Computes the GCV functional.
        """

        return self.V(regparam,reciprocate=reciprocate)/self.T(regparam,reciprocate=reciprocate)



    def project(self, V, x_under=None):
        """
        Builds the projected Tikhonov family on the affine space x_under + col(V).
        """
        # Local import to avoid circular dependency at module load time.
        from .projected_tikhonov import ProjectedTikhonovFamily

        return ProjectedTikhonovFamily(
            aslinearoperator(self.A),
            aslinearoperator(self.L),
            V,
            self.b,
            d=self.d,
            x_under=x_under,
            btrue=self.btrue,
            noise_var=self.noise_var,
        )



    def data_residual(self, regparam, reciprocate=False):
        r"""
        Compute the residual A x_\lambda - b using the GSVD expression

            A x_\lambda - b
              = - U_2 diag( λ / (γ_i^2 + λ) ) U_2^T b
                + U_2 diag( λ γ_i / (γ_i^2 + λ) ) V_2^T d
                - U_\perp U_\perp^T b,

        where γ_i are the generalized singular values in ``self.gamma_check``.

        Parameters
        ----------
        regparam : float or array_like
            Regularization parameter λ. If ``reciprocate=True``, this is β and
            λ = 1 / β is used.
        reciprocate : bool, optional
            If True, interpret ``regparam`` as β = 1/λ.

        Returns
        -------
        residual : ndarray
            If ``regparam`` is scalar, returns an array of shape (M,)
            containing A x_λ - b.  If ``regparam`` is 1D with length K,
            returns an array of shape (M, K) whose j-th column is
            A x_{λ_j} - b.
        """

        # handle λ vs β = 1/λ
        rp = np.asarray(regparam)
        if reciprocate:
            lam = 1.0 / rp
        else:
            lam = rp

        gamma = self.gamma_check              # shape (r_int,)
        gamma2 = gamma**2
        U2 = self.U2                          # shape (M, r_int)
        U2tb = self.U2tb                      # shape (r_int,)
        V2td = self.V2td                      # shape (r_int,)

        # λ–independent term: -U_perp U_perp^T b
        # using U_perp U_perp^T b = b - Uhat Uhat^T b
        base = self.UperpUperptb  # shape (M,)

        # scalar λ
        if np.ndim(lam) == 0:
            lam = float(lam)
            denom = gamma2 + lam                      # shape (r_int,)

            w1 = lam / denom                          # λ / (γ^2 + λ)
            w2 = lam * gamma / denom                  # λ γ / (γ^2 + λ)

            term1 = - U2 @ (w1 * U2tb)                # shape (M,)
            term2 =   U2 @ (w2 * V2td)                # shape (M,)

            residual = base + term1 + term2
            return residual

        # batched λ: lam is 1D of length K
        lam = np.asarray(lam, dtype=gamma2.dtype)      # shape (K,)
        denom = gamma2[:, None] + lam[None, :]         # (r_int, K)

        w1 = lam[None, :] / denom                      # (r_int, K)
        w2 = (lam[None, :] * gamma[:, None]) / denom   # (r_int, K)

        term1 = - U2 @ (w1 * U2tb[:, None])            # (M, K)
        term2 =   U2 @ (w2 * V2td[:, None])            # (M, K)

        residual = base[:, None] + term1 + term2       # (M, K)
        return residual


