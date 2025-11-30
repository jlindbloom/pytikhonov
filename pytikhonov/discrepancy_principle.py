import numpy as np
from scipy import optimize

from .projected_tikhonov import ProjectedTikhonovFamily




def discrepancy_principle(tikh_family, delta=None, f=None, tau=1.01, beta0=0.0):
    """Implements the discrepancy principle root-finding method. Looks for the root of

        phi(lambda) = || A x_lambda - b ||_2^2 + f

    If the root does not ex

    delta: square root of the trace of the noise covariance. Overwrites noise_var if given.
    tau: safeguard parameter, about equal to 1.
    f: shift in the DP functional.
    beta0: initial condition for rootfinder. May not converge if greater than the root.
    """

    # is tikh_family a ProjectedTikhonovFamily?
    if isinstance(tikh_family, ProjectedTikhonovFamily):
        projected = True
    else:
        projected = False

    # handle delta
    if delta is None:
        assert tikh_family.noise_var is not None, "must provide estimate of delta = sqrt(tr(\Sigma_noise))."
        delta = np.sqrt(tikh_family.noise_var*tikh_family.M)

    if f is None: f = 0.0
    tau_sq_delta_sq = (tau*delta)**2
    phi = lambda beta: tikh_family.data_fidelity(beta, reciprocate=True) + f - tau_sq_delta_sq
    phiprime = lambda beta: tikh_family.data_fidelity_derivative(beta, order=1, reciprocate=True)

    phi_zero = tikh_family.b_hat_perp_norm_squared + tikh_family.squared_term.sum() + f - tau_sq_delta_sq
    phi_inf = tikh_family.b_hat_perp_norm_squared + f - tau_sq_delta_sq

    # other stuff for plots
    gamma_sq_min = np.amin(tikh_family.gsvd.gamma_check)**2
    gamma_sq_max = np.amax(tikh_family.gsvd.gamma_check)**2
    phi2 = lambda lambdah: tikh_family.data_fidelity(lambdah, reciprocate=False) + f - tau_sq_delta_sq
    lambdahs = np.logspace( np.log10(gamma_sq_min)-2, np.log10(gamma_sq_max)+2, num=1000, base=10  )
    

    # check if there is a root?
    if not ( ( phi_zero > 0  ) and ( phi_inf < 0 ) ):
        print("DP rootfinder broke, there is no root! Returning lambdah = 1e-12")
        opt_lambdah = 1e-12
        if not projected:
            x_lambdah = tikh_family.solve(opt_lambdah)
        else:
            z_lambdah, x_lambdah = tikh_family.solve(opt_lambdah)
        rho_hat, eta_hat = tikh_family.lcurve(opt_lambdah)
        rho_hat *= 2
        eta_hat *= 2
        rho = np.exp(rho_hat)
        eta = np.exp(eta_hat)
        dp_vals = phi2(lambdahs)

        data = {
            "opt_lambdah": opt_lambdah,
            "opt_lambdah_val": phi2(opt_lambdah),
            "x_lambdah": x_lambdah,
            "n_iters": None,
            "converged": False,
            "lambdahs": lambdahs,
            "opt_rho": rho,
            "opt_eta": eta,
            "opt_rho_hat": rho_hat,
            "opt_eta_hat": eta_hat,
            "dp_vals": dp_vals,
            "gamma_sq_min": gamma_sq_min,
            "gamma_sq_max": gamma_sq_max,
            "gamma_sq_min_val": phi2(gamma_sq_min),
            "gamma_sq_max_val": phi2(gamma_sq_max),
        }

        if projected: data["z_lambdah"] = z_lambdah

        return data
    
    # wraph phi and phi prime to handle beta = 0 differently.
    def _phi(beta):
        if beta == 0.0:
            return phi_zero
        else:
            return phi(beta)
        
    def _phiprime(beta):
        if beta == 0.0:
            result = -2*((tikh_family.gamma_check**2)*tikh_family.squared_term).sum()
            return result
        else:
            return phiprime(beta)
        
    # Using Newton’s method with derivative
    opt_beta, root_info = optimize.newton(_phi, x0=beta0, fprime=_phiprime, maxiter=500, full_output=True)

    # lambdah and x_lambdah
    opt_lambdah = 1.0/opt_beta
    if not projected:
        x_lambdah = tikh_family.solve(opt_lambdah)
    else:
        z_lambdah, x_lambdah = tikh_family.solve(opt_lambdah)
    n_iters = root_info.iterations
    converged = root_info.converged
    assert converged, "rootfinder did not converge!"
    
    # other stuff for plots
    rho_hat_half, eta_hat_half = tikh_family.lcurve(opt_lambdah)
    rho_hat = 2*rho_hat_half
    eta_hat = 2*eta_hat_half
    rho = np.exp(rho_hat)
    eta = np.exp(eta_hat)
    dp_vals = phi2(lambdahs)

    data = {
        "opt_lambdah": opt_lambdah,
        "opt_lambdah_val": phi2(opt_lambdah),
        "x_lambdah": x_lambdah,
        "n_iters": n_iters,
        "converged": converged,
        "lambdahs": lambdahs,
        "opt_rho": rho,
        "opt_eta": eta,
        "opt_rho_hat": rho_hat,
        "opt_eta_hat": eta_hat,
        "dp_vals": dp_vals,
        "gamma_sq_min": gamma_sq_min,
        "gamma_sq_max": gamma_sq_max,
        "gamma_sq_min_val": phi2(gamma_sq_min),
        "gamma_sq_max_val": phi2(gamma_sq_max),
    }

    if projected: data["z_lambdah"] = z_lambdah
    
    return data
