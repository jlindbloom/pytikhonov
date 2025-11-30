import numpy as np
from scipy.optimize import fminbound

from .projected_tikhonov import ProjectedTikhonovFamily



def gcvmin(tikh_family):
    """Minimizes the GCV objective.
    """

    # is tikh_family a ProjectedTikhonovFamily?
    if isinstance(tikh_family, ProjectedTikhonovFamily):
        projected = True
    else:
        projected = False

    gamma_sq_min = np.amin(tikh_family.gsvd.gamma_check)**2
    gamma_sq_max = np.amax(tikh_family.gsvd.gamma_check)**2
    gcv_obj_func = lambda x: np.log10(tikh_family.gcv( np.power(10.0, x) ))
    fmin_res = fminbound(gcv_obj_func, np.log10(gamma_sq_min)-2 , np.log10(gamma_sq_max)+2, xtol=1e-10, maxfun=int(1e5)  )
    gcv_opt_lambdah = np.power(10.0, fmin_res)

    # Compute some other things
    if not projected:
        x_lambdah = tikh_family.solve(gcv_opt_lambdah)
    else:
        z_lambdah, x_lambdah = tikh_family.solve(gcv_opt_lambdah)

    lambdahs = np.logspace( np.log10(gamma_sq_min)-2, np.log10(gamma_sq_max)+2, num=1000, base=10  )
    rho_hat_half, eta_hat_half = tikh_family.lcurve(gcv_opt_lambdah)
    rho_hat = 2.0*rho_hat_half
    eta_hat = 2.0*eta_hat_half
    rho = np.exp(rho_hat)
    eta = np.exp(eta_hat)
    gcv_vals = tikh_family.gcv(lambdahs)

    data = {
        "opt_lambdah": gcv_opt_lambdah,
        "opt_lambdah_val": tikh_family.gcv(gcv_opt_lambdah),
        "x_lambdah": x_lambdah,
        "lambdahs": lambdahs,
        "opt_rho_hat": rho_hat,
        "opt_eta_hat": eta_hat,
        "opt_rho": rho,
        "opt_eta": eta,
        "gcv_vals": gcv_vals,
        "gamma_sq_min": gamma_sq_min,
        "gamma_sq_max": gamma_sq_max,
        "gamma_sq_min_val": tikh_family.gcv(gamma_sq_min),
        "gamma_sq_max_val": tikh_family.gcv(gamma_sq_max),
    }

    if projected: data["z_lambdah"] = z_lambdah

    return data







