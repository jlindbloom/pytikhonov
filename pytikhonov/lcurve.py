import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar


from .util import interior_extremum, find_positive_bump, tallest_true_peak_or_plateau_edge
from .projected_tikhonov import ProjectedTikhonovFamily



def argmax_x_exact(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    # lexsort sorts by the last key first (primary). Here: primary=y, secondary=x.
    idx = np.lexsort((x, y))[-1]   # last is the max y; ties resolved by max x
    return x[idx], idx




def lcorner(tikh_family, f=None, g=None, lambdah_min=1e-12, lambdah_max=1e12, num_points=1000, bounds=None, smooth=None, kind="univariate", method="max_curvature"):
    """Finds the corner of the lcurve. Returns (lambdah, x_lambdah).
    """

    # is tikh_family a ProjectedTikhonovFamily?
    if isinstance(tikh_family, ProjectedTikhonovFamily):
        projected = True
    else:
        projected = False

    valid_methods = ["max_curvature", "min_dist_origin"]
    #method = "min_dist_origin"
    #method = ""
    

    # set f and g to 0 if not passed
    if f is None: f = 0.0
    if g is None: g = 0.0
    gamma_sq_min = np.amin(tikh_family.gsvd.gamma_check)**2
    gamma_sq_max = np.amax(tikh_family.gsvd.gamma_check)**2

    # get the limiting points of the L curve (ok if we get error here)
    with np.errstate(divide='ignore', invalid='ignore'):
        upper_left_point = ( 0.5*np.log(  tikh_family.b_hat_perp_norm_squared + f ), 0.5*np.log(  tikh_family.d_hat_perp_norm_squared + tikh_family.squared_term_rev.sum() + g )   ) # small lambdah limit
        lower_right_point = ( 0.5*np.log(  tikh_family.b_hat_perp_norm_squared + tikh_family.squared_term.sum() + f ),  0.5*np.log(  tikh_family.d_hat_perp_norm_squared + g )  ) # large lambdah limit
    
    # define lambdahs to search over
    log10_lambdah_min = np.log10(lambdah_min)
    log10_lambdah_max = np.log10(lambdah_max)
    lambdahs = np.logspace(start=log10_lambdah_min, stop=log10_lambdah_max, base=10, num=num_points)

    # Compute rhos, etas, and curvature
    ldata = tikh_family.lcurve_data(lambdahs, f=f, g=g)
    rho_hat = ldata["rho_hat"]
    eta_hat = ldata["eta_hat"]
    rho = ldata["rho"]
    eta = ldata["eta"]
    slopes = ldata["slope"]
    second_derivs = ldata["second_deriv"]
    curvatures = ldata["curvature"]

    origin_x, origin_y = -30, 30
    distances_to_origin = ( (rho_hat/2.0)  )**2 + ( (eta_hat/2.0)  )**2
    distances_to_origin = rho + lambdahs*eta



    if method == "max_curvature":

        #try:

        
        # x = np.log10(lambdahs)
        # y = curvatures.copy()

        xy = tallest_true_peak_or_plateau_edge(
            np.log10(lambdahs), curvatures,
            s=0.0,
            edge_margin=0.0,
            nsamp=max(40000, 30*len(lambdahs)),

            prominence=0.02*np.ptp(curvatures),
            curvature_tol=0.0,
            min_drop_left=0.0,
            min_drop_right=0.02*np.ptp(curvatures),

            plateau_override=True,
            flat_slope_rel=1e-3,
            flat_min_frac=0.01,
            fall_slope_rel=5e-2,
            fall_drop_rel=0.02,
            fall_window_frac=0.01,
            plateau_higher_rel=0.0,

            # NEW fallback that scans from x=-12 to first 1% drop
            fallback_enable=True,
            fallback_probe_start=-12.0,
            fallback_drop_rel=0.01,
            fallback_step_frac=1e-4,
        )




        log10_opt_lambdah, _ = xy
        #log10_opt_lambdah, _ = tallest_true_peak( np.log10(lambdahs), curvatures)
        opt_lambdah = np.power(10.0, log10_opt_lambdah)
        
        # except:
        #     print("No peaks found, setting lambdah = 1e-12.")
        #     opt_lambdah = 1e-12

            

        # # look at discrete points to find initial guess of the maximizer of the curvature. then refine
        # opt_lambdah_guess, guess_idx = argmax_x_exact(lambdahs, curvatures)
        # try:
        #     minimization_result = minimize_scalar(lambda x: -tikh_family.lcurve_curvature(x, f=f, g=g), method='brent', bracket=(lambdahs[guess_idx-1], lambdahs[guess_idx+1]) )
        #     opt_lambdah = minimization_result.x
        # except:
        #     opt_lambdah = opt_lambdah_guess
        
        if not projected:
            x_lambdah = tikh_family.solve(opt_lambdah)
        else:
            z_lambdah, x_lambdah = tikh_family.solve(opt_lambdah)
    
    elif method == "min_dist_origin":
        
        i = np.argmin(distances_to_origin)
        opt_lambdah = lambdahs[i]

    else:
        raise NotImplementedError 
    
    # solve for x_lambdah
    if not projected:
        x_lambdah = tikh_family.solve(opt_lambdah)
    else:
        z_lambdah, x_lambdah = tikh_family.solve(opt_lambdah)


    # x = np.log10(lambdahs.copy())
    # y = curvatures.copy()
    # yshift = np.abs(np.amin(y)) + 1.0
    # #y = np.clip(y, a_min=1e-12, a_max=None)
    # #y = np.log10(y)
    # # shift
    # y += yshift
    # y = np.log(y)

    # plt.plot(rhos, etas)
    # plt.axvline( 0.5*np.log(  tikh_family.b_hat_perp_norm_squared + f ) , color="blue")
    # plt.axvline( 0.5*np.log(  tikh_family.b_hat_perp_norm_squared + tikh_family.squared_term.sum() + f ) , color="red")
    # plt.axhline( 0.5*np.log(  tikh_family.d_hat_perp_norm_squared + tikh_family.squared_term_rev.sum() + g ) , color="green")
    # plt.axhline( 0.5*np.log(  tikh_family.d_hat_perp_norm_squared + g ) , color="pink")
    
    # plt.plot(x,y)

    # try:
    #     x_star, y_star, spl = find_positive_bump(x, y)
    #     opt_lambdah = np.power(10, x_star)
    #     x_lambdah = tikh_family.solve(opt_lambdah)
    # except:
    #     opt_lambdah = 1e-12
    #     print("Failed, picking lambda=1e-12")
    #     x_lambdah = tikh_family.solve(opt_lambdah)


    # try:
    #     x_star, y_star, spl = interior_extremum(x, y, which="max", smooth=None, tol=1e-10, clip_to_data=True, grid_n=10000)
    #     opt_lambdah = np.power(10, x_star)
    #     x_lambdah = tikh_family.solve(opt_lambdah)
    # except:
    #     opt_lambdah = 1e-12
    #     print("Failed, picking lambda=1e-12")
    #     x_lambdah = tikh_family.solve(opt_lambdah)

    opt_rho_hat_half, opt_eta_hat_half = tikh_family.lcurve(opt_lambdah, f=f, g=g)
    opt_rho_hat = 2.0*opt_rho_hat_half # need to multiply by 2 to get rho_hat from lcurve
    opt_eta_hat = 2.0*opt_eta_hat_half # need to multiply by 2 to get eta_hat from lcurve
    opt_curvature = tikh_family.lcurve_curvature(opt_lambdah, f=f, g=g)
    #opt_dist_origin = ( (opt_rho_hat/2.0)  )**2 + ( (opt_eta_hat/2.0)  )**2
    opt_dist_origin = np.exp(opt_rho_hat) + opt_lambdah*np.exp(opt_eta_hat)

    #mod_curvatures = np.exp(y)
    #opt_mod_curvature = opt_curvature + yshift


    # Get data for checking dpc?
    dpc_utb = np.abs(tikh_family.gsvd.U2.T @ tikh_family.b)
    dpc_c = tikh_family.gsvd.c_check
    dpc_utb_over_c = dpc_utb/dpc_c

    # related to smallest and largest singular values
    gamma_sq_min_rho_hat, gamma_sq_min_eta_hat = tikh_family.lcurve(gamma_sq_min)
    gamma_sq_min_rho_hat *= 2
    gamma_sq_min_eta_hat *= 2
    gamma_sq_max_rho_hat, gamma_sq_max_eta_hat = tikh_family.lcurve(gamma_sq_max)
    gamma_sq_max_rho_hat *= 2
    gamma_sq_max_eta_hat *= 2
    gamma_sq_min_curvature = tikh_family.lcurve_curvature(gamma_sq_min, f=f, g=g)
    gamma_sq_max_curvature = tikh_family.lcurve_curvature(gamma_sq_max, f=f, g=g)


    # expected lcorner?
    if tikh_family.btrue is not None:
        r_int = tikh_family.N - tikh_family.gsvd.n_A - tikh_family.gsvd.n_L
        # we might get errors here, but ok
        with np.errstate(divide='ignore', invalid='ignore'):
            expected_corner_abscissa = np.log(np.sqrt(tikh_family.noise_var)) + 0.5*(np.log( tikh_family.M - tikh_family.gsvd.r_A + r_int ))
            expected_corner_ordinate = 0.5*np.log(  tikh_family.d_hat_perp_norm_squared +  ( ((  (tikh_family.gsvd.U2.T @ tikh_family.btrue) - tikh_family.gsvd.gamma_check*(tikh_family.gsvd.V2.T @ tikh_family.d)  )**2)/(tikh_family.gsvd.gamma_check**2) ).sum()  )
    else:
        expected_corner_abscissa = None
        expected_corner_ordinate = None

    # was d nonzero?

    
    data = {
        "opt_lambdah": opt_lambdah,
        "x_lambdah": x_lambdah,
        "lambdahs": lambdahs,
        "rho_hat": rho_hat,
        "eta_hat": eta_hat,
        "upper_left_point": upper_left_point,
        "lower_right_point": lower_right_point,
        "curvatures": curvatures,
        "opt_rho_hat": opt_rho_hat,
        "opt_eta_hat": opt_eta_hat,
        "opt_curvature": opt_curvature,
        #"mod_curvatures": mod_curvatures,
        #"opt_mod_curvature": opt_mod_curvature,
        "slopes": slopes,
        "second_derivs": second_derivs,
        "distances_to_origin": distances_to_origin,
        "opt_dist_origin": opt_dist_origin,
        "dpc_utb": dpc_utb,
        "dpc_c": dpc_c,
        "dpc_utb_over_c": dpc_utb_over_c,
        "rho": rho,
        "eta": eta,
        "gamma_sq_min_rho_hat": gamma_sq_min_rho_hat,
        "gamma_sq_min_eta_hat": gamma_sq_min_eta_hat,
        "gamma_sq_max_rho_hat": gamma_sq_max_rho_hat,
        "gamma_sq_max_eta_hat": gamma_sq_max_eta_hat,
        "gamma_sq_min_curvature": gamma_sq_min_curvature,
        "gamma_sq_max_curvature": gamma_sq_max_curvature,
        "gamma_sq_min": gamma_sq_min,
        "gamma_sq_max": gamma_sq_max,
        "expected_corner_abscissa": expected_corner_abscissa,
        "expected_corner_ordinate": expected_corner_ordinate,
    }

    if projected: data["z_lambdah"] = z_lambdah

    return data
