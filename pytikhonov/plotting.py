import matplotlib.pyplot as plt
import numpy as np

from .util import secondary_plateau_level



def plot_rdata(method, rdata, plot_path=None):
    """Generates plot of the rdata. 
    """

    if method == "dp":
        plot_dp(rdata, plot_path=plot_path)
    elif method == "lcorner":
        plot_lcorner(rdata, plot_path=plot_path)
    elif method == "gcv":
        plot_gcv(rdata, plot_path=plot_path)        
    else:
        raise NotImplementedError
    

    # if (method == "lcurve") or (method == "lcurve_unprojected"):

    #     fig, axs = plt.subplots(1,6,figsize=(34,8))
        
    #     ### L-curve plot
    #     axs[0].plot(rdata["rho_hat"]/2.0, rdata["eta_hat"]/2.0)
    #     axs[0].scatter(rdata["opt_rho_hat"]/2.0, rdata["opt_eta_hat"]/2.0, color="red", marker="x", s=25.0)

    #     # draw the bounding box
    #     rho_ul, eta_ul = rdata["upper_left_point"]
    #     rho_lr, eta_lr = rdata["lower_right_point"] 
    
    #     axs[0].hlines(eta_ul, rho_ul, rho_lr, linestyles="dashed", color="k") # top horizontal
    #     axs[0].scatter(rho_ul, eta_ul, s=20.0, marker="o", color="k") # upper left point
        
    #     if eta_lr == -np.inf:
    #         pass
    #     else:
    #         # should check if rho_ul is -inf
    #         if rho_ul == -np.inf:
    #             pass
    #         else:
    #             axs[0].hlines(eta_lr, rho_ul, rho_lr, linestyles="dashed", color="k") # bottom horizontal

    #     # right vertical
    #     if eta_lr == -np.inf:
    #         ymin = axs[0].get_ylim()[0]
    #         axs[0].vlines(rho_lr, ymin, eta_ul, linestyles="dashed", color="k")
    #     else:
    #         axs[0].vlines(rho_lr, eta_lr, eta_ul, linestyles="dashed", color="k")
    #         axs[0].scatter(rho_lr, eta_lr, s=20.0, marker="o", color="k") # bottom right
            
    #     # left vertical
    #     if rho_ul == -np.inf:
    #         pass
    #     else:
    #         # should check if eta_lr is -inf
    #         if eta_lr == -np.inf:
    #             ymin = axs[0].get_ylim()[0]
    #             axs[0].vlines(rho_ul, ymin, eta_ul, linestyles="dashed", color="k")
    #         else:
    #             axs[0].vlines(rho_ul, eta_lr, eta_ul, linestyles="dashed", color="k")
      

    #     #axs[0].set_xlabel("log data fidelity / 2")
    #     axs[0].set_xlabel("$\hat{\\rho}/2 = \log(\| A x_{\lambda} - b \|_2)$")
    #     axs[0].set_ylabel("$\hat{\eta}/2 = \log(\| L x_{\lambda} - d \|_2)$")
    #     #axs[0].set_ylabel("log regularization term / 2")
    #     #axs[0].legend()

    #     ### curvature plot

    #     curvature = rdata["curvatures"]
    #     lambdahs = rdata["lambdahs"]

    #     # plot segments by +/-
    #     y = curvature.copy()
    #     x = lambdahs.copy()
    #     y_abs = np.abs(y)
    #     sign = np.sign(y)

    #     # Find indices where sign changes
    #     sign_change = np.where(np.diff(sign))[0]

    #     # Split into contiguous segments
    #     idx_splits = np.split(np.arange(len(y)), sign_change + 1)

    #     for idx in idx_splits:
    #         if len(idx) == 0:
    #             continue
    #         color = "green" if y[idx[0]] >= 0 else "red"
    #         axs[1].semilogy(x[idx], y_abs[idx], color=color)



    #     #axs[1].plot(rdata["lambdahs"], rdata["mod_curvatures"])
    #     axs[1].scatter(rdata["opt_lambdah"], rdata["opt_curvature"], color="red", marker="x", s=25.0)
    #     axs[1].set_title("Absolute curvature")
    #     axs[1].set_xlabel("$\lambda$")
    #     #axs[1].set_ylabel("$\gamma + C$")
    #     axs[1].set_xscale("log")
    #     axs[1].set_yscale("log")



    #     ### curvature (non-log scale)

    #     axs[2].plot(rdata["lambdahs"], rdata["curvatures"])
    #     axs[2].scatter(rdata["opt_lambdah"], rdata["opt_curvature"], color="red", marker="x", s=25.0)
    #     axs[2].set_title("Curvature")
    #     axs[2].set_xscale("log")
    #     axs[2].set_xlabel("$\lambda$")



    #     ### first derivatives
    #     slopes = rdata["slopes"]
    #     lambdahs = rdata["lambdahs"]
    
    #     # plot segments by +/-
    #     y = slopes.copy()
    #     x = lambdahs.copy()
    #     y_abs = np.abs(y)
    #     sign = np.sign(y)

    #     # Find indices where sign changes
    #     sign_change = np.where(np.diff(sign))[0]

    #     # Split into contiguous segments
    #     idx_splits = np.split(np.arange(len(y)), sign_change + 1)

    #     for idx in idx_splits:
    #         if len(idx) == 0:
    #             continue
    #         color = "green" if y[idx[0]] >= 0 else "red"
    #         axs[3].semilogy(x[idx], y_abs[idx], color=color)

    #     #axs[3].plot(rdata["lambdahs"], rdata["slopes"])
    #     #axs[3].semilogy(lambdahs[pos_mask], abs_slopes[pos_mask], color="green", label="+")
    #     #axs[3].semilogy(lambdahs[neg_mask], abs_slopes[neg_mask], color="red", label="-")
    #     axs[3].set_xscale("log")
    #     axs[3].set_xlabel("$\lambda$")
    #     axs[3].set_title("Absolute slope")
    #     #axs[3].legend()


    #     ### second derivatives
    #     second_derivs = rdata["second_derivs"]
    #     lambdahs = rdata["lambdahs"]

    #     # plot segments by +/-
    #     y = second_derivs.copy()
    #     x = lambdahs.copy()
    #     y_abs = np.abs(y)
    #     sign = np.sign(y)

    #     # Find indices where sign changes
    #     sign_change = np.where(np.diff(sign))[0]

    #     # Split into contiguous segments
    #     idx_splits = np.split(np.arange(len(y)), sign_change + 1)

    #     for idx in idx_splits:
    #         if len(idx) == 0:
    #             continue
    #         color = "green" if y[idx[0]] >= 0 else "red"
    #         axs[4].semilogy(x[idx], y_abs[idx], color=color)


    #     #axs[4].plot(rdata["lambdahs"], rdata["second_derivs"])
    #     #axs[4].semilogy(lambdahs[pos_mask], abs_sderivs[pos_mask], color="green", label="+")
    #     #axs[4].semilogy(lambdahs[neg_mask], abs_sderivs[neg_mask], color="red", label="-")
    #     axs[4].set_xscale("log")
    #     axs[4].set_xlabel("$\lambda$")
    #     axs[4].set_title("Absolute second derivative")




    #     # axs[5].plot(rdata["lambdahs"], rdata["distances_to_origin"])
    #     # axs[5].scatter(rdata["opt_lambdah"], rdata["opt_dist_origin"], color="red", marker="x", s=25.0)
    #     # axs[5].set_xscale("log")
    #     # axs[5].set_yscale("log")
    #     # axs[5].set_xlabel("$\lambda$")
    #     # axs[5].set_title("Distance to origin")


    #     axs[5].set_title("DPC")
    #     idx = [i for i in range(len(rdata["dpc_c"]))]
    #     axs[5].scatter(idx, rdata["dpc_utb"], label="utb")
    #     axs[5].scatter(idx, rdata["dpc_c"], label="c")
    #     axs[5].scatter(idx, rdata["dpc_utb_over_c"], label="utb/c")
    #     axs[5].set_yscale("log")
    #     axs[5].set_xlabel("index")
    #     axs[5].legend()



    #     opt_lambdah = rdata["opt_lambdah"]
    #     fig.suptitle(f"L-curve, $\lambda = {opt_lambdah:.3e}$")
    #     fig.tight_layout()
    #     if plot_path is not None:
    #         fig.savefig(plot_path, dpi=250)
    #         plt.close()
    #         return None
    #     else:
    #         fig.show()
    #         return None
    #         #return fig
        
    #     #plt.close()





    # else:
    #     raise NotImplementedError
    









def plot_monitoring_function(tikh_family, plot_path=None):
    """Plots the monitoring function V(\lambda).
    """
    
    gamma_sq_min = np.amin(tikh_family.gamma_check)**2
    gamma_sq_max = np.amax(tikh_family.gamma_check)**2
    lambdahs = np.logspace( np.log10(gamma_sq_min)-2, np.log10(gamma_sq_max)+2, num=1000, base=10 )
    
    fig, axs = plt.subplots(figsize=(8,5))

    axs.loglog(1.0/lambdahs, tikh_family.V(lambdahs), label="$\\mathcal{V}(\lambda)$", color="blue")
    axs.scatter(1.0/gamma_sq_min, tikh_family.V(gamma_sq_min), label="$\lambda = \gamma_{r_A}^2$", color="orange", s=100, zorder=10, alpha=1.0)
    axs.scatter(1.0/gamma_sq_max, tikh_family.V(gamma_sq_max), label="$\lambda = \gamma_{n_L + 1}^2$", color="brown", s=100, zorder=10, alpha=1.0)
    
    # estimate noise variance
    est_noise_var = secondary_plateau_level( np.log10(np.flip(1.0/lambdahs)), np.log10(np.flip(tikh_family.V(lambdahs))))
    est_noise_var = np.power(10.0, est_noise_var)
    axs.axhline(est_noise_var, label=f"$\hat{{\sigma}} \\approx {np.sqrt(est_noise_var):.5f}$", color="red", ls="--")

    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_xlabel("$\lambda^{-1}$")
    axs.set_title("Monitoring function $\mathcal{{V}}(\lambda)$")
    axs.legend()

    fig.tight_layout()

    if plot_path is not None:
        fig.savefig(plot_path, dpi=250)
        plt.close()
        return None
    else:
        plt.show()
        return None











def picard_plot(tikh_family, plot_path=None):
    """Picard plot.

    btrue: the noiseless RHS.
    noise_var: variance of the i.i.d. Gaussian noise.
    """
    # assert tikh_family.btrue is not None, "must pass btrue to TikhonovFamily when initialized!"
    btrue = tikh_family.btrue
    noise_var = tikh_family.noise_var

    gamma = tikh_family.gamma_check
    U2 = tikh_family.U2
    if noise_var is not None: noise_sigma = np.sqrt(noise_var)
    V2 = tikh_family.V2

    d_nonzero = not np.allclose(tikh_family.d, np.zeros_like(tikh_family.d))


    fig, axs = plt.subplots(figsize=(8,5))
    idx = [i+1 for i in range(len(gamma))]
    axs.scatter(idx, gamma, color="green", label="$\gamma_i$ (generalized SVs)")

    # True noiseless coefficients?
    if btrue is not None:
        if d_nonzero:
            axs.scatter(idx, np.abs( (U2.T @ btrue) - gamma*(V2.T @ tikh_family.d) ), color="orange", label="$|u_i^T b_{{\\text{true}}} - \gamma_i v_i^T d |$", s=5)
        else:
            axs.scatter(idx, np.abs( (U2.T @ btrue) ), color="orange", label="$|u_i^T b_{{\\text{true}}}|$", s=5)

    # Noise coefficients
    if d_nonzero:
        axs.scatter(idx, np.abs( (U2.T @ tikh_family.b) - gamma*(V2.T @ tikh_family.d)  ), color="purple", label="$|u_i^T b - \gamma_i v_i^T d|$", s=5)
    else:
        axs.scatter(idx, np.abs( (U2.T @ tikh_family.b)  ), color="purple", label="$|u_i^T b|$", s=5)

    # If noise_var is
    if noise_var is not None:
        if d_nonzero:
            axs.semilogy( (noise_sigma*np.sqrt(2/np.pi))*np.ones_like(gamma) , color="red", label="predicted 99% CI for $|u_i^T b - \gamma_i v_i^T d|$ (large i)")
        else:
            axs.semilogy( (noise_sigma*np.sqrt(2/np.pi))*np.ones_like(gamma) , color="red", label="predicted 99% CI for $|u_i^T b|$ (large i)")
        axs.semilogy( (0.0063*noise_sigma)*np.ones_like(gamma)  , color="red", ls="--")
        axs.semilogy( (2.807*noise_sigma)*np.ones_like(gamma)  , color="red", ls="--")
    
    axs.legend()
    axs.set_title("Discrete Picard condition")
    axs.set_xlabel("$n_L + i$")
    axs.set_yscale("log")
    fig.tight_layout()

    if plot_path is not None:
        fig.savefig(plot_path, dpi=250)
        plt.close()
        return None
    else:
        plt.show()
        return None
    








def plot_dp(dp_data, plot_path=None):
    """Generates a plot for DP.
    """

    fig, axs = plt.subplots(figsize=(8,5))
    axs.plot(dp_data["lambdahs"], dp_data["dp_vals"], color="blue" )
    dp_opt_lambdah = dp_data["opt_lambdah"]
    axs.scatter(dp_data["opt_lambdah"], dp_data["opt_lambdah_val"], label=f"$\lambda = {dp_opt_lambdah:.2e}$", color="red", s=100 , zorder=10)
    axs.scatter(dp_data["gamma_sq_min"], dp_data["gamma_sq_min_val"], label="$\lambda = \gamma_{r_A}^2$", color="orange", s=100, zorder=10, alpha=0.5)
    axs.scatter(dp_data["gamma_sq_max"],  dp_data["gamma_sq_max_val"], label="$\lambda = \gamma_{n_L + 1}^2$", color="brown", s=100, zorder=10, alpha=0.5)
    axs.set_xscale("log")
    linthresh = np.percentile(np.abs(dp_data["dp_vals"][dp_data["dp_vals"] != 0.0 ]), 5)  # e.g., 5th percentile
    axs.set_yscale('symlog', linthresh=linthresh, linscale=1.0, base=10)
    axs.axhline(0.0, color="black", ls="--", zorder=-10)

    axs.set_title("Discrepancy principle")
    axs.legend()
    axs.grid()
    axs.set_ylim( np.amin(dp_data["dp_vals"])*10.0, np.amax(dp_data["dp_vals"])*10.0 )

    fig.tight_layout()

    if plot_path is not None:
        fig.savefig(plot_path, dpi=250)
        plt.close()
        return None
    else:
        plt.show()
        return None



def plot_gcv(gcv_data, plot_path=None):

    fig, axs = plt.subplots(figsize=(8,5))
    axs.scatter(gcv_data["gamma_sq_min"], gcv_data["gamma_sq_min_val"], label="$\lambda = \gamma_{r_A}^2$", color="orange", s=100, zorder=10, alpha=0.5)
    axs.scatter(gcv_data["gamma_sq_max"],  gcv_data["gamma_sq_max_val"], label="$\lambda = \gamma_{n_L + 1}^2$", color="brown", s=100, zorder=10, alpha=0.5)
    axs.plot( gcv_data["lambdahs"], gcv_data["gcv_vals"], color="blue")
    gcv_opt_lambdah = gcv_data["opt_lambdah"]
    axs.scatter(gcv_data["opt_lambdah"],  gcv_data["opt_lambdah_val"] , label=f"$\lambda = {gcv_opt_lambdah:.2e}$", color="red", s=100 , zorder=10)
    axs.set_xscale("log")
    axs.set_yscale("log")
    axs.set_title("GCV")
    axs.set_xlabel("$\lambda$")
    axs.legend()

    if plot_path is not None:
        fig.savefig(plot_path, dpi=250)
        plt.close()
        return None
    else:
        plt.show()
        return None
    



def plot_lcorner(lcurve_data, plot_path=None):
    """Generates a plot for the L-curve and the corner, as well as the curvature.
    """

    # Get optimal lambdah
    lcurve_opt_lambdah = lcurve_data["opt_lambdah"]
    
    
    fig, axs = plt.subplots(1,2,figsize=(13,5))

    # Lcurve plot
    axs[0].plot( lcurve_data["rho_hat"]/2.0, lcurve_data["eta_hat"]/2.0, color="blue" )
    axs[0].scatter( lcurve_data["opt_rho_hat"]/2.0, lcurve_data["opt_eta_hat"]/2.0, color="red", s=100, zorder=10, label=f"$\lambda = {lcurve_opt_lambdah:.2e}$" )
    if lcurve_data["expected_corner_abscissa"] is not None:
        axs[0].scatter( lcurve_data["expected_corner_abscissa"], lcurve_data["expected_corner_ordinate"], s=50, zorder=10, color="black", marker="x", label="expected corner")
    axs[0].scatter( lcurve_data["gamma_sq_min_rho_hat"]/2.0, lcurve_data["gamma_sq_min_eta_hat"]/2.0, label="$\lambda = \gamma_{r_A}^2$", color="orange", s=100, zorder=10, alpha=0.5 )
    axs[0].scatter( lcurve_data["gamma_sq_max_rho_hat"]/2.0, lcurve_data["gamma_sq_max_eta_hat"]/2.0, label="$\lambda = \gamma_{n_L + 1}^2$", color="brown", s=100, zorder=10, alpha=0.5)
    axs[0].set_title("L-curve")
    axs[0].set_ylabel("$\hat{\\eta}/2 = \log( \| L x_{\lambda} \|_2 )$")
    axs[0].set_xlabel("$\hat{\\rho}/2 = \log( \| A x_{\lambda} - b \|_2 )$")
    axs[0].legend()

    # Curvature
    axs[1].plot(lcurve_data["lambdahs"], lcurve_data["curvatures"], color="blue")
    axs[1].scatter( lcurve_data["opt_lambdah"], lcurve_data["opt_curvature"], color="red", s=100, zorder=10, label=f"$\lambda = {lcurve_opt_lambdah:.2e}$" )
    axs[1].scatter( lcurve_data["gamma_sq_min"], lcurve_data["gamma_sq_min_curvature"], label="$\lambda = \gamma_{r_A}^2$", color="orange", s=100, zorder=10 , alpha=0.5 )
    axs[1].scatter( lcurve_data["gamma_sq_max"], lcurve_data["gamma_sq_max_curvature"], label="$\lambda = \gamma_{n_L + 1}^2$", color="brown", s=100, zorder=10 , alpha=0.5 )
    axs[1].set_title("Curvature")
    axs[1].set_xscale("log")
    axs[1].set_xlabel("$\lambda$")
    axs[1].legend()

    # suptitle
    fig.suptitle("L-corner")
    fig.tight_layout()

    if plot_path is not None:
        fig.savefig(plot_path, dpi=250)
        plt.close()
        return None
    else:
        plt.show()
        return None






def plot_all_methods(all_data, plot_path=None):
    """Generates a plot for the l-curve.
    """

    lcurve_data = all_data["lcurve_data"]
    gcv_data = all_data["gcv_data"]
    if "dp_data" in all_data.keys():
        dp_data = all_data["dp_data"]

    # Get optimal lambdah
    lcurve_opt_lambdah = lcurve_data["opt_lambdah"]
    gcv_opt_lambdah = gcv_data["opt_lambdah"]

    fig, axs = plt.subplots(figsize=(8,5))

    # Lcurve plot
    axs.plot( lcurve_data["rho_hat"]/2.0, lcurve_data["eta_hat"]/2.0, color="blue" )
    axs.scatter( lcurve_data["opt_rho_hat"]/2.0, lcurve_data["opt_eta_hat"]/2.0, color="red", s=100, marker="x", zorder=10, label=f"$\lambda_{{\\text{{lcurve}}}} = {lcurve_opt_lambdah:.2e}$" )
    axs.scatter( gcv_data["opt_rho_hat"]/2.0, gcv_data["opt_eta_hat"]/2.0, color="purple", marker="x", s=100, zorder=10, label=f"$\lambda_{{\\text{{gcv}}}} = {gcv_opt_lambdah:.2e}$" )

    if "dp_data" in all_data.keys():
        dp_opt_lambdah = dp_data["opt_lambdah"]
        axs.scatter( dp_data["opt_rho_hat"]/2.0, dp_data["opt_eta_hat"]/2.0, color="green", marker="x", s=100, zorder=10, label=f"$\lambda_{{\\text{{dp}}}} = {dp_opt_lambdah:.2e}$" )

    if lcurve_data["expected_corner_abscissa"] is not None:
        axs.scatter( lcurve_data["expected_corner_abscissa"], lcurve_data["expected_corner_ordinate"], s=50, zorder=10, color="black", marker="x", label="expected corner")
    axs.scatter( lcurve_data["gamma_sq_min_rho_hat"]/2.0, lcurve_data["gamma_sq_min_eta_hat"]/2.0, label="$\lambda = \gamma_{r_A}^2$", color="orange", s=100, zorder=10 )
    axs.scatter( lcurve_data["gamma_sq_max_rho_hat"]/2.0, lcurve_data["gamma_sq_max_eta_hat"]/2.0, label="$\lambda = \gamma_{n_L + 1}^2$", color="brown", s=100, zorder=10)
    
    axs.set_title("Comparison")
    axs.set_ylabel("$\hat{\\eta}/2 = \log( \| L x_{\lambda} \|_2 )$")
    axs.set_xlabel("$\hat{\\rho}/2 = \log( \| A x_{\lambda} - b \|_2 )$")
    axs.legend()
    

    if plot_path is not None:
        fig.savefig(plot_path, dpi=250)
        plt.close()
        return None
    else:
        plt.show()
        return None



    



