import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.optimize import minimize_scalar, brentq





def piecewise_constant_1d_test_problem():
    """Generates a piecewise constant 1D test vector. From Beck 12.4.3.
    """
    n = 1000
    result = np.zeros(n)
    result[:250] = 1
    result[250:500] = 3
    result[750:] = 2
    return result





def secondary_plateau_level(x, y, window=3, min_run=None, slope_q=0.25):
    """
    Estimate the y-value of the secondary (later) flat plateau in a 1D curve
    sampled densely at ordered x.

    Assumptions:
      - x is strictly increasing (left→right).
      - The curve begins with a flat (primary) plateau, then descends,
        then reaches a second flat (secondary) plateau before doing other stuff.
      - Plateaus are characterized by small |dy/dx|.

    Parameters
    ----------
    x, y : 1D numpy arrays of equal length (len >= 3).
    window : int, odd
        Window size for smoothing the slope with a simple moving average.
        Larger -> more smoothing. Must be >= 3 and odd.
    min_run : int or None
        Minimum number of *points* that must be labeled "flat" to count as a plateau.
        If None, defaults to max(5, int(0.02 * len(x))).
    slope_q : float in (0, 1)
        Quantile to set the flatness threshold on smoothed |dy/dx|.
        Smaller values -> stricter notion of “flat”.

    Returns
    -------
    y_secondary : float
        Estimated y-level of the secondary plateau (mean of y over that plateau).

    Raises
    ------
    ValueError
        If a valid secondary plateau cannot be found.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n != y.size or n < 3:
        raise ValueError("x and y must be same length and at least length 3.")
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing.")
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3.")
    if min_run is None:
        min_run = max(5, int(0.02 * n))

    # 1) Estimate local slope magnitude |dy/dx| between consecutive points.
    dx = np.diff(x)
    dy = np.diff(y)
    slope = np.abs(dy / dx)  # length n-1

    # 2) Smooth the slope with a centered moving average (pad at ends).
    pad = window // 2
    kernel = np.ones(window) / window
    slope_padded = np.pad(slope, (pad, pad), mode="edge")
    slope_smooth = np.convolve(slope_padded, kernel, mode="valid")  # length n-1

    # 3) Define a data-driven flatness threshold from a low quantile of slope.
    #    This adapts to scale: plateaus should have very small slopes.
    thresh = np.quantile(slope_smooth, slope_q)

    # 4) Mark "flat" intervals (between points). Map them to point indices.
    flat_interval = slope_smooth <= thresh  # length n-1
    # Convert interval mask (on edges) to point mask by duplicating to the right.
    flat_points = np.zeros(n, dtype=bool)
    flat_points[:-1] |= flat_interval
    flat_points[1:]  |= flat_interval

    # Helper: find contiguous runs (start,end) of True in a boolean array
    def runs_of_true(mask):
        if not mask.any():
            return []
        idx = np.flatnonzero(mask)
        # split where gaps > 1
        breaks = np.where(np.diff(idx) > 1)[0]
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends   = np.r_[idx[breaks], idx[-1]]
        return list(zip(starts, ends))

    runs = runs_of_true(flat_points)
    # Keep only runs that are long enough to be a plateau
    runs = [(s, e) for (s, e) in runs if (e - s + 1) >= min_run]
    if not runs:
        raise ValueError("No sufficiently flat plateau found; try lowering slope_q or min_run, or increasing window.")

    # 5) The first run near the left edge is the primary plateau.
    #    Choose the earliest run whose start is closest to index 0.
    runs_sorted = sorted(runs, key=lambda se: se[0])
    primary = runs_sorted[0]

    # 6) The secondary plateau is the next run that occurs AFTER a descent region
    #    (i.e., a non-flat gap) that separates it from the primary.
    secondary = None
    for (s, e) in runs_sorted[1:]:
        # Ensure there is a non-flat gap between primary and this run.
        if s > primary[1] + 1:
            secondary = (s, e)
            break

    if secondary is None:
        raise ValueError("Could not identify a distinct secondary plateau after the primary.")

    # 7) Return the robust average y-level on the secondary plateau.
    #    Use trimmed mean for robustness; fall back to plain mean if too short.
    s, e = secondary
    y_seg = y[s:e+1]
    if y_seg.size >= 20:
        # 10% trimmed mean for extra robustness against tiny edges
        lo, hi = np.percentile(y_seg, [10, 90])
        y_trim = y_seg[(y_seg >= lo) & (y_seg <= hi)]
        return float(np.mean(y_trim)) if y_trim.size >= 5 else float(np.mean(y_seg))
    else:
        return float(np.mean(y_seg))






def estimate_noise_variance(tikh_family):
    """Given a Tikhonov family, estimates the noise variance using the monitoring function method.
    """
    gamma_sq_min = np.amin(tikh_family.gamma_check)**2
    gamma_sq_max = np.amax(tikh_family.gamma_check)**2
    lambdahs = np.logspace( np.log10(gamma_sq_min)-2, np.log10(gamma_sq_max)+2, num=1000, base=10 )
    est_noise_var = secondary_plateau_level( np.log10(np.flip(1.0/lambdahs)), np.log10(np.flip(tikh_family.V(lambdahs))))
    est_noise_var = np.power(10.0, est_noise_var)
    return est_noise_var











def interior_extremum(
    x, y, *, which="max", smooth=None, tol=1e-10, clip_to_data=True, grid_n=2001
):
    """
    Find the interior local min/max of a function sampled at (x, y), ignoring the
    monotone-increasing right tail. Uses a cubic UnivariateSpline so derivative roots
    can be computed reliably. Used for lcurve.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of equal length.")
    # sort by x
    idx = np.argsort(x)
    xs, ys = x[idx], y[idx]
    a, b = float(xs[0]), float(xs[-1])

    s = 0.0 if smooth is None else float(smooth)

    # Force cubic spline (k=3) so .roots() is supported
    spl = UnivariateSpline(xs, ys, s=s, k=3)
    dspl = spl.derivative(1)
    d2spl = spl.derivative(2)

    # Try fast path: analytic roots from fitpack for cubic splines
    crit = None
    try:
        crit = np.array(dspl.roots())
    except NotImplementedError:
        crit = np.array([])

    # Robust fallback: bracket sign changes of f'(x) on a grid and use brentq
    if crit is None or crit.size == 0:
        grid = np.linspace(a, b, grid_n)
        dvals = dspl(grid)
        # Identify sign-change intervals
        sign = np.sign(dvals)
        # Treat near-zero as zero to catch flat tangencies
        near_zero = np.isclose(dvals, 0.0, rtol=0, atol=1e-10)
        sign[near_zero] = 0.0

        roots = []
        for i in range(len(grid) - 1):
            g0, g1 = grid[i], grid[i+1]
            s0, s1 = sign[i], sign[i+1]
            if s0 == 0.0:      # exact/near-zero at grid[i]
                roots.append(g0)
            elif s0 * s1 < 0:  # sign change
                try:
                    r = brentq(dspl, g0, g1)
                    roots.append(r)
                except ValueError:
                    pass
        crit = np.array(sorted(set(roots)))

    #print(f"critical points: {crit}")

    # Keep interior points
    if clip_to_data:
        crit = crit[(crit > a + tol) & (crit < b - tol)]

    if crit.size == 0:
        # Fallback: restrict to the left of the last nonpositive derivative, then optimize
        grid = np.linspace(a, b, grid_n)
        dvals = dspl(grid)
        mask = dvals <= 0
        if np.any(mask):
            rightmost_nonpos = grid[np.where(mask)[0][-1]]
            left_b = max(a + tol, rightmost_nonpos)
            if which == "min":
                res = minimize_scalar(spl, bounds=(a, left_b), method="bounded", tol=1e-14)
                x_star = float(res.x)
                return x_star, float(spl(x_star)), spl
            else:
                res = minimize_scalar(lambda z: -spl(z), bounds=(a, left_b), method="bounded", tol=1e-14)
                x_star = float(res.x)
                return x_star, float(spl(x_star)), spl
        else:
            raise RuntimeError("No interior critical points found and derivative > 0 on domain.")

    # Classify via second derivative
    if which == "min":
        crit = crit[d2spl(crit) > 0]
    else:
        crit = crit[d2spl(crit) < 0]

    if crit.size == 0:
        raise RuntimeError(f"No interior {which} found among derivative roots.")

    # Choose the rightmost interior extremum to avoid the increasing tail
    x_star = float(np.max(crit))
    y_star = float(spl(x_star))
    return x_star, y_star, spl








def _numerical_kth_derivative(f, x0, k, h=None, deg=None):
    """
    Numerically estimate the k-th derivative of f at x0 by fitting
    a local polynomial p(t) (t = x - x0) to samples around x0 and
    then taking p^{(k)}(0) = k! * c_k.
    """
    if k < 0 or int(k) != k:
        raise ValueError("k must be a nonnegative integer.")
    k = int(k)

    # Choose spacing and polynomial degree
    eps = np.finfo(float).eps
    scale = max(1.0, abs(x0))
    if h is None:
        # heuristic: smaller step for higher k
        h = (eps ** (1/(k+3))) * scale
    if h <= 0:
        raise ValueError("h must be positive.")

    if deg is None:
        # degree must be at least k; use a bit higher for stability
        deg = max(k, 4 + k)  # e.g., k=0->deg=4, k=1->5, etc.
    if deg < k:
        raise ValueError("deg must be >= k.")

    # Use an odd number of points symmetric about x0
    m = deg // 2
    idx = np.arange(-m, m+1)
    # If deg is even, this gives deg+1 points; if deg is odd, still fine (deg+1 points)
    t = idx * h
    x = x0 + t

    y = np.asarray([f(xx) for xx in x], dtype=float)

    # Build Vandermonde in shifted coords: [1, t, t^2, ... t^deg]
    V = np.vander(t, N=deg+1, increasing=True)

    # Solve least squares for coefficients c of p(t) = sum_j c_j t^j
    # Use robust lstsq; rcond=None uses default cutoff
    c, *_ = np.linalg.lstsq(V, y, rcond=None)

    # k-th derivative at 0 is k! * c_k
    # (since d^k/dt^k t^k = k!, and lower terms vanish at 0)
    from math import factorial
    return factorial(k) * c[k]


def check_kth_derivative(
    f,
    dfk,
    xs,
    k,
    *,
    rtol=1e-5,
    atol=1e-8,
    h=None,
    deg=None,
    dfk_takes_k=False,
    return_details=False,
):
    """
    Compare your claimed k-th derivative function against a numerical estimate.

    Parameters
    ----------
    f : callable
        Original scalar function f(x).
    dfk : callable
        Your claimed k-th derivative function. If dfk_takes_k=False (default),
        it must be dfk(x). If True, it must be dfk(x, k).
    xs : array-like
        Points at which to test.
    k : int
        Derivative order to test.
    rtol, atol : float
        Relative / absolute tolerances for closeness.
    h : float or None
        Optional step size for numerical derivative (heuristic if None).
    deg : int or None
        Optional polynomial degree for the fit (>= k). Heuristic if None.
    dfk_takes_k : bool
        If True, call your function as dfk(x, k). Otherwise as dfk(x).
    return_details : bool
        If True, return a dict of per-point results along with the overall pass flag.

    Returns
    -------
    passed : bool  (and details : dict if return_details=True)
    """
    xs = np.atleast_1d(xs).astype(float)
    preds = []
    nums = []
    abs_errs = []
    rel_errs = []

    for x0 in xs:
        pred = dfk(x0, k) if dfk_takes_k else dfk(x0)
        num = _numerical_kth_derivative(f, x0, k, h=h, deg=deg)
        preds.append(pred)
        nums.append(num)
        abs_err = abs(pred - num)
        denom = max(atol, abs(num))
        rel_err = abs_err / denom
        abs_errs.append(abs_err)
        rel_errs.append(rel_err)

    preds = np.array(preds)
    nums = np.array(nums)
    abs_errs = np.array(abs_errs)
    rel_errs = np.array(rel_errs)

    passed_mask = (abs_errs <= atol) | (rel_errs <= rtol)
    passed = bool(np.all(passed_mask))

    if not return_details:
        return passed

    details = {
        "x": xs,
        "pred": preds,
        "num": nums,
        "abs_err": abs_errs,
        "rel_err": rel_errs,
        "passed_each": passed_mask,
        "passed_overall": passed,
        "k": k,
        "rtol": rtol,
        "atol": atol,
        "h": h,
        "deg": deg,
    }
    return passed, details











import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks

class NoBumpFound(RuntimeError):
    pass

def find_positive_bump(
    x, y, *,
    smooth=None,          # smoothing for the spline; try len(x)*sigma^2 if noisy
    grid_n=4001,          # resolution for the search grid
    min_prominence=None,  # absolute prominence; if None, set relative to data range
    min_distance=None,    # minimal distance between peaks in x-units (optional)
    atol_deriv=1e-10,     # near-zero tolerance for derivative when cutting the tail
):
    """
    Locate the 'positive bump' (interior local max) while ignoring the large
    monotone increase on the far right. Raises NoBumpFound if not present.

    Returns
    -------
    x_bump, y_bump, spline
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of equal length.")

    # sort by x
    order = np.argsort(x)
    xs, ys = x[order], y[order]
    a, b = float(xs[0]), float(xs[-1])

    # cubic smoothing spline (k=3 so we have C^2 and stable derivatives)
    s = 0.0 if smooth is None else float(smooth)
    spl = UnivariateSpline(xs, ys, s=s, k=3)

    # dense grid + derivative
    grid = np.linspace(a, b, grid_n)
    vals = spl(grid)
    dvals = spl.derivative(1)(grid)

    # 1) cut off the right tail: keep up to the last point where derivative <= 0
    nonpos = np.where(dvals <= atol_deriv)[0]
    if nonpos.size == 0:
        # derivative > 0 everywhere => strictly increasing; no interior bump
        raise NoBumpFound("Function is monotone increasing on the domain (no interior bump).")
    right_cut_idx = nonpos[-1]  # last non-positive-derivative point
    grid_L = grid[: right_cut_idx + 1]
    vals_L = vals[: right_cut_idx + 1]

    if grid_L.size < 5:
        raise NoBumpFound("Not enough interior region (after tail cutoff) to contain a bump.")

    # 2) pick a sensible prominence if not provided
    data_range = np.nanmax(vals_L) - np.nanmin(vals_L)
    if min_prominence is None:
        # require a noticeable bump; tune 5–10% as needed
        min_prominence = 0.05 * (data_range if data_range > 0 else 1.0)

    # convert min_distance from x-units to sample indices (optional)
    distance_idx = None
    if min_distance is not None:
        distance_idx = max(1, int(np.round(min_distance / (grid_L[1] - grid_L[0]))))

    # 3) find peaks by prominence on the truncated interval
    peaks, props = find_peaks(vals_L, prominence=min_prominence, distance=distance_idx)

    if peaks.size == 0:
        raise NoBumpFound("No positive bump with sufficient prominence was found.")

    # 4) choose the most prominent peak (or use other tie-breaks if you prefer)
    best = int(peaks[np.argmax(props["prominences"])])
    x_bump = float(grid_L[best])
    y_bump = float(vals_L[best])
    return x_bump, y_bump, spl











import numpy as np

try:
    from scipy.signal import savgol_filter
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


def _savgol_safe(y, window=11, poly=3):
    """Savitzky–Golay smoothing if SciPy is available, else return y."""
    if not _HAS_SCIPY:
        return y
    window = max(3, int(window) | 1)  # odd, >=3
    poly = max(2, int(poly))
    window = min(window, len(y) - (1 - len(y) % 2))  # not larger than n, keep odd
    if window < 3 or window <= poly:
        return y
    return savgol_filter(y, window_length=window, polyorder=poly)


def _curvature_idx(x, y):
    """
    Discrete curvature κ of a parametric curve (x(t), y(t)).
    Here we parameterize by cumulative arc length for invariance.
    Returns curvature array and argmax index (excluding endpoints).
    """
    # Parameter by arc length
    ds = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    s = np.concatenate([[0.0], np.cumsum(ds)])
    # Derivatives w.r.t. s using centered differences
    # gradient handles nonuniform spacing if we pass s
    x_s  = np.gradient(x, s, edge_order=2)
    y_s  = np.gradient(y, s, edge_order=2)
    x_ss = np.gradient(x_s, s, edge_order=2)
    y_ss = np.gradient(y_s, s, edge_order=2)
    kappa = np.abs(x_s * y_ss - y_s * x_ss) / np.maximum((x_s**2 + y_s**2)**1.5, 1e-300)

    # Exclude the first/last few points from peak picking (endpoint bias)
    margin = max(2, len(x)//50)  # ~2% guard or at least 2 points
    valid = np.arange(margin, len(x)-margin)
    if len(valid) == 0:
        return kappa, int(np.argmax(kappa))
    i_peak = valid[np.argmax(kappa[valid])]
    return kappa, int(i_peak)


def _triangle_idx(x, y):
    """
    Maximum perpendicular distance from points to the line through endpoints in (x,y).
    """
    p0 = np.array([x[0], y[0]])
    p1 = np.array([x[-1], y[-1]])
    v = p1 - p0
    v2 = np.dot(v, v)
    if v2 == 0.0:
        return 0, np.zeros_like(x)
    # Perpendicular distance for each point
    pts = np.vstack([x, y]).T
    # Area * 2 / |v| gives perpendicular distance
    cross = np.abs(np.cross(v, pts - p0))
    dist = cross / np.sqrt(v2)
    margin = max(2, len(x)//50)
    valid = np.arange(margin, len(x)-margin)
    i_peak = valid[np.argmax(dist[valid])] if len(valid) else int(np.argmax(dist))
    return int(i_peak), dist


def _prominence_score(arr, idx, left=10, right=10):
    """
    Simple prominence-like score: peak minus max of flanking baselines.
    """
    n = len(arr)
    a = max(0, idx-left)
    b = min(n, idx+right+1)
    baseline = max(np.max(arr[a:idx]) if idx > a else 0.0,
                   np.max(arr[idx+1:b]) if idx+1 < b else 0.0)
    return float(arr[idx] - baseline), float(baseline)


def find_lcurve_corner(
    rho, eta,
    *,
    smooth=True,
    smooth_window=15,
    smooth_poly=3,
    min_prominence=0.15,
    min_endpoint_margin_frac=0.05,
    agree_tol_frac=0.05,
):
    """
    Find the 'corner' index of an L-curve given dense points (rho, eta).
    Operates in log-log space and uses two methods:
      - Maximum curvature (primary)
      - Triangle distance (secondary)
    Returns:
      dict with keys:
        - 'idx' (int or None): index of the chosen corner (None if no clear corner)
        - 'reason' (str): short explanation / failure reason if None
        - 'method' ('curvature' | 'triangle' | 'consensus' | None)
        - 'scores': dict with diagnostic scalars
        - 'curvature': np.ndarray of curvature values (log-log)
        - 'triangle_distance': np.ndarray of distances (log-log)
        - 'xlog', 'ylog': the log-log inputs actually used (smoothed if enabled)
    Heuristics for 'no-corner':
      - Peak too close to endpoints.
      - Peak prominence small relative to local baseline.
      - Curvature and triangle methods strongly disagree (peaks far apart) AND
        each fails its own prominence/endpoint tests.
    Parameters
    ----------
    rho, eta : array_like, shape (n,)
        Positive values; points should be ordered along the L-curve
        (typically rho decreases while eta increases with λ, but order just needs to be monotone along the sampled path).
    smooth : bool, default True
        Apply Savitzky–Golay smoothing in log space if SciPy is available.
    smooth_window : int, default 15
        Window length for smoothing (odd). Auto-adjusted if too large.
    smooth_poly : int, default 3
        Polynomial order for smoothing.
    min_prominence : float in (0, +inf), default 0.15
        Required prominence for acceptance (in absolute units of the score arrays:
        curvature for the curvature method, distance for the triangle method).
        You can tighten/loosen this depending on noise/spacing.
    min_endpoint_margin_frac : float, default 0.05
        Peak must be at least this fraction of n away from each endpoint.
    agree_tol_frac : float, default 0.05
        If curvature and triangle peaks are within this fraction of n,
        we call it a consensus (slightly boosts confidence).
    """
    rho = np.asarray(rho, dtype=float)
    eta = np.asarray(eta, dtype=float)

    n = len(rho)
    if n != len(eta) or n < 5:
        return dict(
            idx=None, reason="Need equal-length rho/eta with at least 5 points.",
            method=None, scores={}, curvature=None, triangle_distance=None,
            xlog=None, ylog=None
        )
    if np.any(rho <= 0) or np.any(eta <= 0):
        return dict(
            idx=None, reason="All rho and eta must be positive (log-log is required).",
            method=None, scores={}, curvature=None, triangle_distance=None,
            xlog=None, ylog=None
        )

    # Log-log space
    x = np.log(rho)
    y = np.log(eta)

    # Optional smoothing (in log space)
    if smooth:
        x_s = _savgol_safe(x, window=smooth_window, poly=smooth_poly)
        y_s = _savgol_safe(y, window=smooth_window, poly=smooth_poly)
    else:
        x_s, y_s = x, y

    # Primary: curvature
    kappa, i_curv = _curvature_idx(x_s, y_s)
    prom_c, base_c = _prominence_score(kappa, i_curv)
    # Endpoint margin check
    margin = max(2, int(np.floor(min_endpoint_margin_frac * n)))
    ok_margin_c = (margin <= i_curv <= n-1-margin)
    ok_prom_c = (prom_c >= min_prominence)

    # Secondary: triangle
    i_tri, tri = _triangle_idx(x_s, y_s)
    prom_t, base_t = _prominence_score(tri, i_tri)
    ok_margin_t = (margin <= i_tri <= n-1-margin)
    ok_prom_t = (prom_t >= min_prominence)

    # Agreement check
    agree_tol = max(1, int(np.ceil(agree_tol_frac * n)))
    agree = (abs(i_curv - i_tri) <= agree_tol)

    # Decision logic
    method = None
    idx = None
    reason = ""

    if ok_margin_c and ok_prom_c and ok_margin_t and ok_prom_t and agree:
        # Best case: both agree and both are strong
        idx = int(round(0.5 * (i_curv + i_tri)))
        method = "consensus"
        reason = "Curvature and triangle methods agree with strong peaks."
    else:
        # Prefer curvature if strong, else triangle if strong
        if ok_margin_c and ok_prom_c:
            idx = i_curv
            method = "curvature"
            reason = "Curvature method found a prominent interior peak."
        elif ok_margin_t and ok_prom_t:
            idx = i_tri
            method = "triangle"
            reason = "Triangle method found a prominent interior peak."
        else:
            # Neither is convincing: declare no-corner
            idx = None
            method = None
            fail_bits = []
            if not ok_margin_c: fail_bits.append("curvature peak near endpoint")
            if not ok_prom_c:   fail_bits.append("curvature peak not prominent")
            if not ok_margin_t: fail_bits.append("triangle peak near endpoint")
            if not ok_prom_t:   fail_bits.append("triangle peak not prominent")
            if not fail_bits:
                fail_bits = ["ambiguous shape / no distinct elbow"]
            reason = "No clear L-curve corner: " + ", ".join(fail_bits) + "."

    return dict(
        idx=idx,
        method=method,
        reason=reason,
        scores=dict(
            curvature_peak_index=int(i_curv),
            curvature_peak_value=float(kappa[i_curv]),
            curvature_prominence=float(prom_c),
            triangle_peak_index=int(i_tri),
            triangle_peak_value=float(tri[i_tri]),
            triangle_prominence=float(prom_t),
            endpoint_margin_points=int(margin),
            agree=bool(agree),
        ),
        curvature=kappa,
        triangle_distance=tri,
        xlog=x_s,
        ylog=y_s,
    )











import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.optimize import brentq

def tallest_true_peak_or_plateau_edge(
    x, y,
    *,
    # Spline / sampling
    s=None, k=3, edge_margin=0.0, nsamp=None,

    # Peak proposal/acceptance
    prominence=None, curvature_tol=0.0,
    min_drop_left=0.0, min_drop_right=0.0,
    slope_right_tol=0.0, slope_probe_frac=0.003,

    # Plateau-edge override (robust, derivative-based)
    plateau_override=True,
    flat_slope_rel=1e-3,
    flat_min_frac=0.01,
    fall_slope_rel=5e-2,
    fall_drop_rel=0.02,
    fall_window_frac=0.01,
    plateau_higher_rel=0.0,   # plateau must exceed peak by this relative margin

    # NEW: last-chance plateau finder (scan from x = -12 to first 1% drop)
    fallback_enable=True,
    fallback_probe_start=-12.0,
    fallback_drop_rel=0.01,         # 1% relative drop trigger
    fallback_step_frac=1e-4,        # step = frac * (xmax - xmin)
    fallback_max_steps=None,        # None -> computed from domain/step
    return_all=False,
):
    """
    Return (x*, y*): the tallest interior true peak (f'(x*)=0, f''(x*)<0) OR,
    if a higher left plateau exists before a precipitous drop, the plateau edge.
    If neither is found and `fallback_enable` is True, start at x=-12 (clamped
    to the domain) and march right until y(x) has fallen by ≥ fallback_drop_rel
    from its start value; return that crossing point (refined with brentq).
    """
    # ---------- Prep ----------
    x = np.asarray(x, float); y = np.asarray(y, float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x and y must be 1D arrays of equal length.")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("x and y must be finite.")
    if not np.all(np.diff(x) > 0):
        o = np.argsort(x); x, y = x[o], y[o]

    spl  = UnivariateSpline(x, y, s=s, k=k)
    dspl = spl.derivative(1)
    d2spl= spl.derivative(2)

    xmin = float(x[0] + edge_margin)
    xmax = float(x[-1] - edge_margin)
    if not (xmin < xmax):
        raise ValueError("edge_margin too large for domain length.")

    Xrng = xmax - xmin
    Yrng = float(np.ptp(y)) if np.ptp(y) > 0 else 1.0

    # Dense grid
    if nsamp is None:
        nsamp = max(4000, 20 * len(x))
    Xd = np.linspace(xmin, xmax, int(nsamp))
    Yd = spl(Xd)
    dY = dspl(Xd)

    # ---------- 1) Tallest true peak ----------
    if prominence is None:
        prominence = 0.02 * float(np.ptp(Yd)) if np.ptp(Yd) > 0 else 0.0
    pidx, props = find_peaks(Yd, prominence=prominence)

    def _refine_peak(idx):
        L = max(0, idx - 1); R = min(len(Xd) - 1, idx + 1)
        for _ in range(200):
            xL, xR = Xd[L], Xd[R]
            fL, fR = float(dspl(xL)), float(dspl(xR))
            if np.isfinite(fL) and np.isfinite(fR) and np.sign(fL) * np.sign(fR) < 0:
                try:
                    return brentq(lambda t: float(dspl(t)), xL, xR)
                except ValueError:
                    pass
            if L > 0: L -= 1
            if R < len(Xd) - 1: R += 1
        return None

    h_right = slope_probe_frac * Xrng
    candidates = []
    for j, idx in enumerate(pidx):
        if idx <= 0 or idx >= len(Xd) - 1:
            continue
        x0 = _refine_peak(idx)
        if x0 is None:
            continue
        curv = float(d2spl(x0))
        if not np.isfinite(curv) or curv > -float(curvature_tol):
            continue
        y0 = float(spl(x0))
        lb = props["left_bases"][j]; rb = props["right_bases"][j]
        yLb = float(spl(Xd[lb])); yRb = float(spl(Xd[rb]))
        dropL = y0 - yLb; dropR = y0 - yRb
        if slope_right_tol > 0 and x0 + h_right < xmax:
            if float(dspl(x0 + h_right)) > -float(slope_right_tol):
                continue
        if dropL >= min_drop_left and dropR >= min_drop_right:
            candidates.append((x0, y0, dropL, dropR, curv))

    tallest_xy = None if not candidates else (max(candidates, key=lambda t: t[1])[0],
                                              max(candidates, key=lambda t: t[1])[1])

    # ---------- 2) Plateau-edge override ----------
    result_xy = tallest_xy
    if plateau_override:
        # Scale-aware thresholds
        flat_eps   = flat_slope_rel * (Yrng / Xrng)              # flat if |f'| ≤ flat_eps
        fall_slope = fall_slope_rel * (Yrng / Xrng)              # fall if f' ≤ −fall_slope
        w_fall     = max(1, int(fall_window_frac * len(Xd)))     # drop window
        min_flat_n = max(1, int(flat_min_frac * len(Xd)))        # min flat length
        drop_abs_min = fall_drop_rel * Yrng

        flat_mask = np.isfinite(dY) & (np.abs(dY) <= flat_eps)
        plateau_edge_candidate = None
        i = 0
        while i < len(Xd):
            if not flat_mask[i]:
                i += 1; continue
            j = i
            while j < len(Xd) and flat_mask[j]:
                j += 1
            run_len = j - i
            if run_len >= min_flat_n and j < len(Xd) - 1:
                edge_idx = j - 1
                x_edge, y_edge = Xd[edge_idx], Yd[edge_idx]
                k1 = min(len(Xd) - 1, edge_idx + 1)
                k2 = min(len(Xd) - 1, edge_idx + max(2, w_fall // 2))
                right_ok = (dY[k1] <= -fall_slope) or (dY[k2] <= -fall_slope)
                if right_ok:
                    k_drop = min(len(Xd) - 1, edge_idx + w_fall)
                    if (y_edge - Yd[k_drop]) >= drop_abs_min:
                        plateau_edge_candidate = (x_edge, y_edge)
            i = j

        if plateau_edge_candidate is not None:
            if result_xy is None:
                result_xy = plateau_edge_candidate
            else:
                x_peak, y_peak = result_xy
                x_pl, y_pl = plateau_edge_candidate
                if y_pl >= y_peak * (1.0 + plateau_higher_rel) + 1e-15:
                    result_xy = (x_pl, y_pl)

    # ---------- 3) Last-chance fallback: scan from x=-12 to first 1% drop ----------
    if result_xy is None and fallback_enable:
        x0 = float(np.clip(fallback_probe_start, xmin, xmax))
        y0 = float(spl(x0))
        # if y0 is NaN (shouldn't happen), start at xmin
        if not np.isfinite(y0):
            x0 = xmin; y0 = float(spl(x0))

        target = y0 * (1.0 - float(fallback_drop_rel))
        step = max( (xmax - xmin) * float(fallback_step_frac), 1e-12 * Xrng )
        max_steps = int(np.ceil((xmax - x0) / step)) if fallback_max_steps is None else int(fallback_max_steps)

        crossed = False
        xL, fL = x0, float(spl(x0)) - target
        for _ in range(max_steps):
            xR = min(xmax, xL + step)
            fR = float(spl(xR)) - target
            if np.isfinite(fL) and np.isfinite(fR) and (fL > 0) and (fR <= 0):
                # refine the first crossing of y(x) - target = 0
                try:
                    xr = brentq(lambda t: float(spl(t)) - target, xL, xR)
                except Exception:
                    xr = xR
                result_xy = (float(xr), float(spl(xr)))
                crossed = True
                break
            xL, fL = xR, fR

        if not crossed and result_xy is None:
            # failed to see a 1% drop before xmax; return xmax
            result_xy = (float(xmax), float(spl(xmax)))

    if result_xy is None:
        raise ValueError(
            "No interior peak, no valid plateau edge, and fallback scan did not detect a 1% drop. "
            "Loosen thresholds or check the data range."
        )

    return (result_xy, candidates) if return_all else result_xy
