import numpy as np

from .lcurve import lcorner
from .discrepancy_principle import discrepancy_principle
from .gcv import gcvmin
from .tikhonov_family import TikhonovFamily





def rand_lcurve(tikh_family, lambdahs=None, n_samples=100, rseed=0):
    """Given a TikhonovFamily, generates random samples of the L-curve.
    """

    assert tikh_family.btrue is not None, "Must pass btrue to TikhonovFamily"
    
    # Get stuff from original
    _gsvd = tikh_family.gsvd
    _A = tikh_family.A
    _L = tikh_family.L
    _d = tikh_family.d
    _noise_var = tikh_family.noise_var
    _btrue = tikh_family.btrue

    # Handle lambdahs
    if lambdahs is None:
        lambdahs = np.logspace(-10,10,num=1000,base=10)

    sample_lcurve_abscissas = np.zeros((n_samples, len(lambdahs)))
    sample_lcurve_ordinates = np.zeros((n_samples, len(lambdahs)))
    np.random.seed(rseed)
    for j in range(n_samples):

        # draw new b and make new family
        _bsample = tikh_family.btrue + ( np.sqrt(_noise_var)*np.random.normal(size=len(tikh_family.btrue)) )
        _tf = TikhonovFamily(_A, _L, _bsample, _d, gsvd=_gsvd)

        # get points on the lcurve
        x, y = _tf.lcurve(lambdahs)
        sample_lcurve_abscissas[j,:] = x.copy()
        sample_lcurve_ordinates[j,:] = y.copy()

    return sample_lcurve_abscissas, sample_lcurve_ordinates






def rand_discrepancy_principle(tikh_family, n_samples=10, tau=1.01, rseed=0):
    """Randomly draws RHS vectors, solves DP problem.
    """

    assert tikh_family.btrue is not None, "Must pass btrue to TikhonovFamily"
    
    # Get stuff from original
    _gsvd = tikh_family.gsvd
    _A = tikh_family.A
    _L = tikh_family.L
    _d = tikh_family.d
    _noise_var = tikh_family.noise_var
    _btrue = tikh_family.btrue

    selected_lambdahs = []
    noise_norms = []
    np.random.seed(rseed)
    for j in range(n_samples):
        # draw new b and make new family
        noise = ( np.sqrt(_noise_var)*np.random.normal(size=len(tikh_family.btrue)) )
        noise_norms.append(np.linalg.norm(noise))
        _bsample = tikh_family.btrue + noise
        _tf = TikhonovFamily(_A, _L, _bsample, _d, gsvd=_gsvd)

        # solve problem
        dp_data = discrepancy_principle(_tf, delta=np.sqrt(len(_btrue)*_noise_var), tau=tau)
        selected_lambdahs.append( dp_data["opt_lambdah"] )

    return selected_lambdahs, noise_norms




def rand_lcorner(tikh_family, n_samples=10, rseed=0):
    """Randomly draws RHS vectors, solves DP problem.
    """

    assert tikh_family.btrue is not None, "Must pass btrue to TikhonovFamily"
    
    # Get stuff from original
    _gsvd = tikh_family.gsvd
    _A = tikh_family.A
    _L = tikh_family.L
    _d = tikh_family.d
    _noise_var = tikh_family.noise_var
    _btrue = tikh_family.btrue

    selected_lambdahs = []
    noise_norms = []
    np.random.seed(rseed)
    for j in range(n_samples):
        # draw new b and make new family
        noise = ( np.sqrt(_noise_var)*np.random.normal(size=len(tikh_family.btrue)) )
        noise_norms.append( noise )
        _bsample = tikh_family.btrue + noise
        _tf = TikhonovFamily(_A, _L, _bsample, _d, gsvd=_gsvd)

        # solve problem
        lcorner_data = lcorner(_tf)
        selected_lambdahs.append( lcorner_data["opt_lambdah"] )

    return selected_lambdahs, noise_norms




def rand_gcvmin(tikh_family, n_samples=10, rseed=0):
    """Randomly draws RHS vectors, solves GCV problem.
    """

    assert tikh_family.btrue is not None, "Must pass btrue to TikhonovFamily"
    
    # Get stuff from original
    _gsvd = tikh_family.gsvd
    _A = tikh_family.A
    _L = tikh_family.L
    _d = tikh_family.d
    _noise_var = tikh_family.noise_var
    _btrue = tikh_family.btrue

    selected_lambdahs = []
    noise_norms = []
    np.random.seed(rseed)
    for j in range(n_samples):
        # draw new b and make new family
        noise = ( np.sqrt(_noise_var)*np.random.normal(size=len(tikh_family.btrue)) )
        noise_norms.append(np.linalg.norm(noise))
        _bsample = tikh_family.btrue + noise
        _tf = TikhonovFamily(_A, _L, _bsample, _d, gsvd=_gsvd)

        # solve problem
        gcv_data = gcvmin(_tf)
        selected_lambdahs.append( gcv_data["opt_lambdah"] )

    return selected_lambdahs, noise_norms