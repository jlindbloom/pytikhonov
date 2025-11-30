import numpy as np

from .lcurve import lcorner
from .gcv import gcvmin
from .discrepancy_principle import discrepancy_principle




def all_regparam_methods(tikh_family):
    """Runs all regularization parameter selection methods.
    """
    lcurve_data = lcorner(tikh_family)
    gcv_data = gcvmin(tikh_family)
    
    data = {
        "lcurve_data": lcurve_data,
        "gcv_data": gcv_data,
    }  
    if tikh_family.btrue is not None: data["dp_data"] = discrepancy_principle(tikh_family, delta=np.sqrt(tikh_family.M*tikh_family.noise_var))

    return data

