import numpy as np
import math

import scipy.sparse as sps



def first_order_derivative_1d(N, boundary="none"):
    """Constructs a sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions. Also returns a dense matrix W
    whose column span the nullspace of the operator (if trivial, W = None).
    """
    
    assert boundary in ["none", "periodic", "zero", "reflexive", "zero_sym"], "Invalid boundary parameter."
    
    d_mat = sps.eye(N)
    d_mat.setdiag(-1,k=1)
    #d_mat = sps.csc_matrix(d_mat)
    d_mat = d_mat.tolil()
    
    if boundary == "periodic":
        d_mat[-1,0] = -1
        W = np.atleast_2d(np.ones(N)).T
    elif boundary == "zero":
        W = None
        pass
    elif boundary == "none":
        d_mat = d_mat[:-1,:]
        W = np.atleast_2d(np.ones(N)).T
    elif boundary == "reflexive":
        d_mat[-1,-1] = 0
        W = np.atleast_2d(np.ones(N)).T
    elif boundary == "zero_sym":
        d_mat = sps.csc_matrix(d_mat)
        new_row = sps.csc_matrix(np.zeros(d_mat.shape[1]))
        d_mat = sps.vstack([new_row, d_mat])
        d_mat[0,0] = -1
        W = None
    else:
        pass
    
    return d_mat, W



def second_order_derivative_1d(N, boundary="none"):
    """Constructs a SciPy sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions.
    """
    
    assert boundary in ["none"], "Invalid boundary parameter."
    
    d_mat = -sps.eye(N)
    d_mat.setdiag(2,k=1)
    d_mat.setdiag(-1,k=2)
    d_mat = d_mat.tolil()
    
    if boundary == "periodic":
        d_mat[-2,0] = -1
        d_mat[-1,0] = 2
        d_mat[-1,1] = -1
        raise NotImplementedError
    elif boundary == "zero":
        raise NotImplementedError
    elif boundary == "none":
        d_mat = d_mat[:-2, :]
        w1 = np.ones(N)/np.linalg.norm(np.ones(N))
        w2 = np.arange(N)/np.linalg.norm(np.arange(N))
        W = np.vstack([w1, w2]).T
    else:
        pass
    
    return d_mat, W



def third_order_derivative_1d(N, boundary="none"):
    """Constructs a SciPy sparse matrix that extracts the (1D) discrete gradient of an input signal.
    Boundary parameter specifies how to handle the boundary conditions.
    """
    
    assert boundary in ["none"], "Invalid boundary parameter."
    
    d_mat = -sps.eye(N)
    d_mat.setdiag(3,k=1)
    d_mat.setdiag(-3,k=2)
    d_mat.setdiag(1,k=3)
    d_mat = d_mat.tolil()
    
    if boundary == "periodic":
        raise NotImplementedError
    elif boundary == "zero":
        raise NotImplementedError
    elif boundary == "none":
        d_mat = d_mat[:-3, :]
        w1 = np.ones(N)/np.linalg.norm(np.ones(N))
        w2 = np.arange(N)/np.linalg.norm(np.arange(N))
        w3 = np.cumsum(w2)/np.linalg.norm(np.cumsum(w2))
        W = np.vstack([w1, w2, w3]).T
    else:
        pass
    
    return d_mat, W




def build_neumann2d_sparse_matrix(grid_shape):
     """Makes a sparse matrix corresponding to the matrix-free Neumann2D operator.
     """

     m, n = grid_shape

     Rv, _ = first_order_derivative_1d(m, boundary="reflexive")
     Rv *= -1.0

     Rh, _ = first_order_derivative_1d(n, boundary="reflexive")
     Rh *= -1.0

     return sps.vstack([sps.kron(Rv, sps.eye(n)), sps.kron(sps.eye(m), Rh) ])






