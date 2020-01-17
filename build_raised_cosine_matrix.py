
import numpy as np


def build_raised_cosine_matrix(nh, endpoints, b, dt):
    
    """
    Make basis of raised cosines with logarithmically stretched time asis.
    
    Ported from [matlab code](https://github.com/pillowlab/raisedCosineBasis) 
    by Jonathan Pillow
    
    Parameters
    ==========
    nh : int
        number of basis vectors
    
    endpoints : array like, shape=(2, )
        absoute temporal position of center of 1st and last cosine basis vector
        
    b : float
        offset for nonlinear stretching of x axis: y=log(x+b)
    
    dt : float
        time bin size of bins representing basis
        
    Return
    ======
    
    ttgrid : shape=(nt, )
        time lattice on which basis is defined
    
    basis : shape=(nt, nh)
        original cosine basis vectors
        
    """
    
    def nl(x):
        return np.log(x + 1e-20)
    
    def invnl(x):
        return np.exp(x) - 1e-20
    
    def raised_cosine_basis(x, c, dc):
        return 0.5 * (np.cos(np.maximum(-np.pi,np.minimum(np.pi,(x-c)*np.pi/dc/2)))+1)
    
    yendpoints = nl(endpoints + b)
    dctr = np.diff(yendpoints) / (nh - 1)
    ctrs = np.linspace(yendpoints[0], yendpoints[1], nh)
    maxt = invnl(yendpoints[1]+2*dctr) - b
    ttgrid = np.arange(0, maxt+dt, dt)
    nt = len(ttgrid)
    
    xx = np.tile(nl(ttgrid+b)[:, np.newaxis], (1, nh))
    cc = np.tile(ctrs, (nt, 1))
    
    basis = raised_cosine_basis(xx, cc, dctr)
    
    return ttgrid, basis
