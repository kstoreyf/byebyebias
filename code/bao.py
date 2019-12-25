import numpy as np
from nbodykit.lab import cosmology

from utils import partial_derivative


def write_bases(rmin, rmax,  saveto, ncont=300, cosmo_base=None):
	if not cosmo_base or if not params:
		raise ValueError("Must pass params and cosmo_base"!)
    bases = get_bases(rmin, rmax, saveto, ncont=ncont, cosmo_base=cosmo_base)
    np.savetxt(saveto, bases.T)
    return saveto

def bao_bases(s, cf_func, p1=1, p2=1, p3=1):
    
    b1 = p1 * 1.0/s**2
    b2 = p2 * 1.0/s
    b3 = p3 * np.ones(len(s))
    
    b4 = CF(s)
    
    alpha = 1.01
    dalpha = 1-alpha
    dxi_dalpha = partial_derivative(cf_func(s), cf_func(alpha*s), dalpha)
    b5 = dalpha*dxi_dalpha
    
    return b1,b2,b3,b4,b5


def get_bases(rmin, rmax,  ncont=300, cosmo_base=None):

	Plin = cosmology.LinearPower(cosmo_base, redshift, transfer='EisensteinHu')
	CF = cosmology.correlation.CorrelationFunction(Plin)

    rcont = np.linspace(rmin, rmax, ncont)
    bs = bao_bases(rcont, CF)

    nbases = len(bs)	
    bases = np.empty((nbases+1, ncont))
    bases[0,:] = rcont
    bases[1:nbases+1,:] = bs

	return bases
	
	
