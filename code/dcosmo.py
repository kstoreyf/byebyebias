import numpy as np
from nbodykit.lab import cosmology

from utils import partial_derivative


def write_bases(rmin, rmax,  saveto, ncont=300, params=None, cosmo_base=None):
	if not cosmo_base or if not params:
		raise ValueError("Must pass params and cosmo_base"!)
    bases = get_bases(rmin, rmax, saveto, ncont=ncont, params=params, cosmo_base=cosmo_base)
    np.savetxt(saveto, bases.T)
    return saveto


def get_bases(rmin, rmax,  ncont=300, params=None, cosmo_base=None):
	nbases = len(params)+1
	rcont = np.linspace(rmin, rmax, ncont)
    bases = np.empty((nbases+1, ncont))
	bases[0,:] = rcont

	Plin = cosmology.LinearPower(cosmo_base, redshift, transfer='EisensteinHu')
	CF = cosmology.correlation.CorrelationFunction(Plin)
	xi_base = CF(rcont)
	bases[1,:] = xi_base

	cosmo_derivs = []
	ds = []
	for i in range(len(params)):
		param = params[i]
	    cosmo_dict = dict(cosmo_base)
	    val_base = cosmo_dict[param]
	    dval = val_base * 0.01
	    val_new = val_base + dval
	    cosmo_dict[param] = val_new
	    cosmo = cosmology.Cosmology.from_dict(cosmo_dict))
		
	    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
	    CF = cosmology.correlation.CorrelationFunction(Plin)
	    xi = CF(rcont)
	
		dcosmo = partial_derivative(xi_base, xi, dval)
		bases[i+2, :] = dcosmo

	return bases
	
	
