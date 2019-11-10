import numpy as np
from nbodykit.lab import cosmology


def write_bases(rmin, rmax, saveto, ncont=300, **kwargs):
    print(saveto)
    bases = get_bases(rmin, rmax, ncont=ncont, **kwargs)
    np.savetxt(saveto, bases.T)
    nprojbins = ncont
    return nprojbins, saveto


def get_bases(rmin, rmax, ncont=300, sigma=3.0):
   
    rcent = 0.0
    rbins = np.linspace(rmin, rmax, ncont+1)
    rs = 0.5*(rbins[1:]+rbins[:-1])
    deltar = (max(rs)-min(rs))/float(ncont)
    gs_pos = np.arange(0, 4*sigma, step=deltar)
    gs_neg = np.arange(0, -4*sigma, step=-deltar)
    gs_neg = gs_neg[1:][::-1] #kill the zero and reverse order
    gs = np.concatenate((gs_neg, gs_pos))
    
    def gaussian(x, mu, sigma):
        return 1.0/(np.sqrt(2.0*np.pi)*sigma) * np.exp(-0.5*((x-mu)/sigma)**2)

    bases = np.empty((2, len(gs)))
    bases[0,:] = gs
    bases[1,:] = gaussian(gs, rcent, sigma)

    return bases
    
