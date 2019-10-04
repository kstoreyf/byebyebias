import numpy as np
import scipy
from scipy.interpolate import BSpline


def main():
	
	#test use
	rmin = 0
	rmax = 4
	nbins = 5
	order = 2
	bases = get_bases(rmin, rmax, nbins, order)


def write_bases(rmin, rmax, nbins, order, saveto, ncont=300):
	bases = get_bases(rmin, rmax, nbins, order, ncont=ncont)
	np.savetxt(saveto, bases.T)
	return saveto

# get knot vectors
def get_kvs(rmin, rmax, nbins, order):
    nknots = order+2
    kvs = np.empty((nbins, nknots))

    width = (rmax-rmin)/(nbins-order)
    print(width)
    for i in range(order):
        val = i+1
        kvs[i,:] = np.concatenate((np.full(nknots-val, rmin), np.linspace(rmin+width, rmin+width*val, val)))
        kvs[nbins-i-1] = np.concatenate((np.linspace(rmax-width*val, rmax-width, val), np.full(nknots-val, rmax)))
    for j in range(nbins-2*order):
        idx = j+order
        kvs[idx] = rmin+width*j + np.arange(0,nknots)*width                     
    return kvs

def get_bases(rmin, rmax, nbins, order, ncont=300):
    if nbins<order*2:
        # does it have to be 2*order + 1? seems fine for piecewise, but for higher orders?
        raise ValueError("nbins must be at least twice the order")
    kvs = get_kvs(rmin, rmax, nbins, order)
    print(kvs)
    rcont = np.linspace(rmin, rmax, ncont)
    bases = np.empty((nbins+1, ncont))
    bases[0,:] = rcont
    for n in range(nbins):
        kv = kvs[n]
        b = BSpline.basis_element(kv)
        bases[n+1,:] = [b(r) if kv[0]<=r<=kv[-1] else 0 for r in rcont]
    return bases


if __name__=='__main__':
	main()
