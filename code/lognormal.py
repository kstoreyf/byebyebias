#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from nbodykit.lab import *
import nbodykit
import Corrfunc
from Corrfunc.theory.DD import DD
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u

import plotter
import corrfuncproj

from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi


plot_dir = '../plots/plots_2019-09-30'
cat_dir = '../catalogs/catalogs_2019-09-30'


def main():

    boxsize = 750
    nbar_str = '3e-4'
    tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    cat_fn = '{}/cat_lognormal{}.dat'.format(cat_dir, tag)
    rand_fn = '{}/rand{}_10x.dat'.format(cat_dir, tag)
    catsky_fn = '{}/catsky_lognormal{}.dat'.format(cat_dir, tag)
    randsky_fn = '{}/randsky{}_10x.dat'.format(cat_dir, tag)
    pk_fn = '{}/pk{}.dat'.format(cat_dir, tag)

    nbar = float(nbar_str)
    boxside = float(boxsize)

    redshift = 0
    cosmo = cosmology.Planck15
    print("Generating power spectrum")
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    b1 = 1.0

    print("Power spectrum and correlation function")
    k = np.logspace(-3, 2, 300)
    Pk = Plin(k)
    np.save(pk_fn, [k, Pk])
    rmin = 1
    rmax = 150
    r_lin = np.linspace(rmin, rmax, 300)
    r_log = np.logspace(np.log10(rmin), np.log10(rmax), 300)
    CF = nbodykit.cosmology.correlation.CorrelationFunction(Plin)
    xi_log = CF(r_log)#, smoothing=0.0, kmin=10**-2, kmax=10**0)
    xi_lin = CF(r_lin)
    np.save('{}/cf_lin_{}{}.npy'.format(cat_dir, 'true', tag), [r_lin, xi_lin, 'true'])
    np.save('{}/cf_log_{}{}.npy'.format(cat_dir, 'true', tag), [r_log, xi_log, 'true'])

    print("Making data catalog")
    s = time.time()
    data = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=boxsize, Nmesh=256, bias=b1, seed=42)
    print('time: {}'.format(time.time()-s)) 
    nd = data.csize

    print("Making random catalog")
    s = time.time()
    random = nbodykit.source.catalog.uniform.UniformCatalog(10*nbar, boxsize, seed=43)
    print('time: {}'.format(time.time()-s))
    nr = random.csize
    print(nd, nr)   

    datasky = to_sky(data['Position'], cosmo)
    randomsky = to_sky(random['Position'], cosmo)

    np.savetxt(catsky_fn, np.array(datasky).T)
    np.savetxt(randsky_fn, np.array(randomsky).T) 

    data = get_positions(data)
    random = get_positions(random)

    np.savetxt(cat_fn, np.array(data).T)
    np.savetxt(rand_fn, np.array(random).T)

    np.savetxt(catsky_fn, np.array(datasky).T)
    np.savetxt(randsky_fn, np.array(randomsky).T)


def to_sky(pos, cosmo, velocity=None, rsd=False, comoving=True):
    if rsd:
        if velocity is None:
            raise ValueError("Must provide velocities for RSD! Or set rsd=False.")
        ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo, velocity=velocity, observer=[0, 0, 0], zmax=100.0, frame='icrs')
    else:
        ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo)
    if comoving:
        z = cosmo.comoving_distance(z)

    ra = ra.compute().astype(float)
    dec = dec.compute().astype(float)
    return ra, dec, z


def get_positions(cat):
    catx = np.array(cat['Position'][:,0]).astype(float)
    caty = np.array(cat['Position'][:,1]).astype(float)
    catz = np.array(cat['Position'][:,2]).astype(float)
    return catx, caty, catz

if __name__=='__main__':
    main()
