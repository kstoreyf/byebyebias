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
result_dir = '../results/results_2019-09-30'
cat_dir = '../catalogs/catalogs_2019-09-30'
nthreads = 24
color_dict = {'True':'black', 'tophat':'blue', 'LS': 'orange', 'piecewise':'red', 'standard': 'orange'}
label_dict = {'generalr': 'cosmo deriv', 'tophat': 'tophat', 'piecewise':'linear spline'}


def main():

    boxsize = 750
    nbar_str = '3e-4' 
    #proj_types = ['tophat','generalr']
    #proj_types = ['tophat']
    proj_types = ['piecewise']
    cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    tag = cat_tag

    print("Get data")
    cat_fn = '{}/cat_lognormal{}.dat'.format(cat_dir, cat_tag)
    rand_fn = '{}/rand{}_10x.dat'.format(cat_dir, cat_tag)
    catsky_fn = '{}/catsky_lognormal{}.dat'.format(cat_dir, cat_tag)
    randsky_fn = '{}/randsky{}_10x.dat'.format(cat_dir, cat_tag)
    
    data = np.loadtxt(cat_fn)
    random = np.loadtxt(rand_fn) 
    datasky = np.loadtxt(catsky_fn)
    randomsky = np.loadtxt(randsky_fn)
    nd = data.shape[0]
    nr = random.shape[0]

    print("Set up bins")
    rmin = 1
    rmax = 150
    nbins = 20
    rbins = np.linspace(rmin, rmax, nbins)
    rbins_avg = 0.5*(rbins[1:]+rbins[:-1])
    rbins_log = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
    rbins_avg_log = 10 ** (0.5 * (np.log10(rbins_log)[1:] + np.log10(rbins_log)[:-1]))
    r_lin, _, _ = np.load('{}/cf_lin_{}{}.npy'.format(cat_dir, 'true', cat_tag))
    r_log, _, _ = np.load('{}/cf_log_{}{}.npy'.format(cat_dir, 'true', cat_tag))

    print("Calc standard corrfunc CF")
    print("LIN")
    dd, dr, rr = counts_corrfunc_3d(data, random, rbins)
    xi_stan = compute_cf(dd, dr, rr, nd, nr, 'ls') 
    print("LOG")
    dd_log, dr_log, rr_log = counts_corrfunc_3d(data, random, rbins_log)
    xi_stan_log = compute_cf(dd_log, dr_log, rr_log, nd, nr, 'ls')
    np.save('{}/cf_lin_{}{}.npy'.format(result_dir, 'standard', tag), [rbins_avg, xi_stan, 'standard'])
    np.save('{}/cf_log_{}{}.npy'.format(result_dir, 'standard', tag), [rbins_avg_log, xi_stan_log, 'standard'])

    print("Calc projected CF")
    rs_lin = [rbins_avg]
    rs_log = [rbins_avg_log]
    cfs_lin = [xi_stan]
    cfs_log = [xi_stan_log]
    labels = ['standard']
    
    for proj_type in proj_types:
        labels.append(label_dict[proj_type])
        print("LIN PROJ")
        r_cont, xi_proj = compute_cf_proj(datasky, randomsky, nd, nr, rbins, r_lin, proj_type)
        rs_lin.append(r_lin)
        cfs_lin.append(xi_proj)

        print("LOG PROJ")
        r_cont_log, xi_proj_log = compute_cf_proj(datasky, randomsky, nd, nr, rbins_log, r_log, proj_type)
        rs_log.append(r_log)
        cfs_log.append(xi_proj_log)
        
        np.save('{}/cf_lin_{}{}.npy'.format(result_dir, proj_type, tag), [r_lin, xi_proj, proj_type])
        np.save('{}/cf_log_{}{}.npy'.format(result_dir, proj_type, tag), [r_log, xi_proj_log, proj_type])
        



def counts_corrfunc_3d(data, random, rbins):

    print data.shape
    datax, datay, dataz = data.T
    randx, randy, randz = random.T
    
    periodic = False
    print('Starting counts')
    s = time.time()
    dd = DD(1, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
               periodic=periodic)
    dd = np.array([x[3] for x in dd])
    print dd
    print('time: {}'.format(time.time()-s))
    s = time.time()
    dr = DD(0, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
               periodic=periodic, X2=randx, Y2=randy, Z2=randz)
    dr = np.array([x[3] for x in dr])
    print dr
    print('time: {}'.format(time.time()-s))
    s = time.time()
    rr = DD(1, nthreads, rbins, randx, randy, randz,
                periodic=periodic)
    rr = np.array([x[3] for x in rr])
    print rr
    print('time: {}'.format(time.time()-s))

    return dd, dr, rr


def compute_cf(dd, dr, rr, nd, nr, est):

    dd = np.array(dd).astype(float)
    dr = np.array(dr).astype(float)
    rr = np.array(rr).astype(float)

    fN = float(nr)/float(nd)
    if est=='ls':
        return (dd * fN**2 - 2*dr * fN + rr)/rr
    elif est=='natural':
        return fN**2*(dd/rr) - 1
    elif est=='dp':
        return fN*(dd/dr) - 1
    elif est=='ham':
        return (dd*rr)/(dr**2) - 1
    else:
        exit("Estimator '{}' not recognized".format(est))


def compute_cf_proj(data, random, nd, nr, rbins, r_cont, proj_type):

    cosmo = LambdaCDM(H0=70, Om0=0.25, Ode0=0.75)
    #cosmo = 1 #doesn't matter bc passing cz, but required

    datara, datadec, datacz = data.T
    randra, randdec, randcz = random.T
    #datara = data[0].compute().astype(float)
    #datadec = data[1].compute().astype(float)
    #datacz = data[2].compute().astype(float)

    #randra = random[0].compute().astype(float)
    #randdec = random[1].compute().astype(float)
    #randcz = random[2].compute().astype(float)

    projfn = None
    if proj_type=="tophat" or proj_type=="piecewise":
        nprojbins = len(rbins)-1
    elif proj_type=="powerlaw":
        nprojbins = 3
    elif proj_type=='generalr':
        nprojbins = 6
        projfn = "/home/users/ksf293/vectorizedEstimator/tables/dcosmos_rsd_norm.dat"
    else:
      raise ValueError("Proj type {} not recognized".format(proj_type))
    print "nprojbins:", nprojbins

    mumax = 1.0 #max of cosine
    weights_data = None
    weights_rand = None
    nproc = nthreads
    res = corrfuncproj.counts_smu(datara, datadec,
                                  datacz, randra, randdec,
                                  randcz, rbins, mumax, cosmo, nproc=nproc,
                                  weights_data=weights_data, weights_rand=weights_rand,
                                  comoving=True, proj_type=proj_type,
                                  nprojbins=nprojbins, projfn=projfn)

    dd, dr, rr, qq, dd_orig, dr_orig, rr_orig = res
    print("Projected results:")
    print(dd)
    print(dr)
    print(rr)
    # Note: dr twice because cross-correlations will be possible
    amps = compute_amps(nprojbins, nd, nd, nr, nr, dd, dr, dr, rr, qq)
    print 'Computed amplitudes'

    amps = np.array(amps)
    #svals = np.linspace(min(rbins), max(rbins), 300)

    rbins = np.array(rbins)
    xi_proj = evaluate_xi(nprojbins, amps, len(r_cont), r_cont, len(rbins), rbins, proj_type, projfn=projfn)
    print "Computed xi"
    return r_cont, xi_proj


def to_sky(pos, cosmo, velocity=None, rsd=False, comoving=True):
    if rsd:
        if velocity is None:
            raise ValueError("Must provide velocities for RSD! Or set rsd=False.")
        ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo, velocity=velocity, observer=[0, 0, 0], zmax=100.0, frame='icrs')
    else:
        ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo)

    if comoving:
        z = cosmo.comoving_distance(z)
    return ra, dec, z


if __name__=='__main__':
    main()
