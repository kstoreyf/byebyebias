#!/usr/bin/env python
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import os

from nbodykit.lab import *
import nbodykit
import Corrfunc
from Corrfunc.theory.DD import DD
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u

import plotter
import corrfuncproj
import spline

from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi


plot_dir = '../plots/plots_2019-10-03'
result_dir = '../results/results_2019-10-03'
cat_dir = '../catalogs/catalogs_2019-09-30'
nthreads = 24
color_dict = {'True':'black', 'tophat':'blue', 'LS': 'orange', 'piecewise':'red', 'standard': 'orange'}


def main():

    multi()


def multi():
    boxsize = 750
    nbar_str = '1e-5'
    #nbar_str = '3e-4'
    #projs = ['quadratic_spline']
    #proj_tags = ['quadratic']
    projs = ['dcosmo']
    proj_tags = ['dcosmo_test']

    # TODO: make sure cosmo is aligned with sim loaded in
    kwargs = {'params':['Omega_cdm', 'Omega_b', 'h'], 'cosmo_base':nbodykit.cosmology.Planck15}

    nrealizations = 1
    seeds = np.arange(nrealizations)
    cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    cat_dir = '../catalogs/cats_lognormal{}'.format(cat_tag)
    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    log = False

    # randoms
    rand_fn = '{}/rand{}_10x.dat'.format(cat_dir, cat_tag)
    random = np.loadtxt(rand_fn)
    nr = random.shape[0]
    randsky_fn = '{}/randsky{}_10x.dat'.format(cat_dir, cat_tag)        
    randomsky = np.loadtxt(randsky_fn)

    print("Set up bins")
    if log:
        rmin = 1
    else:
        rmin = 40
    rmax = 150
    nbins = 16
    rbins = np.linspace(rmin, rmax, nbins)
    rbins_avg = 0.5*(rbins[1:]+rbins[:-1])
    r_lin, _, _ = np.load('{}/cf_lin_{}{}.npy'.format(cat_dir, 'true', cat_tag))#, allow_pickle=True)
    if log:
        rbins_log = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
        rbins_avg_log = 10 ** (0.5 * (np.log10(rbins_log)[1:] + np.log10(rbins_log)[:-1]))
        r_log, _, _ = np.load('{}/cf_log_{}{}.npy'.format(cat_dir, 'true', cat_tag))

    ### random
    rr = counts_corrfunc_auto(random, rbins, boxsize)
    
    rr_projs, qq_projs = [], []
    for i in range(len(projs)):
        rr_proj, qq_proj = counts_cf_proj_auto(randomsky, rbins, r_lin, projs[i], qq=True, **kwargs)
        rr_projs.append(rr_proj)
        qq_projs.append(qq_proj)

    for seed in seeds:

        data_fn = '{}/cat_lognormal{}_seed{}.dat'.format(cat_dir, cat_tag, seed)
        data = np.loadtxt(data_fn)
        nd = data.shape[0]
        datasky_fn = '{}/catsky_lognormal{}_seed{}.dat'.format(cat_dir, cat_tag, seed)
        datasky = np.loadtxt(datasky_fn)

        dd = counts_corrfunc_auto(data, rbins, boxsize)
        dr = counts_corrfunc_cross(data, random, rbins, boxsize)
        xi_stan = compute_cf(dd, dr, rr, nd, nr, 'ls')
        np.save('{}/cf_lin_{}{}_seed{}.npy'.format(result_dir, 'standard', cat_tag, seed), [rbins_avg, xi_stan, 'standard'])

        for i in range(len(projs)):
           
            proj = projs[i]
            dd_proj = counts_cf_proj_auto(datasky, rbins, r_lin, proj, qq=False, **kwargs)
            dr_proj = counts_cf_proj_cross(datasky, randomsky, rbins, r_lin, proj, **kwargs)
            r_cont, xi_proj = compute_cf_proj(dd_proj, dr_proj, rr_projs[i], qq_projs[i], nd, nr, rbins, r_lin, proj, **kwargs)
            np.save('{}/cf_lin_{}{}_seed{}.npy'.format(result_dir, proj_tags[i], cat_tag, seed), [r_lin, xi_proj, proj])



def single():

    boxsize = 750
    nbar_str = '3e-4'
    projs = ['quadratic_spline']
    #projs = ['tophat', 'piecewise']
    #projs = []
    cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    tag = '_nbins8'+cat_tag
    log = False

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
    if log:
        rmin = 1
    else:
        rmin = 40
    rmax = 150
    nbins = 9
    rbins = np.linspace(rmin, rmax, nbins)
    rbins_avg = 0.5*(rbins[1:]+rbins[:-1])
    r_lin, _, _ = np.load('{}/cf_lin_{}{}.npy'.format(cat_dir, 'true', cat_tag))
    if log:
        rbins_log = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
        rbins_avg_log = 10 ** (0.5 * (np.log10(rbins_log)[1:] + np.log10(rbins_log)[:-1]))
        r_log, _, _ = np.load('{}/cf_log_{}{}.npy'.format(cat_dir, 'true', cat_tag))

    print("Calc standard corrfunc CF")
    print("LIN")
    dd, dr, rr = counts_corrfunc_3d(data, random, rbins, boxsize)
    xi_stan = compute_cf(dd, dr, rr, nd, nr, 'ls') 
    np.save('{}/cf_lin_{}{}.npy'.format(result_dir, 'standard', tag), [rbins_avg, xi_stan, 'standard'])
    if log:
        print("LOG")
        dd_log, dr_log, rr_log = counts_corrfunc_3d(data, random, rbins_log, boxsize)
        xi_stan_log = compute_cf(dd_log, dr_log, rr_log, nd, nr, 'ls')
        np.save('{}/cf_log_{}{}.npy'.format(result_dir, 'standard', tag), [rbins_avg_log, xi_stan_log, 'standard'])

    print("Calc projected CF")
    for proj in projs:
        print("LIN PROJ")
        dd, dr, rr, qq = counts_cf_proj(datasky, randomsky, rbins, r_lin, proj)
        r_cont, xi_proj = compute_cf_proj(dd, dr, rr, qq, nd, nr, rbins, r_cont, proj)
        np.save('{}/cf_lin_{}{}.npy'.format(result_dir, proj, tag), [r_lin, xi_proj, proj])
        if log:
            print("LOG PROJ")
            dd, dr, rr, qq = counts_cf_proj(datasky, randomsky, rbins_log, r_log, proj)
            r_cont_log, xi_proj_log = compute_cf_proj(dd, dr, rr, qq, nd, nr, rbins_log, r_cont_log, proj)
            np.save('{}/cf_log_{}{}.npy'.format(result_dir, proj, tag), [r_log, xi_proj_log, proj])
        

def counts_corrfunc_auto(cat, rbins, boxsize):
    x, y, z = cat.T
    periodic = False
    print('Starting counts')
    s = time.time()
    dd = DD(1, nthreads, rbins, X1=x, Y1=y, Z1=z,
               periodic=periodic, boxsize=boxsize)
    dd = np.array([x[3] for x in dd])
    print(dd)
    print('time: {}'.format(time.time()-s))
    return dd


def counts_corrfunc_cross(data, random, rbins, boxsize):
    s = time.time()
    periodic = False
    datax, datay, dataz = data.T
    randx, randy, randz = random.T
    
    dr = DD(0, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
               periodic=periodic, X2=randx, Y2=randy, Z2=randz, boxsize=boxsize)
    dr = np.array([x[3] for x in dr])
    print(dr)
    print('time: {}'.format(time.time()-s))
    return dr


def counts_corrfunc_3d(data, random, rbins, boxsize):
    print(data.shape)
    datax, datay, dataz = data.T
    randx, randy, randz = random.T
    
    periodic = True
    print('Starting counts')
    s = time.time()
    dd = DD(1, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
               periodic=periodic, boxsize=boxsize)
    dd = np.array([x[3] for x in dd])
    print(dd)
    print('time: {}'.format(time.time()-s))
    s = time.time()
    dr = DD(0, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
               periodic=periodic, X2=randx, Y2=randy, Z2=randz, boxsize=boxsize)
    dr = np.array([x[3] for x in dr])
    print(dr)
    print('time: {}'.format(time.time()-s))
    s = time.time()
    rr = DD(1, nthreads, rbins, randx, randy, randz,
                periodic=periodic, boxsize=boxsize)
    rr = np.array([x[3] for x in rr])
    print(rr)
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


def counts_cf_proj_auto(cat, rbins, r_cont, proj, qq=False, **kwargs):
    cosmo = 1 #doesn't matter bc passing cz, but required

    ra, dec, cz = cat.T

    proj_type, nprojbins, projfn = get_proj_parameters(proj, rbins=rbins, **kwargs)

    mumax = 1.0 #max of cosine
    weights = None
    nproc = nthreads
    res = corrfuncproj.counts_smu_auto(ra, dec, cz, 
                                  rbins, mumax, cosmo, nproc=nproc,
                                  weights=weights,
                                  comoving=True, proj_type=proj_type,
                                  nprojbins=nprojbins, projfn=projfn, qq=qq)
    if qq:
        dd_proj, dd_res_corrfunc, dd_projt = res
        return dd_proj, dd_projt
    else: 
        dd_proj, dd_res_corrfunc = res
        return dd_proj
    

def counts_cf_proj_cross(data, random, rbins, r_cont, proj, **kwargs):
    cosmo = 1 #doesn't matter bc passing cz, but required
    
    datara, datadec, datacz = data.T
    randra, randdec, randcz = random.T

    proj_type, nprojbins, projfn = get_proj_parameters(proj, rbins=rbins, **kwargs)

    mumax = 1.0 #max of cosine
    weights_data = None
    weights_rand = None
    nproc = nthreads
    res = corrfuncproj.counts_smu_cross(datara, datadec,
                                  datacz, randra, randdec,
                                  randcz, rbins, mumax, cosmo, nproc=nproc,
                                  weights_data=weights_data, weights_rand=weights_rand,
                                  comoving=True, proj_type=proj_type,
                                  nprojbins=nprojbins, projfn=projfn)

    dr_proj, dr_res_corrfunc = res
    return dr_proj



def counts_cf_proj(data, random, rbins, r_cont, proj, **kwargs):

    #cosmo = LambdaCDM(H0=70, Om0=0.25, Ode0=0.75)
    cosmo = 1 #doesn't matter bc passing cz, but required

    datara, datadec, datacz = data.T
    randra, randdec, randcz = random.T

    proj_type, nprojbins, projfn = get_proj_parameters(proj, rbins=rbins, **kwargs)

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

    return dd, dr, rr, qq


def compute_cf_proj(dd, dr, rr, qq, nd, nr, rbins, r_cont, proj, **kwargs):

    proj_type, nprojbins, projfn = get_proj_parameters(proj, rbins=rbins, **kwargs)
    # Note: dr twice because cross-correlations will be possible
    amps = compute_amps(nprojbins, nd, nd, nr, nr, dd, dr, dr, rr, qq)
    print('Computed amplitudes')

    amps = np.array(amps)

    rbins = np.array(rbins)
    xi_proj = evaluate_xi(nprojbins, amps, len(r_cont), r_cont, len(rbins), rbins, proj_type, projfn=projfn)
    print("Computed xi")
    return r_cont, xi_proj



def get_proj_parameters(proj, rbins=None):
    proj_type = proj
    projfn = None
    if proj=="tophat" or proj_type=="piecewise":
        nprojbins = len(rbins)-1
    elif proj=="powerlaw":
        nprojbins = 3
    elif proj=='generalr':
        nprojbins = 6
        projfn = "/home/users/ksf293/vectorizedEstimator/tables/dcosmos_rsd_norm.dat"
    elif proj=='dcosmo':
        proj_type = 'generalr'
        #params = ['Omega_cdm', 'Omega_b', 'h']
        projfn = '../tables/dcosmo.dat'
        nprojbins = dcosmo.write_bases(rbins[0], rbins[-1], projfn, **kwargs)
    elif proj=='linear_spline':
        nprojbins = len(rbins)-1
        proj_type = 'generalr'
        projfn = '../tables/linear_spline.dat'
        spline.write_bases(rbins[0], rbins[-1], len(rbins)-1, 1, projfn)
    elif proj=='quadratic_spline':
        nprojbins = len(rbins)-1
        proj_type = 'generalr'
        projfn = '../tables/quadratic_spline.dat'
        spline.write_bases(rbins[0], rbins[-1], len(rbins)-1, 2, projfn)
    else:
      raise ValueError("Proj type {} not recognized".format(proj_type))
    print("nprojbins:", nprojbins)
    return proj_type, nprojbins, projfn


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
