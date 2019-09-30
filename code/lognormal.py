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
result_dir = '../results/results_2019-09-26'
#tag = 'nbar1e-4_top'
nthreads = 24
color_dict = {'True':'black', 'tophat':'blue', 'LS': 'orange', 'piecewise':'red', 'standard': 'orange'}
label_dict = {'generalr': 'cosmo deriv', 'tophat': 'tophat', 'piecewise':'linear spline'}


def main():

    nbar_str = '3e-4'
    #proj_types = ['tophat','generalr']
    proj_types = ['piecewise']
    tag = '_nbar{}'.format(nbar_str)

    nbar = float(nbar_str)

    print(Corrfunc.__version__)
    redshift = 0
    cosmo = cosmology.Planck15
    print("Generating power spectrum")
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    b1 = 1.0

    print("Making catalog")
    s = time.time()
    #boxsize = 100.
    #hval = 0.7
    boxsize = 750.
    
    rmin = 1
    rmax = 150
    nbins = 20
    rbins = np.linspace(rmin, rmax, nbins)
    rbins_avg = 0.5*(rbins[1:]+rbins[:-1])
    rbins_log = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
    rbins_avg_log = 10 ** (0.5 * (np.log10(rbins_log)[1:] + np.log10(rbins_log)[:-1]))
    
    r_lin = np.linspace(rmin, rmax, 300)
    r_log = np.logspace(np.log10(rmin), np.log10(rmax), 300)

    data = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=boxsize, Nmesh=256, bias=b1, seed=42)
    print('time: {}'.format(time.time()-s)) 
    nd = data.csize
    s = time.time()
    random = nbodykit.source.catalog.uniform.UniformCatalog(10*nbar, boxsize, seed=43)
    nr = random.csize
    print('time: {}'.format(time.time()-s))
    print(nd, nr)   

    datasky = to_sky(data['Position'], data['Velocity'], cosmo)
    randomsky = to_sky(random['Position'], random['Velocity'], cosmo)

    print("Calcd true CF")
    CF = nbodykit.cosmology.correlation.CorrelationFunction(Plin)
    xi_log = CF(r_log)#, smoothing=0.0, kmin=10**-2, kmax=10**0)
    xi_lin = CF(r_lin)
    np.save('{}/cf_lin_{}{}.npy'.format(result_dir, 'true', tag), [r_lin, xi_lin, 'true'])
    np.save('{}/cf_log_{}{}.npy'.format(result_dir, 'true', tag), [r_log, xi_log, 'true'])
    
    print("Calc corrfunc CF")
    print("LIN")
    dd, dr, rr = counts_corrfunc_3d(data, random, rbins)
    xi_stan = compute_cf(dd, dr, rr, nd, nr, 'ls') 
    print("LOG")
    dd_log, dr_log, rr_log = counts_corrfunc_3d(data, random, rbins_log)
    xi_stan_log = compute_cf(dd_log, dr_log, rr_log, nd, nr, 'ls')
    
    #xi_nat = compute_cf(dd, dr, rr, nd, nr, 'natural')
    rs_lin = [rbins_avg]
    rs_log = [rbins_avg_log]
    cfs_lin = [xi_stan]
    cfs_log = [xi_stan_log]
    np.save('{}/cf_lin_{}{}.npy'.format(result_dir, 'standard', tag), [rbins_avg, xi_stan, 'standard'])
    np.save('{}/cf_log_{}{}.npy'.format(result_dir, 'standard', tag), [rbins_avg_log, xi_stan_log, 'standard'])
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
        

    #SimCF = nbodykit.algorithms.paircount_tpcf.tpcf.SimulationBox2PCF('2d', cat, edges, Nmu=1, randoms1=random, periodic=True, BoxSize=boxsize, show_progress=True)
    #SimCF.run()
    #xi_ls = SimCF.corr

    print("Plot")
    plt.figure()
    k = np.logspace(-3, 2, 300)
    plt.loglog(k, b1**2 * Plin(k), c='k', label=r'$b_1^2 P_\mathrm{lin}$')
    plt.savefig('{}/pk{}.png'.format(plot_dir, tag))

    save_lin = '{}/cf_lin_resid{}.png'.format(plot_dir, tag)
    #plotter.plot_cf_cont(rs_lin, cfs_lin, labels, r_lin, xi_lin, saveto=save_lin, log=False, err=True)

    save_log = '{}/cf_log_resid{}.png'.format(plot_dir, tag)
    #plotter.plot_cf_cont(rs_log, cfs_log, labels, r_log, xi_log, saveto=save_log, log=True, err=True)



def counts_corrfunc_3d(data, random, rbins):

    datax = np.array(data['Position'][:,0]).astype(float)
    datay = np.array(data['Position'][:,1]).astype(float)
    dataz = np.array(data['Position'][:,2]).astype(float)
    
    randx = np.array(random['Position'][:,0]).astype(float)
    randy = np.array(random['Position'][:,1]).astype(float)
    randz = np.array(random['Position'][:,2]).astype(float)
    
    periodic = True
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
    
    datara = data[0].compute().astype(float)
    datadec = data[1].compute().astype(float)
    datacz = data[2].compute().astype(float)

    randra = random[0].compute().astype(float)
    randdec = random[1].compute().astype(float)
    randcz = random[2].compute().astype(float)
    print(datara)

    #datax = np.array(data['Position'][:,0]).astype(float)
    #datay = np.array(data['Position'][:,1]).astype(float)
    #dataz = np.array(data['Position'][:,2]).astype(float)

    #randx = np.array(random['Position'][:,0]).astype(float)
    #randy = np.array(random['Position'][:,1]).astype(float)
    #randz = np.array(random['Position'][:,2]).astype(float)    

    #print("to sky")
    # TODO: use pandas dataframe for faster operation?
    #datara = np.zeros(nd)
    #datadec = np.zeros(nd)
    #datacz = np.zeros(nd)
    #for i in range(nd):
    #    c = SkyCoord(x=datax[i], y=datay[i], z=dataz[i], unit='Mpc', representation='cartesian')
    #    datara[i] = c.spherical.lon.value
    #    datadec[i] = c.spherical.lat.value
    #    datacz[i] = c.spherical.distance.value

    #randra = np.zeros(nr)
    #randdec = np.zeros(nr)
    #randcz = np.zeros(nr)
    #for i in range(nr):
    #    c = SkyCoord(x=randx[i], y=randy[i], z=randz[i], unit='Mpc', representation='cartesian')
    #    randra[i] = c.spherical.lon.value
    #    randdec[i] = c.spherical.lat.value
    #    randcz[i] = c.spherical.distance.value


    #K = 14
    #smin = min(rbins)
    #smax = max(rbins)
    #sbins = np.linspace(smin, smax, K + 1)
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
                                  comoving=False, proj_type=proj_type,
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


def to_sky(pos, velocity, cosmo):
    #ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo, velocity=velocity, observer=[0, 0, 0], zmax=100.0, frame='icrs')
    ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo)
    return ra, dec, z


if __name__=='__main__':
    main()
