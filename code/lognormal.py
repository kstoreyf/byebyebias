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


plot_dir = '../plots/plots_2019-09-16'
tag = 'nbar1e-4_top'
nthreads = 24

def main():
    print(Corrfunc.__version__)
    redshift = 0
    cosmo = cosmology.Planck15
    print("Generating power spectrum")
    Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
    b1 = 1.0

    print("Making catalog")
    s = time.time()
    #boxsize = 100.
    boxsize = 750.
    rmin = 1
    rmax = 150
    rbins = np.linspace(rmin, rmax, 10)
    rbins_avg = 0.5*(rbins[1:]+rbins[:-1])
    r_lin = np.linspace(rmin, rmax, 200)
    r_log = np.logspace(np.log10(rmin), np.log10(rmax), 200)

    nbar = 1e-4
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
    
    print("Calc corrfunc CF")
    dd, dr, rr = counts_corrfunc_3d(data, random, rbins)
    #xi_ls = compute_cf(dd, dr, rr, nd, nr, 'ls')
    #xi_nat = compute_cf(dd, dr, rr, nd, nr, 'natural')
    r_cont, xi_proj = compute_cf_proj(datasky, randomsky, nd, nr, rbins, r_lin)

    #SimCF = nbodykit.algorithms.paircount_tpcf.tpcf.SimulationBox2PCF('2d', cat, edges, Nmu=1, randoms1=random, periodic=True, BoxSize=boxsize, show_progress=True)
    #SimCF.run()
    #xi_ls = SimCF.corr

    print("Plot")
    plt.figure()
    k = np.logspace(-3, 2, 300)
    plt.loglog(k, b1**2 * Plin(k), c='k', label=r'$b_1^2 P_\mathrm{lin}$')
    plt.savefig('{}/pk_{}.png'.format(plot_dir, tag))

    plt.figure()
    plt.loglog(r_log, xi_log, label='True')
    #plt.loglog(rbins_avg, xi_ls, label='LS')
    #plt.loglog(rbins_avg, xi_nat, label='natural')
    plt.loglog(r_lin, xi_proj, label='piecewise')
    plt.legend()
    plt.savefig('{}/cf_log_{}.png'.format(plot_dir, tag))

    plt.figure()
    plt.plot(r_lin, xi_lin, label='True')
    #plt.plot(rbins_avg, xi_ls, label='LS')
    #plt.plot(rbins_avg, xi_nat, label='natural')
    plt.plot(r_lin, xi_proj, label='piecewise')
    plt.legend()
    plt.xlim(40, 150)
    plt.ylim(0, 0.05)
    plt.savefig('{}/cf_lin_{}.png'.format(plot_dir, tag))
    #ra, dec, z = to_sky(cat.Position(), cat.Velocity(), cosmo)
    
    plt.show()


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


def compute_cf_proj(data, random, nd, nr, rbins, r_cont):

    
    cosmo = LambdaCDM(H0=70, Om0=0.25, Ode0=0.75)
    
    datara = data[0].compute().astype(float)
    datadec = data[1].compute().astype(float)
    datacz = data[2].compute().astype(float)

    randra = data[0].compute().astype(float)
    randdec = data[1].compute().astype(float)
    randcz = data[2].compute().astype(float)
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
    proj_type = 'piecewise'
    projfn = None
    if proj_type=="tophat" or proj_type=="piecewise":
      nprojbins = len(rbins)-1
    elif proj_type=="powerlaw":
          nprojbins = 3
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
                                  nprojbins=nprojbins)

    dd, dr, rr, qq, dd_orig, dr_orig, rr_orig = res
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
    ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo, velocity=velocity, observer=[0, 0, 0], zmax=100.0, frame='icrs')
    return ra, dec, z


if __name__=='__main__':
    main()
