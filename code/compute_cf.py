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
import spline

from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi


plot_dir = '../plots/plots_2019-10-03'
result_dir = '../results/results_2019-10-03'
cat_dir = '../catalogs/catalogs_2019-09-30'
nthreads = 24
color_dict = {'True':'black', 'tophat':'blue', 'LS': 'orange', 'piecewise':'red', 'standard': 'orange'}


def main():

	boxsize = 750
	#nbar_str = '3e-4'
	nbar_str = '1e-5'
	#projs = ['quadratic_spline']
	projs = ['dcosmo']	  
	#projs = ['tophat', 'piecewise']
	#projs = []

	# TODO: make sure cosmo is aligned with sim loaded in
	kwargs = {'params':['Omega_cdm', 'Omega_b', 'h'], 'cosmo_base':cosmology.Planck15}

	cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
	#tag = '_nbins8'+cat_tag
	tag = '_dtest'+cat_tag
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
	rs_lin = [rbins_avg]
	cfs_lin = [xi_stan]
	if log:
		cfs_log = [xi_stan_log]
		rs_log = [rbins_avg_log]

	for proj in projs:
		print("LIN PROJ")
		r_cont, xi_proj = compute_cf_proj(datasky, randomsky, nd, nr, rbins, r_lin, proj)
		rs_lin.append(r_lin)
		cfs_lin.append(xi_proj)
		np.save('{}/cf_lin_{}{}.npy'.format(result_dir, proj, tag), [r_lin, xi_proj, proj])
		if log:
			print("LOG PROJ")
			r_cont_log, xi_proj_log = compute_cf_proj(datasky, randomsky, nd, nr, rbins_log, r_log, proj)
			rs_log.append(r_log)
			cfs_log.append(xi_proj_log)
			np.save('{}/cf_log_{}{}.npy'.format(result_dir, proj, tag), [r_log, xi_proj_log, proj])
		



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


def compute_cf_proj(data, random, nd, nr, rbins, r_cont, proj, projfn=None, **kwargs):

	#cosmo = LambdaCDM(H0=70, Om0=0.25, Ode0=0.75)
	cosmo = 1 #doesn't matter bc passing cz, but required

	datara, datadec, datacz = data.T
	randra, randdec, randcz = random.T

	proj_type = proj
	projfn = None
	if proj=="tophat" or proj_type=="piecewise":
		nprojbins = len(rbins)-1
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
	print('Computed amplitudes')

	amps = np.array(amps)
	#svals = np.linspace(min(rbins), max(rbins), 300)

	rbins = np.array(rbins)
	xi_proj = evaluate_xi(nprojbins, amps, len(r_cont), r_cont, len(rbins), rbins, proj_type, projfn=projfn)
	print("Computed xi")
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
