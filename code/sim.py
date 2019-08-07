import numpy as np

from Corrfunc.theory.DD import DD
from astropy.cosmology import LambdaCDM
from astropy.coordinates import SkyCoord, Distance
from astropy import units as u

import plotter
import corrfuncproj

from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi


def main():
	line_segment()
	#poisson()

def line_segment():

	#L = [100, 100, 100]
	#L = [75,75,75]
	L = [50, 50, 50]
	#L = [20, 20, 20]
	clusttag = 'realish'
	savetag = '_piece_lin'
	log = False
	savecf = '../plots/cox_comparison_{}{}.png'.format(clusttag, savetag)
	savesim = '../plots/cox_sim_slice_{}{}.png'.format(clusttag, savetag)

	if clusttag=='ned':
		linedens = 0.001
		l = 10
		pointdens = 1
	if clusttag == 'martinez':
		linedens = 4e-5
		l = 20
		pointdens = 1.88
	elif clusttag=='l30':
		linedens = 0.0003
		l = 30
		pointdens = 1
	elif clusttag=='ker':
		linedens = 0.00002
		l = 10
		pointdens = 10
	elif clusttag=='ker2':
		linedens = 0.00002
		l = 10
		pointdens = 1
	elif clusttag=='labatie':
		linedens = 1e-5
		l = 20
		pointdens = 1.8
	elif clusttag=='realish':
		linedens = 0.001
		l = 10
		pointdens = 0.3

	if log:
		rbins = np.logspace(np.log10(0.5), np.log10(30), 15)
		ravg = 10 ** (0.5 * (np.log10(rbins)[1:] + np.log10(rbins)[:-1]))
		r_cont = np.logspace(np.log10(min(ravg)), np.log10(max(ravg)), 300)
	else:
		rbins = np.linspace(0.5, 30, 15)
		ravg = 0.5*(rbins[1:] + rbins[:-1])
		r_cont = np.linspace(min(ravg), max(ravg), 300)

	iters = 1

	#ests = ['natural', 'ham', 'dp', 'ls']
	ests = ['natural', 'ls', 'proj']
	rs = []
	cfs = []
	for i in range(iters):
		print "iter", i
		data = cox_segment_3d(L, linedens, pointdens, l)
		random = rect_poisson_3d(len(data.T)*10, L)
		nd = len(data.T)
		nr = len(random.T)
		print nd, nr

		if i==0:
			plotter.plot_sim(data, random, zrange=[0, 20], saveto=savesim)

		dd, dr, rr = counts_corrfunc_3d(data, random, rbins)

		cfs_iter = []
		for est in ests:
			if 'proj' in est:
				r0, xi = compute_cf_proj(data, random, nd, nr, rbins, r_cont)
			else:
				xi = compute_cf(dd, dr, rr, nd, nr, est)
				r0 = ravg
			cfs_iter.append(xi)
			if i==0:
				rs.append(r0)
		cfs.append(cfs_iter)

	rs = np.array(rs)
	cfs = np.array(cfs)

	cftrue = cox_cf(ravg, linedens, l)

	cftrue_cont = cox_cf(r_cont, linedens, l)
	#cftrue_svals = cox_cf(svals, linedens, l)

	plotter.plot_cf(rs, cfs, ests, cftrue, r_cont, cftrue_cont, saveto=savecf, log=log, err=True)


def poisson():

	L = [100, 100, 100]
	savecf = '../plots/poisson_comparison.png'
	savezoom = '../plots/poisson_comparison_zoom.png'
	savesim = '../plots/poisson_sim_slice.png'

	nd = 10000
	nr = 10*nd
	print nd, nr

	rbins = np.logspace(np.log10(0.1), np.log10(50), 15)
	ravg = 10 ** (0.5 * (np.log10(rbins)[1:] + np.log10(rbins)[:-1]))

	iters = 100

	ests = ['natural', 'ham', 'dp', 'ls']

	cfs = []
	for i in range(iters):
		print "iter", i
		data = rect_poisson_3d(nd, L)
		random = rect_poisson_3d(nr, L)

		if i==0:
			plotter.plot_sim(data, random, zrange=[0, 10], saveto=savesim)

		dd, dr, rr = counts_corrfunc_3d(data, random, rbins)

		cfs_iter = []
		for est in ests:
			cfs_iter.append(compute_cf(data, random, dd, dr, rr, nd, nr, est))
		cfs.append(cfs_iter)

	cfs = np.array(cfs)

	cftrue = np.zeros(len(ravg))

	r_cont = np.logspace(np.log10(min(ravg)), np.log10(max(ravg)), 100)
	cftrue_cont = np.zeros(len(r_cont))

	plotter.plot_cf(ravg, cfs, ests, cftrue, r_cont, cftrue_cont, saveto=savecf, log=False)
	plotter.plot_cf(ravg, cfs, ests, cftrue, r_cont, cftrue_cont, saveto=savezoom, log=False, zoom=True)


def rect_poisson_2d(n, L):
	xs = np.random.uniform(0, L[0], n)
	ys = np.random.uniform(0, L[1], n)
	return np.array([xs, ys])


def rect_poisson_3d(n, L):
	xs = np.random.uniform(0, L[0], n)
	ys = np.random.uniform(0, L[1], n)
	zs = np.random.uniform(0, L[2], n)
	return np.array([xs, ys, zs])


def disc_poisson_2d(n, R):
	xs = []
	ys = []
	while len(xs)<n:
		x = np.random.random()*R
		y = np.random.random()*R
		if x**2 + y**2 < R:
			xs.append(x)
			ys.append(y)
	return np.array([xs, ys])


def cox_segment_3d(L, linedens, pointdens, l):
	# All the descriptions say linedens and pointdens are the mean
	# density, but I'm not clear what this varies around, so in this
	# it's the exact density of lines in the box and points on a line,
	# respectively (so will have same # for same size box).
	points = []
	nlines = int(linedens * np.product(L))
	npointsline = int(pointdens * l)
	print linedens, pointdens, l
	print nlines, npointsline
	line_ends = zip(np.random.uniform(0, L[0], nlines),
					np.random.uniform(0, L[1], nlines),
					np.random.uniform(0, L[2], nlines))
	for end in line_ends:
		radii = np.random.uniform(0, l, npointsline)
		theta = np.random.random()*2*np.pi
		phi = np.random.random()*np.pi
		x = np.sin(theta)*np.cos(phi)
		y = np.sin(theta)*np.sin(phi)
		z = np.cos(theta)
		for r in radii:
			# for now just cut off if outside bounds
			p = end + r*np.array([x, y, z])
			if np.less(p, L).all() and np.greater_equal(p, 0).all():
				points.append(p)

	return np.array(points).T


def cox_cf(r, linedens, l):
	A = 1./(2*np.pi*linedens*l)
	B = -1./(2*np.pi*linedens*l**2)
	return A/(r**2) + B/r


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

	datax = data[0]
	datay = data[1]
	dataz = data[2]

	# TODO: use pandas dataframe for faster operation?
	datara = np.zeros(nd)
	datadec = np.zeros(nd)
	datacz = np.zeros(nd)
	for i in range(nd):
		c = SkyCoord(x=datax[i], y=datay[i], z=dataz[i], unit='Mpc', representation='cartesian')
		datara[i] = c.spherical.lon.value
		datadec[i] = c.spherical.lat.value
		datacz[i] = c.spherical.distance.value

	randx = random[0]
	randy = random[1]
	randz = random[2]

	randra = np.zeros(nr)
	randdec = np.zeros(nr)
	randcz = np.zeros(nr)
	for i in range(nr):
		c = SkyCoord(x=randx[i], y=randy[i], z=randz[i], unit='Mpc', representation='cartesian')
		randra[i] = c.spherical.lon.value
		randdec[i] = c.spherical.lat.value
		randcz[i] = c.spherical.distance.value


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
	nproc = 1
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


def counts_corrfunc_2d(data, random, rbins):

	tiny = 1e-6

	# To do 2d pair counting, set z to all same
	datax = data[0]
	datay = data[1]
	dataz = np.zeros(len(datax))
	# this is a hack bc for some reason corrfunc won't work if all z's are the same
	dataz[0] = tiny

	randx = random[0]
	randy = random[1]
	randz = np.zeros(len(randx))
	randz[0] = tiny

	nthreads = 2

	print max(np.array(data).flatten())
	dd = DD(1, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
			   periodic=False)
	dd = np.array([x[3] for x in dd])
	print dd
	dr = DD(0, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
			   periodic=False, X2=randx, Y2=randy, Z2=randz)
	dr = np.array([x[3] for x in dr])
	print dr
	rr = DD(1, nthreads, rbins, randx, randy, randz,
				periodic=False)
	rr = np.array([x[3] for x in rr])
	print rr

	return dd, dr, rr


def counts_corrfunc_3d(data, random, rbins):

	datax = data[0]
	datay = data[1]
	dataz = data[2]

	randx = random[0]
	randy = random[1]
	randz = random[2]

	nthreads = 2

	dd = DD(1, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
			   periodic=False)
	dd = np.array([x[3] for x in dd])
	print dd
	dr = DD(0, nthreads, rbins, X1=datax, Y1=datay, Z1=dataz,
			   periodic=False, X2=randx, Y2=randy, Z2=randz)
	dr = np.array([x[3] for x in dr])
	print dr
	rr = DD(1, nthreads, rbins, randx, randy, randz,
				periodic=False)
	rr = np.array([x[3] for x in rr])
	print rr

	return dd, dr, rr


if __name__=="__main__":
	main()
