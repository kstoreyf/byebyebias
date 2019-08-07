import numpy as np
import matplotlib.pyplot as plt

from nbodykit.lab import *
import nbodykit
from Corrfunc.theory.DD import DD

plot_dir = '../plots/plots_2019-08-06'
tag = '_test'

def main():
	redshift = 0
	cosmo = cosmology.Planck15
	print("Generating power spectrum")
	Plin = cosmology.LinearPower(cosmo, redshift, transfer='EisensteinHu')
	b1 = 1.0

	print("Making catalog")
	#boxsize = 100.
	boxsize = 500.
	rmin = 0.1
	rmax = 200
	rbins = np.linspace(rmin, rmax, 20)
	rbins_avg = 0.5*(rbins[1:]+rbins[:-1])
	r_lin = np.linspace(rmin, rmax, 200)
	r_log = np.logspace(np.log10(rmin), np.log10(rmax))

	nbar = 3e-2
	data = LogNormalCatalog(Plin=Plin, nbar=nbar, BoxSize=boxsize, Nmesh=256, bias=b1, seed=42)
	nd = data.csize
	random = nbodykit.source.catalog.uniform.UniformCatalog(10*nbar, boxsize, seed=43)
	nr = random.csize
	print(nd, nr)	

	print("Calc true CF")
	CF = nbodykit.cosmology.correlation.CorrelationFunction(Plin)
	xi_log = CF(r_log)#, smoothing=0.0, kmin=10**-2, kmax=10**0)
	xi_lin = CF(r_lin)
	
	print("Calc corrfunc CF")
	dd, dr, rr = counts_corrfunc_3d(data, random, edges)
	xi_ls = compute_cf(dd, dr, rr, nd, nr, 'ls')
	#SimCF = nbodykit.algorithms.paircount_tpcf.tpcf.SimulationBox2PCF('2d', cat, edges, Nmu=1, randoms1=random, periodic=True, BoxSize=boxsize, show_progress=True)
	#SimCF.run()
	#xi_ls = SimCF.corr

	print("Plot")
	plt.figure()
	plt.loglog(k, b1**2 * Plin(k), c='k', label=r'$b_1^2 P_\mathrm{lin}$')
	plt.savefig('{}/pk_{}.png'.format(plot_dir, tag))

	plt.figure()
	plt.loglog(r, xi, label='True')
	plt.loglog(edges_avg, xi_ls, label='LS')
	plt.legend()
	plt.savefig('{}/cf_log_{}.png'.format(plot_dir, tag))

	plt.figure()
	plt.plot(edges_avg, xilin, label='True')
	plt.plot(edges_avg, xi_ls, label='LS')
	plt.legend()
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


def to_sky(pos, velocity, cosmo):
	ra, dec, z = nbodykit.transform.CartesianToSky(pos, cosmo, velocity=velocity, observer=[0, 0, 0], zmax=100.0, frame='icrs')
	return ra, dec, z


if __name__=='__main__':
	main()
