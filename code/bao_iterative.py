import os
import numpy as np
import matplotlib.pyplot as plt

import nbodykit
import Corrfunc

from Corrfunc.theory.DDsmu import DDsmu
from Corrfunc.utils import compute_amps
from Corrfunc.utils import evaluate_xi
from Corrfunc.utils import qq_analytic
from Corrfunc.bases import bao

base_colors = ['magenta', 'red', 'orange', 'green', 'blue']
base_names = ['a1', 'a2', 'a3', 'Bsq', 'C']


def main():
    #N = 10
    nbar_str = '3e-4'
    #seeds = [98]
    seeds = np.arange(0,100)
    #seeds = [10]
    niter_start = 6
    for seed in seeds:
        biter = BAO_iterator(seed=seed, nbar_str=nbar_str)

        # initial parameters
        if niter_start is not None:
            start_fn = '{}/cf_lin_{}_niter{}{}_seed{}.npy'.format(biter.result_dir, 'baoiter', niter_start, biter.cat_tag, seed)
            res = np.load(start_fn, allow_pickle=True, encoding='latin1')
            _, _, _, _, extra_dict = res
            alpha_model_start = extra_dict['alpha_result']

        else:
            niter_start = 0
            alpha_model_start = 0.98
        
        alpha_model = alpha_model_start
        dalpha = 0.01*alpha_model

        niters = 7
        for niter in range(niter_start+1, niter_start+1+niters):
            print(f'iter {niter}')
            print(f'alpha: {alpha_model}, dalpha: {dalpha}')
            xi, amps = biter.bao_iterative(dalpha, alpha_model)
            C = amps[4]
            alpha_result = alpha_model+dalpha*C
            extra_dict = {'alpha_start': alpha_model_start,
                          'alpha_model': alpha_model,
                          'dalpha': dalpha,
                          'alpha_result': alpha_result,
                          'niter': niter} 
            biter.save_cf(xi, amps, niter, extra_dict)

            print(f"C: {C}")
            alpha_model = alpha_result
            dalpha = 0.01*alpha_model
            print(f'NEW alpha: {alpha_model}, dalpha: {dalpha}')




class BAO_iterator:

    def __init__(self, boxsize=750, nbar_str='1e-5', seed=0, rmin=40, rmax=200, nbins=15):

        # input params
        self.boxsize = boxsize
        self.nbar_str = nbar_str
        self.seed = seed

        self.rmin = rmin
        self.rmax = rmax
        self.nbins = nbins

        # other params
        self.mumax = 1.0
        #weight_type='pair_product'
        self.weight_type=None

        self.periodic = True
        self.nthreads = 24
        self.nmubins = 1
        self.verbose = False
        self.proj_type = 'generalr'

        # set up other data
        self.rbins = np.linspace(rmin, rmax, nbins+1)
        self.rbins_avg = 0.5*(self.rbins[1:]+self.rbins[:-1])
        self.rcont = np.linspace(rmin, rmax, 1000)

        self.cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
        self.cat_dir = '/home/users/ksf293/byebyebias/catalogs/cats_lognormal{}'.format(self.cat_tag)

        #r_true, xi_true = self.load_true_cf(cat_dir, cat_tag, rmin, rmax)
        self.load_data()
        self.load_random()

        self.result_dir = '/home/users/ksf293/byebyebias/results/results_lognormal{}'.format(self.cat_tag)
        if not os.path.exists(self.result_dir):
            os.makedirs(self.result_dir)


    def save_cf(self, xi, amps, niter, extra_dict):
        save_fn = '{}/cf_lin_{}_niter{}{}_seed{}.npy'.format(self.result_dir, 'baoiter', niter, self.cat_tag, self.seed)
        np.save(save_fn, [self.rcont, xi, amps, 'bao', extra_dict])
        print(f"Saved to {save_fn}")


    def load_true_cf(self, rmin, rmax):
        r_true, xi_true, _ = np.load('{}/cf_lin_true{}.npy'.format(self.cat_dir, self.cat_tag), 
            allow_pickle=True, encoding='latin1')
        xi_true = [xi_true[i] for i in range(len(r_true)) if rmin<=r_true[i]<rmax]
        r_true = [r_true[i] for i in range(len(r_true)) if rmin<=r_true[i]<rmax]
        return r_true, xi_true


    def load_data(self):
        data_fn = '{}/cat_lognormal{}_seed{}.dat'.format(self.cat_dir, self.cat_tag, self.seed)
        data = np.loadtxt(data_fn)
        self.x, self.y, self.z = data.T
        self.nd = data.shape[0]
        #weights = np.full(nd, 0.5)
        self.weights = None
        #return x,y,z,weights


    def load_random(self):
        rand_fn = '{}/rand{}_10x.dat'.format(self.cat_dir, self.cat_tag)
        random = np.loadtxt(rand_fn)
        self.x_rand, self.y_rand, self.z_rand = random.T
        self.nr = random.shape[0]
        #weights_rand = np.full(nr, 0.5)
        self.weights_rand = None
        #return x_rand,y_rand,z_rand,weights_rand

    def run_estimator(self):

        _, dd_proj, _ = DDsmu(1, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x, self.y, self.z,
                        proj_type=self.proj_type, nprojbins=self.nprojbins, projfn=self.projfn,
                        verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic)

        _, dr_proj, _ = DDsmu(0, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x, self.y, self.z,
                            X2=self.x_rand, Y2=self.y_rand, Z2=self.z_rand, 
                            proj_type=self.proj_type, nprojbins=self.nprojbins, projfn=self.projfn,
                            verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic)
        #print(dr_proj)
        #_, rr_proj, qq_proj = DDsmu(1, self.nthreads, self.rbins, self.mumax, self.nmubins, self.x_rand, self.y_rand, self.z_rand,
        #        proj_type=self.proj_type, nprojbins=self.nprojbins, projfn=self.projfn,
        #        verbose=self.verbose, boxsize=self.boxsize, periodic=self.periodic)
        
        volume = self.boxsize**3
        sbins = self.rbins
        nsbins = len(self.rbins)-1
        # print(dd_proj)
        rr_ana, qq_ana = qq_analytic(self.rmin, self.rmax, self.nd, volume, self.nprojbins, nsbins, sbins, self.proj_type, projfn=self.projfn)
        qq_ana = qq_ana.flatten()
        print("DD", dd_proj)
        print("DR", dr_proj)
        print("ANA")
        
        #print(qq_proj) 
        #qq_ana *= self.nd*(self.nd+1)
        #rr_ana *= self.nd*(self.nd+1)    
        qq_ana *= (self.nr/self.nd)**2
        rr_ana *= (self.nr/self.nd)**2
        print(rr_ana)
        print(qq_ana)
        #dd = dd_proj/(self.nd*(self.nd+1))
        #dr = dr_proj/(self.nd*self.nr)
        #rr = rr_proj/(self.nr*(self.nr+1))

        # amps_periodic_ana = np.matmul(np.linalg.inv(qq_ana), dd_proj) - 1
        # xi_periodic_ana = evaluate_xi(self.nprojbins, amps_periodic_ana, len(self.rcont), self.rcont, len(self.rbins)-1, self.rbins, self.proj_type, projfn=self.projfn)
                
        #amps = compute_amps(self.nprojbins, self.nd, self.nd, self.nr, self.nr, dd_proj, dr_proj, dr_proj, rr_proj, qq_proj)
        #print(amps)
        #amps = compute_amps(self.nprojbins, 1, 1, 1, 1, dd, dr, dr, rr_ana, qq_ana)
        print("nd nr", self.nd, self.nr)
        amps = compute_amps(self.nprojbins, self.nd, self.nd, self.nr, self.nr, dd_proj, dr_proj, dr_proj, rr_ana, qq_ana)
        print("AMPS:", amps)
        #print(amps)
        xi_proj = evaluate_xi(self.nprojbins, amps, len(self.rcont), self.rcont, len(self.rbins)-1, self.rbins, self.proj_type, projfn=self.projfn)
        #print(xi_proj)
            
        #return amps_periodic_ana, xi_periodic_ana
        return xi_proj, amps


    def bao_iterative(self, dalpha, alpha_model):

        self.projfn = 'bao.dat'
        # The spline routine writes to file, so remember to delete later
        kwargs = {'cosmo_base':nbodykit.cosmology.Planck15, 'redshift':0, 'dalpha':dalpha, 'alpha_model':alpha_model}
        self.nprojbins, _ = bao.write_bases(self.rbins[0], self.rbins[-1], self.projfn, **kwargs)
        #bases = np.loadtxt(self.projfn)
        
        #plot_bases(bases)
        
        #amps, xi = run_estimator(proj_type, nprojbins, projfn)
        xi, amps = self.run_estimator()
        
        return xi, amps



if __name__=="__main__":
    main()
