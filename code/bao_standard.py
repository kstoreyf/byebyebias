import os
import numpy as np

import plotter
import plot_comparison as pc
import utils
import scipy
from scipy import optimize

import nbodykit
from nbodykit.lab import cosmology


def main():
    boxsize = 750
    nbar_str = '3e-4'
    cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    result_dir = '../results/results_lognormal{}'.format(cat_tag)
    cat_dir = '../catalogs/cats_lognormal{}'.format(cat_tag)
    N = 100
    seeds = range(0,N)
    tag = ''
    
    save_fn = '{}/bao_standard{}{}.npy'.format(result_dir, cat_tag, tag)
    s_true, xi_true = load_cf(cat_dir, cat_tag, 'true')

    proj = 'tophat_n44_cont1000'
   
    rmin = 40
    rmax = 150
    nbins = 44
    rbins = np.linspace(rmin, rmax, nbins+1) 
    res = load_cfs(result_dir, cat_tag, proj, seeds)
    rs, xis, amps = res
    
    print("Number of boxes:", len(rs))
    rs_binned, xis_binned, rs_avg, xis_avg = bin_cfs(rs, xis, rbins)
    
    alphas_baoiter = []
    proj = 'baoiter_niter6'
    res = load_cfs(result_dir, cat_tag, proj, seeds)
    rs_baoiter, xis_baoiter, amps_baoiter, extras = res
    for kk in range(len(amps)):
        alphas_baoiter.append(extras[kk]['alpha_result'])
            
    alphas_best = []
    popts_best = []
    xis_best = []
    
    alpha_model = 0.98
    alphas = np.linspace(0.8, 1.2, 41)
    print(alphas)
    
  
    for i in range(N):      
        chis = []
        fits = []
        popts = []
        for alpha in alphas:
            popt, xi_myfit, chi_squared = fit(rs_avg[i], xis_avg[i], s_true, alpha, alpha_model)
            chis.append(chi_squared)
            fits.append(xi_myfit)
            popts.append(popt)
        
        imin = np.argmin(chis)
        print(alphas[imin], chis[imin])
        print(alphas[imin]*alpha_model)
        print(alphas_baoiter[i])
        
        alphas_best.append(alphas[imin])
        xis_best.append(fits[imin])
        popts_best.append(popts[imin])

    print(np.mean(alphas_best), np.std(alphas_best))
    print(np.mean(alphas_baoiter), np.std(alphas_baoiter))
    
    np.save(save_fn, [s_true, xis_best, alphas_best, popts_best, seeds, alpha_model])


def load_cfs(result_dir, cat_tag, proj, seeds):
    rs = []
    xis = []
    amps = []
    extras = []
    return_extras = False
    for i in range(len(seeds)):
        tag = cat_tag+'_seed{}'.format(seeds[i])
        fn = '{}/cf_lin_{}{}.npy'.format(result_dir, proj, tag)
        if os.path.isfile(fn):
            
            res = np.load(fn, allow_pickle=True, encoding='latin1')
            if len(res)>4:
                r, xi, amp, _, extra_dict = res
                return_extras = True
            elif len(res)==4:
                r, xi, amp, _ = res
            else:
                r, xi, amp = res
        else:
            continue
        rs.append(r)
        xis.append(xi)
        amps.append(amp)
        if return_extras:
            extras.append(extra_dict)
    rs = np.array(rs)
    xis = np.array(xis)
    amps = np.array(amps)
    if return_extras:
        extras = np.array(extras)
        return rs, xis, amps, extras
    else:
        return rs, xis, amps


def get_mean_stats(rs, xis):
    rs_mean = np.mean(rs, axis=0)
    xis_mean = np.mean(xis, axis=0)
    xis_std = np.std(xis, axis=0)
    xis_low = xis_mean - xis_std 
    xis_high = xis_mean + xis_std 
    return rs_mean, xis_mean, xis_std, xis_low, xis_high

def load_cf(directory, cat_tag, proj, seed=None):
    if 'true' in proj:
        data = np.load('{}/cf_lin_true{}.npy'.format(directory, cat_tag), allow_pickle=True, encoding='latin1')
        r, xi, _ = data
        return r, xi
    elif 'standard' in proj:
        tag = cat_tag+'_seed{}'.format(seed)
        data = np.load('{}/cf_lin_{}{}.npy'.format(directory, proj, tag), allow_pickle=True, encoding='latin1')
        r, xi, _ = data
        return r, xi
    else:    
        tag = cat_tag+'_seed{}'.format(seed)
        data = np.load('{}/cf_lin_{}{}.npy'.format(directory, proj, tag), allow_pickle=True, encoding='latin1')
        r, xi, amp, _ = data
        return r, xi, amp

def bin_cf_single(r, xi, rbins):
    xis_binned = []
    rs_binned = []
    r_avg = []
    xi_avg = []
    for i in range(len(rbins)-1):
        a = rbins[i]
        b = rbins[i+1]
        rbin, xibin = bin_average(r, xi, a, b)
        r_avg.append(np.mean(rbin))
        xi_avg.append(xibin)
        if xibin:
            for rval in rbin:
                rs_binned.append(rval)
                xis_binned.append(xibin)
    return rs_binned, xis_binned, r_avg, xi_avg

def bin_cfs(rs, xis, rbins):

    rs_binned_all = []
    xis_binned_all = []
    rs_avg_all = []
    xis_avg_all = []
    for i in range(len(xis)):
        r = rs[i]
        xi = xis[i]
        rs_binned, xis_binned, r_avg, xi_avg = bin_cf_single(r, xi, rbins)
        
        xis_binned_all.append(xis_binned)
        rs_binned_all.append(rs_binned)
        rs_avg_all.append(r_avg)
        xis_avg_all.append(xi_avg)

    return np.array(rs_binned_all), np.array(xis_binned_all), np.array(rs_avg_all), np.array(xis_avg_all)


def bin_average(r, x, a, b):
    xint = [x[i] for i in range(len(r)) if r[i]<b and r[i]>=a]
    rint = [r[i] for i in range(len(r)) if r[i]<b and r[i]>=a]
    return rint, np.mean(xint)


def fit(s_samp, xi_samp, s_true, alpha, alpha_model):
    redshift = 0
    cosmo_base = nbodykit.cosmology.Planck15
    Plin = cosmology.LinearPower(cosmo_base, redshift, transfer='EisensteinHu')
    CF = cosmology.correlation.CorrelationFunction(Plin)
    
    def cf_model(s, alpha_model):
        return CF(alpha_model*s)
    
    
    def bao_bases(s, cf_func, alpha, alpha_model):
        b1 = 1.0/s**2
        b2 = 0.1/s
        b3 = 0.001*np.ones(len(s))

        cf = cf_func(s*alpha, alpha_model=alpha_model)
        #cf = CF(s*alpha_model)
        b4 = cf
        return b1,b2,b3,b4
    
    def xi_fit(s, a1, a2, a3, Bsq):#, cf_func, alpha_model):
        b1,b2,b3,b4 = bao_bases(s, cf_model, alpha, alpha_model)
        return a1*b1 + a2*b2 + a3*b3 + Bsq*b4
    
    guess = np.ones(4)
    #args = (cf_model, alpha_model)
    popt, pcov = scipy.optimize.curve_fit(xi_fit, s_samp, xi_samp, p0=guess)#, args=args)
    #for i in range(len(popt)):
    #    print('{}: {:.4f}'.format(base_names[i], popt[i]))
        
    #xerror = np.diag(np.ones(len(xi_samp)))
    xerror = np.ones(len(xi_samp))
    chi_squared = np.sum(((xi_fit(s_samp, *popt) - xi_samp) / xerror) ** 2)
    reduced_chi_squared = chi_squared / (len(s_samp) - len(popt))
    #print('The degrees of freedom for this test is', len(s_samp) - len(popt) )
    #print('The chi squared value is: ', ("%.2e" % chi_squared) )
    #print('The reduced chi squared value is: ', ("%.2e" % reduced_chi_squared) )
    
    xi_myfit = xi_fit(s_true, *popt)
    
    return popt, xi_myfit, chi_squared

if __name__=='__main__':
    main()

