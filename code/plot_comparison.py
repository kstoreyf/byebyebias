#!/usr/bin/env python
import numpy as np

import plotter



plot_dir = '../plots/plots_2019-09-30'
result_dir = '../results/results_2019-09-30'
cat_dir = '../catalogs/catalogs_2019-09-30'
label_dict = {'generalr': 'cosmo deriv', 'tophat': 'tophat', 'piecewise':'linear spline'}

def main():
    plot_lognormal()

    


def plot_lognormal():
    
    boxsize = 750
    nbar_str = '3e-4'
    cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    tag = cat_tag+''
    #proj_types = ['tophat', 'piecewise', 'generalr']
    proj_types = ['tophat', 'piecewise']
    plot_tag = '_tophat_piecewise'

    r_stan_lin, xi_stan_lin, label_stan = np.load('{}/cf_lin_{}{}.npy'.format(result_dir, 'standard', tag))
    r_stan_log, xi_stan_log, label_stan = np.load('{}/cf_log_{}{}.npy'.format(result_dir, 'standard', tag))
    
    rs_lin = [r_stan_lin]
    rs_log = [r_stan_log]
    cfs_lin = [xi_stan_lin]
    cfs_log = [xi_stan_log]
    labels = ['standard']
    for proj_type in proj_types:

        r_lin, xi_proj_lin, projt = np.load('{}/cf_lin_{}{}.npy'.format(result_dir, proj_type, tag))
        r_log, xi_proj_log, projt = np.load('{}/cf_log_{}{}.npy'.format(result_dir, proj_type, tag))

        rs_lin.append(r_lin)
        cfs_lin.append(xi_proj_lin)
        rs_log.append(r_log)
        cfs_log.append(xi_proj_log) 
        labels.append(label_dict[proj_type])

    r_true_lin, xi_true_lin, label_true = np.load('{}/cf_lin_{}{}.npy'.format(cat_dir, 'true', cat_tag))
    r_true_log, xi_true_log, label_true = np.load('{}/cf_log_{}{}.npy'.format(cat_dir, 'true', cat_tag))

    save_lin = '{}/cf_lin{}{}.png'.format(plot_dir, tag, plot_tag)
    plotter.plot_cf_cont(rs_lin, cfs_lin, labels, r_true_lin, xi_true_lin, saveto=save_lin, log=False, err=True)

    save_log = '{}/cf_log{}{}.png'.format(plot_dir, tag, plot_tag)
    plotter.plot_cf_cont(rs_log, cfs_log, labels, r_true_log, xi_true_log, saveto=save_log, log=True, err=True)






if __name__=='__main__':
    main()
