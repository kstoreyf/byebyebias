#!/usr/bin/env python
import numpy as np

import plotter



plot_dir = '../plots/plots_2019-10-04'
result_dir = '../results/results_2019-10-03'
cat_dir = '../catalogs/catalogs_2019-09-30'
label_dict = {'generalr': 'cosmo deriv', 'tophat': 'tophat', 'piecewise':'triangle', 'linear_spline':'linear spline', 'quadratic_spline': 'quadratic spline', 'quadratic_spline_nbins8':'quadratic spline (8 bins)', 'quadratic':'quadratic spline'}
color_dict = {'tophat':'blue', 'standard': 'orange', 'piecewise':'crimson', 'linear_spline':'red', 'cosmo deriv':'purple', 'triangle':'crimson', 'quadratic_spline':'green', 'quadratic_spline_nbins8':'limegreen', 'quadratic':'green'}

def main():
    plot_lognormal_multi()


def plot_lognormal_multi():
    boxsize = 750
    nbar_str = '3e-4'
    #nbar_str = '3e-4'
    projs = ['quadratic_spline']
    proj_tags = ['quadratic']
    plot_tag = '_quadratic_nreals5'
    nrealizations = 5
    seeds = np.arange(nrealizations)
    cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    cat_dir = '../catalogs/cats_lognormal{}'.format(cat_tag)        
    result_dir = '../results/results_lognormal{}'.format(cat_tag)

    labels = []
    colors = []

    rs_stan_lin = []
    cfs_stan_lin = []
    for seed in seeds:
        r_stan_lin, xi_stan_lin, label_stan = np.load('{}/cf_lin_{}{}_seed{}.npy'.format(result_dir, 'standard', cat_tag, seed))
        rs_stan_lin.append(r_stan_lin)
        cfs_stan_lin.append(xi_stan_lin)
        labels.append(None)
        colors.append(color_dict['standard'])

    r_stan_mean = np.mean(rs_stan_lin, axis=0)
    cf_stan_mean = np.mean(cfs_stan_lin, axis=0)

    rs_lin = rs_stan_lin
    rs_lin.append(r_stan_mean)
    cfs_lin = cfs_stan_lin
    cfs_lin.append(cf_stan_mean)
    labels.append('standard')
    colors.append(color_dict['standard'])

    for proj_tag in proj_tags:
        rs_proj_lin = []
        cfs_proj_lin = []
        for seed in seeds:
            r_lin, xi_proj_lin, projt = np.load('{}/cf_lin_{}{}_seed{}.npy'.format(result_dir, proj_tag, cat_tag, seed))
            rs_proj_lin.append(r_lin)
            cfs_proj_lin.append(xi_proj_lin)
            labels.append(None)
            colors.append(color_dict[proj_tag])
        
        rs_lin += rs_proj_lin
        cfs_lin += cfs_proj_lin

        r_proj_mean = np.mean(rs_proj_lin, axis=0)
        cf_proj_mean = np.mean(cfs_proj_lin, axis=0)
        rs_lin.append(r_proj_mean)
        cfs_lin.append(cf_proj_mean)
        labels.append(label_dict[proj_tag])
        colors.append(color_dict[proj_tag])

    #rs_lin = rs_stan_lin + rs_proj_lin
    #cfs_lin = cfs_stan_lin + cfs_proj_lin
    
    r_true_lin, xi_true_lin, label_true = np.load('{}/cf_lin_{}{}.npy'.format(cat_dir, 'true', cat_tag))
   
    save_lin = '{}/cf_lin{}{}.png'.format(plot_dir, cat_tag, plot_tag)
    plotter.plot_cf_cont(rs_lin, cfs_lin, r_true_lin, xi_true_lin, labels, colors, saveto=save_lin, log=False, err=True)


def plot_lognormal():
    
    boxsize = 750
    nbar_str = '3e-4'
    cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    tag = cat_tag+''
    #proj_types = ['tophat', 'piecewise', 'generalr']
    #proj_types = ['tophat', 'linear_spline']
    proj_types = ['tophat', 'quadratic_spline_nbins8']
    plot_tag = '_tophat_quadspline_nbins8'
    log = False

    r_stan_lin, xi_stan_lin, label_stan = np.load('{}/cf_lin_{}{}.npy'.format(result_dir, 'standard', tag))
    rs_lin = [r_stan_lin]
    cfs_lin = [xi_stan_lin]
    if log:
        r_stan_log, xi_stan_log, label_stan = np.load('{}/cf_log_{}{}.npy'.format(result_dir, 'standard', tag))
        rs_log = [r_stan_log]
        cfs_log = [xi_stan_log]
    
    #r_stan_lin_per, xi_stan_lin_per, label_stan_per = np.load('{}/cf_lin_{}{}_periodic.npy'.format(result_dir, 'standard', tag))
    #r_stan_log_per, xi_stan_log_per, label_stan_per = np.load('{}/cf_log_{}{}_periodic.npy'.format(result_dir, 'standard', tag))
    #
    #rs_lin = [r_stan_lin, r_stan_lin_per]
    #rs_log = [r_stan_log, r_stan_log_per]
    #cfs_lin = [xi_stan_lin, xi_stan_lin_per]
    #cfs_log = [xi_stan_log, xi_stan_log_per]

    #colors=['orange', 'pink']
    #labels=['standard (not periodic)', 'standard (periodic)']

    for proj_type in proj_types:

        r_lin, xi_proj_lin, projt = np.load('{}/cf_lin_{}{}.npy'.format(result_dir, proj_type, tag))
        rs_lin.append(r_lin)
        cfs_lin.append(xi_proj_lin)
        if log:
            r_log, xi_proj_log, projt = np.load('{}/cf_log_{}{}.npy'.format(result_dir, proj_type, tag))
            rs_log.append(r_log)
            cfs_log.append(xi_proj_log) 

    colors = [color_dict['standard']] + [color_dict[pt] for pt in proj_types]
    labels = ['standard'] + [label_dict[pt] for pt in proj_types]

    r_true_lin, xi_true_lin, label_true = np.load('{}/cf_lin_{}{}.npy'.format(cat_dir, 'true', cat_tag))
    save_lin = '{}/cf_lin{}{}.png'.format(plot_dir, tag, plot_tag)
    plotter.plot_cf_cont(rs_lin, cfs_lin, r_true_lin, xi_true_lin, labels, colors, saveto=save_lin, log=False, err=True)

    if log:
        r_true_log, xi_true_log, label_true = np.load('{}/cf_log_{}{}.npy'.format(cat_dir, 'true', cat_tag))
        save_log = '{}/cf_log{}{}.png'.format(plot_dir, tag, plot_tag)
        plotter.plot_cf_cont(rs_log, cfs_log, r_true_log, xi_true_log, labels, colors, saveto=save_log, log=True, err=True)






if __name__=='__main__':
    main()
