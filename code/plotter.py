import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



color_dict = {'True':'black', 'tophat':'blue', 'standard': 'orange', 'piecewise':'crimson', 'linear spline':'red', 'cosmo deriv':'purple', 'triangle':'crimson'}

def plot_cf_cont(rs, cfs, r_true, cf_true, labels, colors, alphas=None, saveto=None,
            log=False, err=False, zoom=False, error_regions=None):

    if not alphas:
        alphas = np.ones(len(colors))
    print('plotting')
    if err:
        fig, ax = plt.subplots(2, 1, figsize=(8,8), gridspec_kw={'height_ratios': [2, 1]})
    else:
        plt.figure(figsize=(8,6))
        ax = plt.gca()
        ax = [ax]
    
    if log:
        xmin = 0
    else:
        xmin = 40
    r_t = np.array([r_true[k] for k in range(len(r_true)) if r_true[k]>xmin])
    cf_t = np.array([cf_true[k] for k in range(len(r_true)) if r_true[k]>xmin])
    #cf_t = 1 + cf_t
    #cf_t = r_t**2 * cf_t
    ax[0].plot(r_t, cf_t, color='k', label='True')
    
    for j in range(len(rs)):

        r = np.array([rs[j][k] for k in range(len(rs[j])) if rs[j][k]>xmin])
        cf = np.array([cfs[j][k] for k in range(len(rs[j])) if rs[j][k]>xmin])
        lower = error_regions[j][0]
        upper = error_regions[j][1]
        lower = np.array([lower[k] for k in range(len(rs[j])) if rs[j][k]>xmin])
        upper = np.array([upper[k] for k in range(len(rs[j])) if rs[j][k]>xmin])
        #cf = 1 + cf
        #cf = r**2 * cf

        if len(rs[j])==len(r_true):
            marker = None
            ls = '-'
            rmserr = rmse(cf, cf_t)
            if labels[j] is not None:
                label = '{} (rmse: {:.2e})'.format(labels[j], rmserr)
            else:
                label = labels[j]
        else:
            marker = 'o'
            ls = 'None'
            label = labels[j]
        
        ax[0].plot(r, cf, color=colors[j], alpha=alphas[j], 
                            label=label, marker=marker, ls=ls)
        ax[0].fill_between(r, lower,  upper, color='gray', alpha=0.3)

        if err and len(rs[j])==len(r_true):
            ax[1].plot(r, (cf-cf_t)/cf_t, color=colors[j], alpha=alphas[j])
            #ax[1].plot(r, cf/cf_t, color=colors[j], alpha=alphas[j])

    ax[0].set_xlabel('r')
    #ax[0].set_ylabel(r'$\xi(r)$')  
    ax[0].set_ylabel(r'$\xi(r)$')
    
    if zoom:
        ax[0].set_ylim(-0.05, 0.05)

    if log:
        ax[0].set_xscale('log')
        ax[0].set_yscale('log')
        if err:
            ax[1].set_xscale('log')
    #else:
    #    ax[0].set_xlim(40, max(r_true))
    #    ax[1].set_xlim(40, max(r_true))
        #ax[0].set_ylim(-0.05,0.05)

    if err:
        ax[1].axhline(0, color='k')
        ax[1].set_xlabel('r')
        #ax[1].set_ylabel(r'($\xi-\xi_{true})/\xi_{true}$')
        ax[1].set_ylabel('fractional error')
        ax[1].set_ylim(-5, 5)

    ax[0].legend()

    if saveto:
        plt.savefig(saveto)
    return ax
    

def plot_cf(rs, cfs, ests, cftrue, r_cont, cftrue_cont, saveto=None,
            log=False, err=False, zoom=False):

    colors = ['r','g','b','m']

    if err:
        fig, ax = plt.subplots(2, 1, figsize=(8,8), gridspec_kw={'height_ratios': [2, 1]})
    else:
        plt.figure(figsize=(8,6))
        ax = plt.gca()
        ax = [ax]

    for j in range(len(ests)):
        mean = np.mean(cfs[:,j], axis=0)
        std = np.std(cfs[:,j], axis=0)

        offset = 10**(np.log10(rs[j])+np.log10(0.02*j))
        ax[0].errorbar(rs[j]+offset, mean, yerr=std, capsize=1, color=colors[j], label=ests[j])

        if err:
            if 'proj' in ests[j]:
                ax[1].plot(rs[j], (mean-cftrue_cont)/cftrue_cont, color=colors[j])
            else:
                ax[1].plot(rs[j], (mean-cftrue)/cftrue, color=colors[j])



    ax[0].plot(r_cont, cftrue_cont, color='k', label='true')

    ax[0].set_xlabel('r')
    ax[0].set_ylabel(r'$\xi(r)$')

    ax[0].set_xscale('log')
    if zoom:
        ax[0].set_ylim(-0.05, 0.05)

    if log:
        ax[0].set_yscale('log')

    if err:
        ax[1].axhline(0, color='k')
        ax[1].set_xlabel('r')
        ax[1].set_ylabel(r'($\xi-\xi_{true})/\xi_{true}$')
        ax[1].set_xscale('log')
        ax[1].set_ylim(-0.5, 0.5)

    ax[0].legend()

    if saveto:
        plt.savefig(saveto)


def rmse(y, y_pred):
    return np.sqrt(np.mean(np.square(y - y_pred)))


def plot_sim(data, random, boxsize, zrange=None, saveto=None):
    plt.figure()
    if zrange:
        data = np.array([d for d in data.T if zrange[0]<=d[2]<zrange[1]])
        data = data.T
        random = np.array([r for r in random.T if zrange[0]<=r[2]<zrange[1]])
        random = random.T
    plt.scatter(random[0], random[1], s=1, color='cyan', label='random')
    if len(data)>0:
        plt.scatter(data[0], data[1], s=1, color='red', label='data')
    plt.legend(loc='upper right',framealpha=1)
    #plt.gca().set_aspect('equal', adjustable='box')
    plt.axis('scaled')
    plt.xlim(0, boxsize)
    plt.ylim(0, boxsize)
    plt.xlabel(r'$x$ (h$^{-1}$Mpc)')
    plt.ylabel(r'$y$ (h$^{-1}$Mpc)')
    if saveto:
        plt.savefig(saveto)
