import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt



color_dict = {'True':'black', 'tophat':'blue', 'standard': 'orange', 'piecewise':'crimson', 'linear spline':'red', 'cosmo deriv':'purple', 'triangle':'crimson'}

def plot_cf_cont(rs, cfs, r_true, cf_true, labels, colors, saveto=None,
            log=False, err=False, zoom=False):

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
    cf_t = 1 + cf_t
    ax[0].plot(r_t, cf_t, color='k', label='True')
    
    for j in range(len(rs)):

        if len(rs[j])==len(r_true):
            marker = None
            ls = '-'
        else:
            marker = 'o'
            ls = 'None'
        
        r = np.array([rs[j][k] for k in range(len(rs[j])) if rs[j][k]>xmin])
        cf = np.array([cfs[j][k] for k in range(len(rs[j])) if rs[j][k]>xmin])
        cf = 1 + cf
        
        ax[0].plot(r, cf, color=colors[j], label=labels[j], marker=marker, ls=ls)

        if err and len(rs[j])==len(r_true):
            ax[1].plot(r, (cf-cf_t)/cf_t, color=colors[j])

    ax[0].set_xlabel('r')
    #ax[0].set_ylabel(r'$\xi(r)$')
    ax[0].set_ylabel(r'$1 + \xi(r)$')
    
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
        #ax[1].set_ylim(-0.5, 0.5)

    ax[0].legend()

    if saveto:
        plt.savefig(saveto)



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
        print mean
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


def plot_sim(data, random, zrange=None, saveto=None):
    plt.figure()
    if zrange:
        data = np.array([d for d in data.T if zrange[0]<=d[2]<zrange[1]])
        data = data.T
        random = np.array([r for r in random.T if zrange[0]<=r[2]<zrange[1]])
        random = random.T
    plt.scatter(random[0], random[1], s=1, color='cyan', label='random')
    if len(data)>0:
        plt.scatter(data[0], data[1], s=1, color='red', label='data')
    plt.legend()
    if saveto:
        plt.savefig(saveto)
