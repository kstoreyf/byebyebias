import numpy as np
import plotter


def main():
    #plot_sim_slice()
    plot_multiple()

def plot_sim_slice():

    plot_dir = '../plots/plots_2019-10-03'    
    cat_dir = '../catalogs/catalogs_2019-09-30'
    boxsize = 750
    #nbar_str = '1e-5'     
    nbar_str = '3e-4'
    cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    savefn = '{}/cat_lognormal{}_zslice.png'.format(plot_dir, cat_tag)    

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

    plotter.plot_sim(data.T, random.T, boxsize, zrange=[0,10], saveto=savefn)


def plot_multiple():

    plot_dir = '../plots/plots_2019-10-04'
    boxsize = 750
    nbar_str = '1e-5'     
    #nbar_str = '3e-4'
    nrealizations = 5
    seeds = np.arange(nrealizations)
    cat_tag = '_L{}_nbar{}'.format(boxsize, nbar_str)
    cat_dir = '../catalogs/cats_lognormal{}'.format(cat_tag)

    rand_fn = '{}/rand{}_10x.dat'.format(cat_dir, cat_tag)
    random = np.loadtxt(rand_fn)

    for seed in seeds:
        data_fn = '{}/cat_lognormal{}_seed{}.dat'.format(cat_dir, cat_tag, seed)
        data = np.loadtxt(data_fn) 
        savefn = '{}/cat_lognormal{}_zslice_seed{}.png'.format(plot_dir, cat_tag, seed)
        plotter.plot_sim(data.T, random.T, boxsize, zrange=[0,10], saveto=savefn)



def plot_pk():

    #unfinished!
    print("Plot")
    plt.figure()
    k = np.logspace(-3, 2, 300)
    plt.loglog(k, b1**2 * Plin(k), c='k', label=r'$b_1^2 P_\mathrm{lin}$')
    plt.savefig('{}/pk{}.png'.format(plot_dir, tag))


if __name__=='__main__':
    main()

