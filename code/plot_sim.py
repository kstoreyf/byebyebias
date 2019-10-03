import numpy as np


def main():

    print("Plot")
    plt.figure()
    k = np.logspace(-3, 2, 300)
    plt.loglog(k, b1**2 * Plin(k), c='k', label=r'$b_1^2 P_\mathrm{lin}$')
    plt.savefig('{}/pk{}.png'.format(plot_dir, tag))


if __name__=='__main__':
    main()

