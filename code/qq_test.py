import numpy as np

import Corrfunc

from Corrfunc.utils import qq_analytic
from Corrfunc.bases import spline


rmin = 40.0
rmax = 150.0
nbins = 11
rbins = np.linspace(rmin, rmax, nbins+1)
boxsize = 750.0
volume = boxsize**3
nbins = 11
nprojbins = nbins
nsbins = nbins
nd = 1000
#proj_type = 'tophat'
#proj_type = 'piecewise'
order = 2 # for quadratic
proj_type = 'generalr'
projfn = 'quadratic_spline.dat'
# The spline routine writes to file, so remember to delete later
spline.write_bases(rbins[0], rbins[-1], len(rbins)-1, order, projfn)

sbins = rbins
qq_ana = qq_analytic(rmin, rmax, nd, volume, nprojbins, nsbins, sbins, proj_type, projfn=projfn)
print(qq_ana)
