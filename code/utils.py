def get_label(pt):
	label_dict = {'generalr': 'cosmo deriv', 'tophat': 'tophat', 'piecewise':'triangle', 'linear_spline':'linear spline', 'quadratic_spline': 'quadratic spline', 'quadratic_spline_nbins8':'quadratic spline (8 bins)', 'gaussian_kernel':'gaussian kernel'}
	for k in label_dict.keys():
		pt0 = pt.split('_')[0]
		if pt0 in k:
			return label_dict[k]
	else:
		return pt

def get_color(pt):
	color_dict = {'tophat':'blue', 'standard': 'orange', 'piecewise':'crimson', 'linear_spline':'red', 'cosmo deriv':'purple', 'triangle':'crimson', 'quadratic_spline':'green', 'quadratic_spline_nbins8':'limegreen', 'gaussian_kernel':'orangered'}
	for k in color_dict.keys():
		pt0 = pt.split('_')[0]
		if pt0 in k:
			return color_dict[k]
	else:
		return 'blue'
