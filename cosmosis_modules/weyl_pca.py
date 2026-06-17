import numpy as np
import scipy.interpolate as interp
from cosmosis.datablock import names, option_section

def setup(options):
    mode_file = options.get_string(option_section, "mode_file", "pca_modes.txt")
    n_modes = options.get_int(option_section, "n_modes", 3)
    z_means = np.array([0.295, 0.467, 0.626, 0.771])
    
    data = np.loadtxt(mode_file)
    k_grid = data[:, 0]
    
    modes_funcs = []
    for m in range(n_modes):
        func = interp.interp1d(k_grid, data[:, m+1], kind='cubic', bounds_error=False, fill_value=(data[0, m+1], data[-1, m+1]))
        modes_funcs.append(func)
        
    return modes_funcs, z_means, n_modes

def execute(block, config):
    modes_funcs, z_means, n_modes = config
    
    if not block.has_section("galaxy_shear_cl"):
        return 0
        
    z_dist = block[names.distances, 'z']
    chi_dist = block[names.distances, 'd_m']
    lens_chis = interp.interp1d(z_dist, chi_dist)(z_means)
    
    alphas = np.array([block.get_double("weyl_pca", f"alpha_{m+1}", 0.0) for m in range(n_modes)])
    
    ell_vals = block["galaxy_shear_cl", "ell"]
    n_lens = len(z_means)
    n_source = block.get_int("galaxy_shear_cl", "nbin_b", default=4)

    for i in range(n_lens):
        j_amp = block.get_double("J_bin_bias", f"J_amp_{i+1}")
        
        # 映射 ell 至物理波数 k
        k_vals_at_ell = (ell_vals + 0.5) / lens_chis[i]
        
        # 构建尺度依赖修正项 f(k) = 1 + \sum \alpha_i * e_i(k)
        shape_factor = np.ones_like(k_vals_at_ell)
        for m in range(n_modes):
            shape_factor += alphas[m] * modes_funcs[m](k_vals_at_ell)
            
        # 综合修正矢量
        correction_vector = j_amp * shape_factor
        
        # 作用于剪切-星系交叉功率谱
        for j in range(n_source):
            bin_name = f"bin_{i+1}_{j+1}"
            if block.has_value("galaxy_shear_cl", bin_name):
                block["galaxy_shear_cl", bin_name] *= correction_vector

    return 0