import numpy as np
import scipy.interpolate as interp
from cosmosis.datablock import names, option_section

def setup(options):
    n_nodes = options.get_int(option_section, "n_nodes", 20)
    k_min = options.get_double(option_section, "k_min", 0.01)
    k_max = options.get_double(option_section, "k_max", 0.2)
    
    k_nodes = np.logspace(np.log10(k_min), np.log10(k_max), n_nodes)
    z_means = np.array([0.295, 0.467, 0.626, 0.771])
    
    return k_nodes, n_nodes, z_means

def execute(block, config):
    k_nodes, n_nodes, z_means = config
    
    if not block.has_section("galaxy_shear_cl"):
        return 0

    f_vals = np.array([block.get_double("j_nodes", f"j_{i}") for i in range(n_nodes)])
    
    f_spline = interp.interp1d(k_nodes, f_vals, kind='cubic', bounds_error=False, fill_value=(f_vals[0], f_vals[-1]))
    
    z_dist = block[names.distances, 'z']
    chi_dist = block[names.distances, 'd_m']
    lens_chis = interp.interp1d(z_dist, chi_dist)(z_means)
    
    ell_vals = block["galaxy_shear_cl", "ell"]
    n_lens = len(z_means)
    n_source = block.get_int("galaxy_shear_cl", "nbin_b", default=4)

    for i in range(n_lens):
        j_amp = block.get_double("J_bin_bias", f"J_amp_{i+1}")
        k_vals_at_ell = (ell_vals + 0.5) / lens_chis[i]

        correction = j_amp * f_spline(k_vals_at_ell)

        for j in range(n_source):
            bin_name = f"bin_{i+1}_{j+1}"
            if block.has_value("galaxy_shear_cl", bin_name):
                block["galaxy_shear_cl", bin_name] *= correction

    return 0
