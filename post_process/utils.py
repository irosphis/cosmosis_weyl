def read_samples_from_chain(chain_path, burnin=0.3):  
    import numpy as np
    from getdist import MCSamples
    data = np.loadtxt(chain_path)
    data = data[~np.isnan(data).any(axis=1)]
    
    with open(chain_path, 'r') as f:
        header = f.readline().strip()

    param_names = header.split('\t')

    log_weights = data[:, -3]
    weights = np.exp(log_weights - np.max(log_weights))
    loglikes = data[:, -1]

    samples = MCSamples(
        samples=data,
        weights=weights,
        loglikes=loglikes,
        names=param_names,
        labels=param_names,
        settings={'ignore_rows': burnin}
    )
    
    return samples

def read_split_samples_from_chain(chain_path, burnin=0.3, n_chains=4):
    import numpy as np
    from getdist import MCSamples

    data = np.loadtxt(chain_path)
    
    with open(chain_path, 'r') as f:
        header = f.readline().strip()
    
    if header.startswith('#'):
        header = header[1:].strip()
    param_names = header.split('\t')

    n_total = data.shape[0]
    start_idx = int(n_total * burnin)
    converged_data = data[start_idx:, :]

    split_data = np.array_split(converged_data, n_chains)

    split_loglikes = [d[:, -1] for d in split_data]

    samples = MCSamples(
        samples=split_data,
        loglikes=split_loglikes,
        names=param_names,
        labels=param_names,
        settings={'ignore_rows': 0}
    )

    print(f"Successfully split chain into {n_chains} segments.")
    print(f"Parameters identified: {param_names}")
    
    return samples

def get_theory_curve_camb(Om0, h0, Omb, ns, As, z_arr):
    import numpy as np
    import camb
    pars = camb.CAMBparams()
    ombh2 = Omb * h0**2
    omch2 = (Om0 * h0**2) - ombh2
    pars.set_cosmology(H0=h0*100, ombh2=ombh2, omch2=omch2, mnu=0.06, omk=0)
    pars.set_dark_energy(w=-1.0)
 
    pars.InitPower.set_params(ns=ns, As=As)
    
    pars.set_matter_power(redshifts=z_arr, silent=True, kmax=2.0) 
    
    results = camb.get_results(pars)
    
    z_camb = np.array(results.transfer_redshifts)
    s8_camb = np.array(results.get_sigma8())
    
    s8_z = np.interp(z_arr, z_camb[::-1], s8_camb[::-1])

    Om_z = np.array([
        results.get_Omega("cdm", z) + 
        results.get_Omega("baryon", z) + 
        results.get_Omega("nu", z) 
        for z in z_arr
    ])

    return Om_z * s8_z
