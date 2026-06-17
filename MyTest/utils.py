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

    # 1. 加载数据
    data = np.loadtxt(chain_path)
    
    # 2. 读取表头获取参数名称
    with open(chain_path, 'r') as f:
        header = f.readline().strip()
    
    # 处理可能存在的 # 前缀
    if header.startswith('#'):
        header = header[1:].strip()
    param_names = header.split('\t')

    # 3. 排除全局预热期 (Burn-in)
    # 必须先剔除初始的不稳定阶段，再对剩下的“收敛部分”进行切分
    n_total = data.shape[0]
    start_idx = int(n_total * burnin)
    converged_data = data[start_idx:, :]

    # 4. 将剩余数据切分为 n_chains 段
    # np.array_split 会处理不能整除的情况
    split_data = np.array_split(converged_data, n_chains)

    # 5. 为每一段提取对应的权重和似然值
    # 在 CosmoSIS 的 emcee 输出中，最后一列通常是 weight 或 loglike
    split_loglikes = [d[:, -1] for d in split_data]

    # 6. 初始化 MCSamples
    # 当 samples 为列表形式时，GetDist 会将其识别为多条独立的链
    samples = MCSamples(
        samples=split_data,
        loglikes=split_loglikes,
        names=param_names,
        labels=param_names,
        settings={'ignore_rows': 0}  # 因为之前手动剔除了 burnin，这里设为 0
    )

    print(f"Successfully split chain into {n_chains} segments.")
    print(f"Parameters identified: {param_names}")
    
    return samples

def get_theory_curve_camb(Om0, h0, Omb, ns, As, z_arr):
    """
    使用 CAMB 计算 Jhat(z) = Omega_m(z) * sigma8(z)
    """
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