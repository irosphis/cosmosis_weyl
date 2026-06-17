import numpy as np, pandas as pd
from getdist import MCSamples

def load_mc(f_path, lbl, cfg):
    with open(f_path, 'r') as f: cols = f.readline().replace('#', '').strip().split()
    df = pd.read_csv(f_path, sep='\s+', comment='#', names=[c.lower() for c in cols]).dropna()
    d = [e(df) if callable(e) else df[e.lower()].values for r, s, n, e in cfg if r != "-"]
    n =[n for r, s, n, e in cfg if r != "-"]
    return MCSamples(samples=np.column_stack(d), weights=np.exp(df['log_weight']-df['log_weight'].max()).values, names=n, label=lbl)

def fmt(s, d=3):
    if not s: return "N/A"
    m, dn, up = s.mean, s.mean - s.limits[0].lower, s.limits[0].upper - s.mean
    if 0 < abs(m) < 1e-3:
        p = int(np.floor(np.log10(abs(m))))
        m, dn, up = m * 10**-p, dn * 10**-p, up * 10**-p
        return f"$({m:.{d}f} \\pm {(up+dn)/2:.{d}f}) \\times 10^{{{p}}}$" if abs(up-dn)/max(up,dn)<0.1 else f"${m:.{d}f}^{{+{up:.{d}f}}}_{{-{dn:.{d}f}}} \\times 10^{{{p}}}$"
    return f"${m:.{d}f} \\pm {(up+dn)/2:.{d}f}$" if abs(up-dn)/max(up,dn)<0.1 else f"${m:.{d}f}^{{+{up:.{d}f}}}_{{-{dn:.{d}f}}}$"


if __name__ == '__main__':
    from pathlib import Path
    import os
    os.chdir(Path(__file__).resolve().parent)
    
    cfg =[
        ("Matter density parameter", r"$\Omega_m$", "Om", "cosmological_parameters--omega_m"),
        ("Matter clustering amplitude", r"$\sigma_8$", "s8", "cosmological_parameters--sigma_8"),
        ("Weak lensing parameter", r"$S_8$", "S8", "cosmological_parameters--s_8"),
        ("Primordial amplitude", r"$A_s$", "As", "cosmological_parameters--a_s"),
        ("Spectral index", r"$n_s$", "ns", "cosmological_parameters--n_s"),
        ("Hubble parameter", r"$h_0$", "h", 'cosmological_parameters--h0'),
        ("-", "", "", ""),
        ("Weyl amplitude (LOWZ, $z_{\\rm eff}=0.38$)", r"$\hat{J}_1$", "J1", "j_bin_bias--jhat1"),
        ("Weyl amplitude (CMASS, $z_{\\rm eff}=0.61$)", r"$\hat{J}_2$", "J2", "j_bin_bias--jhat2"),
        ("-", "", "", ""),
        ("$E_G$ statistic (LOWZ)", r"$E_G(z_1)$", "EG1", lambda d: (d['j_bin_bias--jhat1']/d['lss_parameters--fsigma_8_bin_1'])*(d['cosmological_parameters--omega_m']+(1-d['cosmological_parameters--omega_m'])/1.38**3)),
        ("$E_G$ statistic (CMASS)", r"$E_G(z_2)$", "EG2", lambda d: (d['j_bin_bias--jhat2']/d['lss_parameters--fsigma_8_bin_2'])*(d['cosmological_parameters--omega_m']+(1-d['cosmological_parameters--omega_m'])/1.61**3)),
        ("Growth rate (LOWZ)", r"$f\sigma_8(z_1)$", "fs81", "lss_parameters--fsigma_8_bin_1"),
        ("Growth rate (CMASS)", r"$f\sigma_8(z_2)$", "fs82", "lss_parameters--fsigma_8_bin_2")
    ]

    st_np = load_mc('chains/chain_noprior.txt', 'No Prior', cfg).getMargeStats()
    st_p = load_mc('chains/chain_withprior.txt', 'With Prior', cfg).getMargeStats()

    for r, sym, n, e in cfg:
        if r == "-": print("        \\midrule")
        else: print(f"        {r} & {sym} & {fmt(st_p.parWithName(n))} & {fmt(st_np.parWithName(n))} \\\\")