import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from getdist import plots, MCSamples

def process_chain(file_path, label_name, z1=0.38, z2=0.61):
    with open(file_path, 'r') as f:
        header = f.readline().replace('#', '').strip().split()
    df = pd.read_csv(file_path, sep='\s+', comment='#', names=header)
    df = df.dropna()
    log_w = df['log_weight'].values
    weights = np.exp(log_w - np.max(log_w))
    omega_m = df['COSMOLOGICAL_PARAMETERS--OMEGA_M'].values
    jhat1 = df['J_bin_bias--jhat1'].values
    jhat2 = df['J_bin_bias--jhat2'].values
    fsigma81 = df['LSS_PARAMETERS--FSIGMA_8_BIN_1'].values
    fsigma82 = df['LSS_PARAMETERS--FSIGMA_8_BIN_2'].values
    df['DERIVED--E_G_1'] = (jhat1 / fsigma81) * (omega_m + (1.0 - omega_m) / (1.0 + z1)**3)
    df['DERIVED--E_G_2'] = (jhat2 / fsigma82) * (omega_m + (1.0 - omega_m) / (1.0 + z2)**3)
    params_to_plot = [
        'COSMOLOGICAL_PARAMETERS--OMEGA_M',
        'COSMOLOGICAL_PARAMETERS--S_8',
        'COSMOLOGICAL_PARAMETERS--SIGMA_8',
        'J_bin_bias--jhat1',
        'J_bin_bias--jhat2',
        'LSS_PARAMETERS--FSIGMA_8_BIN_1',
        'LSS_PARAMETERS--FSIGMA_8_BIN_2',
        'DERIVED--E_G_1',
        'DERIVED--E_G_2'
    ]
    labels = [r'\Omega_m', r'S_8', r'\sigma_8', r'\hat{J}_1', r'\hat{J}_2', 
              r'f\sigma_{8,1}', r'f\sigma_{8,2}', r'E_{G,1}', r'E_{G,2}']
    samples_array = np.vstack([df[p].values for p in params_to_plot]).T
    param_names = [p.replace('-', '_') for p in params_to_plot]
    return MCSamples(samples=samples_array, weights=weights, 
                     names=param_names, labels=labels, label=label_name)

if __name__ == '__main__':
    from pathlib import Path
    import os
    os.chdir(Path(__file__).resolve().parent)
    samples_no_prior = process_chain('chains/chain_noprior.txt', 'KiDS-1000+BOSS+2dFlenS (No Prior)')
    samples_with_prior = process_chain('chains/chain_withprior.txt', 'KiDS-1000+BOSS+2dFlenS (With Prior)')

    g = plots.get_subplot_plotter()
    g.settings.axes_fontsize = 14
    g.settings.lab_fontsize = 16
    g.settings.legend_fontsize = 16

    params_for_corner = [
        'COSMOLOGICAL_PARAMETERS__OMEGA_M',
        'COSMOLOGICAL_PARAMETERS__S_8',
        'J_bin_bias__jhat1',
        'J_bin_bias__jhat2',
        'DERIVED__E_G_1',
        'DERIVED__E_G_2'
    ]

    g.triangle_plot(
        [samples_no_prior, samples_with_prior], 
        params=params_for_corner, 
        filled=True, 
        contour_colors=['gray', '#006eb8'],
        alpha_filled_add=0.4,
        title_limit=None 
    )

    for j, pname in enumerate(params_for_corner):
        ax = g.subplots[j, j]
        if ax is None: continue
        
        line_no = samples_no_prior.getInlineLatex(pname, limit=1)
        line_with = samples_with_prior.getInlineLatex(pname, limit=1)
        
        ax.text(0.5, 1.12, f"${line_no}$", color='gray', 
                transform=ax.transAxes, ha='center', va='bottom', fontsize=11)
        ax.text(0.5, 1.02, f"${line_with}$", color='#006eb8', 
                transform=ax.transAxes, ha='center', va='bottom', fontsize=11)
    plt.subplots_adjust(wspace=0.1, hspace=0.1, top=0.9)
    plt.savefig('fig/comparison_noprior_vs_withprior.pdf', bbox_inches='tight')
    plt.show()