# KiDS--BOSS Weyl-potential analysis

This repository contains the CosmoSIS modules, configurations, and
post-processing scripts used for the joint KiDS-1000, BOSS, and 2dFLenS
analysis of the Weyl-potential evolution and the $E_G$ statistic.

The analysis uses the linear power spectrum at $z_\star=5$ as a reference
and fits $\hat b$, $\hat f$, and $\hat J$ independently in two BOSS redshift
bins. The paper results use the two `explicit_zstar` configurations in
[`config/`](config/).

## Fork relation

This repository is a fork of
[`itutusaus/cosmosis_weyl`](https://github.com/itutusaus/cosmosis_weyl),
which contains the DES Y3 implementation developed by Tutusaus, Bonvin, and
Grimm. This fork adapts that framework to KiDS-1000 galaxy--galaxy lensing
and BOSS spectroscopic wedges. The original Git history is retained.

## Contents

- `config/`: the two main configurations, Fixed-As and Free-S8 checks,
  and two earlier comparison configurations;
- `cosmosis_modules/`: the additional CosmoSIS modules;
- `patches/`: the small BOSS-wedges interface patch required by this analysis;
- `post_process/`: chain-analysis and plotting scripts;
- `analysis/prior_sensitivity/`: correlation tables, posterior summaries,
  and comparison plots;
- `chain/`: local directory for downloaded chains (not included in Git);
- `pipeline.png`: the current CosmoSIS module flow chart.

## Setup and test

Clone the two upstream dependencies at the revisions used for this release:

```bash
git clone https://github.com/KiDS-WL/kcap.git
git -C kcap checkout 36b40c5
git clone https://github.com/KiDS-WL/kcap_boss_module.git
git -C kcap_boss_module checkout 0e894a7
```

Create the environment, build the required native modules, apply the bundled
BOSS patch, and run one likelihood evaluation:

```bash
conda env create -f environment-weyl.yml
conda activate cosmosis_weyl
./scripts/setup_weyl_runtime.sh --test
```

To run the With-Planck Nautilus configuration after the test succeeds:

```bash
MPI_RANKS=8 ./scripts/run_weyl_nautilus.sh
```

The chains are distributed through
[Zenodo](https://doi.org/10.5281/zenodo.22220484), not Git.
Create a local `chain/` directory and place the text chains there before
running the post-processing scripts.

## Prior-sensitivity checks

- [Fixed-As](config/pipeline2x2pt_weyl_explicit_zstar_fixed_as.ini) fixes
  `A_s = 2.1e-9` and otherwise keeps the No-Prior settings.
- [Free-S8](config/pipeline2x2pt_weyl_explicit_zstar_free_s8.ini) keeps the
  With-Prior Gaussian priors on `omch2`, `ombh2`, `h0`, and `n_s`
  but uses the No-Prior uniform range for `S8`.
  Its archived chain filename contains `shape_priors`.

With the four `chain_explicit_zstar_*` text chains in `chain/`, run:

```bash
python post_process/plot_correlations.py
```

This reproduces the posterior-weighted Pearson and Spearman correlations,
posterior summaries, and three plots in
[`analysis/prior_sensitivity/`](analysis/prior_sensitivity/).
The summaries give means, standard deviations, and quantiles.
The comparison plot shows medians and 16th–84th percentile intervals.

## Citation

This repository accompanies:

Xuwei Zhang, Ming Zhang, Yunliang Ren, and Xiaofeng Yang, "Measurement of
the Weyl Potential Evolution and $E_G$ Statistic from KiDS-1000, BOSS and
2dFLenS," [arXiv:2604.24631](https://arxiv.org/abs/2604.24631), 2026.

```bibtex
@article{zhang2026weyl,
  title = {Measurement of the Weyl Potential Evolution and {$E_G$} Statistic from {KiDS-1000}, {BOSS} and {2dFLenS}},
  author = {Zhang, Xuwei and Zhang, Ming and Ren, Yunliang and Yang, Xiaofeng},
  year = {2026},
  eprint = {2604.24631},
  archivePrefix = {arXiv},
  primaryClass = {astro-ph.CO},
  doi = {10.48550/arXiv.2604.24631}
}
```

Machine-readable citation metadata are provided in
[`CITATION.cff`](CITATION.cff). Fork attribution, dependency revisions, and
the license notice for the modified BOSS interface are recorded in
[`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES). This release does not assign a
new blanket license to files inherited from the upstream `cosmosis_weyl`
repository.
