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

- `config/`: the two paper configurations and two comparison configurations;
- `cosmosis_modules/`: the additional CosmoSIS modules;
- `patches/`: the small BOSS-wedges interface patch required by this analysis;
- `post_process/`: the final chain-analysis and plotting scripts, which write
  generated plots to `figures/`;
- `chain/`: chain catalogue and checksums;
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

The completed text chains and Nautilus checkpoints are not stored in Git
because of their size. Their filenames and SHA-256 checksums are listed in
[`chain/`](chain/). Place the chains there before running the scripts in
`post_process/`.

## Citation and third-party software

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). Fork
attribution, dependency revisions, and the license notice for the modified
BOSS interface are recorded in [`THIRD_PARTY_NOTICES`](THIRD_PARTY_NOTICES).
This release does not assign a new blanket license to files inherited from
the upstream `cosmosis_weyl` repository.

## Version

This is the source layout for release `v1.0.0`.
