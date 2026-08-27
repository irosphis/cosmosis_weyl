# Weyl pipeline runtime

The supported entry point is:

```text
kcap/runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/config/pipeline2x2pt_weyl.ini
```

## System requirements

On Ubuntu, the local modules require GCC/GFortran, Make, pkg-config, GSL,
BLAS, and LAPACK. For example:

```bash
sudo apt install build-essential gfortran pkg-config libgsl-dev libblas-dev liblapack-dev
```

## Python environment

Create and activate the tested environment:

```bash
conda env create -f environment-weyl.yml
conda activate cosmosis_weyl
```

If the environment already exists, verify the important packages with:

```bash
python -c "import camb, cosmosis, h5py, nautilus; print(camb.__version__, cosmosis.__version__, h5py.__version__, nautilus.__version__)"
```

## Build and test

Compile only the native modules required by the Weyl pipeline and run one
complete likelihood evaluation:

```bash
./scripts/setup_weyl_runtime.sh --test
```

The first run also generates deterministic COSEBIs response tables under
`kcap/cosebis/BandPower_outputs/`. Later runs reuse those tables.

To run the already-built test directly:

```bash
cd kcap
cosmosis runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/config/pipeline2x2pt_weyl.ini -p runtime.sampler=test
```

## Simple reparameterized-amplitude pipeline

The BOSS + GGL pipeline that independently samples `bhat` and `fhat` is:

```text
config/pipeline2x2pt_weyl_reparameterized_with_planck.ini
```

Rebuild the BOSS wrapper and run its test point with:

```bash
./scripts/setup_weyl_runtime.sh --reparameterized-test
```

Or, after the native modules have already been compiled:

```bash
cd kcap
cosmosis ../config/pipeline2x2pt_weyl_reparameterized_with_planck.ini -p runtime.sampler=test
```

## Nautilus

### Reparameterized pipeline with MPI

This is the same launch pattern used on the old machine.  The current Ryzen
7 7840HS has 8 physical cores and 16 hardware threads.  OpenMPI exposes the
8 physical cores as slots by default, but all 16 hardware threads can be used
explicitly:

```bash
export PYTHONWARNINGS="ignore"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate cosmosis_weyl
cd /home/wzx/WorkWork/cosmosis_weyl/kcap

mpirun --use-hwthread-cpus -n 16 --map-by hwthread --bind-to hwthread cosmosis --mpi \
  ../config/pipeline2x2pt_weyl_reparameterized_with_planck.ini \
  -p runtime.sampler=nautilus \
  output.filename=../chain/chain_reparameterized_with_planck_nautilus.txt \
  > output/nautilus_reparameterized_with_planck_<YYYYMMDD>.log 2>&1
```

`mpirun` starts 16 independent MPI processes and binds one to each hardware
thread; this is MPI process parallelism rather than one Python process creating
16 threads.  To use only the 8 physical cores, replace the beginning with
`mpirun -n 8`.

Both modes were tested.  For this CPU-heavy likelihood, one 16-point batch took
39.04 seconds with 16 processes, while the observed 8-process throughput was
about 40--43 seconds per 16 points.  Thus 16 processes work correctly but give
only a small speed gain, not a factor of two.  Eight processes leave the desktop
more responsive; 16 processes fully occupy the machine.

For a long unattended run, enter a `screen` session before running the command.
To resume, keep the same output filename and `nautilus.resume=T`.  To start an
unrelated chain, use a new filename (or set `nautilus.resume=F`).

The environment file includes `mpi4py`, which is required by `cosmosis --mpi`.

For a long run, the repository also provides a filtered-output launcher:

```bash
cd /home/wzx/WorkWork/cosmosis_weyl/kcap
../scripts/run_weyl_nautilus.sh
```

It always uses `nautilus.resume=T` and the production chain/checkpoint above.
It keeps Nautilus status lines, resume messages, final counters, and fatal
exceptions, while discarding repetitive per-likelihood module diagnostics.
Filtering affects only the terminal log; the scientific checkpoint and final
chain remain complete.

### Original Weyl pipeline

The INI keeps a `[nautilus]` section. To launch the full sampler without
editing the file:

```bash
cd kcap
cosmosis runs/3x2pt/data_iterated_cov/cosmology/multinest_blindC_EE_nE_w/config/pipeline2x2pt_weyl.ini -p runtime.sampler=nautilus
```

Before a long run, set `[output] filename` to a unique path. CosmoSIS 3.23
derives the Nautilus checkpoint name from this output path (for text output it
adds `.sampler_status.hdf5`). The legacy `[nautilus] filepath` option in the INI
is not used by this CosmoSIS version. With `resume = T`, reusing an old output
name can resume a different run unintentionally.
