#!/usr/bin/env bash

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
KCAP_DIR="${REPO_DIR}/kcap"
# COSMOSIS_BIN_OVERRIDE can be used after moving the environment.
COSMOSIS_BIN="${COSMOSIS_BIN_OVERRIDE:-/home/wzx/miniconda3/envs/cosmosis_weyl/bin/cosmosis}"
MPI_RANKS="${MPI_RANKS:-16}"
CHAIN_FILE="${CHAIN_FILE:-${REPO_DIR}/chain/chain_reparameterized_with_planck_nautilus.txt}"
PROGRESS_LOG="${PROGRESS_LOG:-${KCAP_DIR}/output/nautilus_reparameterized_with_planck_20260808.log}"
INI_FILE="${INI_FILE:-${REPO_DIR}/config/pipeline2x2pt_weyl_reparameterized_with_planck.ini}"
# Leave empty for the prior file named by the pipeline.  A relative path is
# interpreted from KCAP_DIR, matching CosmoSIS's handling of pipeline paths.
PRIORS_FILE="${PRIORS_FILE:-}"

mkdir -p "$(dirname "${CHAIN_FILE}")" "$(dirname "${PROGRESS_LOG}")"

export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1
export COSMOSIS_OMP_PARALLEL=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

MPI_OPTIONS=(-n "${MPI_RANKS}")
if (( MPI_RANKS > 8 )); then
    MPI_OPTIONS=(--use-hwthread-cpus -n "${MPI_RANKS}" --map-by hwthread --bind-to hwthread)
fi

printf '\n=== filtered resume: %s ===\n' "$(date --iso-8601=seconds)" | tee -a "${PROGRESS_LOG}"

cd "${KCAP_DIR}"

COSMOSIS_OVERRIDES=(
    -p runtime.sampler=nautilus
    "output.filename=${CHAIN_FILE}"
    nautilus.resume=T
)
if [[ -n "${PRIORS_FILE}" ]]; then
    COSMOSIS_OVERRIDES+=("pipeline.priors=${PRIORS_FILE}")
fi

# Convert Nautilus carriage-return status updates to lines, then retain only
# progress, resume information, final counters, and genuine fatal exceptions.
# Repetitive per-likelihood module diagnostics are intentionally discarded.
mpirun "${MPI_OPTIONS[@]}" "${COSMOSIS_BIN}" --mpi "${INI_FILE}" \
    "${COSMOSIS_OVERRIDES[@]}" 2>&1 \
    | tr '\r' '\n' \
    | awk '
        /^(Status|Sampling|Computing|Bounding|Finished|Stopped)[[:space:]]*\|/ ||
        /Starting the nautilus sampler/ ||
        /Using MPI with/ ||
        /Saving output/ ||
        /Resuming sampling/ ||
        /Resuming Nautilus/ ||
        /Total posterior evaluations/ ||
        /Successful posterior evaluations/ ||
        /wedges module timed out/ ||
        /Traceback/ ||
        /[A-Za-z]+Error:/ ||
        /Exception:/ ||
        /MPI_ABORT/ ||
        /mpirun has exited/ ||
        /non-zero exit code/ ||
        /Segmentation fault/ ||
        /Killed/ {
            print
            fflush()
        }
    ' \
    | tee -a "${PROGRESS_LOG}"

RUN_CODE="${PIPESTATUS[0]}"
printf 'CosmoSIS exit code: %s at %s\n' "${RUN_CODE}" "$(date --iso-8601=seconds)" \
    | tee -a "${PROGRESS_LOG}"
exit "${RUN_CODE}"
