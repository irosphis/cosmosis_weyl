#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
KCAP_DIR="${REPO_DIR}/kcap"
BOSS_DIR="${REPO_DIR}/kcap_boss_module"
WEYL_INI="${REPO_DIR}/config/pipeline2x2pt_weyl_explicit_zstar_with_planck.ini"
WEYL_REPARAMETERIZED_INI="${REPO_DIR}/config/pipeline2x2pt_weyl_reparameterized_with_planck.ini"
BOSS_PATCH="${REPO_DIR}/patches/kcap_boss_module.patch"

PYTHON_BIN="${PYTHON_BIN:-python}"
COSMOSIS_BIN="${COSMOSIS_BIN:-cosmosis}"

for command_name in "${PYTHON_BIN}" "${COSMOSIS_BIN}" make gcc g++ gfortran pkg-config; do
    if ! command -v "${command_name}" >/dev/null 2>&1; then
        echo "Missing required command: ${command_name}" >&2
        exit 1
    fi
done

"${PYTHON_BIN}" -c "import astropy, camb, cosmosis, getdist, h5py, matplotlib, nautilus, numpy, scipy"

if git -C "${BOSS_DIR}" apply --check "${BOSS_PATCH}" 2>/dev/null; then
    echo "Applying the BOSS wedges interface patch"
    git -C "${BOSS_DIR}" apply "${BOSS_PATCH}"
elif git -C "${BOSS_DIR}" apply --reverse --check "${BOSS_PATCH}" 2>/dev/null; then
    echo "The BOSS wedges interface patch is already applied"
else
    echo "Cannot apply ${BOSS_PATCH}; use kcap_boss_module revision 0e894a7" >&2
    exit 1
fi

COSMOSIS_DIR="$("${PYTHON_BIN}" -c 'import cosmosis, os; print(os.path.dirname(cosmosis.__file__))')"
PYTHON_SITE_DIR="$(dirname "${COSMOSIS_DIR}")"
COSMOSIS_DATABLOCK_LIB="${COSMOSIS_DIR}/datablock"
GSL_INC="$(pkg-config --variable=includedir gsl)"
GSL_LIB="$(pkg-config --variable=libdir gsl)"
LAPACK_LIB="$(pkg-config --variable=libdir lapack)"

# The KCAP projection code predates NumPy 2, which removed np.int. np.intc is
# the matching NumPy dtype for the C int expected by this ctypes declaration.
LIMBER_PY="${KCAP_DIR}/cosmosis-standard-library/structure/projection/limber.py"
if grep -q 'dtype=np.int,' "${LIMBER_PY}"; then
    sed -i 's/dtype=np\.int,/dtype=np.intc,/' "${LIMBER_PY}"
fi

echo "Building the Limber projection library"
make -B -C "${KCAP_DIR}/cosmosis-standard-library/structure/projection" \
    COSMOSIS_SRC_DIR="${COSMOSIS_DIR}" \
    USER_CFLAGS="-I${GSL_INC} -I${PYTHON_SITE_DIR}" \
    USER_LDFLAGS="-lcosmosis -L${COSMOSIS_DATABLOCK_LIB} -Wl,-rpath,${COSMOSIS_DATABLOCK_LIB} -L${GSL_LIB} -lgsl -lgslcblas"

echo "Building the COSEBIs support library"
make -C "${KCAP_DIR}/cosebis/modules" \
    COSMOSIS_SRC_DIR="${COSMOSIS_DIR}" \
    GSL_INC="${GSL_INC}" \
    GSL_LIB="${GSL_LIB}"

echo "Building the GGL bandpower module"
make -B -C "${KCAP_DIR}/cosebis" libbandpower.so \
    COSMOSIS_SRC_DIR="${COSMOSIS_DIR}" \
    USER_CXXFLAGS="-I${GSL_INC} -I${PYTHON_SITE_DIR} -I./modules" \
    USER_LDFLAGS="-lcosmosis -L${COSMOSIS_DATABLOCK_LIB} -Wl,-rpath,${COSMOSIS_DATABLOCK_LIB} -L${GSL_LIB} -lgsl -lgslcblas -lm -L./modules -lcolib -lstdc++ -lz"

echo "Building the bundled Cuba library"
if [[ ! -f "${BOSS_DIR}/Cuba-1.4/libcuba.a" ]]; then
    if [[ ! -f "${BOSS_DIR}/Cuba-1.4/makefile" ]]; then
        (
            cd "${BOSS_DIR}/Cuba-1.4"
            CC=gcc CXX=g++ FC=gfortran CFLAGS=-fPIC ./configure
        )
    fi
    (
        cd "${BOSS_DIR}/Cuba-1.4"
        make lib
    )
fi

echo "Building the BOSS wedges wrapper"
make -C "${BOSS_DIR}/python_interface" \
    FC=gfortran \
    FFLAGS="-fPIC -cpp -fopenmp -O3 -ffree-line-length-none -std=legacy -fallow-argument-mismatch" \
    LAPACK_LINK="-L${LAPACK_LIB} -llapack -lblas"

if [[ "${1:-}" == "--test" ]]; then
    echo "Running the explicit-zstar Weyl pipeline with sampler=test"
    (
        cd "${KCAP_DIR}"
        "${COSMOSIS_BIN}" "${WEYL_INI}" -p runtime.sampler=test
    )
elif [[ "${1:-}" == "--reparameterized-test" ]]; then
    echo "Running the simple reparameterized-amplitude pipeline with sampler=test"
    (
        cd "${KCAP_DIR}"
        "${COSMOSIS_BIN}" "${WEYL_REPARAMETERIZED_INI}" -p runtime.sampler=test
    )
fi

echo "Weyl runtime setup complete"
