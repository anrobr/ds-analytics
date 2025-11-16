#!/bin/bash
# Build script for cuVS from source with Python bindings
# Companion to the faiss-gpu build

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${GREEN}=== cuVS Build Script ===${NC}"
echo "Target: Python 3.13, CUDA 13.0, Ubuntu 24.04"
echo ""

# ---- Configuration ----
CUVS_DIR="${CUVS_DIR:-/workspace/cuvs}"
BUILD_DIR="${CUVS_DIR}/build"
PYTHON_EXECUTABLE=$(which python3.13 || which python)

echo -e "${YELLOW}Configuration:${NC}"
echo "  CUVS_DIR: ${CUVS_DIR}"
echo "  BUILD_DIR: ${BUILD_DIR}"
echo "  PYTHON: ${PYTHON_EXECUTABLE}"
echo ""

# ---- Check if cuVS is already installed via pip ----
if ${PYTHON_EXECUTABLE} -c "import cuvs" 2>/dev/null; then
    echo -e "${GREEN}✓ cuVS is already installed via pip${NC}"
    ${PYTHON_EXECUTABLE} -c "import cuvs; print(f'cuVS version: {cuvs.__version__}')"
    echo ""
    echo "If you want to rebuild from source, uninstall first:"
    echo "  pip uninstall cuvs-cu13"
    exit 0
fi

echo -e "${YELLOW}cuVS not found via pip, building from source...${NC}"
echo ""

# ---- Check if cuVS repo exists ----
if [ ! -d "${CUVS_DIR}" ]; then
    echo -e "${YELLOW}Cloning cuVS repository...${NC}"
    git clone --recursive https://github.com/rapidsai/cuvs.git "${CUVS_DIR}"
fi

cd "${CUVS_DIR}"

# ---- Check dependencies ----
echo -e "${YELLOW}Checking dependencies...${NC}"

if ! ${PYTHON_EXECUTABLE} -c "import rmm" 2>/dev/null; then
    echo -e "${YELLOW}Installing rmm...${NC}"
    ${PYTHON_EXECUTABLE} -m pip install --extra-index-url=https://pypi.nvidia.com rmm-cu13
fi

if ! ${PYTHON_EXECUTABLE} -c "import pylibraft" 2>/dev/null; then
    echo -e "${YELLOW}Installing pylibraft...${NC}"
    ${PYTHON_EXECUTABLE} -m pip install --extra-index-url=https://pypi.nvidia.com pylibraft-cu13
fi

echo -e "${GREEN}✓ Dependencies ready${NC}"
echo ""

# ---- Build cuVS ----
echo -e "${GREEN}=== Building cuVS ===${NC}"

# Use the build script provided by cuVS
if [ -f "${CUVS_DIR}/build.sh" ]; then
    echo "Using cuVS build script..."
    cd "${CUVS_DIR}"
    
    # Build with Python bindings
    ./build.sh cuvs pylibcuvs cuvs-python
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ cuVS build successful${NC}"
    else
        echo -e "${RED}✗ cuVS build failed${NC}"
        exit 1
    fi
else
    echo -e "${YELLOW}Using CMake directly...${NC}"
    
    mkdir -p "${BUILD_DIR}"
    cd "${BUILD_DIR}"
    
    cmake -B . -S "${CUVS_DIR}/cpp" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
        -DCUVS_BUILD_TESTS=OFF \
        -DCUVS_BUILD_PRIMS_BENCH=OFF \
        -GNinja
    
    cmake --build . --config Release -j$(nproc)
    cmake --install .
    
    # Build Python bindings
    cd "${CUVS_DIR}/python"
    ${PYTHON_EXECUTABLE} -m pip install -e .
fi

# ---- Test installation ----
echo -e "${GREEN}=== Testing cuVS installation ===${NC}"

${PYTHON_EXECUTABLE} -c "
import cuvs
import numpy as np

print(f'cuVS version: {cuvs.__version__}')

# Test basic operations
from cuvs.neighbors import cagra

# Create sample data
n_samples = 1000
n_features = 128
dataset = np.random.random((n_samples, n_features)).astype('float32')
queries = np.random.random((10, n_features)).astype('float32')

print('✓ cuVS imported successfully')
print('✓ Can create sample data')
print('✓ CAGRA API accessible')
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}🎉 cuVS BUILD SUCCESSFUL!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "cuVS Python bindings installed and ready"
    echo ""
else
    echo -e "${RED}✗ cuVS installation test failed${NC}"
    exit 1
fi
