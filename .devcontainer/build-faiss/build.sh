#!/bin/bash
# Build script for faiss-gpu with Python 3.13 and CUDA 13
# This script compiles faiss from source with GPU support

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== FAISS-GPU Build Script ===${NC}"
echo "Target: Python 3.13, CUDA 13.0, Ubuntu 24.04"
echo ""

# ---- Configuration ----
FAISS_DIR="${FAISS_DIR:-/workspace/faiss}"
BUILD_DIR="${FAISS_DIR}/build"
INSTALL_PREFIX="${INSTALL_PREFIX:-/workspace/output/install}"
PYTHON_EXECUTABLE=$(which python3.13 || which python)
# Ada Lovelace (RTX 4090 Mobile, RTX 40xx series, L4, L40)
CUDA_ARCH="${CUDA_ARCH:-89}"

echo -e "${YELLOW}Configuration:${NC}"
echo "  FAISS_DIR: ${FAISS_DIR}"
echo "  BUILD_DIR: ${BUILD_DIR}"
echo "  INSTALL_PREFIX: ${INSTALL_PREFIX}"
echo "  PYTHON: ${PYTHON_EXECUTABLE}"
echo "  CUDA_ARCH: ${CUDA_ARCH}"
echo ""

# ---- Check dependencies ----
echo -e "${YELLOW}Checking dependencies...${NC}"

if ! command -v cmake &> /dev/null; then
    echo -e "${RED}ERROR: cmake not found${NC}"
    exit 1
fi

if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}ERROR: nvcc (CUDA compiler) not found${NC}"
    exit 1
fi

if ! ${PYTHON_EXECUTABLE} -c "import numpy" 2>/dev/null; then
    echo -e "${RED}ERROR: numpy not found for Python${NC}"
    exit 1
fi

echo -e "${GREEN}✓ All dependencies found${NC}"
echo ""

# ---- Clean previous build ----
if [ -d "${BUILD_DIR}" ]; then
    echo -e "${YELLOW}Cleaning previous build directory...${NC}"
    rm -rf "${BUILD_DIR}"
fi

mkdir -p "${BUILD_DIR}"

# Ensure install directory exists and is writable
if [ -d "${INSTALL_PREFIX}" ]; then
    rm -rf "${INSTALL_PREFIX}"
fi
sudo mkdir -p "${INSTALL_PREFIX}"
sudo chown -R builder:builder "${INSTALL_PREFIX}"
sudo chmod -R 755 "${INSTALL_PREFIX}"

# ---- Navigate to faiss directory ----
cd "${FAISS_DIR}"

# Optionally pull latest (comment out if you want a specific commit)
# git pull

echo -e "${YELLOW}Building faiss commit: $(git rev-parse --short HEAD)${NC}"
echo ""

# ---- Configure with CMake ----
echo -e "${GREEN}=== Step 1: CMake Configuration ===${NC}"

cd "${BUILD_DIR}"

cmake -B . -S "${FAISS_DIR}" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DBUILD_TESTING=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DFAISS_ENABLE_C_API=ON \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  -DPython_EXECUTABLE="${PYTHON_EXECUTABLE}" \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=$(which g++) \
  -DCMAKE_CXX_FLAGS="-O3 -march=native" \
  -GNinja

if [ $? -ne 0 ]; then
    echo -e "${RED}CMake configuration failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ CMake configuration successful${NC}"
echo ""

# ---- Build faiss ----
echo -e "${GREEN}=== Step 2: Building faiss ===${NC}"
echo "This may take 10-30 minutes depending on your hardware..."
echo ""

cmake --build . --config Release -j$(nproc)

if [ $? -ne 0 ]; then
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Build successful${NC}"
echo ""

# ---- Install faiss ----
echo -e "${GREEN}=== Step 3: Installing faiss ===${NC}"

# Ensure install directory is writable
sudo chown -R builder:builder "${INSTALL_PREFIX}" 2>/dev/null || true
sudo chmod -R 755 "${INSTALL_PREFIX}" 2>/dev/null || true

cmake --install .

if [ $? -ne 0 ]; then
    echo -e "${RED}Installation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Installation successful${NC}"
echo ""

# ---- Build Python package ----
echo -e "${GREEN}=== Step 4: Building Python package ===${NC}"

cd "${BUILD_DIR}/faiss/python"

# Patch setup.py to generate proper platform-specific wheel tags
echo -e "${YELLOW}Patching setup.py for platform-specific wheels...${NC}"
${PYTHON_EXECUTABLE} /workspace/fix-setup.py setup.py

${PYTHON_EXECUTABLE} setup.py build

if [ $? -ne 0 ]; then
    echo -e "${RED}Python package build failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python package built${NC}"
echo ""

# ---- Install Python package ----
echo -e "${GREEN}=== Step 5: Installing Python package ===${NC}"

${PYTHON_EXECUTABLE} -m pip install -e .

if [ $? -ne 0 ]; then
    echo -e "${RED}Python package installation failed!${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python package installed${NC}"
echo ""

# ---- Test installation ----
echo -e "${GREEN}=== Step 6: Testing installation ===${NC}"

${PYTHON_EXECUTABLE} -c "
import faiss
print(f'faiss version: {faiss.__version__}')
print(f'faiss compiled with GPU support: {faiss.get_num_gpus() >= 0}')

# Test GPU availability
ngpus = faiss.get_num_gpus()
if ngpus > 0:
    print(f'Number of GPUs available: {ngpus}')
    # Try to use GPU
    res = faiss.StandardGpuResources()
    print('✓ GPU resources initialized successfully')
else:
    print('⚠ No GPUs detected (may be normal if running without GPU)')
"

if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}🎉 FAISS-GPU BUILD SUCCESSFUL!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Installation location: ${INSTALL_PREFIX}"
    echo "Python package installed and ready to use"
    echo ""
    echo "To create a wheel for distribution:"
    echo "  cd ${BUILD_DIR}/faiss/python"
    echo "  ${PYTHON_EXECUTABLE} setup.py bdist_wheel"
    echo ""
else
    echo -e "${RED}Installation test failed!${NC}"
    exit 1
fi
