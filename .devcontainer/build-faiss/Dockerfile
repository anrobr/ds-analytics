# Dockerfile for building faiss-gpu from source
# Target: Python 3.13, CUDA 13.0, Ubuntu 24.04, RAPIDS 25.x compatible
# Architecture: x86_64

ARG CUDA_TAG=13.0.1-devel-ubuntu24.04
FROM nvidia/cuda:${CUDA_TAG}

LABEL maintainer="faiss-gpu-build"
LABEL description="Build container for faiss-gpu with Python 3.13 and CUDA 13"

# ---- Environment variables ----
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV PATH=/usr/local/cuda/bin:${PATH}

# ---- System dependencies + Python 3.13 ----
RUN apt-get update && apt-get install -y \
    # Build essentials
    build-essential \
    cmake \
    git \
    wget \
    curl \
    pkg-config \
    # Python 3.13 from deadsnakes
    software-properties-common \
    # SWIG for Python bindings
    swig \
    # BLAS libraries
    libopenblas-dev \
    liblapack-dev \
    # Additional build tools
    ninja-build \
    ccache \
    # Utilities
    sudo \
    vim \
    patchelf \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.13 \
    python3.13-dev \
    python3.13-venv \
    && /usr/bin/python3.13 -m ensurepip --upgrade \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.13 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# ---- Install modern CMake (faiss needs >= 3.17) ----
RUN wget -qO- "https://cmake.org/files/v3.28/cmake-3.28.3-linux-x86_64.tar.gz" | \
    tar --strip-components=1 -xz -C /usr/local

# ---- Create non-root user ----
RUN useradd -ms /bin/bash builder && \
    adduser builder sudo && \
    echo "builder ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER builder
WORKDIR /workspace

# ---- Install Python build dependencies ----
RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install numpy scipy packaging

# ---- Install RAPIDS dependencies including cuVS ----
RUN python -m pip install --extra-index-url=https://pypi.nvidia.com \
    cupy-cuda13x \
    rmm-cu13 \
    pylibraft-cu13 \
    raft-dask-cu13 || echo "Installing available RAPIDS packages"

# Try to install cuVS (may need nightly or specific version)
RUN python -m pip install --extra-index-url=https://pypi.nvidia.com \
    cuvs-cu13 || \
    python -m pip install --pre --extra-index-url=https://pypi.nvidia.com \
    cuvs-cu13 || \
    echo "cuVS will be built from source if not available"

# ---- Clone faiss repository ----
RUN git clone https://github.com/facebookresearch/faiss.git /workspace/faiss

# ---- Clone cuVS repository (if needed for source build) ----
RUN git clone --recursive https://github.com/rapidsai/cuvs.git /workspace/cuvs || \
    echo "cuVS already available via pip"

# ---- Copy build scripts and setup.py patch ----
# Copy build scripts
COPY --chown=builder:builder build.sh build-cuvs.sh test-integration.py fix-setup.py repair-wheel.py /workspace/

# Set working directory to faiss
WORKDIR /workspace/faiss

# ---- Default command ----
CMD ["/bin/bash"]
