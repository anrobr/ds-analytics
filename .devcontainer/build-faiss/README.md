# FAISS-GPU + cuVS Build Container

This directory contains configuration for building `faiss-gpu` and `cuVS` from source for Python 3.13, CUDA 13.0, and Ubuntu 24.04.

## Overview

**Target Configuration:**
- Python: 3.13
- CUDA: 13.0.1
- OS: Ubuntu 24.04.3 LTS
- Architecture: x86_64
- RAPIDS compatibility: 25.x

**Libraries Built:**
- **faiss-gpu**: Facebook AI Similarity Search with GPU acceleration
- **cuVS**: CUDA Vector Search from RAPIDS (CAGRA algorithm)

Both libraries provide Python bindings and can be used together for high-performance vector search.

## Prerequisites

- Docker with NVIDIA GPU support (nvidia-docker2)
- NVIDIA GPU with CUDA 13.0 support
- At least 8GB disk space for build artifacts

## Quick Start

### 1. Build the container

```bash
cd .devcontainer/build-faiss
docker build -t faiss-gpu-builder:cuda13-py313 .
```

### 2. Run the build container

```bash
# Run with GPU access
docker run --gpus all -it --rm \
  -v $(pwd)/output:/workspace/faiss-install \
  faiss-gpu-builder:cuda13-py313
```

### 3. Build faiss inside the container

```bash
# Inside the container
./build.sh
```

### 4. Build cuVS (optional but recommended)

```bash
# Inside the container
./build-cuvs.sh
```

### 5. Test integration

```bash
# Inside the container
python test-integration.py
```

Or run everything at once:

```bash
docker run --gpus all -it --rm \
  -v $(pwd)/output:/workspace/faiss-install \
  faiss-gpu-builder:cuda13-py313 \
  bash -c "cd /workspace/faiss && /workspace/build.sh && /workspace/build-cuvs.sh && python /workspace/test-integration.py"
```

## Build Configuration

### CUDA Architectures

By default, the build script compiles for Ada Lovelace architecture (RTX 40xx series):
- `89` - RTX 4090 Mobile, RTX 4090, RTX 4080, RTX 4070, L4, L40

To customize for other GPUs, set the `CUDA_ARCH` environment variable:

```bash
# Build for multiple architectures
docker run --gpus all -it --rm \
  -e CUDA_ARCH="86;89" \
  faiss-gpu-builder:cuda13-py313

# Common architectures:
# 75 - Tesla T4, RTX 20xx
# 80 - A100
# 86 - RTX 30xx
# 89 - RTX 40xx (default)
# 90 - H100
```

### Custom faiss Version

To build a specific faiss commit or tag:

```bash
docker run --gpus all -it --rm \
  faiss-gpu-builder:cuda13-py313 \
  bash -c "cd /workspace/faiss && git checkout v1.8.0 && ./build.sh"
```

## Build Output

After a successful build:
- C++ libraries: `/workspace/faiss-install/lib/`
- Headers: `/workspace/faiss-install/include/`
- Python package: Installed in container Python environment
- Wheel file: `/workspace/faiss/build/faiss/python/dist/`

### Creating a distributable wheel

```bash
# Inside the container after successful build
cd /workspace/faiss/build/faiss/python
python setup.py bdist_wheel

# The wheel will be in dist/
ls -lh dist/*.whl
```

### Extracting the wheel from the container

```bash
# Copy wheel from running container
docker cp <container-id>:/workspace/faiss/build/faiss/python/dist/faiss_gpu-*.whl .

# Or mount a volume to extract it
docker run --gpus all -it --rm \
  -v $(pwd)/wheels:/output \
  faiss-gpu-builder:cuda13-py313 \
  bash -c "./build.sh && cp /workspace/faiss/build/faiss/python/dist/*.whl /output/"
```

## Testing the Build

### Quick test inside container

```bash
# Test faiss-gpu
python -c "
import faiss
import numpy as np

print(f'faiss version: {faiss.__version__}')
print(f'GPUs available: {faiss.get_num_gpus()}')

d = 128
index = faiss.IndexFlatL2(d)
xb = np.random.random((1000, d)).astype('float32')
index.add(xb)
print(f'Index size: {index.ntotal}')
print('✓ faiss-gpu works')
"

# Test cuVS
python -c "
import cuvs
from cuvs.neighbors import cagra
import cupy as cp

print(f'cuVS version: {cuvs.__version__}')

dataset = cp.random.random((1000, 128), dtype=cp.float32)
index = cagra.build(cagra.IndexParams(), dataset)
queries = cp.random.random((10, 128), dtype=cp.float32)
distances, neighbors = cagra.search(cagra.SearchParams(), index, queries, k=5)
print(f'Search result shape: {neighbors.shape}')
print('✓ cuVS works')
"

# Test integration
python test-integration.py
```

### Full test suite

```bash
cd /workspace/faiss
python -m pytest tests/
```

## RAPIDS Integration

This build includes full RAPIDS cuVS integration for GPU-accelerated vector search.

### Using faiss-gpu with cuVS

Both libraries can work together seamlessly:

```python
import faiss
import cuvs
from cuvs.neighbors import cagra
import cupy as cp
import numpy as np

# Create data
d = 128
n = 10000
data = np.random.random((n, d)).astype('float32')

# Option 1: Use faiss-gpu
res = faiss.StandardGpuResources()
faiss_index = faiss.GpuIndexFlatL2(res, d)
faiss_index.add(data)
D, I = faiss_index.search(data[:10], k=5)

# Option 2: Use cuVS (RAPIDS)
data_gpu = cp.asarray(data)
cuvs_index = cagra.build(cagra.IndexParams(), data_gpu)
distances, neighbors = cagra.search(cagra.SearchParams(), cuvs_index, data_gpu[:10], k=5)

# Convert between libraries
# cupy -> numpy (for faiss)
numpy_data = cp.asnumpy(data_gpu)
# numpy -> cupy (for cuVS)  
cupy_data = cp.asarray(data)
```

### When to use which library

**Use faiss-gpu when:**
- You need a wide variety of index types (IVF, HNSW, PQ, etc.)
- You want battle-tested, production-ready code
- You need CPU fallback support
- Cross-platform compatibility is important

**Use cuVS when:**
- You want the fastest GPU-only vector search (CAGRA algorithm)
- You're already using RAPIDS ecosystem (cuDF, cuML)
- You want seamless cupy integration
- You need state-of-the-art graph-based search

## Troubleshooting

### Build fails with CUDA errors

Ensure your host has CUDA 13.0 compatible drivers:
```bash
nvidia-smi
```

### Python package not found after build

Make sure you're using the correct Python:
```bash
which python
# Should show 3.13.x
python --version
```

### Out of memory during build

Reduce parallel jobs:
```bash
# Edit build.sh and change:
cmake --build . -j$(nproc)
# to:
cmake --build . -j4
```

### GPU not detected

Check GPU availability in container:
```bash
docker run --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi
```

## Advanced Usage

### Interactive development

```bash
docker run --gpus all -it --rm \
  -v $(pwd):/host \
  -v /path/to/faiss-fork:/workspace/faiss \
  faiss-gpu-builder:cuda13-py313 \
  bash
```

### Build with debug symbols

Edit `build.sh` and change:
```bash
-DCMAKE_BUILD_TYPE=Release
# to:
-DCMAKE_BUILD_TYPE=RelWithDebInfo
```

### Custom CMake options

The build script supports environment variables:
```bash
export CMAKE_ARGS="-DFAISS_OPT_LEVEL=avx512"
./build.sh
```

## References

- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki)
- [CUDA 13.0 Release Notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/)
- [RAPIDS Documentation](https://docs.rapids.ai/)
