# faiss-gpu vs cuVS Comparison Guide

## Quick Comparison

| Feature | faiss-gpu | cuVS |
|---------|-----------|------|
| **Origin** | Meta AI (Facebook) | NVIDIA RAPIDS |
| **Primary Focus** | General similarity search | GPU-accelerated graph search |
| **Best Algorithm** | Various (IVF, HNSW, PQ) | CAGRA (GPU-optimized) |
| **Language** | C++ with Python bindings | C++/CUDA with Python bindings |
| **Platform** | CPU + GPU | GPU-focused |
| **Maturity** | Very mature (2017+) | Newer (2023+) |
| **Ecosystem** | Standalone | RAPIDS integrated |

## Performance Characteristics

### faiss-gpu
- **Strengths:**
  - Wide variety of index types
  - Excellent CPU fallback
  - Battle-tested at scale (billions of vectors)
  - Advanced quantization (PQ, OPQ, SQ)
  - Mature codebase with years of optimization

- **Best for:**
  - Production deployments requiring reliability
  - Multi-platform support (CPU/GPU/multi-GPU)
  - Diverse index type requirements
  - When CPU fallback is needed

### cuVS
- **Strengths:**
  - State-of-the-art GPU performance (CAGRA algorithm)
  - Native RAPIDS integration (works with cuDF, cuML)
  - Optimized for NVIDIA GPUs
  - Modern GPU-first design
  - Seamless cupy array support

- **Best for:**
  - Maximum GPU performance
  - RAPIDS ecosystem users
  - Research and experimentation
  - When you have dedicated NVIDIA GPUs

## Index Types

### faiss-gpu Indices

```python
import faiss
import numpy as np

d = 128  # dimension
n = 100000  # database size

# 1. Flat (exact search)
index = faiss.IndexFlatL2(d)

# 2. IVF (inverted file index)
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, 100)  # 100 clusters

# 3. IVFPQ (IVF + Product Quantization)
index = faiss.IndexIVFPQ(quantizer, d, 100, 8, 8)  # 8 sub-quantizers, 8 bits each

# 4. HNSW (Hierarchical Navigable Small World)
index = faiss.IndexHNSWFlat(d, 32)  # 32 neighbors per node

# Convert to GPU
res = faiss.StandardGpuResources()
gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
```

### cuVS Indices

```python
from cuvs.neighbors import cagra, ivf_flat, ivf_pq
import cupy as cp

d = 128
n = 100000
dataset = cp.random.random((n, d), dtype=cp.float32)

# 1. IVF Flat
ivf_flat_params = ivf_flat.IndexParams(n_lists=100)
ivf_index = ivf_flat.build(ivf_flat_params, dataset)

# 2. IVF PQ
ivf_pq_params = ivf_pq.IndexParams(n_lists=100, pq_dim=8)
ivf_pq_index = ivf_pq.build(ivf_pq_params, dataset)

# 3. CAGRA (GPU-optimized graph-based)
cagra_params = cagra.IndexParams(
    graph_degree=64,
    intermediate_graph_degree=128
)
cagra_index = cagra.build(cagra_params, dataset)
```

## Code Examples

### Basic Search Comparison

```python
import numpy as np
import cupy as cp
import faiss
from cuvs.neighbors import cagra

# Prepare data
d = 128
nb = 100000  # database size
nq = 100     # queries
k = 10       # neighbors to find

xb = np.random.random((nb, d)).astype('float32')
xq = np.random.random((nq, d)).astype('float32')

# ========== faiss-gpu ==========
print("faiss-gpu:")

# Build index
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatL2(res, d)
index.add(xb)

# Search
import time
t0 = time.time()
D, I = index.search(xq, k)
t1 = time.time()
print(f"Search time: {(t1-t0)*1000:.2f} ms")
print(f"First result distances: {D[0][:3]}")

# ========== cuVS ==========
print("\ncuVS:")

# Convert to GPU
xb_gpu = cp.asarray(xb)
xq_gpu = cp.asarray(xq)

# Build index
index_params = cagra.IndexParams()
cagra_index = cagra.build(index_params, xb_gpu)

# Search
cp.cuda.Stream.null.synchronize()
t0 = time.time()
search_params = cagra.SearchParams()
distances, neighbors = cagra.search(search_params, cagra_index, xq_gpu, k)
cp.cuda.Stream.null.synchronize()
t1 = time.time()
print(f"Search time: {(t1-t0)*1000:.2f} ms")
print(f"First result distances: {distances[0][:3]}")
```

### Memory Efficiency

```python
# faiss: memory-efficient with quantization
import faiss

d = 128
nb = 1000000

# Product Quantization (8x compression)
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFPQ(quantizer, d, 1000, 8, 8)
index.train(training_data)
index.add(database_vectors)

# Estimate memory:
# PQ: nb * (8 bytes for IVF + d/8 bytes for PQ) 
# vs nb * d * 4 bytes for flat
print(f"Memory saved: ~{(1 - (8 + d/8)/(d*4))*100:.1f}%")

# cuVS: optimized graph structure
from cuvs.neighbors import cagra

# CAGRA graph memory is predictable
graph_degree = 64
# Memory: nb * (d * 4 bytes + graph_degree * 4 bytes)
```

### Batch Processing

```python
# faiss: efficient batching
for batch in query_batches:
    D, I = index.search(batch, k)
    process_results(D, I)

# cuVS: GPU stream optimization
import cupy as cp

for batch in query_batches:
    batch_gpu = cp.asarray(batch)
    with cp.cuda.Stream(non_blocking=True):
        distances, neighbors = cagra.search(
            search_params, cagra_index, batch_gpu, k
        )
        # Async processing
```

## Integration Patterns

### Pattern 1: Data Pipeline with Both

```python
import cudf
import cupy as cp
from cuvs.neighbors import cagra
import faiss
import numpy as np

# Load data with cuDF (RAPIDS)
df = cudf.read_parquet('data.parquet')

# Extract feature columns
features = df[feature_cols].values  # cupy array

# Build cuVS index for GPU search
cagra_index = cagra.build(cagra.IndexParams(), features)

# For deployment, export to faiss for broader compatibility
features_np = cp.asnumpy(features)
faiss_index = faiss.IndexFlatL2(d)
faiss_index.add(features_np)

# Save both
# cuVS for internal GPU workloads
cagra.save("index.cagra", cagra_index)
# faiss for external API
faiss.write_index(faiss_index, "index.faiss")
```

### Pattern 2: Hybrid CPU/GPU

```python
# Use faiss for flexibility
import faiss

# Start with CPU index
cpu_index = faiss.IndexFlatL2(d)
cpu_index.add(vectors)

# Move to GPU when available
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
    # Use gpu_index
else:
    # Fall back to CPU
    # Use cpu_index
```

### Pattern 3: RAPIDS-native Pipeline

```python
# Use cuVS when fully in RAPIDS ecosystem
import cudf
import cuml
from cuvs.neighbors import cagra

# Load data
df = cudf.read_parquet('data.parquet')

# Feature engineering with cuML
scaler = cuml.preprocessing.StandardScaler()
features = scaler.fit_transform(df[feature_cols])

# Build cuVS index (all GPU, no CPU transfers)
index = cagra.build(cagra.IndexParams(), features)

# Search stays on GPU
results_df = cudf.DataFrame({
    'distances': distances.ravel(),
    'neighbors': neighbors.ravel()
})
```

## Performance Tuning

### faiss-gpu Tuning

```python
# 1. Adjust IVF probes (accuracy vs speed)
index.nprobe = 10  # Lower = faster, less accurate

# 2. Use GPU resources efficiently
res = faiss.StandardGpuResources()
res.setTempMemory(2 * 1024 * 1024 * 1024)  # 2GB temp memory

# 3. Multi-GPU
cpu_index = faiss.IndexFlatL2(d)
gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
```

### cuVS Tuning

```python
# 1. Adjust CAGRA search parameters
search_params = cagra.SearchParams(
    itopk_size=128,  # Internal topk size
    search_width=4,   # Beam width
    max_iterations=100
)

# 2. Index build parameters
index_params = cagra.IndexParams(
    graph_degree=64,          # Higher = more accurate, slower
    intermediate_graph_degree=128
)

# 3. Memory management
cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
```

## Decision Matrix

Choose **faiss-gpu** when you need:
- [ ] Production stability and maturity
- [ ] CPU fallback capability
- [ ] Diverse index types (especially quantization)
- [ ] Cross-platform deployment
- [ ] Billion+ scale with memory constraints

Choose **cuVS** when you need:
- [ ] Maximum GPU performance
- [ ] Already using RAPIDS ecosystem
- [ ] State-of-the-art graph algorithms (CAGRA)
- [ ] Native cupy/GPU-first workflow
- [ ] Research/experimentation with latest methods

Use **both** when you want:
- [ ] Development with cuVS, deployment with faiss
- [ ] Benchmark and compare algorithms
- [ ] Best of both worlds (flexibility + performance)

## References

- [faiss Documentation](https://github.com/facebookresearch/faiss/wiki)
- [cuVS Documentation](https://docs.rapids.ai/api/cuvs/stable/)
- [CAGRA Paper](https://arxiv.org/abs/2308.15136)
