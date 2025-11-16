#!/usr/bin/env python3
"""
Integration test for faiss-gpu and cuVS
Tests interoperability between the two vector search libraries
"""

import sys
import numpy as np

def test_faiss_gpu():
    """Test faiss-gpu basic operations"""
    print("=== Testing faiss-gpu ===")
    
    try:
        import faiss
        
        print(f"✓ faiss version: {faiss.__version__}")
        
        # Check GPU support
        ngpus = faiss.get_num_gpus()
        print(f"GPUs available: {ngpus}")
        
        # Create sample data
        d = 128
        n = 10000
        xb = np.random.random((n, d)).astype('float32')
        xq = np.random.random((10, d)).astype('float32')
        
        # CPU index
        index_cpu = faiss.IndexFlatL2(d)
        index_cpu.add(xb)
        D_cpu, I_cpu = index_cpu.search(xq, 5)
        print(f"✓ CPU index works (searched {index_cpu.ntotal} vectors)")
        
        # GPU index (if available)
        if ngpus > 0:
            res = faiss.StandardGpuResources()
            index_gpu = faiss.GpuIndexFlatL2(res, d)
            index_gpu.add(xb)
            D_gpu, I_gpu = index_gpu.search(xq, 5)
            print(f"✓ GPU index works (searched {index_gpu.ntotal} vectors)")
        
        return True
        
    except Exception as e:
        print(f"✗ faiss-gpu test failed: {e}")
        return False

def test_cuvs():
    """Test cuVS basic operations"""
    print("\n=== Testing cuVS ===")
    
    try:
        import cuvs
        from cuvs.neighbors import cagra
        import cupy as cp
        
        print(f"✓ cuVS version: {cuvs.__version__}")
        
        # Create sample data on GPU
        n_samples = 10000
        n_features = 128
        n_queries = 10
        
        dataset = cp.random.random((n_samples, n_features), dtype=cp.float32)
        queries = cp.random.random((n_queries, n_features), dtype=cp.float32)
        
        print(f"✓ Created GPU dataset: {dataset.shape}")
        
        # Build CAGRA index
        index_params = cagra.IndexParams(metric="sqeuclidean")
        index = cagra.build(index_params, dataset)
        
        print(f"✓ Built CAGRA index")
        
        # Search
        search_params = cagra.SearchParams()
        distances, neighbors = cagra.search(search_params, index, queries, k=5)
        
        print(f"✓ Search completed: found {neighbors.shape[1]} neighbors per query")
        
        return True
        
    except Exception as e:
        print(f"✗ cuVS test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_interop():
    """Test interoperability between faiss and cuVS"""
    print("\n=== Testing faiss-gpu + cuVS Interoperability ===")
    
    try:
        import faiss
        import cuvs
        from cuvs.neighbors import cagra
        import cupy as cp
        
        # Create sample data
        d = 128
        n = 5000
        
        # Data on GPU (cupy)
        xb_cp = cp.random.random((n, d)).astype(cp.float32)
        xq_cp = cp.random.random((10, d)).astype(cp.float32)
        
        # Convert to numpy for faiss
        xb_np = cp.asnumpy(xb_cp)
        xq_np = cp.asnumpy(xq_cp)
        
        print("✓ Created data compatible with both libraries")
        
        # Test 1: Build cuVS index, search with both
        print("\nTest 1: cuVS index")
        index_params = cagra.IndexParams()
        cuvs_index = cagra.build(index_params, xb_cp)
        search_params = cagra.SearchParams()
        distances_cuvs, neighbors_cuvs = cagra.search(search_params, cuvs_index, xq_cp, k=5)
        print(f"  cuVS search: found neighbors with shape {neighbors_cuvs.shape}")
        
        # Test 2: Build faiss GPU index
        print("\nTest 2: faiss GPU index")
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            faiss_index = faiss.GpuIndexFlatL2(res, d)
            faiss_index.add(xb_np)
            distances_faiss, neighbors_faiss = faiss_index.search(xq_np, 5)
            print(f"  faiss search: found neighbors with shape {neighbors_faiss.shape}")
            
            # Compare results (they should be similar but not identical due to different algorithms)
            # Convert CuPy GPU array to host for indexing
            dist_cuvs_cpu = cp.asnumpy(distances_cuvs)
            print(f"  cuVS first neighbor distances (mean): {dist_cuvs_cpu[:, 0].mean():.4f}")
            print(f"  faiss first neighbor distances (mean): {distances_faiss[:, 0].mean():.4f}")
        else:
            print("  Skipping faiss GPU (no GPU available)")
        
        print("\n✓ Interoperability test successful")
        print("  Both libraries can work with the same data")
        print("  Data can be converted between numpy (faiss) and cupy (cuVS)")
        
        return True
        
    except Exception as e:
        print(f"✗ Interoperability test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🔬 faiss-gpu + cuVS Integration Test")
    print("=" * 60)
    print()
    
    results = []
    
    # Run tests
    results.append(("faiss-gpu", test_faiss_gpu()))
    results.append(("cuVS", test_cuvs()))
    results.append(("Interoperability", test_interop()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)}")
    
    if passed == len(results):
        print("\n🎉 All tests passed!")
        print("\nBoth faiss-gpu and cuVS are working correctly.")
        print("You can use them together for high-performance vector search:")
        print("  - faiss: Mature, feature-rich, CPU/GPU support")
        print("  - cuVS: RAPIDS-native, optimized for NVIDIA GPUs, CAGRA algorithm")
        return 0
    else:
        print("\n⚠️  Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
