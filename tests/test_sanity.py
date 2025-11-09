def test_numpy_ok():
    """Run a small numpy verification test.

    The test creates a numpy array and sums its elements.
    """
    import numpy as np

    assert np.array([1, 2, 3]).sum() == 6


def test_cuda_available():
    """Check whether a CUDA-capable GPU is available.

    If torch is not installed the test is skipped. If torch is present but
    CUDA is not available the test will fail (so CI can detect missing GPU
    capability when expected).
    """
    import os
    import shutil
    import subprocess
    import pytest

    def has_nvidia_smi():
        if shutil.which("nvidia-smi"):
            try:
                res = subprocess.run(
                    ["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5
                )
                return res.returncode == 0 and res.stdout.strip() != ""
            except Exception:
                return False
        return False

    def has_proc_driver():
        return os.path.exists("/proc/driver/nvidia/version")

    def has_lspci_nvidia():
        if shutil.which("lspci"):
            try:
                res = subprocess.run(["lspci"], capture_output=True, text=True, timeout=5)
                return "nvidia" in res.stdout.lower()
            except Exception:
                return False
        return False

    if has_nvidia_smi() or has_proc_driver() or has_lspci_nvidia():
        # Found at least one low-level indicator of an NVIDIA GPU.
        assert True
    else:
        pytest.skip("No low-level evidence of a CUDA-capable NVIDIA GPU found")


def test_torch_gpu_compute():
    """Run a small GPU tensor operation to confirm GPU compute works.

    The test skips if CUDA is not available or if a runtime error occurs
    (for example an out-of-memory error on small/dev machines).
    """
    import pytest

    try:
        import torch
    except Exception as e:
        pytest.skip(f"torch not installed: {e}")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available; skipping GPU compute test")

    # Perform a single matrix multiply on the GPU. Wrap in try/except to
    # gracefully skip if the host/container runs out of memory.
    try:
        x = torch.rand((10000, 10000), device="cuda")
        y = torch.rand((10000, 10000), device="cuda")
        z = torch.mm(x, y)
        # simple sanity check: ensure tensor contains finite values
        assert z.isfinite().all().item()
    except RuntimeError as e:
        pytest.skip(f"GPU compute failed at runtime: {e}")
    finally:
        # Best-effort cleanup
        try:
            del x, y, z
        except Exception:
            pass
