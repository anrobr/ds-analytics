def test_numpy_ok():
    import numpy as np

    assert np.array([1, 2, 3]).sum() == 6
