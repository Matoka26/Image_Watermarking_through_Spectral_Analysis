import numpy as np


def binary_error_rate(a:np.ndarray, b:np.ndarray) -> np.float64:
    assert a.shape == b.shape, "Shape mismatch between watermark arrays"
    total_bits = a.size
    num_errors = np.sum(a != b)
    return num_errors / total_bits


def correlation_coefficient(a: np.ndarray, b: np.ndarray) -> np.float64:
    a_norm = a.flatten().astype(np.float64)
    b_norm = b.flatten().astype(np.float64)

    # Normalize data
    a_norm = (a_norm - np.mean(a_norm)) / np.std(a_norm)
    b_norm = (b_norm - np.mean(b_norm)) / np.std(b_norm)

    # Compute correlation
    return np.dot(a_norm, b_norm) / a_norm.size