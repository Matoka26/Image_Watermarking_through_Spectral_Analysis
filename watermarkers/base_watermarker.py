from abc import ABC, abstractmethod
import numpy as np


class BaseWatermarker(ABC):
    @staticmethod
    def _linear_correlation(a: np.ndarray, b: np.ndarray) -> np.float64:
        a_norm = a.flatten().astype(np.float64)
        b_norm = b.flatten().astype(np.float64)

        # Normalize data
        a_norm = (a_norm - np.mean(a_norm)) / np.std(a_norm)
        b_norm = (b_norm - np.mean(b_norm)) / np.std(b_norm)

        # Compute correlation
        return np.dot(a_norm, b_norm) / a_norm.size
