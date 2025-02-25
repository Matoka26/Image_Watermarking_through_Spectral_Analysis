from .base_watermarker import BaseWatermarker
import numpy as np


class EBlindDLC(BaseWatermarker):
    @staticmethod
    def embed(work: np.ndarray, secret: bool, secret_key: int, embedding_strength: int = 1) -> np.ndarray:
        np.random.seed(secret_key)

        # Generate Reference Pattern
        wr = np.random.randint(10, size=work.shape, dtype=np.uint8)

        # Embedding Scheme
        wm = (-1) ** (int(secret) + 1) * wr
        wa = embedding_strength * wm
        cw = work + wa

        # Clip to correct pixel values
        cw = np.clip(cw, 0, 255)
        cw = cw.astype(np.uint8)

        return cw

    @staticmethod
    def extract(work: np.ndarray, secret_key: int, threshold: np.float64 = 0.1) -> int | None:
        np.random.seed(secret_key)

        # Generate Reference Pattern
        wr = np.random.randint(10, size=work.shape, dtype=np.uint8)

        # Calculate linear correlation of reference pattern and work
        lc = BaseWatermarker._linear_correlation(work, wr)
        if lc > threshold:
            return 1

        if lc < -threshold:
            return 0

        return None
