from .base_watermarker import BaseWatermarker
import numpy as np
from typing import Union, Tuple


class EBlindDLC(BaseWatermarker):
    @staticmethod
    def embed(work: np.ndarray, secret: bool, secret_key: int, embedding_strength: float = 1) -> np.ndarray:
        np.random.seed(secret_key)

        # Generate Reference Pattern
        wr = np.random.randint(100, size=work.shape, dtype=np.int8)

        # Embedding Scheme
        wm = (-1) ** (int(secret) + 1) * wr
        wa = embedding_strength * wm
        cw = work + wa

        # Clip to correct pixel values
        cw = np.clip(cw, 0, 255)
        cw = cw.astype(np.uint8)

        return cw

    @staticmethod
    def extract(work: np.ndarray, secret_key: int, threshold: float = 0.1) -> Union[Tuple[int, float], Tuple[None, float]]:
        np.random.seed(secret_key)

        # Generate Reference Pattern
        wr = np.random.randint(100, size=work.shape, dtype=np.uint8)

        # Calculate the correlation coefficient of reference pattern and work
        lc = BaseWatermarker._correlation_coefficient(work, wr)
        if lc > threshold:
            return 1, lc

        if lc < -threshold:
            return 0, lc

        return None, lc
