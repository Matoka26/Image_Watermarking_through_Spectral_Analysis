from .e_blind_d_lc import EBlindDLC
from .base_watermarker import BaseWatermarker
import numpy as np

'''
- Impractical
- Used standardization instead of 1/N, but the example was not ment to return a coherent work, but a noisy one
- The alpha found by this method does indeed guarantees 1000% finding rate but completly destroys the work
- I declare that this method is not worth working with, except mathematical purposes
'''


class EFixedDLC(EBlindDLC):
    @staticmethod
    def embed(work: np.ndarray,
              secret: bool,
              secret_key: int,
              threshold: float = 0.7,
              over_threshold: float = 0.3) -> np.ndarray:
        np.random.seed(secret_key)

        # Generate Reference Pattern
        wr = np.random.randint(20, size=work.shape, dtype=np.int8)
        wm = (-1) ** (int(secret) + 1) * wr

        # Calculate optimal embedding strength

        alpha = threshold + over_threshold - BaseWatermarker._linear_correlation(work, wm)
        alpha /= BaseWatermarker._linear_correlation(wm, wm)
        wa = alpha * wm
        cw = work + wa

        # Clip to correct pixel values
        cw = np.clip(cw, 0, 255)
        cw = cw.astype(np.uint8)

        return cw