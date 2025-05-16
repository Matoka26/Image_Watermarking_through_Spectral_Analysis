from .base_watermarker import BaseWatermarker
import numpy as np
from typing import Union, Tuple

'''
NOTE: in the embedding, clip-ing the output image would ruin the quantization
     and the method will be uneffective
'''


class EFFTSSBlindDQ(BaseWatermarker):
    @staticmethod
    def embed(host: np.ndarray,
              wm: np.ndarray,
              secret_key: int,
              embedding_strength: np.float64 = 1) -> np.ndarray:

        # Scramble watermark
        wm = BaseWatermarker._arnolds_cat_map_scramble(wm, secret_key=secret_key)
        wm_bits = wm.flatten().astype(np.uint8)

        # FFT
        host_fft = np.fft.fft2(host)
        magnitude = np.abs(host_fft).flatten()
        phase = np.angle(host_fft).flatten()

        # Sort by magnitude
        sorted_indices = np.argsort(-magnitude)
        selected_indices = sorted_indices[:len(wm_bits)]

        # QIM embedding
        for i, idx in enumerate(selected_indices):
            bit = wm_bits[i]
            original_mag = magnitude[idx]
            q = np.floor(original_mag / embedding_strength)
            if bit == 0:
                magnitude[idx] = embedding_strength * (q + 0.25)
            else:
                magnitude[idx] = embedding_strength * (q + 0.75)

        # Reconstruct and return watermarked host
        modified_fft = magnitude * np.exp(1j * phase)
        modified_fft = modified_fft.reshape(host_fft.shape)
        emb_host = np.fft.ifft2(modified_fft).real

        return np.clip(emb_host, 0, 255)

    @staticmethod
    def extract(host: np.ndarray,
                  target_shape: Tuple[int, ...],
                  secret_key: int,
                  embedding_strength: np.float64 = 1) -> np.ndarray:
        host_fft = np.fft.fft2(host)
        magnitude = np.abs(host_fft).flatten()

        sorted_indices = np.argsort(-magnitude)
        selected_indices = sorted_indices[:target_shape[0] * target_shape[1]]

        extracted_bits = []

        for idx in selected_indices:
            fraction = (magnitude[idx] / embedding_strength) % 1
            bit = 1 if fraction > 0.5 else 0
            extracted_bits.append(bit)

        extracted_bits = np.array(extracted_bits, dtype=np.uint8)
        extracted_bits = extracted_bits.reshape(target_shape)

        # Unscramble
        extracted_bits = BaseWatermarker._arnolds_cat_map_inverse(extracted_bits, secret_key=secret_key)

        return extracted_bits
