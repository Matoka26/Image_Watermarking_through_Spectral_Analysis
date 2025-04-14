from .base_watermarker import BaseWatermarker
import numpy as np
from typing import Union, Tuple
import matplotlib.pyplot as plt


class EFFTBlindDCC(BaseWatermarker):
    @staticmethod
    def embed(host: np.ndarray,
              wm: np.ndarray,
              secret_key: int,
              embedding_strength: np.float64 = 1) -> np.ndarray:

        # Encrypt watermark
        wm = BaseWatermarker._arnolds_cat_map_scramble(wm, secret_key=secret_key)

        host_fft = np.fft.fftshift(np.fft.fft2(host))

        # Map values {0, 255} -> {0, 1}
        wm_norm = wm / np.max(wm)

        # Map values {0, 1} -> {-1, 1}
        wm_norm = wm_norm * 2 - 1

        # Pad the watermark with 0's to the center of the host's spectrum
        wm_norm = BaseWatermarker._pad_to_center(wm_norm, host_fft.shape)

        # F' = F + e^alpha * W
        emb_host_fft = host_fft + np.exp(embedding_strength) * wm_norm

        emb_host = np.fft.ifft2(np.fft.ifftshift(emb_host_fft))
        emb_host = np.real(emb_host)

        return emb_host

    @staticmethod
    def plot_strength_correlation(host: np.ndarray,
                              wm: np.ndarray,
                              secret_key: int) -> None:

        wm = BaseWatermarker._arnolds_cat_map_scramble(wm, secret_key=secret_key)
        wm_norm = wm / np.max(wm)
        wm_norm = wm_norm * 2 - 1
        wm_norm = BaseWatermarker._pad_to_center(wm_norm, host.shape)

        host_fft = np.fft.fftshift(np.fft.fft2(host))

        N = 100
        samples = np.arange(0, N)
        emb_strengths = np.linspace(0, 20, N)
        strengths = []
        for e_s in emb_strengths:

            emb_host_fft = host_fft + np.exp(e_s) * wm_norm
            emb_host = np.fft.ifft2(np.fft.ifftshift(emb_host_fft))
            emb_host = np.real(emb_host)
            emb_host_fft = np.fft.fftshift(np.fft.fft2(emb_host))

            strengths.append(BaseWatermarker._correlation_coefficient(np.abs(emb_host_fft), wm_norm))

        plt.plot(samples, strengths)
        plt.grid(True)
        plt.show()