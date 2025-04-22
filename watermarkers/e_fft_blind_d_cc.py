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
        # wm = BaseWatermarker._arnolds_cat_map_scramble(wm, secret_key=secret_key)

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

        return np.clip(emb_host, 0, 255).astype(np.uint8)

    @staticmethod
    def get_best_strengths_for_keys(host: np.ndarray,
                          wm: np.ndarray,
                          secret_keys: list[int],
                          plot: bool = False) -> dict[int, float]:

        N = 1000
        samples = np.linspace(0, 50, N)
        best_alphas = {}

        if plot:
            plt.figure(figsize=(10, 6))

        for secret_key in secret_keys:
            wm_scrambled = BaseWatermarker._arnolds_cat_map_scramble(wm, secret_key=secret_key)
            wm_norm = wm_scrambled / np.max(wm_scrambled)
            wm_norm = wm_norm * 2 - 1
            wm_norm_pad = BaseWatermarker._pad_to_center(wm_norm, host.shape)

            host_fft = np.fft.fftshift(np.fft.fft2(host))
            strengths = []

            for e_s in samples:
                # embed
                emb_host_fft = host_fft + np.exp(e_s) * wm_norm_pad
                emb_host = np.fft.ifft2(np.fft.ifftshift(emb_host_fft))
                emb_host = np.real(emb_host)
                emb_host = np.clip(emb_host, 0, 255).astype(np.uint8)

                # re-transform
                emb_host_fft = np.fft.fftshift(np.fft.fft2(emb_host))
                emb_host = np.log(np.abs(emb_host_fft) + 1e-9) * 10

                emb_host_center = BaseWatermarker._crop_center(emb_host, wm_norm.shape)
                strengths.append(BaseWatermarker._correlation_coefficient(emb_host_center, wm_norm))

            max_idx = np.argmax(strengths)
            best_alpha = samples[max_idx]
            best_alphas[secret_key] = best_alpha

            if plot:
                plt.plot(samples, strengths, label=f'Key {secret_key} (α={best_alpha:.2f})')
                plt.scatter(best_alpha, strengths[max_idx], c='red')

        if plot:
            plt.grid(True)
            plt.ylabel('Correlation Coefficient')
            plt.xlabel('Exponential Embedding Strength (α)')
            plt.title('Best Correlation Coefficients by Secret Key')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f'./figures/Frequency_System1_alpha_key_comparison.pdf')

        return best_alphas
