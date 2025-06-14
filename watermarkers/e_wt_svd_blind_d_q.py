from .base_watermarker import BaseWatermarker
import numpy as np
import matplotlib.pyplot as plt
import pywt
from typing import Tuple
import tensorflow as tf


class EWTSVDBlindDQ(BaseWatermarker):

    @staticmethod
    def _break_into_patches(img: np.ndarray, patch_size: int):
        # Extract patches
        work_tensor = tf.convert_to_tensor(img, dtype=tf.float32)
        # Add batch dimension: (1, H, W)
        work_tensor = tf.expand_dims(work_tensor, axis=0)
        # Add channel dimension: (1, H, W, 1)
        work_tensor = tf.expand_dims(work_tensor, axis=-1)

        patches = tf.image.extract_patches(
            images=work_tensor,  # Add batch dim
            sizes=[1, patch_size, patch_size, 1],  # Patch size
            strides=[1, patch_size, patch_size, 1],  # Stride = patch size (no overlap)
            rates=[1, 1, 1, 1],  # Dilation rate (1 = normal)
            padding="VALID"  # No padding, only full patches
        )
        # Remove batch dimension (axis=0)
        patches = tf.squeeze(patches, axis=0)

        # Reshape to (num_patches, patch_size, patch_size)
        num_patches_x, num_patches_y, _ = patches.shape
        patches = tf.reshape(patches, (num_patches_x * num_patches_y, patch_size, patch_size))
        patches = patches.numpy()

        return patches, num_patches_x, num_patches_y

    @staticmethod
    def embed(host: np.ndarray,
              wm: np.ndarray,
              secret_key: int,
              embedding_strength: np.float64 = 1) -> np.ndarray:


        ''' Consdier Embedding Strength as something around 30'''

        assert host.shape[0] == host.shape[1], "Cover Work must be square"
        assert wm.shape[0] == wm.shape[1], "Watermark must be square"

        delta = embedding_strength

        # Get the LL subband
        coeffs_host = pywt.wavedec2(host, 'db1', level=1)
        LL_band = coeffs_host[0]

        # Calculate the Maximum patch size
        patch_size = int(LL_band.shape[0] / wm.shape[0])

        # Scramble the watermark
        wm = BaseWatermarker._arnolds_cat_map_scramble(wm, secret_key)
        wm_flat = wm.flatten() / 255

        patches, num_patches_x, num_patches_y = EWTSVDBlindDQ._break_into_patches(img=LL_band, patch_size=patch_size)
        embedded_patches = []
        for i, bit in enumerate(wm_flat):
            u, s, v = np.linalg.svd(patches[i])
            s[0] = delta * np.round((s[0] - (delta/2)*bit)/delta) + (delta/2)*bit
            blk_wm = u @ np.diag(s) @ v
            embedded_patches.append(blk_wm)

        # Add remaining unmodified patches
        if len(patches) > len(wm_flat):
            embedded_patches.extend(patches[len(wm_flat):].tolist())

        ll_wm = LL_band
        patch_grid = np.reshape(embedded_patches,
                                (num_patches_x, num_patches_y, patch_size, patch_size))

        # Reconstruct the LL subband
        for i in range(num_patches_x):
            for j in range(num_patches_y):
                x_start = i * patch_size
                x_end = (i + 1) * patch_size
                y_start = j * patch_size
                y_end = (j + 1) * patch_size
                ll_wm[x_start:x_end, y_start:y_end] = patch_grid[i, j]

        # Update wavelet coefficients with modified LL band
        coeffs_wm = list(coeffs_host)
        coeffs_wm[0] = ll_wm

        # Perform inverse wavelet transform
        watermarked = pywt.waverec2(coeffs_wm, 'db1')
        # Clip and convert back to original dtype
        watermarked = np.clip(watermarked, 0, 255).astype(np.uint8)
        return watermarked

    @staticmethod
    def extract(watermarked: np.ndarray,
                target_shape: Tuple[int, ...],
                secret_key: int,
                embedding_strength: int=1) -> np.ndarray:

        # Validate inputs
        assert watermarked.shape[0] == watermarked.shape[1], "Image must be square"
        assert len(target_shape) == 2, "Watermark shape must be 2D"

        # Get the LL subband
        coeffs_host = pywt.wavedec2(watermarked, 'db1', level=1)
        LL_band = coeffs_host[0]

        # Calculate the Maximum patch size
        patch_size = int(LL_band.shape[0] / target_shape[0])
        num_wm_bits = target_shape[0] * target_shape[1]

        patches, num_patches_x, num_patches_y = EWTSVDBlindDQ._break_into_patches(img=LL_band, patch_size=patch_size)
        extracted_bits = np.zeros(num_wm_bits, dtype=np.uint8)

        delta = embedding_strength
        for i in range(num_wm_bits):
            if i >= len(patches):
                break  # Handle case with insufficient patches

            u, s, v = np.linalg.svd(patches[i])

            quantized_value = s[0]
            base = delta * np.floor(quantized_value / delta)
            relative_pos = (quantized_value - base) / delta

            # Determine which quantization lattice was used
            if relative_pos < 0.25 or relative_pos > 0.75:
                extracted_bits[i] = 0
            else:
                extracted_bits[i] = 1

        wm_ext = extracted_bits.reshape(target_shape)
        wm_descrambled = BaseWatermarker._arnolds_cat_map_inverse(wm_ext, secret_key)

        wm_descrambled = (wm_descrambled * 255).astype(np.uint8)
        return wm_descrambled