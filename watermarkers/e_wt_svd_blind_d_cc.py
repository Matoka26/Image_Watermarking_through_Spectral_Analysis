from .base_watermarker import BaseWatermarker
import numpy as np
import matplotlib.pyplot as plt
import pywt
# from skimage.transform import resize
from typing import Tuple


class EWTSVDBlindDCC(BaseWatermarker):
    LH_BAND = 0
    HL_BAND = 1
    HH_BAND = 2

    @staticmethod
    def embed(host: np.ndarray,
              wm: np.ndarray,
              secret_key: int,
              embedding_strength: np.float64 = 1,
              wt_h_level: int = 2) -> np.ndarray:
        wt_w_level = wt_h_level - 1

        # Encrypt watermark
        wm = BaseWatermarker._arnolds_cat_map_scramble(wm, secret_key=secret_key)

        # Change values from {0, 255} -> {0, 1}
        wm = wm / 255

        # Step 1
        coeffs_host = pywt.wavedec2(host, 'db1', level=wt_h_level)
        coeffs_wm = pywt.wavedec2(wm, 'db1', level=wt_w_level)

        # Step 2
        # for host
        LL3 = coeffs_host[0]
        uh1, sh1, vh1 = np.linalg.svd(LL3)
        uh2, sh2, vh2 = np.linalg.svd(np.diag(sh1))

        HH4 = coeffs_host[wt_w_level][EWTSVDBlindDCC.HH_BAND]
        uh3, sh3, vh3 = np.linalg.svd(HH4)
        uh4, sh4, vh4 = np.linalg.svd(np.diag(sh3))

        # for watermark
        LL2 = coeffs_wm[0]
        uw1, sw1, vw1 = np.linalg.svd(LL2)
        uw2, sw2, vw2 = np.linalg.svd(np.diag(sw1))

        HH3 = coeffs_wm[wt_w_level][EWTSVDBlindDCC.HH_BAND]
        uw3, sw3, vw3 = np.linalg.svd(HH3)
        uw4, sw4, vw4 = np.linalg.svd(np.diag(sw3))

        # Step 3
        new_s_h_ll4 = sh1 + sh2
        new_s_h_hh4 = sh3 + sh4

        new_s_w_ll3 = sw1 + sw2
        new_s_w_hh3 = sw3 + sw4

        # Step 4
        w_i_s_ll4 = new_s_h_ll4 + embedding_strength * np.pad(new_s_w_ll3,
                                                              (0, len(new_s_h_ll4) - len(new_s_w_ll3)),
                                                              mode="constant")
        w_i_s_hh4 = new_s_h_hh4 + embedding_strength * np.pad(new_s_w_hh3,
                                                              (0, len(new_s_h_hh4) - len(new_s_w_hh3)),
                                                              mode="constant")

        # Step 5
        w_i_s_ll4_mat = np.diag(w_i_s_ll4)
        w_i_s_hh4_mat = np.diag(w_i_s_hh4)

        w_i_ll4 = uh1 @ w_i_s_ll4_mat @ vh1
        w_i_hh4 = uh3 @ w_i_s_hh4_mat @ vh3

        # Step 6
        # Pad HH3 subband
        target_shape = coeffs_host[wt_h_level][EWTSVDBlindDCC.HH_BAND].shape
        h, w = target_shape
        resized = np.zeros((h, w))
        resized[:min(h, w_i_hh4.shape[0]), :min(w, w_i_hh4.shape[1])] = w_i_hh4[:h, :w]

        coeffs_host[0] = w_i_ll4
        coeffs_host[wt_h_level] = (coeffs_host[wt_h_level][0], coeffs_host[wt_h_level][1], resized)

        rec = pywt.waverec2(coeffs_host, 'db1')
        rec = (rec - np.min(rec)) * (255.0 / (np.max(rec) - np.min(rec)))
        rec = np.clip(rec, 0, 255).astype(np.uint8)
        return rec

    @staticmethod
    def extract(watermarked: np.ndarray,
                host: np.ndarray,
                target_shape: Tuple[int, ...],
                secret_key: int,
                embedding_strength: float = 1,
                wt_h_level: int = 1) -> np.ndarray:

        # Step 1: Decompose watermarked and host images
        coeffs_host = pywt.wavedec2(host, 'db1', level=wt_h_level)
        coeffs_wm = pywt.wavedec2(watermarked, 'db1', level=wt_h_level)

        # Step 2: Compute SVD for watermarked and host coefficients
        # for watermarked
        uw1, sw1, vw1 = np.linalg.svd(coeffs_wm[0])
        uw2, sw2, vw2 = np.linalg.svd(np.diag(sw1))

        uw3, sw3, vw3 = np.linalg.svd(coeffs_wm[wt_h_level][EWTSVDBlindDCC.HH_BAND])
        uw4, sw4, vw4 = np.linalg.svd(np.diag(sw3))

        # for host
        uh1, sh1, vh1 = np.linalg.svd(coeffs_host[0])
        uh2, sh2, vh2 = np.linalg.svd(np.diag(sh1))

        uh3, sh3, vh3 = np.linalg.svd(coeffs_host[wt_h_level][EWTSVDBlindDCC.HH_BAND])
        uh4, sh4, vh4 = np.linalg.svd(np.diag(sh3))

        # Step 3/4: Subtract host SVD from watermarked SVD
        new_sw_ll4 = sw1 + sw2
        new_sw_hh4 = sw3 + sw4

        new_sh_ll4 = sh1 + sh2
        new_sh_hh4 = sh3 + sh4

        # Handle dimension mismatches
        padded_sh_ll4 = np.pad(new_sh_ll4, (0, len(new_sw_ll4) - len(new_sh_ll4)), mode="constant")
        new_sw_ll3 = (new_sw_ll4 - padded_sh_ll4) / embedding_strength
        new_sw_hh3 = (new_sw_hh4 - new_sh_hh4) / embedding_strength

        # Step 5: Reconstruct watermark SVD
        sw1 = np.diag(new_sw_ll3 - sw2)
        sw3 = np.diag(new_sw_hh3 - sw4)

        # Step 6: Rebuild LL3 and HH3 of watermark
        new_ll3 = uw1 @ sw1 @ vw1
        new_hh3 = uw3 @ sw3 @ vw3

        # Step 7: Reconstruct using watermarked coefficients
        coeffs_wm[0] = new_ll3
        coeffs_wm[wt_h_level] = (coeffs_wm[wt_h_level][0], coeffs_wm[wt_h_level][1], new_hh3)
        rec = pywt.waverec2(coeffs_wm, 'db1')

        # Post-processing
        rec = (rec - np.min(rec)) * (255.0 / (np.max(rec) - np.min(rec)))
        rec = np.clip(rec, 0, 255).astype(np.uint8)

        rec = rec[:target_shape[0], :target_shape[1]]
        rec = BaseWatermarker._arnolds_cat_map_inverse(rec, secret_key=secret_key)

        return rec