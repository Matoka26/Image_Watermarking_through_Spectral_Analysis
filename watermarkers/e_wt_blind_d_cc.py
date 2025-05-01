from .base_watermarker import BaseWatermarker
import numpy as np
import matplotlib.pyplot as plt
import pywt
from typing import Tuple


class EWTBlindDCC(BaseWatermarker):
    _level = 2
    LH_BAND = 0
    HL_BAND = 1
    HH_BAND = 2

    @staticmethod
    def embed(host: np.ndarray,
              wm: np.ndarray,
              secret_key: int,
              embedding_strength: np.float64 = 1) -> np.ndarray:

        # Encrypt watermark
        wm = BaseWatermarker._arnolds_cat_map_scramble(wm, secret_key=secret_key)

        # Change values from {0, 255} -> {0, 1}
        wm = wm / 255

        # Decompose the image
        dec = pywt.wavedec2(host, 'db1', level=EWTBlindDCC._level)

        # Pad wm to be added to the center of the bands
        wm_pad_HH = BaseWatermarker._pad_to_center(wm, dec[EWTBlindDCC._level][EWTBlindDCC.HH_BAND].shape)
        wm_pad_LL = BaseWatermarker._pad_to_center(wm, dec[0].shape)

        HH = dec[EWTBlindDCC._level][2]
        HH = HH + embedding_strength * wm_pad_HH
        dec[EWTBlindDCC._level] = (dec[EWTBlindDCC._level][0], dec[EWTBlindDCC._level][1], HH)

        LL = dec[0]
        LL = LL + embedding_strength * wm_pad_LL
        dec[0] = LL
        rec = pywt.waverec2(dec, 'db1')

        return rec



    @staticmethod
    def test_watermark(host: np.ndarray,
                       wm: np.ndarray,
                       secret_key: int) -> Tuple[int, int]:

        dec = pywt.wavedec2(host, 'db1', level=EWTBlindDCC._level)

        HH = dec[EWTBlindDCC._level][2]
        LL = dec[0]

        HH_crop = BaseWatermarker._crop_center(HH, wm.shape)
        LL_crop = BaseWatermarker._crop_center(LL, wm.shape)

        wm_scrambled = BaseWatermarker._arnolds_cat_map_scramble(wm, secret_key)

        hh_cc = BaseWatermarker._correlation_coefficient(np.log1p(np.abs(HH_crop)), wm_scrambled)
        ll_cc = BaseWatermarker._correlation_coefficient(np.log(np.abs(LL_crop)), wm_scrambled)

        return ll_cc, hh_cc

    @staticmethod
    def extract_watermark(host: np.ndarray,
                          target_shape: Tuple[int, ...],
                          secret_key: int) -> np.ndarray:

        noise = np.random.random(target_shape)
        reverse_key = BaseWatermarker._get_cat_map_key(noise)

        dec = pywt.wavedec2(host, 'db1', level=EWTBlindDCC._level)
        HH = dec[EWTBlindDCC._level][EWTBlindDCC.HH_BAND]

        HH_crop = BaseWatermarker._crop_center(HH, target_shape)
        HH_crop = BaseWatermarker._arnolds_cat_map_scramble(HH_crop, secret_key=(reverse_key-secret_key))

        HH_crop_th = np.where(HH_crop > np.median(HH_crop)*10, 255, 0)
        return HH_crop_th