from abc import ABC, abstractmethod
from typing import Tuple
import copy
import numpy as np
import warnings


class BaseWatermarker(ABC):
    @staticmethod
    def _correlation_coefficient(a: np.ndarray, b: np.ndarray) -> np.float64:
        a_norm = a.flatten().astype(np.float64)
        b_norm = b.flatten().astype(np.float64)

        # Normalize data
        a_norm = (a_norm - np.mean(a_norm)) / np.std(a_norm)
        b_norm = (b_norm - np.mean(b_norm)) / np.std(b_norm)

        # Compute correlation
        return np.dot(a_norm, b_norm) / a_norm.size

    @staticmethod
    def _arnolds_cat_map_scramble(img_array: np.ndarray, secret_key: int = 1) -> np.ndarray:
        """
        Scrambles the pixels of an image according to the transformation:
            F(x, y) = [ 1  1 ] * [ x ]  (mod N), where N is the side length of the image
                      [ 1  2 ]   [ y ]
        Parameters:
            img_array (np.ndarray): A NumPy array representing an image
            key(int): The number of scrambles applied to the image
        Returns:
            np.ndarray: The resulting scrambled image
        """
        n = img_array.shape[0]
        if n != img_array.shape[1]:
            warnings.warn('Input image is not square')

        ret = copy.deepcopy(img_array)
        for _ in range(secret_key):
            new_image = np.zeros_like(ret)
            for x in range(n):
                for y in range(n):
                    new_image[(x + y) % n, (x + 2 * y) % n] = ret[x, y]
            ret = new_image

        return ret

    @staticmethod
    def _pad_to_center(small_array: np.ndarray, target_shape: Tuple[int, ...]):
        """
        Pads a 2D array (small_array) with zeros so that it is centered
        within a new array of shape target_shape.

        Args:
            small_array (ndarray): The smaller 2D array to pad.
            target_shape (tuple): The shape (height, width) of the target array.

        Returns:
            ndarray: Padded array with small_array centered.
        """
        h_target, w_target = target_shape
        h_small, w_small = small_array.shape

        if h_small > h_target or w_small > w_target:
            raise ValueError("small_array must be smaller than or equal to target_shape")

        pad_top = (h_target - h_small) // 2
        pad_bottom = h_target - h_small - pad_top
        pad_left = (w_target - w_small) // 2
        pad_right = w_target - w_small - pad_left

        padded = np.pad(small_array, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
        return padded