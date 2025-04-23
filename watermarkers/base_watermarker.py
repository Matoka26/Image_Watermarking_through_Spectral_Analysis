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
    def _get_cat_map_key(image: np.ndarray) -> int:
        """
        Scrambles the pixels of an image over and over again according to the transformation:
            F(x, y) = [ 1  1 ] * [ x ]  (mod N), where N is the side length of the image
                      [ 1  2 ]   [ y ]
            until the original image reappears, finding the decription key, represented
            by the period of the imaage
        Parameters:
            image (np.ndarray): A NumPy array representing an image
        Returns:
            int: The decription key
        """
        new_img = copy.deepcopy(image)
        new_img = BaseWatermarker._arnolds_cat_map_scramble(new_img)

        i = 1
        while not (new_img == image).all():
            new_img = BaseWatermarker._arnolds_cat_map_scramble(new_img)
            i += 1

        return i

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

    def _crop_center(image: np.ndarray, crop_size: Tuple[int, ...]):
        """
        Crop the center of an image to the given shape.

        Parameters:
            image (np.ndarray): Input image.
            crop_size (tuple): (crop_height, crop_width)

        Returns:
            np.ndarray: Center-cropped image.
        """
        h, w = image.shape[:2]
        ch, cw = crop_size

        if ch > h or cw > w:
            raise ValueError("Crop size must be smaller than the image size.")

        start_y = (h - ch) // 2
        start_x = (w - cw) // 2

        return image[start_y:start_y + ch, start_x:start_x + cw]