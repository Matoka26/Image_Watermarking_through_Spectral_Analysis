import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pywt
import matplotlib.gridspec as gridspec
from watermarkers import base_watermarker as be
from watermarkers import e_wt_svd_blind_d_q as e5
from watermarkers import e_wt_blind_d_cc as e3
from watermarkers import e_fft_ss_blind_d_q as e2
from watermarkers import e_fft_blind_d_cc as e1

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


from attacks import attacker as at

import cv2


def rotate_matrix_90(matrix):
    """
    Rotates a given 2D matrix 90 degrees clockwise.

    Args:
        matrix (list of list of any): The matrix to rotate.

    Returns:
        list of list of any: The rotated matrix.
    """
    return [list(row)[::-1] for row in zip(*matrix)]

if __name__ == "__main__":

    # Prepare different watermark sizes
    host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("flower.png", cv2.IMREAD_GRAYSCALE)
    _, wm = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)


    spec = np.fft.fft2(host)
    # mag = np.fft.fftshift(np.log(np.abs(spec) + 1e-16))

    spec = rotate_matrix_90(spec)

    im = np.fft.ifft2(spec).real
    plt.imshow(im, cmap="gray")
    plt.colorbar()
    plt.show()
