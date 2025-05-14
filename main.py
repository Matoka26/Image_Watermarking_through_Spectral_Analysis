import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import pywt
from watermarkers import e_wt_svd_blind_d_q as e
from watermarkers import base_watermarker as ee
from attacks import attacker as at

import cv2

import numpy as np


def compute_spectrum(img:np.ndarray) -> np.ndarray:
    f = np.fft.fftshift(np.fft.fft2(img))

    ret = np.log(np.abs(f) + 1e-9)
    return ret


if __name__ == "__main__":
    host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("watermarks/flower.png", cv2.IMREAD_GRAYSCALE)
    _, wm = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)

    emb = e.EWTSVDBlindDQ.embed(host, wm, secret_key=5, embedding_strength=31)
    ext = e.EWTSVDBlindDQ.extract(watermarked=emb,
                                   target_shape=wm.shape,
                                   secret_key=5,
                                   embedding_strength=31)


    attack = at.Attacker()

    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (emb.shape[1], emb.shape[0]))

    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask = ~mask

    ext = e.EWTSVDBlindDQ.extract(watermarked=emb,
                                   target_shape=wm.shape,
                                   secret_key=5,
                                   embedding_strength=31)

    attacked_images = [
        host,
        attack._white_noise(host, mean=3),
        attack._salt_and_pepper(host),
        attack._uniform_noise(host, high=8),
        attack._gaussian_filter(host),
        attack._median_filter(host, size=(3, 3)),
        attack._average_filter(host, size=(3, 3)),
        attack._sharpening_filter(host, size=(3, 3)),
        attack._jpeg_compression(host),
        attack._brighten(host, value=30),
        attack._darken(host, value=30),
        attack._histogram_equalization(host),
        cv2.resize(attack._scaling(host, percent=30), host.shape),
        attack._crop_corners(host, percent=30),
        attack._inpaint(host, mask)
    ]

    attack_names = [
        "Original",
        "White Noise",
        "Salt & Pepper",
        "Uniform Noise",
        "Gaussian Filter",
        "Median Filter",
        "Average Filter",
        "Sharpening Filter",
        "JPEG Compression",
        "Brighten",
        "Darken",
        "Histogram Equaliaztion",
        "Scaling",
        "Crop Corners",
        "Inpain"
    ]

    fig, axes = plt.subplots(3, 5, figsize=(10, 10))
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        ax.imshow(attacked_images[i])
        ax.set_title(f"{i}. {attack_names[i]}\n PSNR:{peak_signal_noise_ratio(host, attacked_images[i]):.2f}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('figures/Statistics/Attacks.pdf')
    # plt.show()