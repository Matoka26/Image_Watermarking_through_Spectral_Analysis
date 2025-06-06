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

if __name__ == "__main__":

    # Prepare different watermark sizes
    host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("watermarks/flower.png", cv2.IMREAD_GRAYSCALE)
    _, wm = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)

    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (host.shape[1], host.shape[0]))
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask = ~mask

    attack = at.Attacker()
    secret_key = 27

    # # emb5 =
    # # emb3 =
    # #

    embeded_images = [
        e1.EFFTBlindDCC.embed(host, wm, secret_key, embedding_strength=11, fftshift=False),
        e2.EFFTSSBlindDQ.embed(host, wm, secret_key, embedding_strength=7),
        e3.EWTBlindDCC.embed(host, wm, secret_key, embedding_strength=30),
        e5.EWTSVDBlindDQ.embed(host, wm, secret_key, embedding_strength=40)
    ]

    fig, ax = plt.subplots(nrows=15, ncols=4, figsize=(6, 8))  # bigger figure for clarity

    embedding_titles = [
        "System1",
        "System2",
        "System3",
        "System4"
    ]

    attack_names = [
        "Original",
        "White Noise",
        "Salt & Pepper",
        "Uniform Noise",
        "Gaussian Filter",
        "Median Filter",
        "Sharpen Filter",
        "JPEG Compression",
        "Brighten",
        "Darken",
        "Hist. Equalization",
        "Scaling",
        "Crop Corners",
        "Rotate",
        "Inpaint"
    ]

    for i, emb in enumerate(embeded_images):
        emb = emb.copy()
        attacked_images = [
            emb,
            attack._white_noise(emb, mean=3),
            attack._salt_and_pepper(emb),
            attack._uniform_noise(emb, high=8),
            attack._gaussian_filter(emb),
            attack._median_filter(emb, size=(3, 3)),
            attack._sharpening_filter(emb, size=(3, 3)),
            attack._jpeg_compression(emb),
            attack._brighten(emb, value=30),
            attack._darken(emb, value=30),
            attack._histogram_equalization(emb),
            cv2.resize(attack._scaling(emb, percent=30), emb.shape),
            attack._crop_corners(emb, percent=30),
            attack._rotate(emb, n_rotations=3),
            attack._inpaint(emb, mask)
        ]
        for j, a in enumerate(attacked_images):
            if i == 0:
                # Show attack name on left side for first column
                ax[j, i].set_ylabel(attack_names[j], rotation=0, labelpad=40, fontsize=8, va='center')
            if i == 0:
                ext = e1.EFFTBlindDCC.extract(a, wm.shape, secret_key=secret_key, fftshift=False)
            elif i == 1:
                ext = e2.EFFTSSBlindDQ.extract(a, wm.shape, embedding_strength=7, secret_key=secret_key)
            elif i == 2:
                ext = e3.EWTBlindDCC.extract(a, wm.shape, embedding_strength=30, secret_key=secret_key)
            else:
                ext = e5.EWTSVDBlindDQ.extract(a, wm.shape, embedding_strength=40, secret_key=secret_key)

            ax[j, i].imshow(ext, cmap='gray')
            ax[j, i].axis('off')

        # Set column title once per embedding method
        ax[0, i].set_title(embedding_titles[i], fontsize=12)

    for j, attack_name in enumerate(attack_names):
        pos = ax[j, 0].get_position()
        y = pos.y0 + pos.height / 2
        x = pos.x0 - 0.01
        fig.text(x, y, attack_name, va='center', ha='right', fontsize=8)

    # plt.subplots_adjust(wspace=0.1)  # tighten horizontal space between columns
    # plt.show()
    plt.savefig("figures/Statistics/Overall.pdf")
