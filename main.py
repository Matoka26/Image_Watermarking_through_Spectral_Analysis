import matplotlib.pyplot as plt
import numpy as np
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


def compute_spectrum(img):
    sp = np.fft.fft2(img)
    return np.fft.fftshift((np.abs(np.log(sp))))

def binary_error_rate(a:np.ndarray, b:np.ndarray) -> np.float64:
    assert a.shape == b.shape, "Shape mismatch between watermark arrays"
    total_bits = a.size
    num_errors = np.sum(a != b)
    return num_errors / total_bits


if __name__ == "__main__":
    host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("watermarks/flower.png", cv2.IMREAD_GRAYSCALE)
    _, wm = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)

    # mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    # mask = cv2.resize(mask, (host.shape[1], host.shape[0]))
    # _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    # mask = ~mask
    #
    attack = at.Attacker()
    secret_key = 27
    embedding_strength = 8
    # plot_name = 'Frequency_System2_attacked_extractions_key0'
    plot_name = None

    # # emb5 = e5.EWTSVDBlindDQ.embed(host, wm, secret_key, embedding_strength)
    # # emb3 = e3.EWTBlindDCC.embed(host, wm, secret_key, embedding_strength)
    # # emb2 = e2.EFFTSSBlindDQ.embed(host, wm, secret_key, embedding_strength)

    emb1 = e1.EFFTBlindDCC.embed(host, wm, secret_key, embedding_strength=14, fftshift=True)
    emb2 = e1.EFFTBlindDCC.embed(host, wm, secret_key, embedding_strength=11, fftshift=False)

    ext1 = e1.EFFTBlindDCC.extract(emb1, wm.shape, secret_key, fftshift=True)
    ext2 = e1.EFFTBlindDCC.extract(emb2, wm.shape, secret_key, fftshift=False)

    ncols = 4

    # Set up GridSpec with a small third row for the colorbar
    fig = plt.figure(figsize=(10, 7))
    gs = gridspec.GridSpec(3, ncols, height_ratios=[1, 1, 0.05])

    # Create 8 axes for images
    ax = [fig.add_subplot(gs[i // ncols, i % ncols]) for i in range(ncols * 2)]

    # Disable ticks and borders
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
        for spine in a.spines.values():
            spine.set_visible(False)

    # First row
    ax[0].imshow(emb1, cmap="gray")
    ax[0].set_xlabel(
        f"MSE: {mean_squared_error(emb1, host):.2f}, SSIM: {ssim(host, emb1, data_range=emb1.max() - emb1.min()):.2f}")

    ax[1].imshow(ext1, cmap="gray")
    ax[1].set_xlabel(f'{binary_error_rate(ext1, wm):.2f}')

    spec1 = compute_spectrum(emb1)
    im1 = ax[2].imshow(spec1, cmap="inferno")

    ax[3].imshow(np.abs(host - emb1), cmap="inferno")

    # Second row
    ax[4].imshow(emb2, cmap="gray")
    ax[4].set_xlabel(
        f"MSE: {mean_squared_error(emb2, host):.2f}, SSIM: {ssim(host, emb2, data_range=emb2.max() - emb2.min()):.2f}")

    ax[5].imshow(ext2, cmap="gray")
    ax[5].set_xlabel(f'{binary_error_rate(ext2, wm):.2f}')

    spec2 = compute_spectrum(emb2)
    ax[6].imshow(spec2, cmap="inferno")

    ax[7].imshow(np.abs(host - emb2), cmap="inferno")

    # Column titles
    titles = ["Embedded", "Extracted", "Spectrum", "Difference"]
    for i in range(ncols):
        ax[i].set_title(titles[i])


    # Colorbar underneath the Spectrum column
    cbar_ax = fig.add_subplot(gs[2, 2])  # third row, third column
    cbar = fig.colorbar(im1, cax=cbar_ax, orientation='horizontal')

    # Colorbar underneath the Difference column (last column)
    diff_im = ax[3].images[0]  # use the image from the first row difference
    cbar_ax_diff = fig.add_subplot(gs[2, 3])  # third row, last column
    cbar_diff = fig.colorbar(diff_im, cax=cbar_ax_diff, orientation='horizontal')

    # Add side titles for each row
    fig.text(0, 0.75, 'Low Frequency', va='center', rotation='vertical', fontsize=12)
    fig.text(0, 0.35, 'High Frequency', va='center', rotation='vertical', fontsize=12)

    plt.tight_layout()
    plt.savefig('figures/Frequency_System1/Frequency_System1_Overall.pdf')