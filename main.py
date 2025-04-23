import matplotlib.pyplot as plt
import numpy as np
from watermarkers import e_fft_blind_d_cc as e
import cv2


if __name__ == "__main__":
    host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("watermarks/flower.png", cv2.IMREAD_GRAYSCALE)
    _, wm = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)

    alpha = 7

    wm_flat = wm.flatten()
    wm_norm = (wm_flat - np.mean(wm_flat)) / np.std(wm_flat)

    # Compute FFT
    host_fft = np.fft.fft2(host)

    # Flatten FFT and separate magnitude/phase
    fft_flat = host_fft.flatten()
    magnitude = np.abs(fft_flat)
    phase = np.angle(fft_flat)

    # Sort magnitudes (descending) and track indices
    sorted_indices = np.argsort(-magnitude)
    selected_indices = sorted_indices[:len(wm_norm)]  # Top n coefficients

    # Embed watermark into magnitudes (preserve phase)
    watermark = np.exp(alpha) * wm_norm
    magnitude[selected_indices] += watermark  # Additive embedding

    # Rebuild FFT spectrum
    modified_fft = magnitude * np.exp(1j * phase)
    modified_fft = modified_fft.reshape(host_fft.shape)

    # Inverse FFT
    watermarked_image = np.fft.ifft2(modified_fft).real

    # plt.imshow(watermarked_image, cmap='gray')
    # plt.show()

    # plt.imshow(np.abs(watermarked_image - host), cmap='inferno')
    # plt.colorbar()
    plt.imshow(watermarked_image, cmap='gray')
    plt.show()
