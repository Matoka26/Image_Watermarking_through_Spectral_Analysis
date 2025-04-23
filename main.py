import matplotlib.pyplot as plt
import numpy as np
from watermarkers import e_fft_ss_blind_d_cc as e
import cv2


if __name__ == "__main__":
    host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("watermarks/flower.png", cv2.IMREAD_GRAYSCALE)
    _, wm = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)

    alpha = 7
    c = e.EFFTSSBlindDCC.embed(host=host, wm=wm, embedding_strength=alpha, secret_key=0)

    extracted_watermark = e.EFFTSSBlindDCC.extract(c, target_shape=wm.shape, secret_key=0, embedding_strength=alpha)