import matplotlib.pyplot as plt
import numpy as np
from watermarkers import e_fft_blind_d_cc as e
import cv2

if __name__ == "__main__":
    host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("watermarks/flower.png", cv2.IMREAD_GRAYSCALE)
    _, wm = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)

    c = e.EFFTBlindDCC.embed(host=host, wm=wm, embedding_strength=11.01, secret_key=0, fftshit=False)

    cc = e.EFFTBlindDCC.test_watermark(c, wm, secret_key=0, fftshit=False)
    print(cc)
