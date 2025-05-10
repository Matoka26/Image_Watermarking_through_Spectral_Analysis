import matplotlib.pyplot as plt
import numpy as np
import pywt
from watermarkers import e_wt_blind_d_cc as e
from watermarkers import base_watermarker as ee
import cv2

import numpy as np


if __name__ == "__main__":
    host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("watermarks/flower.png", cv2.IMREAD_GRAYSCALE)
    _, wm = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)

    emb = e.EWTBlindDCC.embed(host, wm, secret_key=1, embedding_strength=40)
    ext = e.EWTBlindDCC.extract(emb, wm.shape, secret_key=1)

    plt.imshow(ext)
    plt.show()
