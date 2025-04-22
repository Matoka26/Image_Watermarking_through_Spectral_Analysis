import matplotlib.pyplot as plt
import numpy as np
from watermarkers import e_fft_blind_d_cc as e
import cv2

if __name__ == "__main__":
    host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("watermarks/flower.png", cv2.IMREAD_GRAYSCALE)
    _, wm = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)

    # c = e.EFFTBlindDCC.embed(host=host, wm=wm, embedding_strength=10, secret_key=1)

    alpha = e.EFFTBlindDCC.get_best_strengths_for_keys(host, wm, secret_keys=[0, 7, 51, 97], plot=True)

    print(alpha)
    # plt.imshow(cropped_host_fft)
    # plt.colorbar()
    #
    # plt.show()
    # cv2.imshow("lena", cropped_host)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
