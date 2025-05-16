import matplotlib.pyplot as plt

from watermarkers import e_wt_svd_blind_d_q as e5
from watermarkers import e_wt_blind_d_cc as e3
from watermarkers import e_fft_ss_blind_d_q as e2
from watermarkers import e_fft_blind_d_cc as e1


from attacks import attacker as at

import cv2


if __name__ == "__main__":
    host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
    wm = cv2.imread("watermarks/flower.png", cv2.IMREAD_GRAYSCALE)
    _, wm = cv2.threshold(wm, 127, 255, cv2.THRESH_BINARY)

    mask = cv2.imread("mask.png", cv2.IMREAD_GRAYSCALE)
    mask = cv2.resize(mask, (host.shape[1], host.shape[0]))
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    mask = ~mask

    attack = at.Attacker()
    secret_key = 27
    embedding_strength = 8
    # plot_name = 'Frequency_System2_attacked_extractions_key0'
    plot_name = None

    # emb5 = e5.EWTSVDBlindDQ.embed(host, wm, secret_key, embedding_strength)
    # emb3 = e3.EWTBlindDCC.embed(host, wm, secret_key, embedding_strength)
    # emb2 = e2.EFFTSSBlindDQ.embed(host, wm, secret_key, embedding_strength)
    # emb1 = e1.EFFTBlindDCC.embed(host, wm, secret_key, embedding_strength)


    attack.plot_all_extractions(e2.EFFTSSBlindDQ,
                                watermarked=emb2,
                                target_shape=wm.shape,
                                original_wm=wm,
                                embedding_strength=embedding_strength,
                                secret_key=secret_key,
                                mask=mask,
                                plot_name=plot_name)