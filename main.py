from watermarkers import e_blind_d_lc as e
import cv2


if __name__ == "__main__":
    c = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)

    c = e.EBlindDLC.embed(work=c, secret=False, secret_key=100, embedding_strength=1)
    ext = e.EBlindDLC.extract(work=c, secret_key=100, threshold=0.05)

    print(ext)