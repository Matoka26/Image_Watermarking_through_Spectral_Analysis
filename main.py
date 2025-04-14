from watermarkers import e_fixed_d_lc as e
import cv2

if __name__ == "__main__":
    c = cv2.imread("lena.png", cv2.IMREAD_GRAYSCALE)


    c = e.EFixedDLC.embed(work=c, secret=True, secret_key=100)
    ext = e.EFixedDLC.extract(work=c, secret_key=100, threshold=0.1)
    print(ext)

    cv2.imshow("ceva", c)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
