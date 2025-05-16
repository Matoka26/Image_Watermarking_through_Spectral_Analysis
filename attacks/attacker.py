from scipy.ndimage import gaussian_filter, median_filter
from skimage.metrics import peak_signal_noise_ratio
from skimage import exposure
from metrics.metrics import binary_error_rate
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import io
import cv2


class Attacker:
    def __init__(self):
        pass

    def _white_noise(self, image, mean=0, std=1):
        return image.copy() + np.random.normal(size=image.shape, loc=mean, scale=std)

    def _salt_and_pepper(self, image, prob=0.005):
        image = image.copy()
        output = np.zeros(image.shape, np.uint8)
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = np.random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]

        return output

    def _uniform_noise(self, image, low=0, high=8):
        return image.copy() + np.random.uniform(low, high, image.shape)

    def _gaussian_filter(self, image, sigma=3):
        return gaussian_filter(image.copy(), sigma=sigma)

    def _median_filter(self, image, size=(3,3)):
        return median_filter(image.copy(), size=size)

    def _average_filter(self, img, size=(3,3)):
        img = img.copy()
        m, n = img.shape
        k_h, k_w = size
        pad_h = k_h // 2
        pad_w = k_w // 2

        mask = np.ones((k_h, k_w), dtype=float) / (k_h * k_w)

        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        img_new = np.zeros_like(img, dtype=float)

        for i in range(m):
            for j in range(n):
                region = padded_img[i:i + k_h, j:j + k_w]
                img_new[i, j] = np.sum(region * mask)

        return img_new

    def _sharpening_filter(self, img, size=(3, 3)):
        img = img.copy()
        assert size[0] % 2 == 1 and size[1] % 2 == 1, "Sharpen filter size must be odd in both dimensions."

        m, n = img.shape
        k_h, k_w = size
        pad_h = k_h // 2
        pad_w = k_w // 2

        kernel = -1 * np.ones((k_h, k_w), dtype=float)

        center_y, center_x = pad_h, pad_w
        kernel_sum = np.sum(kernel)
        kernel[center_y, center_x] = -kernel_sum + 1  # So total kernel sum becomes 1

        padded_img = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
        img_new = np.zeros_like(img, dtype=float)

        for i in range(m):
            for j in range(n):
                region = padded_img[i:i + k_h, j:j + k_w]
                img_new[i, j] = np.sum(region * kernel) / kernel.size

        return img_new

    def _jpeg_compression(self, img, quality=50):
        img = img.copy()
        img_uint8 = np.clip(img, 0, 255).astype(np.uint8)
        img_pil = Image.fromarray(img_uint8, mode='L')

        # Save to in-memory buffer as JPEG
        buffer = io.BytesIO()
        img_pil.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)

        # Load compressed image back into NumPy array
        img_compressed = Image.open(buffer)
        return np.array(img_compressed)

    def _brighten(self, img, value=20):
        return np.clip(img.copy().astype(np.int16) + value, 0, 255).astype(np.uint8)

    def _darken(self, img, value=20):
        return np.clip(img.copy().astype(np.int16) - value, 0, 255).astype(np.uint8)

    def _histogram_equalization(self, img):
        return exposure.equalize_hist(img.copy())

    def _scaling(self, image, percent=25, down:bool=False):
        image = image.copy()
        width, height = image.shape

        scale = percent / 100 * ((-1) ** int(down))

        new_size = (int(width + width * scale), int(width + height * scale))
        resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)

        return resized_image

    def _crop_corners(self, image, percent=20):
        image = image.copy()
        width, height = image.shape

        corner_size = int(min(width, height) * (percent/100))

        image[:corner_size, :corner_size] = 0  # top-left corner
        image[:corner_size, -corner_size:] = 0  # top-right corner
        image[-corner_size:, :corner_size] = 0  # bottom-left corner
        image[-corner_size:, -corner_size:] = 0  # bottom-right corner

        return image

    def _inpaint(self, image, mask):
        image = image.copy()
        image[mask == 255] = 255
        return image

    def _rotate(self, image, n_rotations=1):
        return np.rot90(image.copy(), k=n_rotations)
    
    def plot_all_attacks(self, host, mask):
        attacked_images = [
            host,
            self._white_noise(host, mean=3),
            self._salt_and_pepper(host),
            self._uniform_noise(host, high=8),
            self._gaussian_filter(host),
            self._median_filter(host, size=(3, 3)),
            self._sharpening_filter(host, size=(3, 3)),
            self._jpeg_compression(host),
            self._brighten(host, value=30),
            self._darken(host, value=30),
            self._histogram_equalization(host),
            cv2.resize(self._scaling(host, percent=30), host.shape),
            self._crop_corners(host, percent=30),
            self._rotate(host, n_rotations=3),
            self._inpaint(host, mask)
        ]
        attack_names = [
            "Original",
            "White Noise",
            "Salt & Pepper",
            "Uniform Noise",
            "Gaussian Filter",
            "Median Filter",
            "Sharpening Filter",
            "JPEG Compression",
            "Brighten",
            "Darken",
            "Histogram Equaliaztion",
            "Scaling",
            "Crop Corners",
            "Rotate",
            "Inpain"
        ]

        fig, axes = plt.subplots(3, 5, figsize=(10, 10))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.imshow(attacked_images[i])
            ax.set_title(f"{i}. {attack_names[i]}\n PSNR:{peak_signal_noise_ratio(host, attacked_images[i]):.2f}")
            ax.axis('off')

        plt.tight_layout()
        # plt.savefig('figures/Statistics/Attacks.pdf')
        plt.show()

    def plot_all_extractions(self, embedder, watermarked, target_shape, embedding_strength, mask, original_wm, secret_key=1, plot_name=None):

        attacked_images = [
            watermarked,
            self._white_noise(watermarked, mean=3),
            self._salt_and_pepper(watermarked),
            self._uniform_noise(watermarked, high=8),
            self._gaussian_filter(watermarked),
            self._median_filter(watermarked, size=(3, 3)),
            self._sharpening_filter(watermarked, size=(3, 3)),
            self._jpeg_compression(watermarked),
            self._brighten(watermarked, value=30),
            self._darken(watermarked, value=30),
            self._histogram_equalization(watermarked),
            cv2.resize(self._scaling(watermarked, percent=30), watermarked.shape),
            self._crop_corners(watermarked, percent=30),
            self._rotate(watermarked, n_rotations=3),
            self._inpaint(watermarked, mask)
        ]
        attack_names = [
            "No Attack",
            "White Noise",
            "Salt & Pepper",
            "Uniform Noise",
            "Gaussian Filter",
            "Median Filter",
            "Sharpening Filter",
            "JPEG Compression",
            "Brighten",
            "Darken",
            "Histogram Equaliaztion",
            "Scaling",
            "Crop Corners",
            "Rotate",
            "Inpain"
        ]

        extracted_wms = [embedder.extract(img, target_shape, secret_key, embedding_strength) for img in attacked_images]

        fig, axes = plt.subplots(3, 5, figsize=(10, 10))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            ax.imshow(extracted_wms[i])
            if attack_names[i] != "Rotate":
                ax.set_title(f"{i}. {attack_names[i]}\n BER:{binary_error_rate(extracted_wms[i], original_wm):.2f}")
            else:
                rotated_wm = self._rotate(original_wm, n_rotations=3)
                ax.set_title(f"{i}. {attack_names[i]}\n BER:{binary_error_rate(extracted_wms[i], rotated_wm):.2f}")
            ax.axis('off')

        plt.tight_layout()
        if plot_name is None:
            plt.show()
        else:
            plt.savefig(f'{"figures/Statistics/" + plot_name + ".pdf"}')
