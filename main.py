import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap
from watermarkers import e_wt_svd_blind_d_q as e5

# ==== STEP 1: Create wrapped text image using Pillow ====

# Create a 64x64 white image
img_pil = Image.new('L', (80, 80), color=255)
draw = ImageDraw.Draw(img_pil)

# Font & wrapped text
font = ImageFont.load_default()
sentence = input("Enter a sentence to watermark: ")

# Wrap to ~10 characters per line (tweak if needed)
wrapped_text = textwrap.fill(sentence, width=10)

# Draw wrapped text
draw.text((2, 2), wrapped_text, fill=0, font=font)

# Convert Pillow image to OpenCV format (NumPy array)
image = np.array(img_pil)

# Threshold to binary watermark
_, wm = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# ==== STEP 2: Watermark embedding ====
host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
secret_key = 27
emb_str = 40

emb = e5.EWTSVDBlindDQ.embed(host, wm, secret_key, embedding_strength=emb_str)
ext = e5.EWTSVDBlindDQ.extract(emb, wm.shape, secret_key=secret_key, embedding_strength=emb_str)

# ==== STEP 3: Plot all in one row ====
fig, axs = plt.subplots(1, 4, figsize=(12, 4))

axs[0].imshow(wm, cmap='gray')
axs[0].set_title("Original Wrapped Text")
axs[0].axis('off')

axs[1].imshow(emb, cmap='gray')
axs[1].set_title("Embedded Image")
axs[1].axis('off')

axs[2].imshow(ext, cmap='gray')
axs[2].set_title("Extracted Watermark")
axs[2].axis('off')

axs[3].imshow(np.abs(host - emb), cmap='inferno')
axs[3].set_title("Difference Image")
axs[3].axis('off')

plt.tight_layout()
plt.show()
