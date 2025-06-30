import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw, ImageFont
import textwrap
from watermarkers import e_wt_svd_blind_d_q as e5
from skimage.metrics import structural_similarity as ssim


def set_borders_255(matrix, border_thickness=1):
    # Top border
    matrix[:border_thickness, :] = 0
    # Bottom border
    matrix[-border_thickness:, :] = 0
    # Left border
    matrix[:, :border_thickness] = 0
    # Right border
    matrix[:, -border_thickness:] = 0

# === Prepare watermark text ===
img_pil = Image.new('L', (80, 80), color=255)
draw = ImageDraw.Draw(img_pil)
font = ImageFont.load_default()
sentence = input("Enter a sentence to watermark: ")
wrapped_text = textwrap.fill(sentence, width=10)
draw.text((2, 2), wrapped_text, fill=0, font=font)
image = np.array(img_pil)

_, wm = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
set_borders_255(wm)

# === Read secret key from keyboard ===
secret_key = 42

host = cv2.imread("hosts/lena.png", cv2.IMREAD_GRAYSCALE)
emb_str = 30
emb = e5.EWTSVDBlindDQ.embed(host, wm, secret_key, embedding_strength=emb_str)
ext = e5.EWTSVDBlindDQ.extract(emb, wm.shape, secret_key=secret_key, embedding_strength=emb_str)

# Compute difference between extracted and original watermark
diff = np.abs(emb - host)

# --- Compute BER ---
wm_bin = (wm > 127).astype(np.uint8)
ext_bin = (ext > 127).astype(np.uint8)
ber = np.sum(wm_bin != ext_bin) / wm_bin.size

# --- Compute MSE ---
mse = np.mean((host.astype(np.float32) - emb.astype(np.float32)) ** 2)

# --- Compute SSIM ---
ssim_val = ssim(host, emb, data_range=emb.max() - emb.min())

# === Plot ===
fig = plt.figure(figsize=(10, 5))
gs = GridSpec(2, 4, figure=fig, width_ratios=[2, 1, 1, 1])

# Swap positions: top-left = host, bottom-left = watermark
ax_host = fig.add_subplot(gs[0, 0])
ax_wm = fig.add_subplot(gs[1, 0])
ax_emb = fig.add_subplot(gs[:, 1])
ax_ext = fig.add_subplot(gs[:, 2])
ax_diff = fig.add_subplot(gs[:, 3])

ax_host.imshow(host, cmap='gray')
ax_host.set_title("Host Image")
ax_host.axis('off')

ax_wm.imshow(wm, cmap='gray')
ax_wm.set_title("Original Watermark")
ax_wm.axis('off')

# Show secret key below watermark (black color), moved lower
bbox = ax_wm.get_position()
fig.text(bbox.x0 + bbox.width / 2, bbox.y0 - 0.08,
         f"Secret Key: {secret_key}",
         ha='center', va='top', fontsize=12, color='black')

ax_emb.imshow(emb, cmap='gray')
ax_emb.set_title(f"Embedded Image\nMSE: {mse:.2f} | SSIM: {ssim_val:.2f}")
ax_emb.axis('off')

ax_ext.imshow(ext, cmap='gray')
ax_ext.set_title(f"Extracted Watermark\nBER: {ber:.2f}")
ax_ext.axis('off')

ax_diff.imshow(diff, cmap='inferno')
ax_diff.set_title("Difference Image")
ax_diff.axis('off')

plt.tight_layout()
plt.show()
