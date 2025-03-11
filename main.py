from watermarkers import e_blind_d_lc as e
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import seaborn as sns
import numpy as np
import cv2


if __name__ == "__main__":
    nof_samples = 9469
    dataset = tfds.load("imagenette", split=f"train[:{nof_samples}]", download=False)
    dataset = dataset.take(nof_samples)

    threshold = 0.1
    label_0 = []
    label_1 = []
    label_none = []

    for i, img in enumerate(dataset):
        img = np.array(img["image"], dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        key = np.random.randint(922337)
        _, lc = e.EBlindDLC.extract(work=img, secret_key=key, threshold=threshold)
        label_none.append(lc)

        img_1 = e.EBlindDLC.embed(work=img, secret=True, secret_key=key, embedding_strength=1.5)
        _, lc = e.EBlindDLC.extract(work=img_1, secret_key=key, threshold=threshold)
        label_1.append(lc)

        img_0 = e.EBlindDLC.embed(work=img, secret=False, secret_key=key, embedding_strength=1.5)
        _, lc = e.EBlindDLC.extract(work=img_0, secret_key=key, threshold=threshold)
        label_0.append(lc)

        if i % 50 == 0:
            print(f'{i}/{nof_samples}')

    plt.figure(figsize=(8, 6))

    plt.axvspan(-threshold, threshold, color='gray', alpha=0.3, label="Threshold region")
    plt.axvline(-threshold, color="black", linestyle="--")
    plt.axvline(threshold, color="black", linestyle="--")

    sns.kdeplot(label_0, color="blue", label=r"$m = 0$", linestyle="-")
    sns.kdeplot(label_1, color="red", label=r"$m = 1$", linestyle="-")
    # sns.kdeplot(label_none, color="black", label="No watermark", linestyle="-")

    # Labels and formatting
    plt.xlabel("Detection value", fontsize=12)
    plt.ylabel("")
    plt.yticks([])
    plt.legend()
    plt.show()