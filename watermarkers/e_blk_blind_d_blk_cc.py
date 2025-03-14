from .base_watermarker import BaseWatermarker
import numpy as np
from typing import Union, Tuple
import tensorflow as tf


class EBLKBlindDBLKCC(BaseWatermarker):

    @staticmethod
    def embed(work: np.ndarray,
              secret: bool,
              secret_key: int,
              embedding_strength: int = 1,
              patch_size: int = 8) -> np.ndarray:
        np.random.seed(secret_key)

        # Extract patches
        work_tensor = tf.convert_to_tensor(work, dtype=tf.float32)
        # Add batch dimension: (1, H, W)
        work_tensor = tf.expand_dims(work_tensor, axis=0)
        # Add channel dimension: (1, H, W, 1)
        work_tensor = tf.expand_dims(work_tensor, axis=-1)

        patches = tf.image.extract_patches(
            images=work_tensor,  # Add batch dim
            sizes=[1, patch_size, patch_size, 1],  # Patch size
            strides=[1, patch_size, patch_size, 1],  # Stride = patch size (no overlap)
            rates=[1, 1, 1, 1],  # Dilation rate (1 = normal)
            padding="VALID"  # No padding, only full patches
        )
        # Remove batch dimension (axis=0)
        patches = tf.squeeze(patches, axis=0)

        # Reshape to (num_patches, patch_size, patch_size)
        num_patches_x, num_patches_y, _ = patches.shape
        patches = tf.reshape(patches, (num_patches_x * num_patches_y, patch_size, patch_size))
        patches = patches.numpy()

        # Compute mean patch
        mean_patch = np.mean(patches, axis=0)

        # Get reference pattern
        wr = np.random.randint(7, size=mean_patch.shape, dtype=np.int8)

        # Embedding Scheme
        wm = (-1) ** (int(secret) + 1) * wr
        wa = embedding_strength * wm
        cw = mean_patch + wa

        # Compute how many times to repeat in height and width
        target_h, target_w = work.shape
        repeat_h = tf.math.ceil(target_h / patch_size)
        repeat_w = tf.math.ceil(target_w / patch_size)

        # Convert to int (tf.tile needs integer multiples)
        multiples = tf.convert_to_tensor([int(repeat_h), int(repeat_w)], dtype=tf.int32)
        ref_pattern = tf.tile(cw, multiples)
        cw = work + ref_pattern

        # Normalize
        cw = cw.numpy()
        cw = cw - np.min(cw)  # Shift minimum to 0
        cw = (cw / np.max(cw)) * 255  # Scale to 0-255
        cw = cw.astype(np.uint8)

        return cw

    @staticmethod
    def extract(work: np.ndarray,
                secret_key: int,
                threshold: float,
                patch_size: int = 8) -> Union[Tuple[int, float], Tuple[None, float]]:

        np.random.seed(secret_key)

        # Extract patches
        work_tensor = tf.convert_to_tensor(work, dtype=tf.float32)
        # Add batch dimension: (1, H, W)
        work_tensor = tf.expand_dims(work_tensor, axis=0)
        # Add channel dimension: (1, H, W, 1)
        work_tensor = tf.expand_dims(work_tensor, axis=-1)

        patches = tf.image.extract_patches(
            images=work_tensor,  # Add batch dim
            sizes=[1, patch_size, patch_size, 1],  # Patch size
            strides=[1, patch_size, patch_size, 1],  # Stride = patch size (no overlap)
            rates=[1, 1, 1, 1],  # Dilation rate (1 = normal)
            padding="VALID"  # No padding, only full patches
        )
        # Remove batch dimension (axis=0)
        patches = tf.squeeze(patches, axis=0)

        # Reshape to (num_patches, patch_size, patch_size)
        num_patches_x, num_patches_y, _ = patches.shape
        patches = tf.reshape(patches, (num_patches_x * num_patches_y, patch_size, patch_size))
        patches = patches.numpy()

        # Compute mean patch
        mean_patch = np.mean(patches, axis=0)

        # Get reference pattern
        wr = np.random.randint(7, size=mean_patch.shape, dtype=np.int8)

        # Calculate the correlation coefficient of reference pattern and work
        lc = BaseWatermarker._correlation_coefficient(mean_patch, wr)
        if lc > threshold:
            return 1, lc

        if lc < -threshold:
            return 0, lc

        return None, lc