"""
Preprocessing for the APTOS 2019 diabetic retinopathy dataset.

This module is the SINGLE source of truth for image preprocessing. It is
imported by train.py, evaluate.py, and app.py so that the transform applied
at training time is bit-for-bit identical to the one applied at inference
time. Duplicating this logic in the serving code is the most common cause
of silent training/serving skew.
"""

import cv2
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input

IMAGE_SIZE = 299  # Xception's native input resolution
SIGMA_X = 10      # Gaussian sigma for Ben Graham local contrast normalisation


def crop_image_from_gray(img, tol=7):
    """Crop the black border surrounding a fundus image.

    Fundus photographs are a circular retina on a black rectangular canvas,
    and the amount of padding varies by camera. We threshold on grayscale
    intensity and keep only rows/columns containing signal, so that the
    subsequent resize spends all 299x299 pixels on retina rather than on
    letterboxing.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol

    # Degenerate case: an entirely dark image would crop to nothing.
    if img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0] == 0:
        return img

    channels = [img[:, :, i][np.ix_(mask.any(1), mask.any(0))] for i in range(3)]
    return np.stack(channels, axis=-1)


def preprocess_image(image_path, sigma_x=SIGMA_X):
    """Full preprocessing pipeline: load -> crop -> resize -> Ben Graham -> scale.

    The Ben Graham step is `4*I - 4*blur(I) + 128`, a high-pass filter.
    The heavy blur (sigma=10) captures low-frequency content -- overall
    illumination, colour cast, retinal pigmentation -- which varies enormously
    between cameras and patients and is nuisance signal. Subtracting it leaves
    local structure: vessels, microaneurysms, exudates, haemorrhages. The x4
    amplifies that residual and the +128 recentres it into visible range.

    Returns a float32 array scaled to [-1, 1] by Xception's preprocess_input.
    """
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.addWeighted(
        image, 4,
        cv2.GaussianBlur(image, (0, 0), sigma_x),
        -4, 128,
    )

    # Xception was pretrained with inputs scaled to [-1, 1], not [0, 1].
    # Using the wrong range here degrades accuracy silently.
    return preprocess_input(image.astype(np.float32))


def preprocess_array(image_rgb, sigma_x=SIGMA_X):
    """Same pipeline for an in-memory RGB array (used by the Gradio app)."""
    image = crop_image_from_gray(image_rgb)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    image = cv2.addWeighted(
        image, 4,
        cv2.GaussianBlur(image, (0, 0), sigma_x),
        -4, 128,
    )
    return preprocess_input(image.astype(np.float32))


def denormalize_for_display(image):
    """Map a preprocessed [-1, 1] image back to [0, 1] for matplotlib."""
    return (image + 1) / 2
