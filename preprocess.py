"""Single source of truth for image preprocessing.

Both train.py and app.py import from this module instead of keeping their
own copies. That shared-module structure is the point: it makes
training/serving skew structurally impossible rather than merely unlikely.
"""

import cv2
import numpy as np
from tensorflow.keras.applications.xception import preprocess_input

IMAGE_SIZE = 299
SIGMA_X = 10


def crop_image_from_gray(img, tol=7):
    """Crop the black border surrounding the circular retinal fundus."""
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]

    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray_img > tol
    check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
    if check_shape == 0:
        # Degenerate all-dark image: cropping would leave nothing, so
        # return the input unchanged rather than an empty array.
        return img

    img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
    img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
    img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
    return np.stack([img1, img2, img3], axis=-1)


def _ben_graham(image_rgb, sigma_x):
    """Crop, resize, and apply the Ben Graham high-pass transform.

    Shared by preprocess_image and preprocess_array so the two entry
    points can never drift apart.
    """
    image = crop_image_from_gray(image_rgb)
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
    # 4*I - 4*blur(I) + 128: the heavy Gaussian blur captures only the
    # low-frequency content of the image (illumination, colour cast,
    # retinal pigmentation), all of which vary by camera and patient.
    # Subtracting that blur is a high-pass filter, leaving behind the
    # fine structure that actually matters: vessels, microaneurysms,
    # exudates and haemorrhages. The +128 re-centers the result into a
    # displayable range instead of clipping around zero.
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigma_x), -4, 128)
    return image


def preprocess_image(image_path, sigma_x=SIGMA_X):
    """Load an image from disk and preprocess it for the model.

    Returns a float32 array scaled to [-1, 1] by Xception's
    preprocess_input.
    """
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = _ben_graham(image, sigma_x)
    return preprocess_input(image.astype(np.float32))


def preprocess_array(image_rgb, sigma_x=SIGMA_X):
    """Preprocess an in-memory RGB array (e.g. from the Gradio app).

    Must produce output identical to preprocess_image for the same
    underlying image, since it shares the crop/resize/high-pass helper.
    """
    image = _ben_graham(image_rgb, sigma_x)
    return preprocess_input(image.astype(np.float32))


def denormalize_for_display(image):
    """Map a [-1, 1]-scaled image back to [0, 1] for matplotlib."""
    return (image + 1.0) / 2.0
