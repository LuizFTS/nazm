"""
preprocessing.py
Image preprocessing utilities shared across detection stages.

Design rationale
----------------
All detection stages operate on preprocessed representations rather than
raw BGR pixels.  This makes detection structurally robust to:

  • UI theme / color scheme changes   → grayscale removes color dependency
  • Anti-aliasing / sub-pixel rendering → edge maps preserve structure
  • DPI scaling / resolution mismatch  → scale-normalised resizing

Each function is stateless and returns a new array; originals are not mutated.
"""

import logging
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canny edge detection defaults.
# These values work well for UI screenshots (sharp edges, flat fill regions).
# Lower threshold determines weak-edge hysteresis; upper anchors strong edges.
# ---------------------------------------------------------------------------
_CANNY_LOW: int = 50
_CANNY_HIGH: int = 150


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """
    Convert a BGR uint8 image to single-channel grayscale.

    Accepts images that are already grayscale (2-D or 3-channel identical).

    Args:
        img: BGR or grayscale uint8 numpy array.

    Returns:
        2-D uint8 grayscale array of the same height/width.
    """
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 1:
        return img[:, :, 0]
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def to_edges(
    img: np.ndarray,
    low_threshold: int = _CANNY_LOW,
    high_threshold: int = _CANNY_HIGH,
    blur_ksize: int = 3,
) -> np.ndarray:
    """
    Produce a Canny edge map from a BGR or grayscale image.

    A mild Gaussian blur is applied first to suppress single-pixel noise from
    screen capture compression artefacts without blurring UI element edges.

    Args:
        img:            BGR or grayscale uint8 array.
        low_threshold:  Canny hysteresis lower bound (default 50).
        high_threshold: Canny hysteresis upper bound (default 150).
        blur_ksize:     Gaussian kernel size for pre-blur (must be odd, ≥ 1).

    Returns:
        2-D binary uint8 edge map (0 or 255 per pixel).

    Raises:
        PreprocessingError: If the input produces an empty or unusable edge map.
    """
    gray = to_grayscale(img)

    # Blur before edge detection to avoid noise-driven false edges
    if blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    edges = cv2.Canny(gray, low_threshold, high_threshold)

    if edges.max() == 0:
        logger.debug(
            "to_edges: produced an empty edge map "
            "(image may be uniform — edge matching will be ineffective)."
        )

    return edges


def resize_image(
    img: np.ndarray,
    scale: float,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Resize an image by a scalar factor.

    Uses INTER_LINEAR by default (good balance of speed and quality for
    downscaling UI elements).  Caller may pass INTER_AREA for aggressive
    downscaling or INTER_CUBIC for upscaling.

    Args:
        img:           Input array (any number of channels).
        scale:         Scaling factor (e.g. 0.5 = half size, 2.0 = double).
        interpolation: OpenCV interpolation flag.

    Returns:
        Resized array.  Returns the original unchanged if scale ≈ 1.0.

    Raises:
        ValueError: If scale ≤ 0.
    """
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")

    if abs(scale - 1.0) < 1e-6:
        return img

    h, w = img.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=interpolation)


def crop_region(
    img: np.ndarray,
    cx: int,
    cy: int,
    width: int,
    height: int,
) -> Optional[np.ndarray]:
    """
    Crop a rectangular region centred on (cx, cy) from img.

    Clamps to image boundaries; returns None if the resulting crop would be
    degenerate (zero area after clamping).

    Args:
        img:    Source array (BGR or grayscale).
        cx:     Horizontal centre of the crop in img coordinates.
        cy:     Vertical centre of the crop in img coordinates.
        width:  Desired crop width in pixels.
        height: Desired crop height in pixels.

    Returns:
        Cropped array, or None if coordinates are entirely outside img.
    """
    img_h, img_w = img.shape[:2]

    x1 = max(0, cx - width // 2)
    y1 = max(0, cy - height // 2)
    x2 = min(img_w, cx + (width - width // 2))
    y2 = min(img_h, cy + (height - height // 2))

    if x2 <= x1 or y2 <= y1:
        return None

    return img[y1:y2, x1:x2]
