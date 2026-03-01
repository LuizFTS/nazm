"""
similarity.py
Stage 3 — Structural Similarity Index (SSIM) verification.

Role in the pipeline
--------------------
Template matching and feature matching both return candidate coordinates.
SSIM acts as a *confirmation gate*: it compares the actual screen region at
those coordinates against the original template to reject spurious matches.

Why SSIM instead of raw pixel difference (MSE)?
------------------------------------------------
MSE is sensitive to uniform brightness shifts (display calibration, night mode)
and to single bright pixels, making it unreliable as a gating signal.

SSIM models three aspects of perception separately:
  1. Luminance similarity    (μx ≈ μy)
  2. Contrast similarity     (σx ≈ σy)
  3. Structural correlation  (σxy / (σx·σy))

Two screenshots of the same button under different DPI or theme settings
will have high SSIM because they share structural correlation even when
absolute luminance or contrast differs.

Implementation note
-------------------
We implement SSIM with NumPy/OpenCV to avoid adding scikit-image as a
dependency.  The formula follows Wang et al. (2004) with the standard
stabilisation constants C1 = (0.01·255)² and C2 = (0.03·255)².
"""

import logging
from typing import Optional

import cv2
import numpy as np

from .preprocessing import to_grayscale, crop_region, resize_image

logger = logging.getLogger(__name__)

# Standard SSIM stabilisation constants (Wang et al. 2004)
_C1 = (0.01 * 255) ** 2   # Luminance stabiliser
_C2 = (0.03 * 255) ** 2   # Contrast stabiliser

# Window size for local statistics.  11px is the canonical value from the
# original paper; smaller values respond to finer structural differences.
_WINDOW_SIZE = 11


def _ssim_numpy(img_a: np.ndarray, img_b: np.ndarray) -> float:
    """
    Compute the mean SSIM between two single-channel float images of identical
    shape using a uniform (box) filter for local statistics.

    A Gaussian window (as in the original paper) requires scipy; a box filter
    is a common lightweight approximation that performs well on UI content
    because UI elements have large flat regions where local statistics are
    already uniform.

    Args:
        img_a: Grayscale float32 array.
        img_b: Grayscale float32 array, same shape as img_a.

    Returns:
        Mean SSIM value in [-1, 1].  Values > 0.8 indicate strong similarity.
    """
    ksize = (_WINDOW_SIZE, _WINDOW_SIZE)

    mu_a = cv2.blur(img_a, ksize)
    mu_b = cv2.blur(img_b, ksize)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab   = mu_a * mu_b

    sigma_a_sq = cv2.blur(img_a * img_a, ksize) - mu_a_sq
    sigma_b_sq = cv2.blur(img_b * img_b, ksize) - mu_b_sq
    sigma_ab   = cv2.blur(img_a * img_b, ksize) - mu_ab

    # Clamp negative variances that can arise from floating-point noise
    sigma_a_sq = np.maximum(sigma_a_sq, 0.0)
    sigma_b_sq = np.maximum(sigma_b_sq, 0.0)

    numerator   = (2.0 * mu_ab + _C1) * (2.0 * sigma_ab + _C2)
    denominator = (mu_a_sq + mu_b_sq + _C1) * (sigma_a_sq + sigma_b_sq + _C2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(np.mean(ssim_map))


def ssim_verify(
    screen: np.ndarray,
    template: np.ndarray,
    cx: int,
    cy: int,
    region_size: Optional[tuple[int, int]] = None,
    threshold: float = 0.60,
) -> tuple[bool, float]:
    """
    Verify that the screen region at (cx, cy) is structurally similar
    to the template.

    Steps:
      1. Crop the screen at (cx, cy) using the template dimensions.
      2. Resize the crop to match the template exactly.
      3. Convert both to grayscale float.
      4. Compute SSIM.
      5. Return (passes_threshold, ssim_score).

    Args:
        screen:      Full-screen BGR uint8 array.
        template:    Template BGR uint8 array.
        cx:          Horizontal centre of the candidate region.
        cy:          Vertical centre of the candidate region.
        region_size: (width, height) of the candidate region on screen.
                     Defaults to template dimensions if None.
        threshold:   Minimum SSIM to accept the match (default 0.60).
                     Lower than a typical photo-similarity threshold because
                     UI elements rendered at different DPI have structural
                     similarity around 0.65–0.75.

    Returns:
        (passed: bool, score: float).
        passed=True means the candidate region meets the SSIM threshold.
    """
    tmpl_gray = to_grayscale(template).astype(np.float32)
    t_h, t_w = tmpl_gray.shape

    # Use template size as crop window if no explicit region_size given
    crop_w, crop_h = region_size if region_size else (t_w, t_h)

    crop = crop_region(screen, cx, cy, crop_w, crop_h)
    if crop is None:
        logger.debug(
            f"SSIM: crop at ({cx},{cy}) size ({crop_w}x{crop_h}) "
            f"is outside screen bounds — rejecting."
        )
        return False, 0.0

    crop_gray = to_grayscale(crop).astype(np.float32)

    # Resize crop to template dimensions for pixel-aligned comparison
    if crop_gray.shape != (t_h, t_w):
        crop_gray = cv2.resize(crop_gray, (t_w, t_h), interpolation=cv2.INTER_LINEAR)

    # Images must be large enough for the window filter to be meaningful
    if t_h < _WINDOW_SIZE or t_w < _WINDOW_SIZE:
        logger.debug(
            f"SSIM: template ({t_w}x{t_h}) smaller than window "
            f"({_WINDOW_SIZE}px) — using pixel-wise correlation instead."
        )
        # Fallback: normalised cross-correlation as a simpler structural proxy
        score = float(np.corrcoef(tmpl_gray.ravel(), crop_gray.ravel())[0, 1])
        score = max(0.0, score)   # corrcoef can return negative values for inverted content
    else:
        score = _ssim_numpy(tmpl_gray, crop_gray)

    passed = score >= threshold
    logger.debug(
        f"SSIM: score={score:.4f} threshold={threshold} → "
        f"{'PASS' if passed else 'FAIL'}"
    )
    return passed, score
