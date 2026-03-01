"""
feature_matching.py
Stage 2 — AKAZE keypoint detection with BFMatcher and RANSAC homography.

Why AKAZE over ORB?
-------------------
ORB uses binary descriptors optimised for speed on natural images.  UI
elements (flat buttons, icons, text) have few of the blob-like structures
ORB relies on, leading to poor repeatability at different DPI settings.

AKAZE (Accelerated-KAZE) uses a nonlinear diffusion scale space which:
  • Preserves fine-grained UI structures (corners, text strokes, icon edges)
  • Produces float descriptors that are robust to scale and slight rotation
  • Outperforms ORB on synthetic/graphic content in published benchmarks

Descriptor type: AKAZE defaults to M-LDB binary descriptors (256-bit),
so BFMatcher with NORM_HAMMING is correct and fast.  The alternative
KAZE (float) would require NORM_L2.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .preprocessing import to_grayscale

logger = logging.getLogger(__name__)

# Maximum Hamming distance for a match to be considered "good".
# AKAZE descriptors are 486-bit by default; 120 ≈ 75% bit agreement.
_AKAZE_MAX_DISTANCE = 120


@dataclass
class FeatureMatchCandidate:
    """
    Result of a successful AKAZE feature match.

    Attributes:
        center:     (x, y) screen coordinates of the element centre.
        inliers:    Number of RANSAC inlier matches (quality indicator).
        match_size: Estimated (width, height) of the found region on screen.
    """
    center: tuple[int, int]
    inliers: int
    match_size: tuple[int, int]


def akaze_match(
    screen: np.ndarray,
    template: np.ndarray,
    min_inliers: int = 8,
    max_distance: int = _AKAZE_MAX_DISTANCE,
) -> Optional[FeatureMatchCandidate]:
    """
    Locate a template in a screen image using AKAZE feature descriptors,
    brute-force Hamming matching, and RANSAC-based homography validation.

    The RANSAC step is critical: it discards geometrically inconsistent
    matches (which are common with UI elements that repeat patterns, such as
    toolbars or icon grids) and leaves only the spatially coherent subset.

    Args:
        screen:       Full-screen BGR uint8 array.
        template:     Template BGR uint8 array.
        min_inliers:  Minimum RANSAC inliers to accept a detection.
                      8 is intentionally lower than the old ORB value of 15
                      because AKAZE produces fewer but more reliable matches
                      on UI content.
        max_distance: Maximum Hamming distance between descriptor pairs.

    Returns:
        FeatureMatchCandidate if a valid homography is found, else None.
    """
    screen_gray = to_grayscale(screen)
    tmpl_gray = to_grayscale(template)

    # AKAZE is available in all OpenCV builds without extra contrib modules
    akaze = cv2.AKAZE_create()

    kp_t, des_t = akaze.detectAndCompute(tmpl_gray, None)
    kp_s, des_s = akaze.detectAndCompute(screen_gray, None)

    if des_t is None or des_s is None or len(kp_t) < 4 or len(kp_s) < 4:
        logger.debug(
            f"AKAZE: insufficient keypoints "
            f"(template={len(kp_t) if des_t is not None else 0}, "
            f"screen={len(kp_s) if des_s is not None else 0})."
        )
        return None

    # crossCheck=True enforces the mutual-best-match constraint, which
    # significantly reduces false positive matches on repetitive UI textures
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    raw_matches = bf.match(des_t, des_s)

    good = [m for m in raw_matches if m.distance < max_distance]
    good.sort(key=lambda m: m.distance)

    logger.debug(
        f"AKAZE: {len(raw_matches)} raw → {len(good)} good "
        f"(min_inliers={min_inliers})"
    )

    if len(good) < min_inliers:
        return None

    src_pts = np.float32([kp_t[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_s[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # RANSAC reprojection threshold of 5.0px is a standard recommendation
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None:
        logger.warning("AKAZE: RANSAC homography returned None (degenerate config).")
        return None

    inlier_count = int(mask.sum()) if mask is not None else 0
    logger.debug(f"AKAZE: homography inliers={inlier_count}")

    if inlier_count < min_inliers:
        logger.debug(f"AKAZE: inlier count {inlier_count} < {min_inliers} — discarding.")
        return None

    # Project the four template corners through H to find the on-screen region
    t_h, t_w = tmpl_gray.shape
    corners = np.float32([[0, 0], [t_w, 0], [t_w, t_h], [0, t_h]]).reshape(-1, 1, 2)
    proj = cv2.perspectiveTransform(corners, H)

    cx = int(np.mean(proj[:, 0, 0]))
    cy = int(np.mean(proj[:, 0, 1]))

    # Compute bounding box of the projected region for SSIM cropping
    x_coords = proj[:, 0, 0]
    y_coords = proj[:, 0, 1]
    est_w = int(np.max(x_coords) - np.min(x_coords))
    est_h = int(np.max(y_coords) - np.min(y_coords))

    # Sanity check: centre must fall inside screen
    s_h, s_w = screen_gray.shape
    if not (0 <= cx < s_w and 0 <= cy < s_h):
        logger.warning(
            f"AKAZE: projected centre ({cx}, {cy}) outside screen "
            f"({s_w}x{s_h}) — discarding."
        )
        return None

    return FeatureMatchCandidate(
        center=(cx, cy),
        inliers=inlier_count,
        match_size=(max(1, est_w), max(1, est_h)),
    )
