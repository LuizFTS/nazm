"""
template_matching.py
Stage 1 — Multi-scale edge-based template matching.

Why edge maps instead of raw pixels?
--------------------------------------
Raw pixel matching is sensitive to:
  - UI theme (dark mode vs light mode changes pixel values)
  - DPI-aware rendering (sub-pixel smoothing shifts colour values)
  - JPEG/PNG compression in captured screenshots

Edge maps encode *structure* (borders, corners, text strokes) rather than
absolute pixel values.  Two screenshots of the same button in different themes
will share the same edge structure even if their colours differ completely.

Why multiple scales?
--------------------
Templates are captured at a specific DPI.  On a 150% scaled display the same
element appears 1.5x larger.  By testing a range of scales we find the best
fitting size without requiring the caller to know the target DPI in advance.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .preprocessing import to_edges, resize_image

logger = logging.getLogger(__name__)

# Scale range: 50% → 200% of template size in 9 steps.
# Covers the full range from a 1x template on a 2x display down to a 2x
# template on a 1x display, with 125% steps for common Windows DPI settings.
_DEFAULT_SCALES: list[float] = [0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0]


@dataclass
class TemplateMatchCandidate:
    """
    A candidate match returned by multiscale_template_match.

    Attributes:
        center:     (x, y) screen coordinates of the match centre.
        score:      TM_CCOEFF_NORMED confidence score (0–1).
        scale:      Scale factor at which the match was found.
        match_loc:  Top-left corner of the matched region (for SSIM cropping).
        match_size: (width, height) of the matched region at the winning scale.
    """
    center: tuple[int, int]
    score: float
    scale: float
    match_loc: tuple[int, int]
    match_size: tuple[int, int]


def multiscale_template_match(
    screen: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.75,
    scales: list[float] = _DEFAULT_SCALES,
    canny_low: int = 50,
    canny_high: int = 150,
) -> Optional[TemplateMatchCandidate]:
    """
    Find a template inside a screen image across multiple scales using
    edge-map cross-correlation.

    The function:
      1. Converts both images to Canny edge maps.
      2. Resizes the template edge map to each scale in `scales`.
      3. Runs cv2.matchTemplate (TM_CCOEFF_NORMED) at each scale.
      4. Returns the best-scoring candidate above `threshold`.

    Note: The screen edge map is computed once and reused across scales to
    avoid redundant work.  Only the template is resized per iteration.

    Args:
        screen:     Full-screen BGR uint8 array.
        template:   Template BGR uint8 array.
        threshold:  Minimum confidence required (default 0.75).
                    Lower than the raw-pixel default because edge maps are
                    sparser and slightly noisier than full-colour images.
        scales:     List of scale factors to test.
        canny_low:  Canny lower hysteresis threshold.
        canny_high: Canny upper hysteresis threshold.

    Returns:
        TemplateMatchCandidate with the best score above threshold, or None.
    """
    screen_edges = to_edges(screen, low_threshold=canny_low, high_threshold=canny_high)
    tmpl_edges_base = to_edges(template, low_threshold=canny_low, high_threshold=canny_high)

    screen_h, screen_w = screen_edges.shape
    best: Optional[TemplateMatchCandidate] = None

    for scale in scales:
        scaled_tmpl = resize_image(tmpl_edges_base, scale)
        t_h, t_w = scaled_tmpl.shape[:2]

        # Template must be strictly smaller than the screen in both dimensions
        if t_h >= screen_h or t_w >= screen_w:
            logger.debug(
                f"Scale {scale:.3f}: scaled template ({t_w}x{t_h}) "
                f"≥ screen ({screen_w}x{screen_h}) — skipping."
            )
            continue

        # Template must be at least 2x2 pixels after scaling
        if t_h < 2 or t_w < 2:
            logger.debug(f"Scale {scale:.3f}: template too small after scaling — skipping.")
            continue

        result_map = cv2.matchTemplate(screen_edges, scaled_tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result_map)

        logger.debug(
            f"Scale {scale:.3f}: best_score={max_val:.4f} "
            f"at {max_loc} (threshold={threshold})"
        )

        if max_val < threshold:
            continue

        # Keep the highest-confidence match across all scales
        if best is None or max_val > best.score:
            cx = max_loc[0] + t_w // 2
            cy = max_loc[1] + t_h // 2
            best = TemplateMatchCandidate(
                center=(cx, cy),
                score=max_val,
                scale=scale,
                match_loc=max_loc,
                match_size=(t_w, t_h),
            )

    if best is not None:
        logger.debug(
            f"Multi-scale template match: best score={best.score:.4f} "
            f"at scale={best.scale:.3f}, centre={best.center}"
        )

    return best
