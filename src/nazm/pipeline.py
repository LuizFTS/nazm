"""
pipeline.py
Detection pipeline orchestrator.

Pipeline per polling cycle
--------------------------
  Stage 1  →  Multi-scale edge-based template matching   (fast path)
              Returns a TemplateMatchCandidate or None.

  Stage 2  →  AKAZE feature matching + RANSAC homography  (robust path)
              Runs only when Stage 1 fails.
              Returns a FeatureMatchCandidate or None.

  Stage 3  →  SSIM verification                           (confirmation gate)
              Runs on whichever candidate Stage 1 or Stage 2 produced.
              Rejects geometrically plausible but structurally wrong matches.

Only when all three stages yield no accepted candidate does the cycle fail.
The outer loop retries until timeout.

OCR is intentionally excluded from the core pipeline.  It can be added
by callers via the `post_processors` extension point (not implemented in MVP).
"""

import logging
import time
import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


import cv2

from .screen import capture_screen
from .template_matching import multiscale_template_match, TemplateMatchCandidate
from .feature_matching import akaze_match, FeatureMatchCandidate
from .similarity import ssim_verify
from .exceptions import ElementNotFoundError, TemplateLoadError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SearchConfig:
    """
    All tuneable parameters for a single find_element() call.

    Template matching
    -----------------
    threshold_template : float
        Minimum TM_CCOEFF_NORMED score on edge maps.  Edge-map scores are
        inherently lower than raw-pixel scores; 0.75 is a calibrated default.
    scales : list[float]
        Scale factors to probe.  Default covers 50%–200% in 9 steps.

    Feature matching
    ----------------
    min_akaze_inliers : int
        Minimum RANSAC inliers for an AKAZE match to be accepted.

    SSIM verification
    -----------------
    threshold_ssim : float
        Minimum SSIM to confirm a Stage 1 or Stage 2 candidate.
        Set to 0.0 to disable SSIM verification entirely.

    Capture
    -------
    monitor_index : int
        mss monitor index (0 = all combined, 1+ = physical monitors).

    Timing
    ------
    timeout : float       Total seconds to keep retrying.
    poll_interval : float Seconds between polling cycles.
    """
    # Template matching
    threshold_template: float = 0.75
    scales: list[float] = field(
        default_factory=lambda: [0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0]
    )
    canny_low: int = 50
    canny_high: int = 150

    # Feature matching
    min_akaze_inliers: int = 8

    # SSIM verification
    threshold_ssim: float = 0.60

    # Capture
    monitor_index: int = 0

    # Timing
    timeout: float = 10.0
    poll_interval: float = 0.5


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Result returned by find_element() on success.

    Attributes:
        center:        (x, y) screen coordinates of element centre.
        stage:         1 = template match, 2 = feature match.
        stage_name:    Human-readable label.
        ssim_score:    SSIM score of the confirmed region (0–1).
        elapsed:       Seconds from search start to detection.
        template_path: Path of the matched template.
    """
    center: tuple[int, int]
    stage: int
    stage_name: str
    ssim_score: float
    elapsed: float
    template_path: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_template(template_path: str):
    """Read a template image from disk as a BGR numpy array."""
    img = cv2.imread(template_path)
    if img is None:
        raise TemplateLoadError(
            f"OpenCV could not read: '{template_path}'. "
            "Ensure the file exists and is PNG/JPEG/BMP with no unicode in the path."
        )
    return img


def _try_confirm(
    screen,
    template,
    cx: int,
    cy: int,
    region_size: tuple[int, int],
    threshold_ssim: float,
) -> tuple[bool, float]:
    """
    Run SSIM verification.  Always returns (True, 1.0) when SSIM is disabled
    (threshold_ssim ≤ 0) so Stage 1/2 candidates pass through unconditionally.
    """
    if threshold_ssim <= 0.0:
        return True, 1.0
    return ssim_verify(
        screen=screen,
        template=template,
        cx=cx,
        cy=cy,
        region_size=region_size,
        threshold=threshold_ssim,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def find_element(
    template_path: str,
    config: Optional[SearchConfig] = None,
    # Convenience flat overrides — applied after config is resolved
    monitor_index: Optional[int] = None,
    threshold_template: Optional[float] = None,
    threshold_ssim: Optional[float] = None,
    min_akaze_inliers: Optional[int] = None,
    timeout: Optional[float] = None,
    poll_interval: Optional[float] = None,
) -> MatchResult:
    """
    Search for a UI element using the three-stage vision pipeline.

    Runs Stage 1 → Stage 2 → Stage 3 (SSIM) on each polling cycle until a
    confirmed match is found or the timeout expires.

    Args:
        template_path:      Path to the template image (PNG/JPEG/BMP).
        config:             SearchConfig for full parameter control.
                            A default SearchConfig is used when None.
        monitor_index:      Quick override for config.monitor_index.
        threshold_template: Quick override for config.threshold_template.
        threshold_ssim:     Quick override for config.threshold_ssim.
        min_akaze_inliers:  Quick override for config.min_akaze_inliers.
        timeout:            Quick override for config.timeout.
        poll_interval:      Quick override for config.poll_interval.

    Returns:
        MatchResult with centre coordinates, winning stage, SSIM score, timing.

    Raises:
        TemplateLoadError:    Template image cannot be read.
        ElementNotFoundError: No confirmed match within timeout.
    """
    cfg = config if config is not None else SearchConfig()

    # Apply flat keyword overrides
    if monitor_index is not None:    cfg.monitor_index    = monitor_index
    if threshold_template is not None: cfg.threshold_template = threshold_template
    if threshold_ssim is not None:   cfg.threshold_ssim   = threshold_ssim
    if min_akaze_inliers is not None: cfg.min_akaze_inliers = min_akaze_inliers
    if timeout is not None:          cfg.timeout          = timeout
    if poll_interval is not None:    cfg.poll_interval    = poll_interval

    actual_path = resolve_template_path(template_path)
    template_bgr = _load_template(actual_path)

    logger.info(
        f"Searching for '{template_path}' (Resolved: {actual_path}) "
        f"[monitor={cfg.monitor_index}, "
        f"tmpl_threshold={cfg.threshold_template}, "
        f"ssim_threshold={cfg.threshold_ssim}, "
        f"timeout={cfg.timeout}s]"
    )

    start = time.monotonic()
    deadline = start + cfg.timeout
    cycle = 0

    while time.monotonic() < deadline:
        cycle += 1
        elapsed = time.monotonic() - start
        logger.debug(f"Poll cycle {cycle} at t={elapsed:.2f}s")

        screen = capture_screen(cfg.monitor_index)

        # ── Stage 1: Multi-scale edge-based template matching ──────────────
        tmpl_candidate: Optional[TemplateMatchCandidate] = multiscale_template_match(
            screen=screen,
            template=template_bgr,
            threshold=cfg.threshold_template,
            scales=cfg.scales,
            canny_low=cfg.canny_low,
            canny_high=cfg.canny_high,
        )

        if tmpl_candidate is not None:
            cx, cy = tmpl_candidate.center
            passed, ssim_score = _try_confirm(
                screen=screen,
                template=template_bgr,
                cx=cx,
                cy=cy,
                region_size=tmpl_candidate.match_size,
                threshold_ssim=cfg.threshold_ssim,
            )
            if passed:
                elapsed = time.monotonic() - start
                logger.info(
                    f"[✓ Stage 1 — Multi-scale Template Match] "
                    f"Found at {(cx, cy)} | "
                    f"scale={tmpl_candidate.scale:.3f} | "
                    f"conf={tmpl_candidate.score:.4f} | "
                    f"ssim={ssim_score:.4f} | "
                    f"t={elapsed:.2f}s (cycle {cycle})"
                )
                return MatchResult(
                    center=(cx, cy),
                    stage=1,
                    stage_name="Multi-scale Template Match",
                    ssim_score=ssim_score,
                    elapsed=elapsed,
                    template_path=template_path,
                )
            else:
                logger.debug(
                    f"Stage 1 candidate at {(cx, cy)} rejected by SSIM "
                    f"(score={ssim_score:.4f} < {cfg.threshold_ssim})."
                )

        # ── Stage 2: AKAZE feature matching + RANSAC ───────────────────────
        feat_candidate: Optional[FeatureMatchCandidate] = akaze_match(
            screen=screen,
            template=template_bgr,
            min_inliers=cfg.min_akaze_inliers,
        )

        if feat_candidate is not None:
            cx, cy = feat_candidate.center
            passed, ssim_score = _try_confirm(
                screen=screen,
                template=template_bgr,
                cx=cx,
                cy=cy,
                region_size=feat_candidate.match_size,
                threshold_ssim=cfg.threshold_ssim,
            )
            if passed:
                elapsed = time.monotonic() - start
                logger.info(
                    f"[✓ Stage 2 — AKAZE Feature Match] "
                    f"Found at {(cx, cy)} | "
                    f"inliers={feat_candidate.inliers} | "
                    f"ssim={ssim_score:.4f} | "
                    f"t={elapsed:.2f}s (cycle {cycle})"
                )
                return MatchResult(
                    center=(cx, cy),
                    stage=2,
                    stage_name="AKAZE Feature Match",
                    ssim_score=ssim_score,
                    elapsed=elapsed,
                    template_path=template_path,
                )
            else:
                logger.debug(
                    f"Stage 2 candidate at {(cx, cy)} rejected by SSIM "
                    f"(score={ssim_score:.4f} < {cfg.threshold_ssim})."
                )

        remaining = deadline - time.monotonic()
        logger.debug(
            f"Cycle {cycle}: both stages failed. "
            f"Retrying in {cfg.poll_interval}s ({remaining:.1f}s remaining)."
        )

        if remaining > cfg.poll_interval:
            time.sleep(cfg.poll_interval)
        else:
            break

    total = time.monotonic() - start
    raise ElementNotFoundError(
        f"Element not found after {cycle} cycle(s) ({total:.1f}s). "
        "Both detection stages + SSIM verification exhausted.",
        template_path=template_path,
        timeout=cfg.timeout,
    )

def resolve_template_path(name: str) -> str:
    """
    Tenta encontrar o template. 
    1. Verifica se 'name' já é um caminho real.
    2. Se não, procura na pasta de templates do AppData.
    """
    # Se o arquivo já existe no caminho passado, usa ele mesmo
    if os.path.exists(name):
        return name
        
    # Se não existe, procura no AppData
    appdata_dir = Path(os.getenv('APPDATA')) / "nazm" / "templates"
    
    # Tenta com o nome exato e também garantindo a extensão .png
    possible_names = [name, f"{name}.png", f"{name}.jpg"]
    
    for n in possible_names:
        full_path = appdata_dir / n
        if full_path.exists():
            return str(full_path)
            
    # Se chegar aqui, o arquivo realmente não foi encontrado
    return name