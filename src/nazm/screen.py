"""
screen.py
Multi-monitor screenshot capture via mss.
Returns BGR uint8 numpy arrays compatible with OpenCV.
"""

import logging
from typing import Optional

import mss
import numpy as np

from .exceptions import ScreenCaptureError

logger = logging.getLogger(__name__)


def capture_screen_with_offset(monitor_index: int = 1):
    with mss.mss() as sct:
        monitor = sct.monitors[monitor_index]
        raw = sct.grab(monitor)
        img = np.array(raw, dtype=np.uint8)[:, :, :3]

        return img, monitor["left"], monitor["top"]

def list_monitors() -> list[dict]:
    """
    Return metadata for all available monitors.
    Index 0 is the combined virtual desktop; 1+ are physical monitors.
    """
    with mss.mss() as sct:
        return list(sct.monitors)


def capture_screen(monitor_index: int = 0) -> np.ndarray:
    """
    Capture a screenshot of the specified monitor.

    Args:
        monitor_index: 0 = combined virtual screen (all monitors).
                       1, 2, ... = individual physical monitors.

    Returns:
        BGR uint8 numpy array of shape (height, width, 3).

    Raises:
        ScreenCaptureError: If mss fails to grab the screen.
        IndexError: If monitor_index exceeds available monitors.
    """
    try:
        with mss.mss() as sct:
            monitors = sct.monitors
            if monitor_index >= len(monitors):
                raise IndexError(
                    f"monitor_index={monitor_index} out of range. "
                    f"Available: 0–{len(monitors) - 1}"
                )
            raw = sct.grab(monitors[monitor_index])
            img = np.array(raw, dtype=np.uint8)
            bgr = img[:, :, :3]   # Drop alpha channel
            logger.debug(
                f"Captured monitor {monitor_index}: "
                f"{bgr.shape[1]}x{bgr.shape[0]}px"
            )
            return bgr
    except mss.exception.ScreenShotError as exc:
        raise ScreenCaptureError(f"mss failed to capture screen: {exc}") from exc


def capture_region(x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Capture a rectangular region of the screen using absolute coordinates.

    Returns:
        BGR uint8 numpy array of shape (height, width, 3).
    """
    try:
        with mss.mss() as sct:
            region = {"top": y, "left": x, "width": width, "height": height}
            raw = sct.grab(region)
            img = np.array(raw, dtype=np.uint8)
            return img[:, :, :3]
    except mss.exception.ScreenShotError as exc:
        raise ScreenCaptureError(f"mss failed to capture region: {exc}") from exc
