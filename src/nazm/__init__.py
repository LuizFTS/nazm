"""
vision
======
Resolution-independent visual UI automation via a three-stage pipeline.

  Stage 1 → Multi-scale edge-based template matching   (fast, DPI-robust)
  Stage 2 → AKAZE feature matching + RANSAC homography (handles distortion)
  Stage 3 → SSIM structural verification               (rejects false positives)

Public API — element interaction
---------------------------------
    click(path)                 Left-click a detected element
    double_click(path)          Double-click a detected element
    right_click(path)           Right-click a detected element
    move_to(path)               Hover cursor over element (no click)
    type_into(path, text)       Click element, then type text
    click_offset(path, dx, dy)  Click at (dx, dy) pixels from element centre
    scroll_at(path, clicks)     Scroll mouse wheel at element location
    drag_to_element(src, dst)   Drag from one element to another
    find(path)                  Locate element, return MatchResult (no click)

Public API — waiting and polling
---------------------------------
    wait_for(path)              Wait until element appears → MatchResult
    wait_cursor_normal()        Wait for loading cursor (hourglass) to clear
    wait_until_gone(path)       Wait until element disappears from screen
    wait_screen_stable()        Wait for animations/transitions to finish
    wait_any(paths)             Wait for first of multiple templates to appear
    wait_all(paths)             Wait until ALL templates are visible at once

Public API — state queries
---------------------------------
    exists(path)                True if element is currently visible
    count(path)                 Number of non-overlapping instances on screen

Public API — clipboard
---------------------------------
    copy_from(path)             Click element, Ctrl+A, Ctrl+C, return text
    paste_into(path, text)      Place text on clipboard, click element, Ctrl+V

Public API — debug / diagnostics
---------------------------------
    screenshot_of(path)         Save a screenshot of the detected element region
    highlight(path)             Flash a visible border around element (debug)

Public API — configuration
---------------------------------
    set_config(cfg)             Override the module-level default SearchConfig
    list_monitors()             List all monitors with coordinates
    SearchConfig                Dataclass for all detection parameters
    MatchResult                 Return type with .center, .stage, .ssim_score…
    ElementNotFoundError        Raised when detection fails within timeout
"""

import logging
import time
import os
from typing import Optional

import cv2
import numpy as np

from .pipeline import find_element, MatchResult, SearchConfig
from .actions import (
    click as _click_xy,
    double_click as _double_click_xy,
    right_click as _right_click_xy,
    move_to as _move_to_xy,
    type_text as _type_text,
    press_key as _press_key,
    hotkey as _hotkey,
    scroll as _scroll_xy,
    drag as _drag_xy,
    clipboard_set as _clipboard_set,
    clipboard_get as _clipboard_get,
)
from .cursor import wait_cursor_normal as _wait_cursor_normal, is_cursor_busy
from .exceptions import ElementNotFoundError
from .screen import list_monitors, capture_screen_with_offset, capture_screen
from . import capture

logger = logging.getLogger(__name__)

__version__ = "0.1.1"

__all__ = [
    # Core
    "find", "click", "wait_and_click",
    # Interaction
    "double_click", "right_click", "move_to", "type_into",
    "click_offset", "scroll_at", "drag_to_element",
    # Waiting
    "wait_for", "wait_cursor_normal", "wait_until_gone",
    "wait_screen_stable", "wait_any", "wait_all",
    # State
    "exists", "count",
    # Clipboard
    "copy_from", "paste_into",
    # Debug
    "screenshot_of", "highlight",
    # Config
    "set_config", "list_monitors",
    "SearchConfig", "MatchResult", "ElementNotFoundError",
]

# ---------------------------------------------------------------------------
# Module-level default config — override via set_config()
# ---------------------------------------------------------------------------

_default_config = SearchConfig(
    monitor_index=0,
    threshold_template=0.75,
    threshold_ssim=0.60,
    min_akaze_inliers=8,
    scales=[0.5, 0.625, 0.75, 0.875, 1.0, 1.25, 1.5, 1.75, 2.0],
    timeout=20.0,
    poll_interval=0.2,
)


def set_config(config: SearchConfig) -> None:
    """
    Override the module-level default SearchConfig.

    All functions that accept `config=` will use this default when no
    config is explicitly passed.

    Args:
        config: New default SearchConfig.

    Example:
        vision.set_config(vision.SearchConfig(timeout=30.0, monitor_index=1))
    """
    global _default_config
    _default_config = config


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_cfg(config: Optional[SearchConfig]) -> SearchConfig:
    """Return config if provided, otherwise the module default."""
    return config if config is not None else _default_config


def _find_with_offset(
    template_path: str,
    config: SearchConfig,
    **kwargs,
) -> tuple[MatchResult, int, int]:
    """
    Run find_element and also return the monitor offset (offset_x, offset_y).
    The offset must be added to result.center before calling pyautogui.
    """
    _, offset_x, offset_y = capture_screen_with_offset(config.monitor_index)
    result = find_element(template_path=template_path, config=config, **kwargs)
    return result, offset_x, offset_y


# ---------------------------------------------------------------------------
# Core: find
# ---------------------------------------------------------------------------

def find(
    template_path: str,
    config: Optional[SearchConfig] = None,
    **kwargs,
) -> MatchResult:
    """
    Locate a UI element without interacting with it.

    Args:
        template_path: Path to template image (PNG/JPEG/BMP).
        config:        Optional SearchConfig. Module default used if None.
        **kwargs:      Flat overrides: monitor_index, threshold_template,
                       threshold_ssim, min_akaze_inliers, timeout,
                       poll_interval.

    Returns:
        MatchResult with .center (x, y) in monitor-local coordinates,
        .stage, .stage_name, .ssim_score, .elapsed, .template_path.

    Raises:
        TemplateLoadError:    Template image cannot be read.
        ElementNotFoundError: Element not found within timeout.
    """
    return find_element(
        template_path=template_path,
        config=_resolve_cfg(config),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Interaction
# ---------------------------------------------------------------------------

def click(
    template_path: str,
    config: Optional[SearchConfig] = None,
    button: str = "left",
    move_duration: float = 0.1,
    **kwargs,
) -> MatchResult:
    """Locate a UI element and left-click it."""
    cfg = _resolve_cfg(config)
    result, ox, oy = _find_with_offset(template_path, cfg, **kwargs)
    _click_xy(result.center[0] + ox, result.center[1] + oy,
               button=button, move_duration=move_duration)
    return result


# Canonical alias — communicates the polling/timeout behaviour
wait_and_click = click


def double_click(
    template_path: str,
    config: Optional[SearchConfig] = None,
    move_duration: float = 0.1,
    **kwargs,
) -> MatchResult:
    """Locate a UI element and double-click it."""
    cfg = _resolve_cfg(config)
    result, ox, oy = _find_with_offset(template_path, cfg, **kwargs)
    _double_click_xy(result.center[0] + ox, result.center[1] + oy,
                     move_duration=move_duration)
    return result


def right_click(
    template_path: str,
    config: Optional[SearchConfig] = None,
    move_duration: float = 0.1,
    **kwargs,
) -> MatchResult:
    """Locate a UI element and right-click it."""
    cfg = _resolve_cfg(config)
    result, ox, oy = _find_with_offset(template_path, cfg, **kwargs)
    _right_click_xy(result.center[0] + ox, result.center[1] + oy,
                    move_duration=move_duration)
    return result


def move_to(
    template_path: str,
    config: Optional[SearchConfig] = None,
    duration: float = 0.1,
    **kwargs,
) -> MatchResult:
    """Move the cursor over an element without clicking (hover)."""
    cfg = _resolve_cfg(config)
    result, ox, oy = _find_with_offset(template_path, cfg, **kwargs)
    _move_to_xy(result.center[0] + ox, result.center[1] + oy, duration=duration)
    return result


def type_into(
    template_path: str,
    text: str,
    config: Optional[SearchConfig] = None,
    clear_first: bool = False,
    interval: float = 0.05,
    **kwargs,
) -> MatchResult:
    """
    Click an element (e.g. a text field) and type text into it.

    Args:
        template_path: Template of the input field.
        text:          Text to type.
        config:        Optional SearchConfig.
        clear_first:   If True, select all (Ctrl+A) and delete before typing.
        interval:      Delay between keystrokes in seconds.
        **kwargs:      Forwarded to find().

    Returns:
        MatchResult from the click.
    """
    result = click(template_path, config=config, **kwargs)
    if clear_first:
        _hotkey("ctrl", "a")
        _press_key("delete")
    _type_text(text, interval=interval)
    return result


def click_offset(
    template_path: str,
    dx: int,
    dy: int,
    config: Optional[SearchConfig] = None,
    button: str = "left",
    move_duration: float = 0.1,
    **kwargs,
) -> MatchResult:
    """
    Click at an offset from the detected element centre.

    Useful when the clickable area is adjacent to the visual landmark
    (e.g. clicking the dropdown arrow next to a labelled field).

    Args:
        template_path: Template image.
        dx:            Horizontal pixel offset from element centre.
        dy:            Vertical pixel offset from element centre.
        config:        Optional SearchConfig.
        button:        Mouse button.
        move_duration: Cursor animation duration.
        **kwargs:      Forwarded to find().
    """
    cfg = _resolve_cfg(config)
    result, ox, oy = _find_with_offset(template_path, cfg, **kwargs)
    x = result.center[0] + dx + ox
    y = result.center[1] + dy + oy
    _click_xy(x, y, button=button, move_duration=move_duration)
    return result


def scroll_at(
    template_path: str,
    clicks: int = 3,
    direction: str = "down",
    config: Optional[SearchConfig] = None,
    **kwargs,
) -> MatchResult:
    """
    Locate an element and scroll the mouse wheel at its location.

    Args:
        template_path: Template of the scrollable area or element within it.
        clicks:        Scroll notches.  Positive = up/right.
        direction:     "up", "down", "left", "right".
        config:        Optional SearchConfig.
        **kwargs:      Forwarded to find().

    Returns:
        MatchResult of the located element.
    """
    cfg = _resolve_cfg(config)
    result, ox, oy = _find_with_offset(template_path, cfg, **kwargs)
    _scroll_xy(result.center[0] + ox, result.center[1] + oy,
                clicks=clicks, direction=direction)
    return result


def drag_to_element(
    source_path: str,
    target_path: str,
    config: Optional[SearchConfig] = None,
    duration: float = 0.5,
    button: str = "left",
    **kwargs,
) -> tuple[MatchResult, MatchResult]:
    """
    Drag from one detected element to another.

    Finds both elements in a single screenshot to ensure consistent
    coordinates, then performs the drag.

    Args:
        source_path: Template of the element to drag FROM.
        target_path: Template of the element to drag TO.
        config:      Optional SearchConfig.
        duration:    Drag animation duration in seconds.
        button:      Mouse button held during drag.
        **kwargs:    Forwarded to both find() calls.

    Returns:
        (source_result, target_result) — MatchResults for both elements.

    Raises:
        ElementNotFoundError: If either element cannot be located.
    """
    cfg = _resolve_cfg(config)
    src_result, ox, oy = _find_with_offset(source_path, cfg, **kwargs)
    tgt_result, _, _ = _find_with_offset(target_path, cfg, **kwargs)

    x1 = src_result.center[0] + ox
    y1 = src_result.center[1] + oy
    x2 = tgt_result.center[0] + ox
    y2 = tgt_result.center[1] + oy

    logger.info(
        f"Drag: '{source_path}' ({x1},{y1}) → '{target_path}' ({x2},{y2})"
    )
    _drag_xy(x1, y1, x2, y2, duration=duration, button=button)
    return src_result, tgt_result


# ---------------------------------------------------------------------------
# Waiting and polling
# ---------------------------------------------------------------------------

def wait_for(
    template_path: str,
    config: Optional[SearchConfig] = None,
    **kwargs,
) -> MatchResult:
    """
    Wait until a UI element appears, then return its location.
    Equivalent to find() — provided as a semantic alias.

    Example:
        result = vision.wait_for("assets/save_dialog.png", timeout=30.0)
    """
    return find(template_path=template_path, config=config, **kwargs)


def wait_cursor_normal(
    timeout: float = 30.0,
    poll_interval: float = 0.25,
) -> bool:
    """
    Block until the system cursor stops showing a loading/busy indicator.

    Detects IDC_WAIT (hourglass) and IDC_APPSTARTING (arrow + hourglass).
    On non-Windows platforms falls back to a short fixed sleep.

    Args:
        timeout:       Maximum seconds to wait before giving up.
        poll_interval: Seconds between cursor state checks.

    Returns:
        True  — cursor returned to normal within timeout.
        False — timeout elapsed while cursor was still busy.

    Example:
        vision.click("assets/run_report.png")
        if not vision.wait_cursor_normal(timeout=120):
            raise TimeoutError("Report took too long to generate")
        vision.click("assets/export.png")
    """
    return _wait_cursor_normal(timeout=timeout, poll_interval=poll_interval)


def wait_until_gone(
    template_path: str,
    config: Optional[SearchConfig] = None,
    timeout: float = 30.0,
    poll_interval: float = 0.5,
    **kwargs,
) -> bool:
    """
    Block until an element disappears from the screen.

    Useful for:
        • Waiting for a loading spinner to vanish
        • Confirming a modal dialog closed
        • Waiting for a progress bar to complete

    Args:
        template_path: Template of the element to wait for disappearance.
        config:        Optional SearchConfig.  timeout and poll_interval
                       from this config are overridden by the direct args.
        timeout:       Maximum seconds to wait.
        poll_interval: Seconds between visibility checks.
        **kwargs:      Forwarded to find().

    Returns:
        True  — element disappeared within timeout.
        False — element still visible after timeout.

    Example:
        vision.click("assets/start_upload.png")
        if not vision.wait_until_gone("assets/progress_bar.png", timeout=60):
            raise TimeoutError("Upload did not complete")
    """
    cfg = _resolve_cfg(config)
    # Use a short timeout for each existence check — we want a fast negative
    check_cfg = SearchConfig(
        monitor_index=cfg.monitor_index,
        threshold_template=cfg.threshold_template,
        threshold_ssim=cfg.threshold_ssim,
        min_akaze_inliers=cfg.min_akaze_inliers,
        scales=cfg.scales,
        timeout=poll_interval,       # Quick check — don't wait long per poll
        poll_interval=poll_interval,
    )

    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        visible = exists(template_path, config=check_cfg, **kwargs)
        if not visible:
            logger.info(f"wait_until_gone: '{template_path}' has disappeared.")
            return True
        remaining = deadline - time.monotonic()
        logger.debug(
            f"wait_until_gone: '{template_path}' still visible. "
            f"{remaining:.1f}s remaining."
        )
        time.sleep(poll_interval)

    logger.warning(
        f"wait_until_gone: '{template_path}' still visible after {timeout}s."
    )
    return False


def wait_screen_stable(
    timeout: float = 10.0,
    poll_interval: float = 0.3,
    stable_for: float = 0.6,
    diff_threshold: float = 0.01,
    monitor_index: int = 0,
) -> bool:
    """
    Block until the screen stops changing (animations and transitions finish).

    Compares consecutive screenshots and considers the screen "stable" when
    the fraction of changed pixels falls below `diff_threshold` and stays
    there for `stable_for` seconds.

    Args:
        timeout:        Maximum seconds to wait.
        poll_interval:  Seconds between consecutive screenshots.
        stable_for:     Seconds the screen must remain stable to confirm.
        diff_threshold: Maximum fraction of changed pixels (0–1) to be
                        considered stable.  0.01 = less than 1% of pixels
                        changed, which ignores blinking cursors / clock digits.
        monitor_index:  Monitor to watch.

    Returns:
        True  — screen stabilised within timeout.
        False — screen was still changing when timeout elapsed.

    Example:
        vision.click("assets/animate_button.png")
        vision.wait_screen_stable(stable_for=1.0)
        vision.click("assets/next.png")
    """
    deadline = time.monotonic() + timeout
    stable_since: Optional[float] = None
    prev_gray: Optional[np.ndarray] = None

    while time.monotonic() < deadline:
        screen = capture_screen(monitor_index)
        gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY).astype(np.float32)

        if prev_gray is not None:
            # Mean absolute difference normalised to [0, 1]
            diff = float(np.mean(np.abs(gray - prev_gray))) / 255.0
            is_stable = diff < diff_threshold
            logger.debug(
                f"wait_screen_stable: diff={diff:.4f} "
                f"({'stable' if is_stable else 'changing'})"
            )

            if is_stable:
                if stable_since is None:
                    stable_since = time.monotonic()
                elif time.monotonic() - stable_since >= stable_for:
                    logger.info(
                        f"wait_screen_stable: stable for {stable_for}s "
                        f"(diff={diff:.4f})."
                    )
                    return True
            else:
                stable_since = None   # Reset — screen changed again

        prev_gray = gray
        time.sleep(poll_interval)

    logger.warning(f"wait_screen_stable: timeout ({timeout}s) elapsed.")
    return False


def wait_any(
    template_paths: list[str],
    config: Optional[SearchConfig] = None,
    **kwargs,
) -> tuple[str, MatchResult]:
    """
    Wait until ANY of the given templates appears on screen.

    Runs the full pipeline independently for each template on every polling
    cycle.  Returns as soon as one match is confirmed.

    Args:
        template_paths: List of template image paths to search for.
        config:         Optional SearchConfig.
        **kwargs:       Forwarded to find() for each template.

    Returns:
        (matched_path, MatchResult) — which template matched and where.

    Raises:
        ValueError:           If template_paths is empty.
        ElementNotFoundError: If none of the templates appear within timeout.

    Example:
        # Handle either a success or error dialog appearing after an action
        path, result = vision.wait_any(
            ["assets/success_ok.png", "assets/error_close.png"],
            timeout=30.0,
        )
        if "error" in path:
            raise RuntimeError("Operation failed")
        vision.click(path)
    """
    if not template_paths:
        raise ValueError("wait_any: template_paths must not be empty.")

    cfg = _resolve_cfg(config)
    # Short per-template timeout so we cycle through all templates quickly
    check_timeout = cfg.poll_interval
    check_cfg = SearchConfig(
        monitor_index=cfg.monitor_index,
        threshold_template=cfg.threshold_template,
        threshold_ssim=cfg.threshold_ssim,
        min_akaze_inliers=cfg.min_akaze_inliers,
        scales=cfg.scales,
        timeout=check_timeout,
        poll_interval=check_timeout,
    )

    deadline = time.monotonic() + cfg.timeout

    while time.monotonic() < deadline:
        for path in template_paths:
            try:
                result = find_element(
                    template_path=path, config=check_cfg, **kwargs
                )
                logger.info(f"wait_any: matched '{path}' at {result.center}")
                return path, result
            except ElementNotFoundError:
                continue
        time.sleep(cfg.poll_interval)

    raise ElementNotFoundError(
        f"wait_any: none of {template_paths} appeared within {cfg.timeout}s.",
        template_path=str(template_paths),
        timeout=cfg.timeout,
    )


def wait_all(
    template_paths: list[str],
    config: Optional[SearchConfig] = None,
    **kwargs,
) -> dict[str, MatchResult]:
    """
    Wait until ALL of the given templates are simultaneously visible.

    Performs a full-screen scan on each poll cycle.  All templates must be
    present in the SAME screenshot (i.e. at the same moment) to succeed.

    Args:
        template_paths: List of template image paths — all must be visible.
        config:         Optional SearchConfig.
        **kwargs:       Forwarded to find() for each template.

    Returns:
        Dict mapping each template path to its MatchResult.

    Raises:
        ValueError:           If template_paths is empty.
        ElementNotFoundError: If not all templates appear within timeout.

    Example:
        results = vision.wait_all([
            "assets/username_field.png",
            "assets/password_field.png",
            "assets/login_button.png",
        ])
        # All three are now confirmed visible — safe to proceed
    """
    if not template_paths:
        raise ValueError("wait_all: template_paths must not be empty.")

    cfg = _resolve_cfg(config)
    check_cfg = SearchConfig(
        monitor_index=cfg.monitor_index,
        threshold_template=cfg.threshold_template,
        threshold_ssim=cfg.threshold_ssim,
        min_akaze_inliers=cfg.min_akaze_inliers,
        scales=cfg.scales,
        timeout=cfg.poll_interval,
        poll_interval=cfg.poll_interval,
    )

    deadline = time.monotonic() + cfg.timeout

    while time.monotonic() < deadline:
        found: dict[str, MatchResult] = {}
        for path in template_paths:
            try:
                found[path] = find_element(
                    template_path=path, config=check_cfg, **kwargs
                )
            except ElementNotFoundError:
                break   # At least one missing — don't bother checking the rest

        if len(found) == len(template_paths):
            logger.info(f"wait_all: all {len(template_paths)} templates found.")
            return found

        missing = [p for p in template_paths if p not in found]
        logger.debug(f"wait_all: still waiting for {missing}")
        time.sleep(cfg.poll_interval)

    raise ElementNotFoundError(
        f"wait_all: not all templates visible within {cfg.timeout}s. "
        f"Paths: {template_paths}",
        template_path=str(template_paths),
        timeout=cfg.timeout,
    )


# ---------------------------------------------------------------------------
# State queries
# ---------------------------------------------------------------------------

def exists(
    template_path: str,
    config: Optional[SearchConfig] = None,
    **kwargs,
) -> bool:
    """
    Return True if the element is currently visible on screen, False otherwise.

    Does NOT raise — always returns a boolean.  Useful for if-statements
    and conditional automation flows.

    Args:
        template_path: Template image path.
        config:        Optional SearchConfig.
        **kwargs:      Forwarded to find().

    Example:
        if vision.exists("assets/error_banner.png"):
            vision.click("assets/dismiss.png")
    """
    try:
        find(template_path=template_path, config=config, **kwargs)
        return True
    except ElementNotFoundError:
        return False


def count(
    template_path: str,
    config: Optional[SearchConfig] = None,
    max_count: int = 50,
    **kwargs,
) -> int:
    """
    Count the number of non-overlapping instances of a template on screen.

    Uses a suppression window equal to the template size so that adjacent
    matches from a repeated UI element (e.g. a list of items) are counted
    separately without double-counting overlapping detections.

    Args:
        template_path: Template image to search for.
        config:        Optional SearchConfig.
        max_count:     Safety cap — stops after finding this many instances.
        **kwargs:      Forwarded to find().

    Returns:
        Number of distinct matches found (0 if none).

    Example:
        n = vision.count("assets/checkbox_unchecked.png")
        print(f"{n} unchecked items remaining")
    """
    import cv2 as _cv2
    from .preprocessing import to_edges as _to_edges, resize_image as _resize_image
    from .pipeline import _load_template

    cfg = _resolve_cfg(config)
    template_bgr = _load_template(template_path)
    screen, ox, oy = capture_screen_with_offset(cfg.monitor_index)

    # Use edge-based matching consistent with Stage 1 of the pipeline
    screen_edges = _to_edges(screen)
    tmpl_edges = _to_edges(template_bgr)
    t_h, t_w = tmpl_edges.shape[:2]

    if t_h >= screen_edges.shape[0] or t_w >= screen_edges.shape[1]:
        logger.debug("count: template larger than screen — returning 0.")
        return 0

    result_map = _cv2.matchTemplate(screen_edges, tmpl_edges, _cv2.TM_CCOEFF_NORMED)

    # Suppress already-counted regions by zeroing out a window around each peak
    suppression_w = max(1, t_w // 2)
    suppression_h = max(1, t_h // 2)
    found = 0
    work = result_map.copy()

    for _ in range(max_count):
        _, max_val, _, max_loc = _cv2.minMaxLoc(work)
        if max_val < cfg.threshold_template:
            break
        found += 1
        # Zero out the suppression window around this peak
        x1 = max(0, max_loc[0] - suppression_w)
        y1 = max(0, max_loc[1] - suppression_h)
        x2 = min(work.shape[1], max_loc[0] + suppression_w)
        y2 = min(work.shape[0], max_loc[1] + suppression_h)
        work[y1:y2, x1:x2] = 0

    logger.info(f"count: found {found} instance(s) of '{template_path}'")
    return found


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------

def copy_from(
    template_path: str,
    config: Optional[SearchConfig] = None,
    select_all: bool = True,
    **kwargs,
) -> str:
    """
    Click an element, select its text content, copy to clipboard, and return it.

    Workflow: click → (Ctrl+A if select_all) → Ctrl+C → read clipboard.

    Args:
        template_path: Template of the element containing text.
        config:        Optional SearchConfig.
        select_all:    If True, press Ctrl+A before Ctrl+C to select all text.
                       Set to False for elements that already have a selection.
        **kwargs:      Forwarded to find().

    Returns:
        String contents of the clipboard after copying.

    Example:
        text = vision.copy_from("assets/result_field.png")
        print(f"Result: {text}")
    """
    click(template_path, config=config, **kwargs)
    if select_all:
        _hotkey("ctrl", "a")
    _hotkey("ctrl", "c")
    time.sleep(0.1)   # Give OS time to update clipboard
    return _clipboard_get()


def paste_into(
    template_path: str,
    text: str,
    config: Optional[SearchConfig] = None,
    clear_first: bool = True,
    **kwargs,
) -> MatchResult:
    """
    Place text on the clipboard, click an element, and paste with Ctrl+V.

    Preferred over type_into() for:
        • Long strings (much faster than simulated keystrokes)
        • Unicode / special characters that typewrite() cannot handle
        • Preserving exact formatting

    Args:
        template_path: Template of the target input field.
        text:          Text to paste.
        config:        Optional SearchConfig.
        clear_first:   If True, select all and delete before pasting.
        **kwargs:      Forwarded to find().

    Returns:
        MatchResult from the click.
    """
    _clipboard_set(text)
    result = click(template_path, config=config, **kwargs)
    if clear_first:
        _hotkey("ctrl", "a")
        _press_key("delete")
    _hotkey("ctrl", "v")
    return result


# ---------------------------------------------------------------------------
# Debug / diagnostics
# ---------------------------------------------------------------------------

def screenshot_of(
    template_path: str,
    save_path: str = "",
    padding: int = 20,
    config: Optional[SearchConfig] = None,
    **kwargs,
) -> tuple[MatchResult, np.ndarray]:
    """
    Locate an element and capture a cropped screenshot of the region around it.

    Saves the image to `save_path` if provided, otherwise returns the array
    without writing to disk.  Useful for building visual test reports or
    debugging false positives.

    Args:
        template_path: Template of the element to capture.
        save_path:     File path to write (PNG/JPEG).  Empty = no file written.
        padding:       Extra pixels around the detected element bounding box.
        config:        Optional SearchConfig.
        **kwargs:      Forwarded to find().

    Returns:
        (MatchResult, crop_bgr) — detection result and the cropped BGR array.

    Example:
        result, img = vision.screenshot_of(
            "assets/error_banner.png",
            save_path="debug/error_captured.png",
        )
    """
    from .pipeline import _load_template
    cfg = _resolve_cfg(config)
    result = find(template_path, config=cfg, **kwargs)
    screen, ox, oy = capture_screen_with_offset(cfg.monitor_index)

    # Estimate element size from template dimensions
    tmpl = _load_template(template_path)
    t_h, t_w = tmpl.shape[:2]

    cx, cy = result.center
    x1 = max(0, cx - t_w // 2 - padding)
    y1 = max(0, cy - t_h // 2 - padding)
    x2 = min(screen.shape[1], cx + (t_w - t_w // 2) + padding)
    y2 = min(screen.shape[0], cy + (t_h - t_h // 2) + padding)

    crop = screen[y1:y2, x1:x2]

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        cv2.imwrite(save_path, crop)
        logger.info(f"screenshot_of: saved region to '{save_path}'")

    return result, crop


def highlight(
    template_path: str,
    save_path: str = "debug/highlight.png",
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3,
    config: Optional[SearchConfig] = None,
    **kwargs,
) -> MatchResult:
    """
    Locate an element and save a full screenshot with a coloured rectangle
    drawn around it — a visual debug aid.

    This does NOT draw on screen in real time; it writes a PNG file that
    you can inspect after the fact.  For live highlighting, use
    `move_to()` to place the cursor on the element as a visual indicator.

    Args:
        template_path: Template of the element to highlight.
        save_path:     Output image file path (default: "debug/highlight.png").
        color:         BGR colour of the highlight rectangle (default: green).
        thickness:     Rectangle border thickness in pixels.
        config:        Optional SearchConfig.
        **kwargs:      Forwarded to find().

    Returns:
        MatchResult of the found element.

    Example:
        # Inspect what the pipeline is actually matching
        vision.highlight("assets/submit.png", save_path="debug/submit_box.png")
    """
    from .pipeline import _load_template
    cfg = _resolve_cfg(config)
    result = find(template_path, config=cfg, **kwargs)
    screen, _, _ = capture_screen_with_offset(cfg.monitor_index)

    tmpl = _load_template(template_path)
    t_h, t_w = tmpl.shape[:2]
    cx, cy = result.center

    x1 = max(0, cx - t_w // 2)
    y1 = max(0, cy - t_h // 2)
    x2 = min(screen.shape[1], x1 + t_w)
    y2 = min(screen.shape[0], y1 + t_h)

    annotated = screen.copy()
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

    # Add label above the box
    label = f"Stage {result.stage} | ssim={result.ssim_score:.2f}"
    cv2.putText(
        annotated, label,
        (x1, max(0, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, color, 1, cv2.LINE_AA,
    )

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    cv2.imwrite(save_path, annotated)
    logger.info(
        f"highlight: saved annotated screenshot to '{save_path}' "
        f"[box=({x1},{y1})→({x2},{y2})]"
    )
    return result