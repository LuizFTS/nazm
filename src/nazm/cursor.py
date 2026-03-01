"""
cursor.py
System cursor state detection — used to wait for loading indicators to clear.

Platform support
----------------
Windows:  Full support via ctypes / Win32 API.
          Detects IDC_WAIT (hourglass) and IDC_APPSTARTING (arrow + hourglass).
macOS:    Not supported — NSCursor API requires PyObjC which is out of scope.
          wait_cursor_normal() falls back to a fixed sleep.
Linux/X11: Not supported without python-xlib / wnck.
           wait_cursor_normal() falls back to a fixed sleep.

Design rationale
----------------
The Win32 approach loads the *system-defined* cursor handles for IDC_WAIT and
IDC_APPSTARTING and compares them against the *currently active* cursor handle
obtained via GetCursorInfo().  This works for standard Windows applications.

Custom application cursors (e.g. a browser's custom spinning wheel) will NOT
be detected because they do not use the system IDC_WAIT handle.  In that case,
use wait_screen_stable() from __init__.py instead, which detects cessation of
pixel changes regardless of cursor type.
"""

import logging
import sys
import time
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

_IS_WINDOWS = sys.platform == "win32"

# Windows system cursor IDs
_IDC_ARROW        = 32512   # Normal pointer
_IDC_IBEAM        = 32513   # Text insertion bar
_IDC_WAIT         = 32514   # Hourglass (application fully busy)
_IDC_CROSS        = 32515   # Crosshair
_IDC_SIZENWSE     = 32642   # Resize diagonal ↖↗
_IDC_SIZENESW     = 32643   # Resize diagonal ↗↙
_IDC_SIZEWE       = 32644   # Resize horizontal ↔
_IDC_SIZENS       = 32645   # Resize vertical ↕
_IDC_HAND         = 32649   # Pointer hand (links)
_IDC_APPSTARTING  = 32650   # Arrow + hourglass (background busy)
_IDC_HELP         = 32651   # Arrow + question mark

_CURSOR_ID_NAMES = {
    _IDC_ARROW:       "arrow",
    _IDC_IBEAM:       "ibeam",
    _IDC_WAIT:        "wait",
    _IDC_CROSS:       "cross",
    _IDC_HAND:        "hand",
    _IDC_APPSTARTING: "appstarting",
    _IDC_HELP:        "help",
    _IDC_SIZENWSE:    "sizenwse",
    _IDC_SIZENESW:    "sizenesw",
    _IDC_SIZEWE:      "sizewe",
    _IDC_SIZENS:      "sizens",
}

# Cursor IDs considered "busy" (loading)
_BUSY_CURSOR_IDS = {_IDC_WAIT, _IDC_APPSTARTING}


# ---------------------------------------------------------------------------
# Windows-specific internals
# ---------------------------------------------------------------------------

if _IS_WINDOWS:
    import ctypes
    import ctypes.wintypes

    class _POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    class _CURSORINFO(ctypes.Structure):
        _fields_ = [
            ("cbSize",      ctypes.c_uint32),
            ("flags",       ctypes.c_uint32),
            ("hCursor",     ctypes.c_void_p),
            ("ptScreenPos", _POINT),
        ]

    def _current_cursor_handle() -> int:
        """Return the Win32 handle of the cursor currently shown on screen."""
        ci = _CURSORINFO()
        ci.cbSize = ctypes.sizeof(_CURSORINFO)
        ctypes.windll.user32.GetCursorInfo(ctypes.byref(ci))
        return ci.hCursor or 0

    def _system_cursor_handle(cursor_id: int) -> int:
        """Return the Win32 handle for a system cursor by IDC_* constant."""
        return ctypes.windll.user32.LoadCursorW(None, cursor_id) or 0

    # Cache system handles at import time — they never change within a session
    _SYSTEM_HANDLES: dict[int, int] = {
        cid: _system_cursor_handle(cid) for cid in _CURSOR_ID_NAMES
    }
    _BUSY_HANDLES: set[int] = {
        _SYSTEM_HANDLES[cid] for cid in _BUSY_CURSOR_IDS if _SYSTEM_HANDLES.get(cid)
    }

    def _cursor_type_name() -> str:
        """
        Map the current cursor handle to a human-readable name.
        Returns "unknown" for application-defined cursors.
        """
        handle = _current_cursor_handle()
        for cid, h in _SYSTEM_HANDLES.items():
            if h and h == handle:
                return _CURSOR_ID_NAMES.get(cid, "unknown")
        return "unknown"

    def _is_cursor_busy() -> bool:
        """Return True if the current cursor is a wait/appstarting cursor."""
        handle = _current_cursor_handle()
        return handle in _BUSY_HANDLES


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_cursor_type() -> str:
    """
    Return a human-readable name for the current system cursor.

    Returns one of: "arrow", "ibeam", "wait", "cross", "hand",
    "appstarting", "help", "sizenwse", "sizenesw", "sizewe", "sizens",
    "unknown" (application-defined cursor), or "unsupported" (non-Windows).

    Returns:
        Cursor type string.
    """
    if not _IS_WINDOWS:
        logger.debug("get_cursor_type: platform not supported, returning 'unsupported'.")
        return "unsupported"
    return _cursor_type_name()


def is_cursor_busy() -> bool:
    """
    Return True if the cursor is currently showing a loading / busy indicator.

    Detects:
        IDC_WAIT        — full hourglass (app completely busy)
        IDC_APPSTARTING — arrow + hourglass (app loading in background)

    Returns False on non-Windows platforms (safe default for polling loops).

    Returns:
        bool: True = busy / loading, False = normal or unsupported platform.
    """
    if not _IS_WINDOWS:
        return False
    return _is_cursor_busy()


def wait_cursor_normal(
    timeout: float = 30.0,
    poll_interval: float = 0.25,
    fallback_sleep: float = 1.0,
) -> bool:
    """
    Block until the cursor stops showing a loading indicator.

    On Windows this actively polls the Win32 cursor handle.
    On other platforms it falls back to a fixed sleep of `fallback_sleep`
    seconds (since cursor state is unavailable without additional dependencies).

    Args:
        timeout:        Maximum seconds to wait (default 30).
        poll_interval:  Seconds between cursor state checks (default 0.25).
        fallback_sleep: Sleep duration on non-Windows platforms (default 1.0).

    Returns:
        True  — cursor returned to normal within timeout.
        False — timeout elapsed while cursor was still busy.

    Example:
        vision.click("assets/heavy_operation.png")
        if not vision.wait_cursor_normal(timeout=60):
            raise TimeoutError("Operation did not finish within 60 seconds")
        vision.click("assets/next_step.png")
    """
    if not _IS_WINDOWS:
        logger.debug(
            f"wait_cursor_normal: platform not Windows — "
            f"falling back to sleep({fallback_sleep}s)."
        )
        time.sleep(fallback_sleep)
        return True

    deadline = time.monotonic() + timeout
    was_busy = False

    while time.monotonic() < deadline:
        busy = _is_cursor_busy()
        if busy:
            was_busy = True
            ctype = _cursor_type_name()
            logger.debug(f"wait_cursor_normal: cursor is busy ({ctype}) — waiting…")
            time.sleep(poll_interval)
        else:
            if was_busy:
                logger.info(
                    f"wait_cursor_normal: cursor returned to normal "
                    f"({_cursor_type_name()})."
                )
            else:
                logger.debug(
                    "wait_cursor_normal: cursor was already normal — "
                    "returning immediately."
                )
            return True

    logger.warning(
        f"wait_cursor_normal: timeout ({timeout}s) elapsed — "
        f"cursor still busy ({_cursor_type_name()})."
    )
    return False