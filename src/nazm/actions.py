"""
actions.py
Low-level mouse and keyboard actions via pyautogui.

All coordinates are absolute screen pixel positions (global space).
Higher-level compound actions (find + act) live in __init__.py.
"""

import logging

import pyautogui

logger = logging.getLogger(__name__)

pyautogui.FAILSAFE = True   # Move cursor to a screen corner to abort
pyautogui.PAUSE    = 0.05   # Brief inter-call pause — prevents OS queue floods

# ---------------------------------------------------------------------------
# Mouse — basic
# ---------------------------------------------------------------------------

def click(x: int, y: int, button: str = "left", move_duration: float = 0.1) -> None:
    """Move to (x, y) and perform a mouse click."""
    logger.info(f"Click [{button}] at ({x}, {y})")
    pyautogui.moveTo(x, y, duration=move_duration)
    pyautogui.click(button=button)


def double_click(x: int, y: int, move_duration: float = 0.1) -> None:
    """Move to (x, y) and double-click."""
    logger.info(f"Double-click at ({x}, {y})")
    pyautogui.moveTo(x, y, duration=move_duration)
    pyautogui.doubleClick()


def right_click(x: int, y: int, move_duration: float = 0.1) -> None:
    """Move to (x, y) and right-click."""
    click(x, y, button="right", move_duration=move_duration)


def move_to(x: int, y: int, duration: float = 0.1) -> None:
    """Move cursor to (x, y) without clicking."""
    logger.debug(f"Move cursor → ({x}, {y})")
    pyautogui.moveTo(x, y, duration=duration)


# ---------------------------------------------------------------------------
# Mouse — scroll
# ---------------------------------------------------------------------------

def scroll(
    x: int,
    y: int,
    clicks: int = 3,
    direction: str = "down",
) -> None:
    """
    Move to (x, y) and scroll the mouse wheel.

    Args:
        x:         Horizontal coordinate.
        y:         Vertical coordinate.
        clicks:    Number of scroll notches.  Positive = scroll up/right.
        direction: "up", "down", "left", "right".
                   "down" and "left" invert the clicks value automatically.
    """
    pyautogui.moveTo(x, y, duration=0.1)
    direction = direction.lower()

    if direction in ("down", "left"):
        clicks = -abs(clicks)
    else:
        clicks = abs(clicks)

    if direction in ("left", "right"):
        logger.info(f"Horizontal scroll {direction} ×{abs(clicks)} at ({x}, {y})")
        pyautogui.hscroll(clicks)
    else:
        logger.info(f"Scroll {direction} ×{abs(clicks)} at ({x}, {y})")
        pyautogui.scroll(clicks)


# ---------------------------------------------------------------------------
# Mouse — drag
# ---------------------------------------------------------------------------

def drag(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    duration: float = 0.4,
    button: str = "left",
) -> None:
    """
    Press and hold `button` at (x1, y1), drag to (x2, y2), then release.

    Args:
        x1, y1:  Drag start coordinates.
        x2, y2:  Drag end coordinates.
        duration: Total animation time in seconds (smooth drag).
        button:   Mouse button held during drag ("left", "right", "middle").
    """
    logger.info(f"Drag [{button}] ({x1},{y1}) → ({x2},{y2}) over {duration}s")
    pyautogui.moveTo(x1, y1, duration=0.1)
    pyautogui.dragTo(x2, y2, duration=duration, button=button)


def drag_relative(
    x: int,
    y: int,
    dx: int,
    dy: int,
    duration: float = 0.4,
    button: str = "left",
) -> None:
    """
    Drag from (x, y) by a relative offset (dx, dy).

    Useful for sliders, reordering list items, or panning a canvas.
    """
    drag(x, y, x + dx, y + dy, duration=duration, button=button)



# ---------------------------------------------------------------------------
# Keyboard — typing
# ---------------------------------------------------------------------------

def type_text(text: str, interval: float = 0.05) -> None:
    """
    Type a string character by character.

    Args:
        text:     String to type.  Printable ASCII only via typewrite;
                  for Unicode use `hotkey('ctrl','v')` + clipboard.
        interval: Delay between keystrokes in seconds.
    """
    logger.info(f"Type: {text!r}")
    pyautogui.write(text, interval=interval)


def press_key(key: str) -> None:
    """Press and release a single key (e.g. 'enter', 'tab', 'esc', 'f5')."""
    logger.info(f"Key: {key!r}")
    pyautogui.press(key)


def hotkey(*keys: str) -> None:
    """
    Press a key combination simultaneously.

    Examples:
        hotkey('ctrl', 'c')       # Copy
        hotkey('ctrl', 'shift', 's')  # Save-as
        hotkey('alt', 'f4')       # Close window
    """
    logger.info(f"Hotkey: {' + '.join(keys)}")
    pyautogui.hotkey(*keys)


def key_down(key: str) -> None:
    """
    Press and HOLD a key without releasing.
    Must be paired with key_up() to avoid the key remaining stuck.

    Typical use: holding Shift while clicking to multi-select.
    """
    logger.debug(f"Key down: {key!r}")
    pyautogui.keyDown(key)


def key_up(key: str) -> None:
    """Release a held key (counterpart to key_down)."""
    logger.debug(f"Key up: {key!r}")
    pyautogui.keyUp(key)


# ---------------------------------------------------------------------------
# Clipboard
# ---------------------------------------------------------------------------

def clipboard_set(text: str) -> None:
    """
    Place text onto the system clipboard.

    Uses pyautogui's internal hotcopy mechanism (platform-aware).
    Falls back to pyperclip if available.

    Args:
        text: String to place on clipboard.
    """
    try:
        import pyperclip  # Optional fast path
        pyperclip.copy(text)
        logger.debug(f"Clipboard set via pyperclip: {text!r}")
    except ImportError:
        # pyautogui fallback: write to a temp location then Ctrl+A / Ctrl+C
        # is fragile; just use Win32 directly on Windows as best effort.
        import sys
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.OpenClipboard(0)
            ctypes.windll.user32.EmptyClipboard()
            # Encode text as UTF-16LE for CF_UNICODETEXT
            data = (text + "\x00").encode("utf-16-le")
            hglob = ctypes.windll.kernel32.GlobalAlloc(0x0042, len(data))
            ptr = ctypes.windll.kernel32.GlobalLock(hglob)
            ctypes.memmove(ptr, data, len(data))
            ctypes.windll.kernel32.GlobalUnlock(hglob)
            ctypes.windll.user32.SetClipboardData(13, hglob)  # CF_UNICODETEXT=13
            ctypes.windll.user32.CloseClipboard()
            logger.debug(f"Clipboard set via Win32: {text!r}")
        else:
            logger.warning(
                "clipboard_set: pyperclip not installed and platform is not Windows. "
                "Install pyperclip for clipboard support: pip install pyperclip"
            )


def clipboard_get() -> str:
    """
    Retrieve the current text content of the system clipboard.

    Returns:
        Clipboard text, or empty string if clipboard is empty / non-text.
    """
    try:
        import pyperclip
        text = pyperclip.paste()
        logger.debug(f"Clipboard get via pyperclip: {text!r}")
        return text
    except ImportError:
        import sys
        if sys.platform == "win32":
            import ctypes
            ctypes.windll.user32.OpenClipboard(0)
            handle = ctypes.windll.user32.GetClipboardData(13)  # CF_UNICODETEXT
            if not handle:
                ctypes.windll.user32.CloseClipboard()
                return ""
            ptr = ctypes.windll.kernel32.GlobalLock(handle)
            text = ctypes.wstring_at(ptr)
            ctypes.windll.kernel32.GlobalUnlock(handle)
            ctypes.windll.user32.CloseClipboard()
            logger.debug(f"Clipboard get via Win32: {text!r}")
            return text
        logger.warning("clipboard_get: pyperclip not installed.")
        return ""
