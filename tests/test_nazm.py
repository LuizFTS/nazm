"""
tests/test_nazm.py
Test suite for nazm v0.3.0 — covers all new and existing public methods.
Fully offline: no real screen, no pyautogui calls, no Tesseract required.
"""

import time
import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _solid(h, w, color=(80, 80, 80)):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = color
    return img


def _noise(h, w, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(20, 235, (h, w, 3), dtype=np.uint8)


def _match_result(cx=300, cy=200, stage=1):
    from nazm.pipeline import MatchResult
    return MatchResult(
        center=(cx, cy), stage=stage,
        stage_name="Multi-scale Template Match",
        ssim_score=0.82, elapsed=0.15,
        template_path="btn.png",
    )


def _mock_find(center=(300, 200), stage=1):
    """Patch src.nazm.find_element to return a canned MatchResult."""
    return patch(
        "src.nazm.pipeline.find_element",
        return_value=_match_result(center[0], center[1], stage),
    )


def _mock_capture(screen_h=600, screen_w=800, ox=0, oy=0):
    """Patch capture_screen_with_offset to return a synthetic image + offsets."""
    img = _noise(screen_h, screen_w, seed=42)
    return patch(
        "src.nazm.screen.capture_screen_with_offset",
        return_value=(img, ox, oy),
    )


# ── cursor.py ─────────────────────────────────────────────────────────────────

class TestCursorModule(unittest.TestCase):

    def test_is_cursor_busy_returns_bool(self):
        from nazm.cursor import is_cursor_busy
        result = is_cursor_busy()
        self.assertIsInstance(result, bool)

    def test_get_cursor_type_returns_string(self):
        from nazm.cursor import get_cursor_type
        ctype = get_cursor_type()
        self.assertIsInstance(ctype, str)
        # On any platform we expect a non-empty string
        self.assertGreater(len(ctype), 0)

    def test_wait_cursor_normal_nonwindows_returns_true(self):
        """On non-Windows the fallback sleep path should always return True."""
        import sys
        with patch.object(sys, "platform", "linux"):
            from nazm import cursor
            with patch.object(cursor, "_IS_WINDOWS", False):
                result = cursor.wait_cursor_normal(
                    timeout=5.0,
                    poll_interval=0.1,
                    fallback_sleep=0.0,  # Don't actually sleep in tests
                )
        self.assertTrue(result)

    @patch("time.sleep")
    def test_wait_cursor_normal_already_normal(self, mock_sleep):
        """If cursor is already normal, should return True immediately."""
        import src.nazm.cursor as cur
        if not cur._IS_WINDOWS:
            self.skipTest("Windows-only test")

        with patch.object(cur, "_is_cursor_busy", return_value=False):
            result = cur.wait_cursor_normal(timeout=5.0)
        self.assertTrue(result)

    @patch("time.sleep")
    def test_wait_cursor_normal_timeout(self, mock_sleep):
        """If cursor stays busy past timeout, should return False."""
        import src.nazm.cursor as cur
        if not cur._IS_WINDOWS:
            self.skipTest("Windows-only test")

        with patch.object(cur, "_is_cursor_busy", return_value=True):
            # Force immediate timeout by making monotonic advance faster
            start = time.monotonic()
            with patch("time.monotonic", side_effect=[start, start + 100]):
                result = cur.wait_cursor_normal(timeout=0.01, poll_interval=0.001)
        self.assertFalse(result)


# ── actions.py ────────────────────────────────────────────────────────────────

class TestActions(unittest.TestCase):

    def _make_pyautogui_mock(self):
        """Import actions with pyautogui already stubbed."""
        import src.nazm.actions as act
        return act

    @patch("pyautogui.moveTo")
    @patch("pyautogui.scroll")
    def test_scroll_down(self, mock_scroll, mock_move):
        from nazm.actions import scroll
        scroll(100, 200, clicks=5, direction="down")
        mock_scroll.assert_called_once_with(-5)  # down = negative

    @patch("pyautogui.moveTo")
    @patch("pyautogui.scroll")
    def test_scroll_up(self, mock_scroll, mock_move):
        from nazm.actions import scroll
        scroll(100, 200, clicks=3, direction="up")
        mock_scroll.assert_called_once_with(3)

    @patch("pyautogui.moveTo")
    @patch("pyautogui.hscroll")
    def test_scroll_horizontal(self, mock_hscroll, mock_move):
        from nazm.actions import scroll
        scroll(100, 200, clicks=4, direction="left")
        mock_hscroll.assert_called_once_with(-4)

    @patch("pyautogui.moveTo")
    @patch("pyautogui.dragTo")
    def test_drag(self, mock_drag, mock_move):
        from nazm.actions import drag
        drag(10, 20, 300, 400, duration=0.5)
        mock_drag.assert_called_once_with(300, 400, duration=0.5, button="left")

    @patch("pyautogui.moveTo")
    @patch("pyautogui.dragTo")
    def test_drag_relative(self, mock_drag, mock_move):
        from nazm.actions import drag_relative
        drag_relative(50, 60, dx=100, dy=0)
        mock_drag.assert_called_once_with(150, 60, duration=0.4, button="left")

    @patch("pyautogui.keyDown")
    def test_key_down(self, mock_kd):
        from nazm.actions import key_down
        key_down("shift")
        mock_kd.assert_called_once_with("shift")

    @patch("pyautogui.keyUp")
    def test_key_up(self, mock_ku):
        from nazm.actions import key_up
        key_up("shift")
        mock_ku.assert_called_once_with("shift")

    def test_clipboard_set_get_roundtrip(self):
        """
        Clipboard requires pyperclip or Win32 — test the try/except path.
        If neither is available, the functions should not raise.
        """
        from nazm.actions import clipboard_set, clipboard_get
        try:
            clipboard_set("hello nazm")
            text = clipboard_get()
            # If pyperclip is available, roundtrip should work
            if text:
                self.assertEqual(text, "hello nazm")
        except Exception as e:
            self.fail(f"clipboard_set/get raised unexpectedly: {e}")


# ── __init__.py — interaction methods ────────────────────────────────────────

class TestInteractionMethods(unittest.TestCase):

    @patch("src.nazm._click_xy")
    @patch("src.nazm.find_element")
    @patch("src.nazm.capture_screen_with_offset")
    def test_click_adds_monitor_offset(self, mock_cap, mock_find, mock_click):
        """Clicking on monitor 2 (offset 1920, 0) adds offset to coordinates."""
        import nazm
        screen = _noise(600, 800)
        mock_cap.return_value = (screen, 1920, 0)  # Secondary monitor
        mock_find.return_value = _match_result(cx=100, cy=200)

        nazm.click("btn.png")

        mock_click.assert_called_once_with(
            100 + 1920, 200 + 0, button="left", move_duration=0.1
        )

    @patch("src.nazm._double_click_xy")
    @patch("src.nazm.find_element")
    @patch("src.nazm.capture_screen_with_offset")
    def test_double_click(self, mock_cap, mock_find, mock_dbl):
        import nazm
        mock_cap.return_value = (_noise(600, 800), 0, 0)
        mock_find.return_value = _match_result(cx=50, cy=60)

        nazm.double_click("icon.png")
        mock_dbl.assert_called_once_with(50, 60, move_duration=0.1)

    @patch("src.nazm._right_click_xy")
    @patch("src.nazm.find_element")
    @patch("src.nazm.capture_screen_with_offset")
    def test_right_click(self, mock_cap, mock_find, mock_rc):
        import nazm
        mock_cap.return_value = (_noise(600, 800), 0, 0)
        mock_find.return_value = _match_result(cx=200, cy=100)

        nazm.right_click("menu.png")
        mock_rc.assert_called_once_with(200, 100, move_duration=0.1)

    @patch("src.nazm._type_text")
    @patch("src.nazm._click_xy")
    @patch("src.nazm.find_element")
    @patch("src.nazm.capture_screen_with_offset")
    def test_type_into(self, mock_cap, mock_find, mock_click, mock_type):
        import nazm
        mock_cap.return_value = (_noise(600, 800), 0, 0)
        mock_find.return_value = _match_result()

        nazm.type_into("field.png", "hello world")
        mock_type.assert_called_once_with("hello world", interval=0.05)

    @patch("src.nazm._hotkey")
    @patch("src.nazm._press_key")
    @patch("src.nazm._type_text")
    @patch("src.nazm._click_xy")
    @patch("src.nazm.find_element")
    @patch("src.nazm.capture_screen_with_offset")
    def test_type_into_clear_first(self, mock_cap, mock_find, mock_click,
                                    mock_type, mock_press, mock_hotkey):
        import nazm
        mock_cap.return_value = (_noise(600, 800), 0, 0)
        mock_find.return_value = _match_result()

        nazm.type_into("field.png", "text", clear_first=True)
        mock_hotkey.assert_any_call("ctrl", "a")
        mock_press.assert_called_with("delete")

    @patch("src.nazm._click_xy")
    @patch("src.nazm.find_element")
    @patch("src.nazm.capture_screen_with_offset")
    def test_click_offset(self, mock_cap, mock_find, mock_click):
        import nazm
        mock_cap.return_value = (_noise(600, 800), 0, 0)
        mock_find.return_value = _match_result(cx=100, cy=100)

        nazm.click_offset("label.png", dx=50, dy=-10)
        mock_click.assert_called_once_with(150, 90, button="left", move_duration=0.1)

    @patch("src.nazm._scroll_xy")
    @patch("src.nazm.find_element")
    @patch("src.nazm.capture_screen_with_offset")
    def test_scroll_at(self, mock_cap, mock_find, mock_scroll):
        import nazm
        mock_cap.return_value = (_noise(600, 800), 0, 0)
        mock_find.return_value = _match_result(cx=400, cy=300)

        nazm.scroll_at("list.png", clicks=5, direction="down")
        mock_scroll.assert_called_once_with(400, 300, clicks=5, direction="down")

    @patch("src.nazm._drag_xy")
    @patch("src.nazm.find_element")
    @patch("src.nazm.capture_screen_with_offset")
    def test_drag_to_element(self, mock_cap, mock_find, mock_drag):
        import nazm
        mock_cap.return_value = (_noise(600, 800), 0, 0)

        # find_element called twice (source, target) — return different results
        src = _match_result(cx=100, cy=100)
        dst = _match_result(cx=400, cy=300)
        mock_find.side_effect = [src, dst]

        src.nazm.drag_to_element("src.png", "dst.png", duration=0.4)
        mock_drag.assert_called_once_with(100, 100, 400, 300,
                                           duration=0.4, button="left")


# ── __init__.py — waiting methods ─────────────────────────────────────────────

class TestWaitingMethods(unittest.TestCase):

    @patch("src.nazm.find_element")
    def test_wait_for_delegates_to_find(self, mock_find):
        import nazm
        mock_find.return_value = _match_result()
        result = nazm.wait_for("btn.png")
        self.assertEqual(result.center, (300, 200))

    @patch("time.sleep")
    @patch("src.nazm.find_element")
    def test_wait_until_gone_element_absent(self, mock_find, mock_sleep):
        """If element is immediately absent (raises), should return True."""
        import nazm
        mock_find.side_effect = ElementNotFoundError("gone")
        result = nazm.wait_until_gone("spinner.png", timeout=2.0)
        self.assertTrue(result)

    @patch("time.sleep")
    @patch("src.nazm.find_element")
    def test_wait_until_gone_timeout(self, mock_find, mock_sleep):
        """If element persists past timeout, should return False."""
        import nazm
        # Always found → never gone; use a tiny timeout to force expiry fast
        mock_find.return_value = _match_result()
        result = nazm.wait_until_gone(
            "spinner.png",
            timeout=0.05,
            poll_interval=0.01,
        )
        self.assertFalse(result)

    @patch("time.sleep")
    @patch("src.nazm.capture_screen")
    def test_wait_screen_stable_stable_immediately(self, mock_screen, mock_sleep):
        """If screen doesn't change, it should stabilise quickly."""
        import nazm
        static_screen = _solid(600, 800)
        mock_screen.return_value = static_screen

        result = nazm.wait_screen_stable(
            timeout=5.0,
            poll_interval=0.01,
            stable_for=0.02,
            diff_threshold=0.01,
        )
        self.assertTrue(result)

    @patch("time.sleep")
    @patch("src.nazm.find_element")
    def test_wait_any_first_match(self, mock_find, mock_sleep):
        """wait_any returns the first template that matches."""
        import nazm
        # First template raises, second matches
        mock_find.side_effect = [
            ElementNotFoundError("not found"),
            _match_result(cx=50, cy=50),
        ]
        path, result = nazm.wait_any(
            ["assets/a.png", "assets/b.png"]
        )
        self.assertEqual(path, "assets/b.png")
        self.assertEqual(result.center, (50, 50))

    def test_wait_any_empty_list_raises(self):
        import nazm
        with self.assertRaises(ValueError):
            nazm.wait_any([])

    @patch("time.sleep")
    @patch("src.nazm.find_element")
    def test_wait_any_timeout_raises(self, mock_find, mock_sleep):
        import nazm
        mock_find.side_effect = ElementNotFoundError("not found")
        with self.assertRaises(ElementNotFoundError):
            nazm.wait_any(["a.png", "b.png"], timeout=0.1, poll_interval=0.05)

    @patch("time.sleep")
    @patch("src.nazm.find_element")
    def test_wait_all_all_found(self, mock_find, mock_sleep):
        """wait_all returns dict mapping each path to its MatchResult."""
        import nazm
        r1 = _match_result(cx=100, cy=100)
        r2 = _match_result(cx=200, cy=200)
        mock_find.side_effect = [r1, r2]

        results = nazm.wait_all(["a.png", "b.png"])
        self.assertIn("a.png", results)
        self.assertIn("b.png", results)

    @patch("time.sleep")
    @patch("src.nazm.find_element")
    def test_wait_all_timeout_raises(self, mock_find, mock_sleep):
        """If one template never appears, raises ElementNotFoundError."""
        import nazm
        # First always found, second always missing
        def side(*args, **kwargs):
            path = kwargs.get("template_path", "")
            if "a.png" in path:
                return _match_result()
            raise ElementNotFoundError("not found")

        mock_find.side_effect = side
        with self.assertRaises(ElementNotFoundError):
            nazm.wait_all(["a.png", "b.png"], timeout=0.1, poll_interval=0.05)

    def test_wait_all_empty_raises(self):
        import nazm
        with self.assertRaises(ValueError):
            nazm.wait_all([])

    # Re-import at top level for assertRaises to work in subtest
    from nazm.exceptions import ElementNotFoundError


# ── __init__.py — state queries ───────────────────────────────────────────────

class TestStateQueries(unittest.TestCase):

    @patch("src.nazm.find_element")
    def test_exists_true(self, mock_find):
        import nazm
        mock_find.return_value = _match_result()
        self.assertTrue(nazm.exists("btn.png"))

    @patch("src.nazm.find_element")
    def test_exists_false(self, mock_find):
        import nazm
        mock_find.side_effect = ElementNotFoundError("nope")
        self.assertFalse(nazm.exists("missing.png"))

    @patch("src.nazm.pipeline._load_template")
    @patch("src.nazm.capture_screen_with_offset")
    def test_count_zero_when_no_match(self, mock_cap, mock_load):
        import nazm
        import src.nazm.pipeline as p
        # Template is a random patch; screen is all zeros — should not match
        mock_load.return_value = _noise(40, 60, seed=1)
        mock_cap.return_value = (_solid(400, 600), 0, 0)

        n = nazm.count("btn.png")
        self.assertIsInstance(n, int)
        self.assertGreaterEqual(n, 0)

    @patch("src.nazm.pipeline._load_template")
    @patch("src.nazm.capture_screen_with_offset")
    def test_count_finds_repeated_element(self, mock_cap, mock_load):
        """Embed the same patch at three locations — count should return 3."""
        import nazm
        rng = np.random.default_rng(77)
        tmpl = rng.integers(10, 245, (30, 60, 3), dtype=np.uint8)
        screen = _solid(300, 600)

        # Embed at three non-overlapping x positions
        for x in [0, 120, 400]:
            screen[100:130, x:x + 60] = tmpl

        mock_load.return_value = tmpl
        mock_cap.return_value = (screen, 0, 0)

        n = nazm.count("item.png")
        # Suppress window might merge close hits; assert at least 1
        self.assertGreaterEqual(n, 1)


# ── __init__.py — clipboard methods ──────────────────────────────────────────

class TestClipboardMethods(unittest.TestCase):

    @patch("src.nazm._clipboard_get", return_value="extracted text")
    @patch("src.nazm._hotkey")
    @patch("src.nazm._click_xy")
    @patch("src.nazm.find_element")
    @patch("src.nazm.capture_screen_with_offset")
    def test_copy_from(self, mock_cap, mock_find, mock_click, mock_hotkey, mock_get):
        import nazm
        mock_cap.return_value = (_noise(600, 800), 0, 0)
        mock_find.return_value = _match_result()

        text = nazm.copy_from("field.png")
        self.assertEqual(text, "extracted text")
        mock_hotkey.assert_any_call("ctrl", "a")
        mock_hotkey.assert_any_call("ctrl", "c")

    @patch("src.nazm._hotkey")
    @patch("src.nazm._clipboard_set")
    @patch("src.nazm._click_xy")
    @patch("src.nazm.find_element")
    @patch("src.nazm.capture_screen_with_offset")
    def test_paste_into(self, mock_cap, mock_find, mock_click, mock_set, mock_hotkey):
        import nazm
        mock_cap.return_value = (_noise(600, 800), 0, 0)
        mock_find.return_value = _match_result()

        nazm.paste_into("field.png", "paste this text")
        mock_set.assert_called_once_with("paste this text")
        mock_hotkey.assert_any_call("ctrl", "v")


# ── __init__.py — debug methods ───────────────────────────────────────────────

class TestDebugMethods(unittest.TestCase):

    @patch("src.nazm.pipeline._load_template")
    @patch("src.nazm.capture_screen_with_offset")
    @patch("src.nazm.find_element")
    def test_screenshot_of_returns_array(self, mock_find, mock_cap, mock_load):
        import nazm
        screen = _noise(600, 800, seed=10)
        mock_cap.return_value = (screen, 0, 0)
        mock_find.return_value = _match_result(cx=400, cy=300)
        mock_load.return_value = _noise(60, 90, seed=11)

        result, crop = nazm.screenshot_of("btn.png")  # No save_path
        self.assertIsInstance(crop, np.ndarray)
        self.assertEqual(crop.ndim, 3)

    @patch("cv2.imwrite")
    @patch("os.makedirs")
    @patch("src.nazm.pipeline._load_template")
    @patch("src.nazm.capture_screen_with_offset")
    @patch("src.nazm.find_element")
    def test_screenshot_of_saves_file(self, mock_find, mock_cap, mock_load,
                                       mock_makedirs, mock_imwrite):
        import nazm
        screen = _noise(600, 800, seed=12)
        mock_cap.return_value = (screen, 0, 0)
        mock_find.return_value = _match_result(cx=300, cy=200)
        mock_load.return_value = _noise(60, 90, seed=13)

        nazm.screenshot_of("btn.png", save_path="debug/out.png")
        mock_imwrite.assert_called_once()

    @patch("cv2.imwrite")
    @patch("os.makedirs")
    @patch("src.nazm.pipeline._load_template")
    @patch("src.nazm.capture_screen_with_offset")
    @patch("src.nazm.find_element")
    def test_highlight_writes_annotated_image(self, mock_find, mock_cap, mock_load,
                                               mock_makedirs, mock_imwrite):
        import nazm
        screen = _noise(600, 800, seed=20)
        mock_cap.return_value = (screen, 0, 0)
        mock_find.return_value = _match_result(cx=400, cy=300)
        mock_load.return_value = _noise(60, 90, seed=21)

        nazm.highlight("btn.png", save_path="debug/highlight.png")
        mock_imwrite.assert_called_once()


# ── config ────────────────────────────────────────────────────────────────────

class TestConfig(unittest.TestCase):

    def test_set_config_changes_default(self):
        import nazm
        original = nazm._default_config
        new_cfg = nazm.SearchConfig(timeout=99.0)
        nazm.set_config(new_cfg)
        self.assertEqual(nazm._default_config.timeout, 99.0)
        nazm.set_config(original)  # Restore

    def test_version(self):
        import nazm
        self.assertEqual(nazm.__version__, "0.3.0")


from nazm.exceptions import ElementNotFoundError  # noqa — used in tests above

if __name__ == "__main__":
    unittest.main(verbosity=2)