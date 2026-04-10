"""
LidarSensor — LIDAR-style wall distance observations from live game screenshots.

Ported from TMAI's GameCapture.py (e:/GitHub/TMAI/tmai/env/utils/GameCapture.py).

Pipeline:
    MSS screenshot → grayscale → binary threshold → Canny edges → morphological
    dilation → Gaussian blur → binary → resize 128×128 → crop middle 32 rows
    → raycasting from bottom-centre → normalised wall distances

Output: np.ndarray of shape (n_rays,), dtype float32, values in ~[0, 1].
"""

from __future__ import annotations

import cv2
import numpy as np
import win32.win32gui as win32gui
from mss import mss

_WINDOW_TITLE_PREFIX = "TmForever"


def _find_game_hwnd() -> int:
    """Return the HWND of the TMNF/TMInterface window.

    Uses EnumWindows with a prefix match so the TMInterface version suffix
    (e.g. ' (TMInterface 1.1.1)') does not need to be known in advance.
    Raises RuntimeError if the window is not found.
    """
    found = []

    def _cb(hwnd: int, _: None) -> None:
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title.startswith(_WINDOW_TITLE_PREFIX):
                found.append(hwnd)

    win32gui.EnumWindows(_cb, None)
    if not found:
        raise RuntimeError(
            f"Could not find a window whose title starts with {_WINDOW_TITLE_PREFIX!r}. "
            "Is the game running?"
        )
    return found[0]


class LidarSensor:
    """
    Captures the game window and returns LIDAR-style wall distances.

    Parameters
    ----------
    n_rays:
        Number of rays to cast, spread evenly from 0 to π (left to right
        across the horizon of the car's view). Default 16.
    """

    def __init__(self, n_rays: int = 16) -> None:
        self.n_rays = n_rays
        self._sct = mss()
        self._hwnd = _find_game_hwnd()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_distances(self) -> np.ndarray:
        """Capture the current game frame and return wall distances.

        Returns
        -------
        np.ndarray
            Shape (n_rays,), dtype float32. Each value is the normalised
            distance to the nearest wall in that ray direction, ~[0, 1].
        """
        raw = self._capture()
        frame = self._process_screen(raw)
        return self._raycast(frame)

    # ------------------------------------------------------------------
    # Internal: capture
    # ------------------------------------------------------------------

    def _window_rect(self) -> tuple[int, int, int, int]:
        left, top, right, bottom = win32gui.GetWindowRect(self._hwnd)
        # Trim window chrome (title bar + borders)
        return left + 10, top + 40, right - 10, bottom - 10

    def _capture(self) -> np.ndarray:
        left, top, right, bottom = self._window_rect()
        region = {"left": left, "top": top, "width": right - left, "height": bottom - top}
        return cv2.cvtColor(np.array(self._sct.grab(region)), cv2.COLOR_RGBA2BGR)

    # ------------------------------------------------------------------
    # Internal: image processing
    # ------------------------------------------------------------------

    def _process_screen(self, screenshot: np.ndarray) -> np.ndarray:
        """Convert raw BGR screenshot to a 128×32 binary edge image."""
        img = cv2.resize(screenshot, (256, 256))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.threshold(img, 32, 255, cv2.THRESH_BINARY)[1]
        img = cv2.Canny(img, 100, 300)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        img = cv2.dilate(img, kernel, iterations=3)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)[1]
        img = cv2.resize(img, (128, 128))
        h = img.shape[0]
        # Crop to the middle horizontal strip — represents the horizon
        return img[h // 2: h // 2 + 32, :]  # shape: (32, 128)

    # ------------------------------------------------------------------
    # Internal: raycasting
    # ------------------------------------------------------------------

    def _find_contact(self, angle: float, frame: np.ndarray) -> tuple[int, int]:
        """Walk along *angle* from the bottom-centre until a white pixel is hit."""
        h, w = frame.shape
        dx = np.cos(angle)
        dy = np.sin(angle)
        cx = float(w // 2)
        cy = float(h - 1)
        while 0 <= int(cx) < w and 0 <= int(cy) < h and frame[int(cy)][int(cx)] == 0:
            cx += dx
            cy -= dy
        return int(cx), int(cy)

    def _raycast(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape
        origin = np.array([w // 2, h - 1], dtype=float)
        ref_size = np.hypot(h, w) / 2

        distances = np.empty(self.n_rays, dtype=np.float32)
        for i in range(self.n_rays):
            angle = i * np.pi / (self.n_rays - 1)
            cx, cy = self._find_contact(angle, frame)
            # Angle-dependent perspective correction: side rays appear shorter
            scale = (1.0 + 3.0 * np.sin(angle)) / 4.0
            distances[i] = scale * np.linalg.norm(np.array([cx, cy]) - origin) / ref_size

        return distances
