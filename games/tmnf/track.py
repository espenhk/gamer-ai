from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.spatial import KDTree

from games.tmnf.state import Vec3

# Default world up-vector (Y-up). Callers may override this via the
# constructor for coordinate systems where a different axis is up.
_DEFAULT_UP: np.ndarray = np.array([0.0, 1.0, 0.0])


@dataclass
class CenterlineProjection:
    """Result of projecting a position onto the centerline."""

    progress: float  # fraction of total track length completed, in [0, 1]
    lateral_offset: float  # metres to the right of the centreline (negative = left)
    vertical_offset: float  # metres above the centreline (negative = below)
    forward: np.ndarray  # unit forward direction of the track at the nearest point
    nearest_idx: int  # index of the nearest centerline point (for hint-based search)


class Centerline:
    def __init__(self, path: str, up_vector: np.ndarray | None = None) -> None:
        """Load a centerline from a .npy file of shape (N, 3).

        Args:
            path:      Path to the .npy file containing (N, 3) float32 waypoints.
            up_vector: World up-vector for the game's coordinate system.
                       Defaults to Y-up [0, 1, 0] which is correct for TMNF.
        """
        raw_up = up_vector if up_vector is not None else _DEFAULT_UP.copy()
        up_len = float(np.linalg.norm(raw_up))
        self._up = raw_up / up_len if up_len > 1e-9 else _DEFAULT_UP.copy()
        self._points = np.load(path)  # (N, 3) float32
        diffs = np.diff(self._points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self._arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self._total_length = self._arc[-1]
        # KDTree for O(log N) nearest-point queries used in the default path.
        self._kdtree = KDTree(self._points)

    def project_with_forward(self, pos: Vec3, hint_idx: int | None = None, window: int = 50) -> CenterlineProjection:
        """Return all centerline quantities for *pos* in one pass.

        Returns:
            CenterlineProjection with progress, lateral_offset, vertical_offset,
            forward, and nearest_idx.

        Prefer this over calling project() and forward_at() separately — they each
        did an independent nearest-point search.

        Default path (hint_idx=None): O(log N) via KDTree.
        Hint path (hint_idx provided): O(window) linear scan around the previous
            index, which is faster when the car is moving predictably along the track.
        """
        p = np.array([pos.x, pos.y, pos.z], dtype=np.float64)

        if hint_idx is not None:
            n = len(self._points)
            hint_idx = max(0, min(n - 2, hint_idx))
            w = max(1, window)
            lo = max(0, hint_idx - w)
            hi = min(n - 2, hint_idx + w)
            if lo <= hi:
                local_dists = np.linalg.norm(self._points[lo : hi + 1] - p, axis=1)
                idx = lo + int(np.argmin(local_dists))
            else:
                _, idx = self._kdtree.query(p)
                idx = int(idx)
        else:
            _, idx = self._kdtree.query(p)
            idx = int(idx)

        idx = min(idx, len(self._points) - 2)

        a = self._points[idx].astype(np.float64)
        b = self._points[idx + 1].astype(np.float64)
        ab = b - a
        seg_len = float(np.linalg.norm(ab))
        t = float(np.dot(p - a, ab) / (seg_len**2)) if seg_len > 1e-9 else 0.0
        t = max(0.0, min(1.0, t))

        foot = a + t * ab
        progress = (self._arc[idx] + t * seg_len) / self._total_length

        offset = p - foot
        forward = ab / seg_len if seg_len > 1e-9 else np.array([1.0, 0.0, 0.0])

        right = np.cross(forward, self._up)
        right_len = np.linalg.norm(right)
        if right_len > 1e-9:
            right /= right_len
        else:
            right = np.array([1.0, 0.0, 0.0])

        lateral_offset = float(np.dot(offset, right))
        vertical_offset = float(np.dot(offset, self._up))

        return CenterlineProjection(
            progress=float(progress),
            lateral_offset=lateral_offset,
            vertical_offset=vertical_offset,
            forward=forward,
            nearest_idx=idx,
        )

    def project_ahead(self, pos: Vec3, nearest_idx: int, steps: int) -> tuple[float, float]:
        """Return (lateral_offset, heading_change) for the waypoint *steps* ahead.

        lateral_offset: metres from the car's current position to the target
          waypoint, projected onto the right axis at *nearest_idx*.  Positive
          means the waypoint is to the right of the car.
        heading_change: signed angle (rad) between the track forward direction at
          *nearest_idx* and at *target_idx*, i.e. how much the track turns between
          here and the lookahead point.  Positive = right turn, negative = left turn.
        """
        n = len(self._points)
        target_idx = min(nearest_idx + steps, n - 2)

        # Forward and right vectors at the current nearest point
        a0 = self._points[nearest_idx].astype(np.float64)
        b0 = self._points[nearest_idx + 1].astype(np.float64)
        fwd0 = b0 - a0
        fwd0_len = float(np.linalg.norm(fwd0))
        fwd0 = fwd0 / fwd0_len if fwd0_len > 1e-9 else np.array([1.0, 0.0, 0.0])

        right0 = np.cross(fwd0, self._up)
        right0_len = float(np.linalg.norm(right0))
        right0 = right0 / right0_len if right0_len > 1e-9 else np.array([1.0, 0.0, 0.0])

        # Forward direction at the target point
        a1 = self._points[target_idx].astype(np.float64)
        b1 = self._points[target_idx + 1].astype(np.float64)
        fwd1 = b1 - a1
        fwd1_len = float(np.linalg.norm(fwd1))
        fwd1 = fwd1 / fwd1_len if fwd1_len > 1e-9 else np.array([1.0, 0.0, 0.0])

        # Lateral offset: how far the lookahead waypoint is to the right/left
        p = np.array([pos.x, pos.y, pos.z], dtype=np.float64)
        lateral_offset = float(np.dot(a1 - p, right0))

        # Heading change: signed angle between fwd0 and fwd1
        cos_a = float(np.clip(np.dot(fwd0, fwd1), -1.0, 1.0))
        cross = np.cross(fwd0, fwd1)
        sign = 1.0 if float(np.dot(cross, self._up)) >= 0.0 else -1.0
        heading_change = sign * math.acos(cos_a)

        return lateral_offset, heading_change

    def project(self, pos: Vec3) -> tuple[float, float, float]:
        """Returns (progress, lateral_offset, vertical_offset). See project_with_forward()."""
        proj = self.project_with_forward(pos)
        return proj.progress, proj.lateral_offset, proj.vertical_offset

    def forward_at(self, pos: Vec3) -> np.ndarray:
        """Return the unit forward direction of the track at the closest point to pos."""
        return self.project_with_forward(pos).forward
