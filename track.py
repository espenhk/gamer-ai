import math

import numpy as np
from scipy.spatial import KDTree

from constants import UP_VECTOR
from utils import Vec3


class Centerline:
    def __init__(self, path: str) -> None:
        self._points = np.load(path)  # (N, 3) float32
        diffs = np.diff(self._points, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        self._arc = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        self._total_length = self._arc[-1]
        # KDTree for O(log N) nearest-point queries instead of O(N) argmin.
        self._kdtree = KDTree(self._points)

    def project_with_forward(
        self, pos: Vec3, hint_idx: int | None = None, window: int = 50
    ) -> tuple[float, float, float, np.ndarray, int]:
        """
        Single O(N) scan returning all centerline quantities at once.

        Returns (progress, lateral_offset, vertical_offset, forward_dir, nearest_idx).

        prefer this over calling project() and forward_at() separately — they each
        did an independent O(N) nearest-point search.  Passing hint_idx (the index
        returned by the previous call) with a suitable window reduces the search to
        O(window) once the car is moving predictably.
        """
        p = np.array([pos.x, pos.y, pos.z], dtype=np.float64)

        if hint_idx is not None:
            n = len(self._points)
            hint_idx = max(0, min(n - 2, hint_idx))
            w = max(1, window)
            lo = max(0, hint_idx - w)
            hi = min(n - 2, hint_idx + w)
            if lo <= hi:
                local_dists = np.linalg.norm(self._points[lo:hi + 1] - p, axis=1)
                idx = lo + int(np.argmin(local_dists))
            else:
                dists = np.linalg.norm(self._points - p, axis=1)
                idx = int(np.argmin(dists))
        else:
            dists = np.linalg.norm(self._points - p, axis=1)
            idx = int(np.argmin(dists))

        idx = min(idx, len(self._points) - 2)

        a = self._points[idx].astype(np.float64)
        b = self._points[idx + 1].astype(np.float64)
        ab = b - a
        seg_len = float(np.linalg.norm(ab))
        t = float(np.dot(p - a, ab) / (seg_len ** 2)) if seg_len > 1e-9 else 0.0
        t = max(0.0, min(1.0, t))

        foot = a + t * ab
        progress = (self._arc[idx] + t * seg_len) / self._total_length

        offset = p - foot
        forward = ab / seg_len if seg_len > 1e-9 else np.array([1.0, 0.0, 0.0])

        # World up is Y in TMNF (imported from constants)
        right = np.cross(forward, UP_VECTOR)
        right_len = np.linalg.norm(right)
        if right_len > 1e-9:
            right /= right_len
        else:
            right = np.array([1.0, 0.0, 0.0])

        lateral_offset = float(np.dot(offset, right))
        vertical_offset = float(offset[1])

        return float(progress), lateral_offset, vertical_offset, forward, idx

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

        right0 = np.cross(fwd0, UP_VECTOR)
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

        # Heading change: signed angle between fwd0 and fwd1 (Y-up cross product)
        cos_a = float(np.clip(np.dot(fwd0, fwd1), -1.0, 1.0))
        cross = np.cross(fwd0, fwd1)
        sign = 1.0 if float(cross[1]) >= 0.0 else -1.0
        heading_change = sign * math.acos(cos_a)

        return lateral_offset, heading_change

    def project(self, pos: Vec3) -> tuple[float, float, float]:
        """Returns (progress, lateral_offset, vertical_offset). See project_with_forward()."""
        progress, lat, vert, _, _ = self.project_with_forward(pos)
        return progress, lat, vert

    def forward_at(self, pos: Vec3) -> np.ndarray:
        """Return the unit forward direction of the track at the closest point to pos."""
        _, _, _, fwd, _ = self.project_with_forward(pos)
        return fwd
