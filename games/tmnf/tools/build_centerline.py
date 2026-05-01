"""
Offline script: extract car positions from a TMNF ghost replay and save
an evenly-spaced centerline as a .npy file.

Usage:
    python -m games.tmnf.tools.build_centerline path/to/replay.Replay.Gbx
    python -m games.tmnf.tools.build_centerline path/to/replay.Replay.Gbx --output tracks/b05_centerline.npy --track-name b05 --spacing 2.0
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import yaml
from scipy.interpolate import splev, splprep

logger = logging.getLogger(__name__)


def extract_positions(gbx_path: str) -> np.ndarray:
    from pygbx import Gbx, GbxType
    g = Gbx(gbx_path)
    ghost = g.get_class_by_id(GbxType.CTN_GHOST)
    if ghost is None:
        raise ValueError(f"No CTN_GHOST found in {gbx_path!r}. Is this a ghost replay?")

    samples = ghost.records
    if not samples:
        raise ValueError("Ghost has no records entries.")

    positions = np.array(
        [[s.position.x, s.position.y, s.position.z] for s in samples],
        dtype=np.float32,
    )
    return positions


def resample_centerline(positions: np.ndarray, spacing: float) -> np.ndarray:
    # Remove duplicate consecutive points (splprep requires strictly increasing u)
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    keep = np.concatenate([[True], diffs > 1e-4])
    positions = positions[keep]

    # Compute cumulative arc length
    diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(diffs)])
    total_length = arc[-1]

    # Fit parametric spline
    u = arc / total_length
    tck, _ = splprep([positions[:, 0], positions[:, 1], positions[:, 2]], u=u, s=0, k=3)

    # Evaluate at evenly-spaced arc positions
    n_points = max(2, int(total_length / spacing) + 1)
    u_even = np.linspace(0.0, 1.0, n_points)
    coords = splev(u_even, tck)

    centerline = np.stack(coords, axis=1).astype(np.float32)
    return centerline


def update_registry(registry_path: Path, track_name: str, output_path: Path, replay_path: str) -> None:
    """Upsert an entry for *track_name* in *registry_path*."""

    registry = yaml.safe_load(registry_path.read_text()) if registry_path.exists() else {}
    registry[track_name] = {
        "centerline_path": output_path.as_posix(),
        "default_par_time_s": None,   # user fills in manually
        "source_replay": replay_path,
    }
    registry_path.write_text(yaml.dump(registry, sort_keys=True))
    logger.info("Updated registry %s  (track=%r)", registry_path, track_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a track centerline from a TMNF ghost replay.")
    parser.add_argument("replay", help="Path to .Replay.Gbx file")
    parser.add_argument("--output", default="runs/centerline.npy", help="Output .npy path (default: runs/centerline.npy)")
    parser.add_argument("--track-name", default=None,
                        help="Track identifier for tracks/registry.yaml (defaults to stem of --output)")
    parser.add_argument("--spacing", type=float, default=2.0, help="Point spacing in metres (default: 2.0)")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    output_path = Path(args.output)
    track_name = args.track_name or output_path.stem

    logger.info("Reading %r ...", args.replay)
    try:
        positions = extract_positions(args.replay)
    except Exception as e:
        logger.error("%s", e)
        raise SystemExit(1)

    logger.info("Extracted %d raw sample positions.", len(positions))

    centerline = resample_centerline(positions, args.spacing)
    logger.info("Resampled to %d points at ~%s m spacing.", len(centerline), args.spacing)
    logger.info("Total track length: %.1f m", np.linalg.norm(np.diff(centerline, axis=0), axis=1).sum())

    np.save(args.output, centerline)
    logger.info("Saved centerline to %r  shape=%s", args.output, centerline.shape)

    update_registry(Path("tracks/registry.yaml").as_posix(), track_name, output_path.as_posix(), args.replay)


if __name__ == "__main__":
    main()
