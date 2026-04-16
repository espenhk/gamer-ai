"""Tests for build_centerline.update_registry() (issue #14 multi-track support)."""
import os
import tempfile
import unittest
from pathlib import Path

import yaml

from games.tmnf.tools.build_centerline import update_registry


class TestUpdateRegistry(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.mkdtemp()
        self.registry_path = Path(self._tmp) / "registry.yaml"

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmp, ignore_errors=True)

    def test_creates_registry_when_absent(self):
        self.assertFalse(self.registry_path.exists())
        update_registry(self.registry_path, "a03", Path("tracks/a03_centerline.npy"), "replays/a03.Replay.Gbx")
        self.assertTrue(self.registry_path.exists())

    def test_entry_fields(self):
        update_registry(self.registry_path, "a03", Path("tracks/a03_centerline.npy"), "replays/a03.Replay.Gbx")
        data = yaml.safe_load(self.registry_path.read_text())
        entry = data["a03"]
        self.assertEqual(entry["centerline_path"], "tracks/a03_centerline.npy")
        self.assertEqual(entry["source_replay"], "replays/a03.Replay.Gbx")
        self.assertIn("default_par_time_s", entry)

    def test_upsert_overwrites_existing_track(self):
        update_registry(self.registry_path, "a03", Path("tracks/a03_old.npy"), "replays/old.Replay.Gbx")
        update_registry(self.registry_path, "a03", Path("tracks/a03_new.npy"), "replays/new.Replay.Gbx")
        data = yaml.safe_load(self.registry_path.read_text())
        self.assertEqual(data["a03"]["centerline_path"], "tracks/a03_new.npy")

    def test_upsert_preserves_other_tracks(self):
        update_registry(self.registry_path, "a03", Path("tracks/a03_centerline.npy"), "replays/a03.Replay.Gbx")
        update_registry(self.registry_path, "b05", Path("tracks/b05_centerline.npy"), "replays/b05.Replay.Gbx")
        data = yaml.safe_load(self.registry_path.read_text())
        self.assertIn("a03", data)
        self.assertIn("b05", data)
        self.assertEqual(data["b05"]["centerline_path"], "tracks/b05_centerline.npy")

    def test_multiple_tracks_sorted_keys(self):
        update_registry(self.registry_path, "z99", Path("tracks/z99.npy"), "replays/z99.Replay.Gbx")
        update_registry(self.registry_path, "a03", Path("tracks/a03.npy"), "replays/a03.Replay.Gbx")
        data = yaml.safe_load(self.registry_path.read_text())
        keys = list(data.keys())
        self.assertEqual(keys, sorted(keys))


if __name__ == "__main__":
    unittest.main(verbosity=2)
