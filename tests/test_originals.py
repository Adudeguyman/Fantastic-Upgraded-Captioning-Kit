import tempfile
import unittest
from pathlib import Path

from captioning_kit.store import CaptionStore


class OriginalBackupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.img = self.tmp / "photo.png"
        self.img.write_bytes(b"ORIGINAL")
        self.store = CaptionStore(self.tmp, ".json")

    def test_backup_creates_original_dir_copy(self):
        dst = self.store.backup_original(self.img)
        self.assertEqual(dst, self.tmp / ".original" / "photo.png")
        self.assertTrue(dst.exists())
        self.assertEqual(dst.read_bytes(), b"ORIGINAL")
        self.assertTrue(self.store.has_original_backup(self.img))

    def test_first_backup_wins(self):
        self.store.backup_original(self.img)
        self.img.write_bytes(b"EDIT1")
        self.store.backup_original(self.img)  # must NOT overwrite
        backup = self.store.original_backup_path(self.img)
        self.assertEqual(backup.read_bytes(), b"ORIGINAL")

    def test_restore_copies_back_and_keeps_backup(self):
        self.store.backup_original(self.img)
        self.img.write_bytes(b"EDITED")
        self.assertTrue(self.store.restore_original(self.img))
        self.assertEqual(self.img.read_bytes(), b"ORIGINAL")
        self.assertTrue(self.store.has_original_backup(self.img))

    def test_restore_without_backup_is_false(self):
        self.assertFalse(self.store.restore_original(self.img))

    def test_original_dir_excluded_from_images(self):
        (self.tmp / ".original").mkdir()
        (self.tmp / ".original" / "hidden.png").write_bytes(b"x")
        names = [p.name for p in self.store.images()]
        self.assertEqual(names, ["photo.png"])


if __name__ == "__main__":
    unittest.main()
