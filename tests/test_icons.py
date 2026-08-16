"""Every icon the app asks for must exist and actually render.

A missing Lucide glyph renders as a blank button rather than raising, which is
easy to ship without noticing — this test scans app.py for every icon name used
and checks each one produces visible pixels.
"""
import os
import re
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from captioning_kit import app as A

_APP_SOURCE = Path(A.__file__).read_text(encoding="utf-8")
_ICON_CALL = re.compile(r"""lucide_(?:icon|pixmap)\(\s*["']([a-z0-9-]+)["']""")


def _referenced_names() -> set[str]:
    return set(_ICON_CALL.findall(_APP_SOURCE))


class IconTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_referenced_icons_are_defined(self):
        missing = sorted(n for n in _referenced_names() if n not in A._LUCIDE_ICONS)
        self.assertEqual(missing, [], f"icon names used but not defined: {missing}")

    def test_referenced_icons_render_visible_pixels(self):
        blank = []
        for name in sorted(_referenced_names()):
            if name not in A._LUCIDE_ICONS:
                continue
            image = A.lucide_pixmap(name, "#FFFFFF", 18).toImage()
            visible = any(
                image.pixelColor(x, y).alpha() > 0
                for x in range(image.width())
                for y in range(image.height())
            )
            if not visible:
                blank.append(name)
        self.assertEqual(blank, [], f"icons render blank: {blank}")

    def test_play_glyph_is_filled_not_outlined(self):
        """The SVG template sets fill='none', so a filled shape must supply its own
        fill or it reads as a hollow outline in the transport bar."""
        image = A.lucide_pixmap("play", "#FFFFFF", 18).toImage()
        opaque = sum(
            1
            for x in range(image.width())
            for y in range(image.height())
            if image.pixelColor(x, y).alpha() > 200
        )
        self.assertGreater(opaque, 200)


if __name__ == "__main__":
    unittest.main()
