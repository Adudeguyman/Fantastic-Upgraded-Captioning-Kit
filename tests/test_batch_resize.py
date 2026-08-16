"""Geometry tests for the batch-resize planner.

BatchResizePlan is deliberately pure (no I/O, no Qt widgets), so the sizing rules
that decide whether an image is rewritten can be tested directly.
"""
import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from captioning_kit.app import BatchResizePlan


class LongestShortestTests(unittest.TestCase):
    def test_longest_side_landscape_and_portrait(self):
        plan = BatchResizePlan("longest", value=1024)
        self.assertEqual(plan.target_for((4000, 3000)), (1024, 768))
        self.assertEqual(plan.target_for((3000, 4000)), (768, 1024))

    def test_shortest_side(self):
        plan = BatchResizePlan("shortest", value=512)
        self.assertEqual(plan.target_for((2000, 1000)), (1024, 512))

    def test_already_conforming_is_skipped(self):
        self.assertIsNone(BatchResizePlan("longest", value=1024).target_for((1024, 768)))

    def test_smaller_image_skipped_unless_upscale_allowed(self):
        self.assertIsNone(BatchResizePlan("longest", value=1024).target_for((800, 600)))
        up = BatchResizePlan("longest", value=1024, allow_upscale=True)
        self.assertEqual(up.target_for((800, 600)), (1024, 768))

    def test_aspect_ratio_is_preserved(self):
        w, h = BatchResizePlan("longest", value=1000).target_for((1600, 900))
        self.assertAlmostEqual(w / h, 1600 / 900, places=2)


class PercentTests(unittest.TestCase):
    def test_halving(self):
        self.assertEqual(BatchResizePlan("percent", percent=50).target_for((1000, 800)),
                         (500, 400))

    def test_no_op_at_100(self):
        self.assertIsNone(BatchResizePlan("percent", percent=100).target_for((1000, 800)))

    def test_upscale_blocked_by_default(self):
        self.assertIsNone(BatchResizePlan("percent", percent=150).target_for((1000, 800)))
        up = BatchResizePlan("percent", percent=150, allow_upscale=True)
        self.assertEqual(up.target_for((1000, 800)), (1500, 1200))


class ExactCropTests(unittest.TestCase):
    def setUp(self):
        self.plan = BatchResizePlan("exact", width=1024, height=1024)

    def test_target_is_the_exact_size(self):
        self.assertEqual(self.plan.target_for((4000, 3000)), (1024, 1024))

    def test_wide_source_is_side_cropped(self):
        self.assertEqual(self.plan.crop_box_for((4000, 3000), (1024, 1024)),
                         (500, 0, 3500, 3000))

    def test_tall_source_is_top_bottom_cropped(self):
        self.assertEqual(self.plan.crop_box_for((3000, 4000), (1024, 1024)),
                         (0, 500, 3000, 3500))

    def test_matching_aspect_needs_no_crop(self):
        self.assertIsNone(self.plan.crop_box_for((2048, 2048), (1024, 1024)))

    def test_other_modes_never_crop(self):
        plan = BatchResizePlan("longest", value=1024)
        self.assertIsNone(plan.crop_box_for((4000, 3000), (1024, 768)))


class SaveKwargsTests(unittest.TestCase):
    def test_lossy_formats_get_quality(self):
        plan = BatchResizePlan("longest")
        from pathlib import Path
        self.assertEqual(plan.save_kwargs(Path("a.jpg")), {"quality": 95})
        self.assertEqual(plan.save_kwargs(Path("a.webp")), {"quality": 95})
        self.assertEqual(plan.save_kwargs(Path("a.png")), {})


if __name__ == "__main__":
    unittest.main()
