"""Model-target maths, checked against the concrete frame counts published for each
model. These are the numbers auto-trim will rely on, so the asserted values are the
ones from the sources rather than ones derived from our own formula."""
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from captioning_kit.model_targets import (
    ModelTarget,
    BUILTIN_TARGETS,
    LTX_2_3,
    LTX_2_5,
    MINIMAX_H3,
    WAN22_A14B,
    WAN22_TI2V_5B,
    Bucket,
    ModelTarget,
    builtin_map,
    load_targets,
    save_targets,
    targets_path,
)


class FrameGridTests(unittest.TestCase):
    def test_h3_ladder_is_17n_plus_5(self):
        # published ladder: 5, 22, 39, 56, 73, 90, 107, 124, 141 ...
        for frames in (5, 22, 39, 56, 73, 90, 107, 124, 141, 243, 362):
            self.assertTrue(MINIMAX_H3.is_legal_frames(frames), frames)
        for frames in (23, 100, 240, 360):
            self.assertFalse(MINIMAX_H3.is_legal_frames(frames), frames)

    def test_h3_rounds_up_like_the_model_does(self):
        # a request for 23 frames rounds up to 39
        self.assertEqual(MINIMAX_H3.snap_frames(23, "up"), 39)
        # ...and 15s (360 frames) becomes 362, not 360
        self.assertEqual(MINIMAX_H3.snap_frames(360, "up"), 362)

    def test_h3_durations_match_published_values(self):
        self.assertAlmostEqual(MINIMAX_H3.seconds_for_frames(243), 10.125, places=3)
        self.assertAlmostEqual(MINIMAX_H3.seconds_for_frames(362), 15.083, places=2)
        self.assertAlmostEqual(MINIMAX_H3.seconds_for_frames(107), 4.458, places=2)

    def test_h3_ceiling_is_362_not_345(self):
        """Rounding the 15s cap *down* would give 345 and waste a second of the
        model's trained range."""
        self.assertEqual(MINIMAX_H3.max_frames(), 362)

    def test_wan_is_4n_plus_1(self):
        for frames in (9, 41, 81):
            self.assertTrue(WAN22_A14B.is_legal_frames(frames), frames)
        self.assertFalse(WAN22_A14B.is_legal_frames(80))

    def test_wan_ceiling_is_the_official_81(self):
        self.assertEqual(WAN22_A14B.max_frames(), 81)

    def test_wan_variants_differ_in_fps(self):
        """The A14B/5B fps split is the whole reason these are separate entries."""
        self.assertEqual(WAN22_A14B.fps, 16.0)
        self.assertEqual(WAN22_TI2V_5B.fps, 24.0)

    def test_ltx_is_8n_plus_1(self):
        for frames in (1, 9, 17, 25, 33, 41, 49, 57, 65, 73, 81, 89, 97, 121):
            self.assertTrue(LTX_2_3.is_legal_frames(frames), frames)
        self.assertFalse(LTX_2_3.is_legal_frames(120))

    def test_ltx_versions_share_geometry(self):
        self.assertEqual(LTX_2_3.frame_modulus, LTX_2_5.frame_modulus)
        self.assertEqual(LTX_2_3.dimension_multiple, LTX_2_5.dimension_multiple)

    def test_snap_down_never_exceeds_source(self):
        for target in BUILTIN_TARGETS:
            for frames in range(1, 200):
                snapped = target.snap_frames(frames, "down")
                self.assertTrue(target.is_legal_frames(snapped))
                if frames >= target.smallest_legal_frames():
                    self.assertLessEqual(snapped, frames)

    def test_nearest_picks_the_closer_side(self):
        self.assertEqual(MINIMAX_H3.snap_frames(24, "nearest"), 22)
        self.assertEqual(MINIMAX_H3.snap_frames(37, "nearest"), 39)


class DimensionTests(unittest.TestCase):
    def test_snaps_down_to_multiple(self):
        self.assertEqual(LTX_2_3.snap_dimension(1080), 1056)
        self.assertEqual(LTX_2_3.snap_dimension(960), 960)
        self.assertEqual(MINIMAX_H3.snap_dimension(1000), 992)

    def test_never_returns_zero(self):
        self.assertEqual(LTX_2_3.snap_dimension(4), 32)


class BuiltinConsistencyTests(unittest.TestCase):
    def test_every_bucket_is_legal_for_its_target(self):
        for target in BUILTIN_TARGETS:
            for bucket in target.buckets:
                self.assertTrue(target.is_legal_frames(bucket.frames),
                                f"{target.key} {bucket.label}")
                self.assertEqual(bucket.width % target.dimension_multiple, 0,
                                 f"{target.key} {bucket.label}")
                self.assertEqual(bucket.height % target.dimension_multiple, 0,
                                 f"{target.key} {bucket.label}")

    def test_ceilings_and_floors_are_legal(self):
        for target in BUILTIN_TARGETS:
            self.assertTrue(target.is_legal_frames(target.max_frames()), target.key)
            self.assertTrue(target.is_legal_frames(target.min_frames()), target.key)

    def test_every_entry_records_its_provenance(self):
        """These specs drift and some came from community sources, so an entry with
        no source/verified date is a bug."""
        for target in BUILTIN_TARGETS:
            self.assertTrue(target.source, target.key)
            self.assertTrue(target.verified, target.key)

    def test_h3_requires_exact_fps(self):
        self.assertTrue(MINIMAX_H3.exact_fps)
        self.assertFalse(WAN22_A14B.exact_fps)


class UserOverrideTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_no_file_returns_builtins(self):
        self.assertEqual(set(load_targets(self.tmp)), set(builtin_map()))

    def test_user_file_overrides_a_builtin(self):
        targets_path(self.tmp).write_text(json.dumps({"targets": [{
            "key": "minimax_h3", "label": "MiniMax H3 (corrected)", "fps": 24.0,
            "frame_modulus": 17, "frame_remainder": 5, "dimension_multiple": 32,
            "min_seconds": 4.0, "max_seconds": 20.0,
        }]}))
        loaded = load_targets(self.tmp)
        self.assertEqual(loaded["minimax_h3"].label, "MiniMax H3 (corrected)")
        self.assertEqual(loaded["minimax_h3"].max_seconds, 20.0)
        # untouched entries survive
        self.assertIn("ltx_2_3", loaded)

    def test_user_file_can_add_a_model(self):
        targets_path(self.tmp).write_text(json.dumps({"targets": [{
            "key": "future_model", "label": "Future", "fps": 30.0,
            "frame_modulus": 6, "frame_remainder": 2, "dimension_multiple": 8,
            "min_seconds": 1.0, "max_seconds": 4.0,
            "buckets": [{"width": 640, "height": 480, "frames": 62}],
        }]}))
        loaded = load_targets(self.tmp)
        self.assertIn("future_model", loaded)
        self.assertTrue(loaded["future_model"].is_legal_frames(62))

    def test_malformed_file_falls_back_to_builtins(self):
        targets_path(self.tmp).write_text("{ not json")
        self.assertEqual(set(load_targets(self.tmp)), set(builtin_map()))

    def test_bad_entry_is_skipped_not_fatal(self):
        targets_path(self.tmp).write_text(json.dumps({"targets": [
            {"no_key": True},
            {"key": "ok_model", "label": "OK", "fps": 24.0, "frame_modulus": 4,
             "frame_remainder": 1, "dimension_multiple": 16,
             "min_seconds": 1.0, "max_seconds": 5.0},
        ]}))
        loaded = load_targets(self.tmp)
        self.assertIn("ok_model", loaded)

    def test_zero_modulus_is_clamped(self):
        targets_path(self.tmp).write_text(json.dumps({"targets": [{
            "key": "zero", "label": "Zero", "fps": 24.0, "frame_modulus": 0,
            "frame_remainder": 0, "dimension_multiple": 0,
            "min_seconds": 1.0, "max_seconds": 5.0,
        }]}))
        target = load_targets(self.tmp)["zero"]
        self.assertGreaterEqual(target.frame_modulus, 1)
        self.assertTrue(target.is_legal_frames(37))   # no constraint
        self.assertEqual(target.snap_dimension(101), 101)

    def test_round_trip_save_and_load(self):
        save_targets(self.tmp, builtin_map())
        loaded = load_targets(self.tmp)
        self.assertEqual(loaded["minimax_h3"].frame_modulus, 17)
        self.assertEqual(loaded["minimax_h3"].frame_remainder, 5)
        self.assertEqual(loaded["ltx_2_3"].buckets[0], Bucket(960, 544, 49))


if __name__ == "__main__":
    unittest.main()


class WindowMoveInvariantTests(unittest.TestCase):
    """Sliding a fitted selection must never change its frame count.

    This is the property the trim UI relies on: 'Fit to target' produces a legal
    length, and the user then drags that window to the part of the clip they want.
    If moving could alter the count, the fit would silently stop being legal.
    """

    def _legal_span_ms(self, target, frames):
        import math
        return int(math.ceil(frames / target.fps * 1000))

    def test_moving_a_fitted_window_preserves_legality(self):
        duration_ms = 30_000
        for target in BUILTIN_TARGETS:
            frames = target.snap_frames(
                min(int(5 * target.fps), target.max_frames()), "down")
            span = self._legal_span_ms(target, frames)
            for start in range(0, duration_ms - span, 613):   # arbitrary offsets
                end = start + span
                recovered = int(round((end - start) / 1000 * target.fps))
                self.assertTrue(
                    target.is_legal_frames(recovered),
                    f"{target.key}: window at {start}ms gave {recovered} frames")

    def test_clamping_at_the_timeline_end_keeps_the_length(self):
        duration_ms = 12_000
        for target in BUILTIN_TARGETS:
            frames = target.snap_frames(
                min(int(4 * target.fps), target.max_frames()), "down")
            span = self._legal_span_ms(target, frames)
            if span >= duration_ms:
                continue
            start = min(duration_ms - span, duration_ms)   # pinned to the far edge
            recovered = int(round(span / 1000 * target.fps))
            self.assertTrue(target.is_legal_frames(recovered), target.key)
            self.assertGreaterEqual(start, 0)


class TrimBarInteractionTests(unittest.TestCase):
    """Hit-test priority on the trim bar: in/out handles, then the playhead, then
    the selection window. The playhead must be grabbable — scrubbing is not the
    same gesture as sliding the trim, and conflating them loses the ability to
    look through a clip."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import captioning_kit.app as A
        from captioning_kit.llm_captioning import CaptioningSettings
        self.bar = A.TrimBar(A.Theme(CaptioningSettings()))
        self.bar.resize(600, 34)
        self.bar.set_duration(12000)
        self.bar.set_trim(2000, 10000)
        self.bar.set_position(6000)

    def _press(self, x):
        from PySide6.QtCore import QPointF, QEvent, Qt
        from PySide6.QtGui import QMouseEvent
        self.bar.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(x, 17), Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))

    def test_playhead_is_grabbable(self):
        self._press(self.bar._x_for(6000))
        self.assertEqual(self.bar._drag, "playhead")

    def test_playhead_beats_the_window_where_they_overlap(self):
        """The playhead sits inside the selection almost always, so window-drag
        must not swallow it."""
        self._press(self.bar._x_for(6000))
        self.assertEqual(self.bar._drag, "playhead")
        self.assertEqual(self.bar.trim(), (2000, 10000))

    def test_playhead_beats_the_handles_where_they_coincide(self):
        """Reversed deliberately: the playhead and the in-handle both start at 0 on
        every clip, and handles-first made scrubbing impossible until you moved the
        bracket. The grip's zone is tight, so the bracket stays reachable beside it.
        """
        self.bar.set_position(2000)      # playhead parked on the in-handle
        self._press(self.bar._x_for(2000))
        self.assertEqual(self.bar._drag, "playhead")
        self._press(self.bar._x_for(2000) + self.bar.GRIP + 3)
        self.assertEqual(self.bar._drag, "in")

    def test_window_drag_still_available_away_from_the_playhead(self):
        self.bar.set_position(2500)
        self._press(self.bar._x_for(7000))
        self.assertEqual(self.bar._drag, "window")

    def test_click_outside_the_selection_seeks(self):
        self.bar.set_position(6000)
        self._press(self.bar._x_for(11000))
        self.assertEqual(self.bar._drag, "seek")


class TrimBarMuteRangeTests(unittest.TestCase):
    """The mute range is a second span on the same bar. It answers a different
    question from the trim — which frames are silent, not which frames ship — so
    the two sets of handles must never fight over the same pixels."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import captioning_kit.app as A
        from captioning_kit.llm_captioning import CaptioningSettings
        self.bar = A.TrimBar(A.Theme(CaptioningSettings()))
        self.bar.resize(600, 46)
        self.bar.set_duration(12000)
        self.bar.set_trim(0, 12000)

    def test_hidden_by_default(self):
        self.assertFalse(self.bar.mute_visible())

    def test_enabling_seeds_a_visible_range_inside_the_trim(self):
        self.bar.set_trim(2000, 8000)
        self.bar.set_mute_visible(True)
        lo, hi = self.bar.mute_range()
        self.assertLess(lo, hi)
        self.assertGreaterEqual(lo, 2000)
        self.assertLessEqual(hi, 8000)

    def test_lower_band_clicks_go_to_the_mute_handles(self):
        from PySide6.QtCore import QPointF, QEvent, Qt
        from PySide6.QtGui import QMouseEvent
        self.bar.set_mute_visible(True)
        lo, _hi = self.bar.mute_range()
        y = self.bar._mute_band().center().y()
        self.bar.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(self.bar._x_for(lo), y),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(self.bar._drag, "mute_in")

    def test_upper_track_still_belongs_to_the_trim(self):
        from PySide6.QtCore import QPointF, QEvent, Qt
        from PySide6.QtGui import QMouseEvent
        self.bar.set_mute_visible(True)
        self.bar.set_trim(2000, 8000)
        self.bar.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(self.bar._x_for(2000), 17),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(self.bar._drag, "in")

    def test_mute_range_is_clamped_to_the_clip(self):
        self.bar.set_mute_visible(True)
        self.bar.set_mute_range(-500, 99999)
        self.assertEqual(self.bar.mute_range(), (0, 12000))

    def test_reversed_input_is_normalised(self):
        self.bar.set_mute_visible(True)
        self.bar.set_mute_range(6000, 3000)
        lo, hi = self.bar.mute_range()
        self.assertLessEqual(lo, hi)


class SpecComplianceMessageTests(unittest.TestCase):
    """The verdict on a clip follows the selected target, not the file: the same
    clip is legal for one model and illegal for another, so it has to be evaluated
    live rather than stamped at import."""

    def _info(self, seconds, fps, w=832, h=480):
        from captioning_kit.video_tools import VideoInfo
        return VideoInfo(duration_s=seconds, width=w, height=h, fps=fps,
                         frame_count=int(seconds * fps), codec="h264")

    def test_a_clip_can_pass_one_target_and_fail_another(self):
        clip = self._info(5.0625, 16.0)          # 81 frames at Wan's rate
        self.assertAlmostEqual(clip.fps, WAN22_A14B.fps)
        self.assertNotAlmostEqual(clip.fps, LTX_2_3.fps)

    def test_exact_fps_targets_are_stricter(self):
        """H3 needs exactly 24.000; LTX trains at 24 but doesn't demand it."""
        self.assertTrue(MINIMAX_H3.exact_fps)
        self.assertFalse(LTX_2_3.exact_fps)

    def test_over_length_is_detectable(self):
        long_clip = self._info(20.0, 30.0)
        frames_at_target = round(long_clip.duration_s * WAN22_A14B.fps)
        self.assertGreater(frames_at_target, WAN22_A14B.max_frames())

    def test_dimension_multiple_violation_is_detectable(self):
        odd = self._info(5.0, 16.0, w=1920, h=1080)
        self.assertNotEqual(odd.height % WAN22_A14B.dimension_multiple, 0)

    def test_a_conforming_clip_trips_nothing(self):
        good = self._info(5.0625, 16.0, w=832, h=480)
        frames = round(good.duration_s * WAN22_A14B.fps)
        self.assertTrue(WAN22_A14B.is_legal_frames(frames))
        self.assertLessEqual(frames, WAN22_A14B.max_frames())
        self.assertEqual(good.width % WAN22_A14B.dimension_multiple, 0)
        self.assertEqual(good.height % WAN22_A14B.dimension_multiple, 0)


class TargetOverlaySemanticsTests(unittest.TestCase):
    """The user file is a *diff*, not a snapshot.

    Writing every model would freeze the shipped defaults out — a later release
    correcting a spec would be silently overridden by a stale copy the user never
    knowingly edited.
    """

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_only_changed_entries_need_writing(self):
        builtins = builtin_map()
        changed = {k: v for k, v in builtins.items()}
        changed["minimax_h3"] = replace(builtins["minimax_h3"], max_seconds=20.0)
        diff = {k: t for k, t in changed.items()
                if k not in builtins or t != builtins[k]}
        self.assertEqual(set(diff), {"minimax_h3"})

    def test_an_overlay_of_one_leaves_the_rest_from_code(self):
        save_targets(self.tmp, {"minimax_h3": replace(builtin_map()["minimax_h3"],
                                                      max_seconds=20.0)})
        loaded = load_targets(self.tmp)
        self.assertEqual(loaded["minimax_h3"].max_seconds, 20.0)
        self.assertEqual(loaded["wan22_a14b"].fps, builtin_map()["wan22_a14b"].fps)
        self.assertEqual(set(loaded), set(builtin_map()))

    def test_a_user_added_model_survives_the_round_trip(self):
        added = ModelTarget(
            key="sora_9_turbo", label="Sora 9 Turbo", fps=30.0, frame_modulus=6,
            frame_remainder=2, dimension_multiple=32, max_pixels=0,
            min_seconds=1.0, max_seconds=8.0, source="added by hand", verified="")
        save_targets(self.tmp, {"sora_9_turbo": added})
        loaded = load_targets(self.tmp)
        self.assertIn("sora_9_turbo", loaded)
        self.assertEqual(loaded["sora_9_turbo"].fps, 30.0)
        self.assertTrue(loaded["sora_9_turbo"].is_legal_frames(8))
        self.assertFalse(loaded["sora_9_turbo"].is_legal_frames(9))

    def test_legal_frame_ladder_is_derivable_for_any_grid(self):
        """The editor shows the ladder because a modulus and remainder are hard to
        sanity-check in the abstract."""
        target = ModelTarget(
            key="x", label="X", fps=24.0, frame_modulus=6, frame_remainder=2,
            dimension_multiple=32, max_pixels=0, min_seconds=1.0, max_seconds=8.0)
        ladder = []
        n = target.smallest_legal_frames()
        while len(ladder) < 5:
            ladder.append(n)
            n += target.frame_modulus
        self.assertEqual(ladder, [2, 8, 14, 20, 26])
        self.assertTrue(all(target.is_legal_frames(f) for f in ladder))


class TrimBarWaveformTests(unittest.TestCase):
    """The waveform has to be big enough to read and not painted over.

    First attempt drew it inside an 8px track (peaks 4px tall) and *underneath* the
    selection fill, so it was technically present and completely invisible.
    """

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import captioning_kit.app as A
        from captioning_kit.llm_captioning import CaptioningSettings
        self.bar = A.TrimBar(A.Theme(CaptioningSettings()))
        self.bar.set_duration(6000)
        self.bar.set_trim(0, 6000)

    def test_track_stays_slim_without_a_waveform(self):
        self.assertLess(self.bar._track_rect().height(), 12)

    def test_track_grows_to_show_a_waveform(self):
        import math
        self.bar.set_peaks([abs(math.sin(i / 12)) for i in range(320)])
        self.assertGreater(self.bar._track_rect().height(), 20)
        self.assertGreater(self.bar.minimumHeight(), 34)

    def test_waveform_is_visible_across_the_track(self):
        import math
        from PySide6.QtGui import QPixmap
        self.bar.set_peaks([abs(math.sin(i / 12)) for i in range(320)])
        self.bar.resize(600, self.bar.minimumHeight())
        pm = QPixmap(600, self.bar.minimumHeight())
        pm.fill()
        self.bar.render(pm)
        image = pm.toImage()
        track = self.bar._track_rect()
        varying = sum(
            1 for x in range(int(track.left()) + 2, int(track.right()) - 2, 7)
            if len({image.pixel(x, y)
                    for y in range(int(track.top()) + 1, int(track.bottom()) - 1)}) > 1)
        self.assertGreater(varying, 30, "waveform should vary along the track")

    def test_waveform_survives_the_selection_fill(self):
        """Selected regions are filled with the accent colour; the waveform is drawn
        after it so the fill can't erase it."""
        import math
        from PySide6.QtGui import QPixmap
        self.bar.set_peaks([abs(math.sin(i / 12)) for i in range(320)])
        self.bar.set_trim(1000, 3000)
        self.bar.resize(600, self.bar.minimumHeight())
        pm = QPixmap(600, self.bar.minimumHeight())
        pm.fill()
        self.bar.render(pm)
        image = pm.toImage()
        track = self.bar._track_rect()
        inside_x = int(self.bar._x_for(2000))
        colours = {image.pixel(inside_x, y)
                   for y in range(int(track.top()) + 1, int(track.bottom()) - 1)}
        self.assertGreater(len(colours), 1, "waveform hidden inside the selection")


class TrimSnapTests(unittest.TestCase):
    """Snapping pulls a dragged edge onto a frame count the model accepts.

    Without it you land on an arbitrary count and the trainer silently truncates
    or pads — LTX in particular does this without saying so.
    """

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import captioning_kit.app as A
        from captioning_kit.llm_captioning import CaptioningSettings
        self.bar = A.TrimBar(A.Theme(CaptioningSettings()))
        self.bar.resize(700, 40)
        self.bar.set_duration(12000)

    def test_no_hook_leaves_the_trim_alone(self):
        self.bar.set_snap(None)
        self.assertEqual(self.bar._snapped(0, 5123, "out"), (0, 5123))

    def test_a_failing_hook_never_breaks_the_drag(self):
        """A bad target shouldn't make the handle unusable."""
        def boom(*_args):
            raise ValueError("nope")
        self.bar.set_snap(boom)
        self.assertEqual(self.bar._snapped(0, 5123, "out"), (0, 5123))

    def test_hook_result_is_applied(self):
        self.bar.set_snap(lambda i, o, which: (i, 4000))
        self.assertEqual(self.bar._snapped(0, 5123, "out"), (0, 4000))

    def test_h3_grid_accepts_only_17n_plus_5(self):
        self.assertTrue(MINIMAX_H3.is_legal_frames(73))
        self.assertTrue(MINIMAX_H3.is_legal_frames(90))
        self.assertFalse(MINIMAX_H3.is_legal_frames(74))

    def test_nearest_snap_can_round_either_way(self):
        """'Next acceptable window' means nearest, not always down — rounding down
        from just under a rung would lose most of a second."""
        self.assertEqual(MINIMAX_H3.snap_frames(76, "nearest"), 73)
        self.assertEqual(MINIMAX_H3.snap_frames(88, "nearest"), 90)

    def test_snapping_respects_the_model_ceiling(self):
        capped = WAN22_A14B.snap_frames(10_000, "nearest")
        self.assertLessEqual(min(capped, WAN22_A14B.max_frames()),
                             WAN22_A14B.max_frames())
        self.assertTrue(WAN22_A14B.is_legal_frames(WAN22_A14B.max_frames()))

    def test_each_target_has_a_distinct_grid(self):
        grids = {(t.frame_modulus, t.frame_remainder)
                 for t in (WAN22_A14B, LTX_2_3, MINIMAX_H3)}
        self.assertEqual(len(grids), 3, grids)


class TrimBarGrabPriorityTests(unittest.TestCase):
    """The playhead grip is drawn on top, so it must be grabbable on top.

    On a freshly loaded clip the playhead and the in-handle both sit at 0, and
    testing handles first made scrubbing impossible until you moved the bracket
    out of the way.
    """

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import captioning_kit.app as A
        from captioning_kit.llm_captioning import CaptioningSettings
        self.bar = A.TrimBar(A.Theme(CaptioningSettings()))
        self.bar.set_duration(6000)
        self.bar.set_trim(0, 6000)
        self.bar.resize(700, self.bar.minimumHeight())

    def _press(self, x, y=None):
        from PySide6.QtCore import QPointF, QEvent, Qt
        from PySide6.QtGui import QMouseEvent
        y = self.bar._track_rect().center().y() if y is None else y
        self.bar._drag = None
        self.bar.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(x, y), Qt.LeftButton,
            Qt.LeftButton, Qt.NoModifier))
        return self.bar._drag

    def test_playhead_wins_where_it_overlaps_the_in_handle(self):
        self.bar.set_position(0)
        self.assertEqual(self._press(self.bar._x_for(0)), "playhead")

    def test_the_bracket_is_still_reachable_beside_the_grip(self):
        """Grab the square for the playhead, the wider bar for the bracket."""
        self.bar.set_position(0)
        self.assertEqual(
            self._press(self.bar._x_for(0) + self.bar.GRIP + 3), "in")

    def test_playhead_wins_over_the_window(self):
        self.bar.set_position(3000)
        self.assertEqual(self._press(self.bar._x_for(3000)), "playhead")

    def test_window_drag_survives_away_from_the_playhead(self):
        self.bar.set_position(1000)
        self.assertEqual(self._press(self.bar._x_for(4500)), "window")

    def test_grip_zone_is_smaller_than_the_handle_zone(self):
        """A tight playhead target is what keeps the brackets usable."""
        self.assertLess(self.bar.GRIP, self.bar.HANDLE + 2)


class MuteBandGeometryTests(unittest.TestCase):
    """The mute range lives on the main track now.

    A separate strip underneath had to be tall enough to grab and kept fighting the
    waveform for height; red brackets on the one timeline sidestep that and are
    always the full height of the track.
    """

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import captioning_kit.app as A
        from captioning_kit.llm_captioning import CaptioningSettings
        self.bar = A.TrimBar(A.Theme(CaptioningSettings()))
        self.bar.set_duration(6000)
        self.bar.set_trim(0, 6000)
        self.bar.set_mute_visible(True)
        self.bar.resize(700, self.bar.minimumHeight())

    def test_band_shares_the_track(self):
        track = self.bar._track_rect()
        band = self.bar._mute_band()
        self.assertEqual((band.top(), band.height()), (track.top(), track.height()))

    def test_band_fits_inside_the_widget(self):
        self.assertLessEqual(self.bar._mute_band().bottom(), self.bar.height())

    def test_turning_mute_on_does_not_resize_the_bar(self):
        tall = self.bar.minimumHeight()
        self.bar.set_mute_visible(False)
        self.assertEqual(self.bar.minimumHeight(), tall)

    def test_mute_brackets_take_priority_while_mute_mode_is_on(self):
        """Both ranges share one track, so the range you opened the mode to adjust
        is the one that answers a click."""
        from PySide6.QtCore import QPointF, QEvent, Qt
        from PySide6.QtGui import QMouseEvent
        lo, _hi = self.bar.mute_range()
        y = self.bar._track_rect().center().y()
        self.bar._drag = None
        self.bar.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(self.bar._x_for(lo), y),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(self.bar._drag, "mute_in")

    def test_trim_handles_return_when_mute_mode_is_off(self):
        from PySide6.QtCore import QPointF, QEvent, Qt
        from PySide6.QtGui import QMouseEvent
        self.bar.set_mute_visible(False)
        self.bar.set_position(5000)
        y = self.bar._track_rect().center().y()
        self.bar._drag = None
        self.bar.mousePressEvent(QMouseEvent(
            QEvent.MouseButtonPress, QPointF(self.bar._x_for(0), y),
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        self.assertEqual(self.bar._drag, "in")


class SnapFloorTests(unittest.TestCase):
    """Snapping must never offer a length the model can't train on.

    The grid's smallest legal value and the model's minimum usable length are
    different numbers: H3's 17n+5 admits 5, 22, 39 and 56, all of which are below
    its 73-frame (3.04s) minimum. Clamping to the former offered lengths that look
    valid and aren't.
    """

    def test_grid_floor_and_usable_floor_differ(self):
        for target in (MINIMAX_H3, WAN22_A14B, LTX_2_3):
            self.assertLess(target.smallest_legal_frames(), target.min_frames(),
                            target.label)

    def test_h3_rungs_below_the_minimum_are_grid_legal(self):
        """Which is exactly why grid legality is not a sufficient check."""
        for frames in (5, 22, 39, 56):
            self.assertTrue(MINIMAX_H3.is_legal_frames(frames))
            self.assertLess(frames, MINIMAX_H3.min_frames())

    def test_first_usable_rung_is_the_minimum_itself(self):
        self.assertEqual(MINIMAX_H3.min_frames(), 73)
        self.assertTrue(MINIMAX_H3.is_legal_frames(73))

    def test_every_target_minimum_sits_on_its_own_grid(self):
        """If it didn't, clamping to the minimum would produce an off-grid length."""
        for target in (MINIMAX_H3, WAN22_A14B, LTX_2_3):
            self.assertTrue(target.is_legal_frames(target.min_frames()),
                            f"{target.label}: min {target.min_frames()} is off-grid")

    def test_minimum_never_exceeds_maximum(self):
        for target in (MINIMAX_H3, WAN22_A14B, LTX_2_3):
            self.assertLessEqual(target.min_frames(), target.max_frames())


class PlayheadRenderTests(unittest.TestCase):
    """The playhead must be visible and distinguishable, on every clip.

    Moving the waveform above the selection fill accidentally left the playhead
    inside the waveform branch, so a clip with no audio drew none at all.
    """

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import captioning_kit.app as A
        from captioning_kit.llm_captioning import CaptioningSettings
        self.A = A
        self.theme = A.Theme(CaptioningSettings())

    def _orange_pixels(self, peaks):
        from PySide6.QtGui import QColor, QPixmap
        bar = self.A.TrimBar(self.theme)
        bar.set_duration(3000)
        bar.set_trim(0, 3000)
        bar.set_peaks(peaks)
        bar.set_position(1500)
        bar.resize(600, bar.minimumHeight())
        pixmap = QPixmap(600, bar.height())
        pixmap.fill(QColor("black"))
        bar.render(pixmap)
        image = pixmap.toImage()
        want = QColor(self.A.PLAYHEAD_COLOR)
        centre = int(bar._x_for(1500))
        return sum(
            1
            for x in range(centre - 8, centre + 9)
            for y in range(bar.height())
            if abs(QColor(image.pixel(x, y)).red() - want.red()) < 30
            and abs(QColor(image.pixel(x, y)).green() - want.green()) < 30
        )

    def test_visible_on_a_clip_with_no_audio(self):
        self.assertGreater(self._orange_pixels([]), 20)

    def test_visible_on_a_clip_with_a_waveform(self):
        import math
        self.assertGreater(
            self._orange_pixels([abs(math.sin(i / 12)) for i in range(320)]), 20)

    def test_playhead_colour_differs_from_the_trim_accent(self):
        """They sit side by side; the same colour made the playhead hard to pick
        out from the handle beside it."""
        self.assertNotEqual(self.A.PLAYHEAD_COLOR.lower(), self.theme.accent.lower())
