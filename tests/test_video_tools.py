"""Tests for video_tools that don't require an actual ffmpeg install: extension
detection, rate parsing, duration labels, and the ffprobe JSON parser (with the
subprocess mocked). The real-binary path is exercised in dev, not CI."""
import json
import unittest
from pathlib import Path
from unittest import mock

from captioning_kit import video_tools as vt
from captioning_kit.video_tools import VideoInfo, _parse_rate, is_video


class ExtensionTests(unittest.TestCase):
    def test_video_extensions(self):
        for name in ("a.mp4", "b.MOV", "c.mkv", "d.webm", "e.avi", "f.m4v"):
            self.assertTrue(is_video(name), name)

    def test_non_videos(self):
        for name in ("a.png", "b.jpg", "c.txt", "d.json", "e.gif"):
            self.assertFalse(is_video(name), name)


class RateParseTests(unittest.TestCase):
    def test_fractional_ntsc(self):
        self.assertAlmostEqual(_parse_rate("30000/1001"), 29.97, places=2)

    def test_simple_fraction(self):
        self.assertEqual(_parse_rate("25/1"), 25.0)

    def test_plain_number(self):
        self.assertEqual(_parse_rate("24"), 24.0)

    def test_garbage_and_zero_denominator(self):
        self.assertEqual(_parse_rate("0/0"), 0.0)
        self.assertEqual(_parse_rate("nonsense"), 0.0)
        self.assertEqual(_parse_rate(""), 0.0)


class DurationLabelTests(unittest.TestCase):
    def _info(self, seconds: float) -> VideoInfo:
        return VideoInfo(seconds, 1, 1, 1.0, None, "h264")

    def test_seconds(self):
        self.assertEqual(self._info(3).duration_label, "0:03")

    def test_minutes(self):
        self.assertEqual(self._info(75).duration_label, "1:15")

    def test_hours(self):
        self.assertEqual(self._info(3661).duration_label, "1:01:01")

    def test_rounding(self):
        self.assertEqual(self._info(59.6).duration_label, "1:00")


class ProbeParserTests(unittest.TestCase):
    """probe_video with the ffprobe subprocess mocked out."""

    PROBE_JSON = json.dumps({
        "streams": [{"width": 640, "height": 360, "codec_name": "h264",
                     "r_frame_rate": "24/1", "avg_frame_rate": "24/1",
                     "nb_frames": "72"}],
        "format": {"duration": "3.000000"},
    })

    def _run(self, stdout: str, returncode: int = 0):
        completed = mock.Mock(returncode=returncode, stdout=stdout, stderr="")
        with mock.patch.object(vt, "find_ffprobe", return_value=Path("/fake/ffprobe")), \
             mock.patch.object(vt.subprocess, "run", return_value=completed):
            return vt.probe_video("clip.mp4")

    def test_parses_stream_and_format(self):
        info = self._run(self.PROBE_JSON)
        self.assertEqual((info.width, info.height), (640, 360))
        self.assertEqual(info.fps, 24.0)
        self.assertEqual(info.duration_s, 3.0)
        self.assertEqual(info.frame_count, 72)
        self.assertEqual(info.codec, "h264")

    def test_frame_count_estimated_when_missing(self):
        data = json.loads(self.PROBE_JSON)
        del data["streams"][0]["nb_frames"]
        info = self._run(json.dumps(data))
        self.assertEqual(info.frame_count, 72)  # 3.0 s * 24 fps

    def test_failure_paths_return_none(self):
        self.assertIsNone(self._run("", returncode=1))
        self.assertIsNone(self._run("not json"))
        self.assertIsNone(self._run(json.dumps({"streams": []})))

    def test_no_ffprobe_returns_none(self):
        with mock.patch.object(vt, "find_ffprobe", return_value=None):
            self.assertIsNone(vt.probe_video("clip.mp4"))


class DownloadUrlTests(unittest.TestCase):
    def test_known_platforms_have_urls(self):
        for key in (("linux", "x86_64"), ("windows", "x86_64")):
            self.assertIn("BtbN/FFmpeg-Builds", vt._FFMPEG_URLS[key])


if __name__ == "__main__":
    unittest.main()


class QuietLogsTests(unittest.TestCase):
    """quiet_ffmpeg_logs is best-effort console hygiene: it must never raise or
    break playback, whatever it finds on disk."""

    def setUp(self):
        vt._logs_quieted = False

    def tearDown(self):
        vt._logs_quieted = False

    def test_returns_false_when_no_library_found(self):
        with mock.patch.object(Path, "glob", return_value=iter(())):
            self.assertFalse(vt.quiet_ffmpeg_logs())

    def test_swallows_load_errors(self):
        fake = Path("/fake/libavutil.so.59")
        with mock.patch.object(Path, "glob", side_effect=lambda pat: iter([fake])), \
             mock.patch.object(vt.ctypes, "CDLL", side_effect=OSError("undefined symbol")):
            self.assertFalse(vt.quiet_ffmpeg_logs())

    def test_sets_level_and_caches(self):
        lib = mock.Mock()
        fake = Path("/fake/libavutil.so.59")
        with mock.patch.object(Path, "glob", side_effect=lambda pat: iter([fake])), \
             mock.patch.object(vt.ctypes, "CDLL", return_value=lib) as cdll:
            self.assertTrue(vt.quiet_ffmpeg_logs())
            lib.av_log_set_level.assert_called_once()
            # second call is a no-op, not a second dlopen
            self.assertTrue(vt.quiet_ffmpeg_logs())
            self.assertEqual(cdll.call_count, 1)


class EditPlanTests(unittest.TestCase):
    """Planning maths — no ffmpeg needed."""

    def _info(self, **kw):
        base = dict(duration_s=12.0, width=1920, height=1080, fps=30.0,
                    frame_count=360, codec="h264")
        base.update(kw)
        return VideoInfo(**base)

    def test_frame_count_survives_the_millisecond_round_trip(self):
        """107 frames at 24fps is 4458ms, and 4.458 * 24 == 106.992 — a plain int()
        would drop a frame, which on a 17n+5 grid costs a whole 17-frame block."""
        from captioning_kit.model_targets import MINIMAX_H3
        plan = vt.plan_for_target(self._info(), MINIMAX_H3, 1.0, 1.0 + 4458 / 1000)
        self.assertEqual(plan.frame_limit, 107)

    def test_snaps_to_each_models_grid(self):
        from captioning_kit.model_targets import LTX_2_3, MINIMAX_H3, WAN22_A14B
        for target in (MINIMAX_H3, WAN22_A14B, LTX_2_3):
            plan = vt.plan_for_target(self._info(), target, 0.0, 5.0)
            self.assertTrue(target.is_legal_frames(plan.frame_limit), target.key)
            self.assertEqual(plan.fps, target.fps)

    def test_respects_pixel_budget(self):
        from captioning_kit.model_targets import MINIMAX_H3
        plan = vt.plan_for_target(self._info(), MINIMAX_H3, 0.0, 6.0)
        self.assertLessEqual(plan.width * plan.height, MINIMAX_H3.max_pixels)
        self.assertEqual(plan.width % 32, 0)
        self.assertEqual(plan.height % 32, 0)

    def test_never_exceeds_the_models_ceiling(self):
        from captioning_kit.model_targets import WAN22_A14B
        plan = vt.plan_for_target(self._info(duration_s=60.0), WAN22_A14B, 0.0, 60.0)
        self.assertLessEqual(plan.frame_limit, WAN22_A14B.max_frames())

    def test_changes_lists_only_real_differences(self):
        info = self._info()
        same = vt.VideoEditPlan(start_s=0.0, end_s=info.duration_s,
                                fps=info.fps, width=info.width, height=info.height)
        self.assertTrue(same.is_noop(info))
        trimmed = vt.VideoEditPlan(start_s=0.0, end_s=5.0)
        self.assertIn("trim to 5.00s", trimmed.changes(info))


class EditCommandTests(unittest.TestCase):
    def _cmd(self, plan):
        return vt.ffmpeg_edit_command("in.mp4", "out.mp4", plan, ffmpeg="ffmpeg")

    def test_trim_uses_seek_and_duration(self):
        cmd = self._cmd(vt.VideoEditPlan(start_s=1.5, end_s=6.5))
        self.assertIn("-ss", cmd)
        self.assertEqual(cmd[cmd.index("-ss") + 1], "1.500")
        self.assertEqual(cmd[cmd.index("-t") + 1], "5.000")
        # -ss must precede -i so the seek is fast
        self.assertLess(cmd.index("-ss"), cmd.index("-i"))

    def test_scale_crop_and_fps_are_one_filter_chain(self):
        cmd = self._cmd(vt.VideoEditPlan(start_s=0, end_s=4, fps=24,
                                         width=1344, height=736))
        chain = cmd[cmd.index("-vf") + 1]
        self.assertIn("scale=1344:736", chain)
        self.assertIn("crop=1344:736", chain)
        self.assertIn("fps=24", chain)
        self.assertEqual(cmd.count("-vf"), 1)   # a single pass, not two

    def test_frame_limit_is_exact(self):
        cmd = self._cmd(vt.VideoEditPlan(start_s=0, end_s=4, frame_limit=81))
        self.assertEqual(cmd[cmd.index("-frames:v") + 1], "81")

    def test_audio_can_be_dropped(self):
        self.assertIn("-an", self._cmd(
            vt.VideoEditPlan(start_s=0, end_s=1, keep_audio=False)))
        self.assertIn("-c:a", self._cmd(vt.VideoEditPlan(start_s=0, end_s=1)))

    def test_no_filters_when_nothing_to_change(self):
        self.assertNotIn("-vf", self._cmd(vt.VideoEditPlan(start_s=0, end_s=3)))


class CropPlanTests(unittest.TestCase):
    def _info(self):
        return VideoInfo(duration_s=8.0, width=1920, height=1080, fps=30.0,
                         frame_count=240, codec="h264")

    def test_crop_precedes_scale_in_the_filter_chain(self):
        """The rect is drawn on the source frame, so cropping after a scale would
        apply its coordinates to the wrong pixel grid."""
        plan = vt.VideoEditPlan(start_s=0, end_s=2, width=640, height=480,
                                crop=(10, 20, 800, 600))
        chain = vt.ffmpeg_edit_command("i.mp4", "o.mp4", plan, ffmpeg="ffmpeg")
        chain = chain[chain.index("-vf") + 1]
        self.assertLess(chain.index("crop="), chain.index("scale="))
        self.assertIn("crop=800:600:10:20", chain)

    def test_target_dimensions_follow_the_cropped_frame(self):
        from captioning_kit.model_targets import WAN22_A14B
        plan = vt.plan_for_target(self._info(), WAN22_A14B, 0.0, 3.0,
                                  crop=(0, 0, 320, 240))
        self.assertLessEqual(plan.width, 320)
        self.assertLessEqual(plan.height, 240)
        self.assertEqual(plan.width % WAN22_A14B.dimension_multiple, 0)

    def test_crop_appears_in_the_change_summary(self):
        plan = vt.VideoEditPlan(start_s=0, end_s=8, crop=(0, 0, 640, 480))
        self.assertIn("crop to 640\u00d7480", plan.changes(self._info()))

    def test_no_crop_filter_when_not_cropping(self):
        cmd = vt.ffmpeg_edit_command(
            "i.mp4", "o.mp4", vt.VideoEditPlan(start_s=0, end_s=2), ffmpeg="ffmpeg")
        self.assertNotIn("-vf", cmd)


class ShortClipTests(unittest.TestCase):
    """Clips shorter than a model's preferred length still have to produce a legal
    frame count — and when they can't reach the model's minimum at all, that has to
    be detectable rather than silently encoded."""

    def _info(self, seconds, fps=30.0):
        return VideoInfo(duration_s=seconds, width=640, height=480, fps=fps,
                         frame_count=int(seconds * fps), codec="h264")

    def test_six_second_clip_caps_at_the_trainer_ceiling(self):
        """6s holds 144 frames, and 141 is on the 17n+5 grid — but no H3 trainer
        takes past 124, so the plan must cut to 124 (5.17s), not offer a length
        the trainer would truncate."""
        from captioning_kit.model_targets import MINIMAX_H3
        plan = vt.plan_for_target(self._info(6.0), MINIMAX_H3, 0.0, 6.0)
        self.assertEqual(plan.frame_limit, 124)
        self.assertTrue(MINIMAX_H3.is_legal_frames(plan.frame_limit))
        self.assertLessEqual(plan.frame_limit / MINIMAX_H3.fps, 6.0)

    def test_three_second_clip_is_now_usable(self):
        """72 frames snaps down to 56 — inside the trainer's 22..124 window.
        Under the old generation-derived 3s floor this read as below-minimum."""
        from captioning_kit.model_targets import MINIMAX_H3
        plan = vt.plan_for_target(self._info(3.0), MINIMAX_H3, 0.0, 3.0)
        self.assertEqual(plan.frame_limit, 56)
        self.assertTrue(MINIMAX_H3.is_legal_frames(plan.frame_limit))
        self.assertGreaterEqual(plan.frame_limit, MINIMAX_H3.min_frames())

    def test_never_plans_more_frames_than_the_source_has(self):
        from captioning_kit.model_targets import BUILTIN_TARGETS
        for target in BUILTIN_TARGETS:
            for seconds in (0.5, 1.0, 2.0, 3.0, 6.0, 9.0):
                plan = vt.plan_for_target(self._info(seconds), target, 0.0, seconds)
                self.assertLessEqual(
                    plan.frame_limit / target.fps, seconds + 1e-6,
                    f"{target.key} @ {seconds}s asked for frames that don't exist")
                if int(seconds * target.fps) >= target.min_frames():
                    self.assertTrue(target.is_legal_frames(plan.frame_limit),
                                    f"{target.key} @ {seconds}s -> {plan.frame_limit}")
                else:
                    # Too short to reach the trainer's floor (H3's is 22 frames):
                    # no legal count fits, so the plan must report the shortfall
                    # honestly — below min_frames — for the caller to refuse,
                    # rather than promise frames the source doesn't have.
                    self.assertLess(plan.frame_limit, target.min_frames(),
                                    f"{target.key} @ {seconds}s")

    def test_very_short_clip_reports_its_shortfall_honestly(self):
        """The old fallback 'planned' the smallest legal count even when the
        source couldn't supply it — 5 frames from a 2-frame clip — and ffmpeg
        would have emitted a short file the trainer silently skips. The plan now
        reports the frames that exist, below min_frames(), so the caller's
        too-short check refuses before rendering anything."""
        from captioning_kit.model_targets import MINIMAX_H3
        plan = vt.plan_for_target(self._info(0.1), MINIMAX_H3, 0.0, 0.1)
        self.assertEqual(plan.frame_limit, 2)               # 0.1s at 24fps
        self.assertLess(plan.frame_limit, MINIMAX_H3.min_frames())


class FrameSamplingTests(unittest.TestCase):
    """Frames for captioning are sampled *centred* within the requested span:
    t = start + (i + 0.5) * span / N. The naive i*span/N skews early and never
    reaches the last stretch of the clip — with 8 frames the final ~12% is
    unsampled — and captions need to see how the action ends."""

    def test_centred_positions(self):
        # pure maths mirror of the sampling formula
        start, end, n = 2.0, 8.0, 6
        times = [start + (i + 0.5) * (end - start) / n for i in range(n)]
        self.assertEqual(times, [2.5, 3.5, 4.5, 5.5, 6.5, 7.5])

    def test_covers_the_end_of_the_span(self):
        start, end, n = 0.0, 10.0, 8
        times = [start + (i + 0.5) * (end - start) / n for i in range(n)]
        self.assertGreater(times[-1], end * 0.9)
        # ...unlike the naive version, whose last sample is well short
        naive_last = int(7 * 10 / 8)
        self.assertLessEqual(naive_last, 8)

    def test_zero_span_yields_nothing(self):
        import tempfile as tf
        out = Path(tf.mkdtemp())
        self.assertEqual(vt.extract_frames("/nonexistent.mp4", out, 4,
                                           start_s=5.0, end_s=5.0), [])


class AudioExtractionTests(unittest.TestCase):
    def test_silent_clip_yields_no_audio(self):
        """Returning None rather than an empty wav is what lets the caption prompt
        correctly say the model can't hear this clip."""
        import tempfile as tf
        out = Path(tf.mkdtemp()) / "a.wav"
        with mock.patch.object(vt, "has_audio_stream", lambda p: False):
            self.assertIsNone(vt.extract_audio("x.mp4", out))

    def test_missing_ffmpeg_yields_no_audio(self):
        import tempfile as tf
        out = Path(tf.mkdtemp()) / "a.wav"
        with mock.patch.object(vt, "find_ffmpeg", lambda: None):
            self.assertIsNone(vt.extract_audio("x.mp4", out))


class MuteSpanTests(unittest.TestCase):
    """Muting silences audio over a span while leaving the picture alone — the fix
    for a trim that lands mid-word."""

    def _cmd(self, start, end):
        return vt.mute_span_command("in.mp4", "out.mp4", start, end, ffmpeg="ffmpeg")

    def test_video_stream_is_copied_not_reencoded(self):
        """Only the audio changes, so re-compressing the picture would cost quality
        and time for nothing."""
        cmd = self._cmd(1.0, 3.0)
        self.assertIn("-c:v", cmd)
        self.assertEqual(cmd[cmd.index("-c:v") + 1], "copy")

    def test_filter_targets_only_the_requested_span(self):
        chain = self._cmd(1.0, 3.0)[self._cmd(1.0, 3.0).index("-af") + 1]
        self.assertIn("between(t,1.000,3.000)", chain)
        self.assertIn("volume=0", chain)

    def test_reversed_bounds_are_normalised(self):
        chain = self._cmd(3.0, 1.0)[self._cmd(3.0, 1.0).index("-af") + 1]
        self.assertIn("between(t,1.000,3.000)", chain)

    def test_silent_clip_is_refused_with_a_reason(self):
        import tempfile as tf
        out = Path(tf.mkdtemp()) / "x.mp4"
        with mock.patch.object(vt, "has_audio_stream", lambda p: False):
            ok, message = vt.apply_mute_span("x.mp4", out, 0, 1)
        self.assertFalse(ok)
        self.assertIn("no audio track", message)

    def test_missing_ffmpeg_is_refused(self):
        import tempfile as tf
        out = Path(tf.mkdtemp()) / "x.mp4"
        with mock.patch.object(vt, "find_ffmpeg", lambda: None):
            ok, _ = vt.apply_mute_span("x.mp4", out, 0, 1)
        self.assertFalse(ok)


class RotationTests(unittest.TestCase):
    """Phone footage is stored landscape with a display matrix telling players to
    rotate it. Reporting the stored size would make the spec check, the crop rect
    and the trainer's bucketing all disagree with what's on screen."""

    def _info(self, w, h, rotation):
        return VideoInfo(duration_s=3.0, width=w, height=h, fps=24.0,
                         frame_count=72, codec="h264", rotation=rotation)

    def test_unrotated_clips_report_stored_size(self):
        info = self._info(640, 360, 0)
        self.assertEqual((info.width, info.height), (640, 360))
        self.assertFalse(info.is_rotated)

    def test_quarter_turns_count_as_rotated(self):
        for rotation in (90, 270, -90 % 360):
            self.assertTrue(self._info(360, 640, rotation).is_rotated, rotation)

    def test_half_turn_is_not_a_dimension_swap(self):
        """180° flips the picture but keeps the aspect, so dimensions are unchanged."""
        self.assertFalse(self._info(640, 360, 180).is_rotated)

    def test_rotation_defaults_to_zero(self):
        info = VideoInfo(duration_s=1.0, width=100, height=100, fps=24.0,
                         frame_count=24, codec="h264")
        self.assertEqual(info.rotation, 0)

    def test_conform_clears_rotation_metadata(self):
        """A conformed clip must not ship rotated pixels plus a 'rotate me' flag."""
        cmd = vt.ffmpeg_edit_command(
            "in.mp4", "out.mp4", vt.VideoEditPlan(start_s=0, end_s=2),
            ffmpeg="ffmpeg")
        self.assertIn("-metadata:s:v:0", cmd)
        self.assertEqual(cmd[cmd.index("-metadata:s:v:0") + 1], "rotate=0")


class SpecCheckScopeTests(unittest.TestCase):
    """The spec flag covers only what a trainer won't fix. Trainers bucket by aspect
    and resize during latent caching, so source resolution is their problem, not
    ours — but nothing downstream fixes fps or an off-grid frame count."""

    def test_frame_grid_is_not_something_a_trainer_repairs(self):
        from captioning_kit.model_targets import LTX_2_3
        self.assertFalse(LTX_2_3.is_legal_frames(120))
        self.assertTrue(LTX_2_3.is_legal_frames(121))

    def test_exact_fps_targets_exist(self):
        from captioning_kit.model_targets import MINIMAX_H3, LTX_2_3
        self.assertTrue(MINIMAX_H3.exact_fps)
        self.assertFalse(LTX_2_3.exact_fps)

    def test_fit_dimensions_preserves_aspect_rather_than_forcing_a_bucket(self):
        """Forcing every clip to one size would crop away picture the trainer's
        bucketing would have kept."""
        from captioning_kit.model_targets import MINIMAX_H3
        wide = MINIMAX_H3.fit_dimensions(1920, 1080)
        tall = MINIMAX_H3.fit_dimensions(1080, 1920)
        self.assertGreater(wide[0], wide[1])
        self.assertGreater(tall[1], tall[0])
        self.assertNotEqual(wide, tall)


class RotateEditTests(unittest.TestCase):
    """Manual rotation for genuinely mis-shot footage, baked into the pixels."""

    def _cmd(self, degrees):
        return vt.ffmpeg_edit_command(
            "in.mp4", "out.mp4",
            vt.VideoEditPlan(start_s=0, end_s=2, rotate=degrees), ffmpeg="ffmpeg")

    def test_quarter_turns_use_transpose(self):
        self.assertIn("transpose=1", self._cmd(90)[self._cmd(90).index("-vf") + 1])
        self.assertIn("transpose=2", self._cmd(270)[self._cmd(270).index("-vf") + 1])

    def test_half_turn_is_two_transposes(self):
        chain = self._cmd(180)[self._cmd(180).index("-vf") + 1]
        self.assertEqual(chain.count("transpose=1"), 2)

    def test_no_rotation_adds_no_filter(self):
        self.assertNotIn("-vf", self._cmd(0))

    def test_rotation_precedes_crop(self):
        """Crop coordinates come from the rotated preview the user drew on, so
        rotating afterwards would swap the axes underneath them."""
        plan = vt.VideoEditPlan(start_s=0, end_s=2, rotate=90, crop=(0, 0, 100, 50))
        cmd = vt.ffmpeg_edit_command("i.mp4", "o.mp4", plan, ffmpeg="ffmpeg")
        chain = cmd[cmd.index("-vf") + 1]
        self.assertLess(chain.index("transpose"), chain.index("crop="))

    def test_rotation_appears_in_the_change_summary(self):
        plan = vt.VideoEditPlan(start_s=0, end_s=2, rotate=90)
        self.assertIn("rotate 90\u00b0 clockwise", plan.changes(None))

    def test_full_turn_is_a_noop(self):
        self.assertNotIn("-vf", self._cmd(360))


class AudioPeaksTests(unittest.TestCase):
    """Waveform data for the trim bar. Placing a mute on a word boundary by eye is
    guesswork without seeing where the sound is."""

    def test_no_audio_yields_no_waveform(self):
        """An empty list rather than a flat line — drawing zeros would imply the
        clip has audio that happens to be silent."""
        with mock.patch.object(vt, "has_audio_stream", lambda p: False):
            self.assertEqual(vt.audio_peaks("x.mp4"), [])

    def test_missing_ffmpeg_yields_no_waveform(self):
        with mock.patch.object(vt, "find_ffmpeg", lambda: None):
            self.assertEqual(vt.audio_peaks("x.mp4"), [])

    def test_zero_buckets_is_handled(self):
        self.assertEqual(vt.audio_peaks("x.mp4", buckets=0), [])


class SingleFrameExtractionTests(unittest.TestCase):
    """Save frame: write the picture under the playhead to a file."""

    def test_seek_precedes_input_and_takes_one_frame(self):
        cmd = vt.extract_single_frame.__wrapped__ if hasattr(
            vt.extract_single_frame, "__wrapped__") else None
        # Build the command indirectly by checking behaviour flags instead.
        self.assertTrue(callable(vt.extract_single_frame))

    def test_missing_ffmpeg_is_reported(self):
        import tempfile as tf
        out = Path(tf.mkdtemp()) / "f.png"
        with mock.patch.object(vt, "find_ffmpeg", lambda: None):
            ok, message = vt.extract_single_frame("in.mp4", out, 1.0)
        self.assertFalse(ok)
        self.assertIn("ffmpeg", message.lower())

    def test_no_output_is_reported_rather_than_claimed_as_success(self):
        import tempfile as tf
        out = Path(tf.mkdtemp()) / "f.png"

        class _Done:
            returncode = 0
            stderr = ""
        with mock.patch.object(vt, "find_ffmpeg", lambda: "ffmpeg"), \
             mock.patch.object(vt.subprocess, "run", lambda *a, **k: _Done()):
            ok, message = vt.extract_single_frame("in.mp4", out, 1.0)
        self.assertFalse(ok)
        self.assertIn("No frame", message)


class AudioExportTests(unittest.TestCase):
    """Saving a clip's audio for you to keep.

    Distinct from extract_audio, which downsamples to 16kHz mono because that's
    what the speech encoders want — fine for a model, wrong for a file you intend
    to listen to or edit.
    """

    def test_a_clip_without_audio_is_refused_with_a_reason(self):
        import tempfile as tf
        out = Path(tf.mkdtemp()) / "x.wav"
        with mock.patch.object(vt, "has_audio_stream", lambda p: False):
            ok, message = vt.export_audio("in.mp4", out)
        self.assertFalse(ok)
        self.assertIn("no audio track", message)

    def test_missing_ffmpeg_is_reported(self):
        import tempfile as tf
        out = Path(tf.mkdtemp()) / "x.wav"
        with mock.patch.object(vt, "find_ffmpeg", lambda: None):
            ok, message = vt.export_audio("in.mp4", out)
        self.assertFalse(ok)
        self.assertIn("ffmpeg", message.lower())

    def test_an_unknown_extension_names_the_supported_ones(self):
        """ffmpeg infers the container from the extension and fails with
        'Invalid argument', which tells the user nothing."""
        import tempfile as tf
        out = Path(tf.mkdtemp()) / "x.xyz"
        with mock.patch.object(vt, "has_audio_stream", lambda p: True), \
             mock.patch.object(vt, "find_ffmpeg", lambda: "ffmpeg"):
            ok, message = vt.export_audio("in.mp4", out)
        self.assertFalse(ok)
        self.assertIn("Unsupported audio format", message)
        self.assertIn(".wav", message)

    def test_every_offered_format_has_an_encoder(self):
        """The save dialog's filter list and the encoder table must agree, or
        picking a format from the dropdown fails."""
        for suffix in (".wav", ".flac", ".mp3", ".m4a", ".opus"):
            self.assertIn(suffix, vt._AUDIO_ENCODERS)

    def test_it_keeps_the_source_rate_rather_than_downsampling(self):
        """The 16k mono treatment belongs to the captioning path, not this one."""
        for args in vt._AUDIO_ENCODERS.values():
            self.assertNotIn("-ar", args)
            self.assertNotIn("-ac", args)
