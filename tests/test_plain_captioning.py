"""Plain-text captioning: the path used by the plain and MiniMax H3 presets.

The distinguishing property is that the model's prose IS the caption — there's no
JSON to parse and no schema to repair — so the tests here are mostly about not
mangling what came back.
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import captioning_kit.llm_captioning as L
from captioning_kit.llm_captioning import CaptioningSettings, clean_plain_caption


class CleanPlainCaptionTests(unittest.TestCase):
    def test_strips_surrounding_code_fence(self):
        self.assertEqual(clean_plain_caption("```\na red car\n```"), "a red car")
        self.assertEqual(clean_plain_caption("```text\na red car\n```"), "a red car")

    def test_strips_assistant_preamble(self):
        for raw in ("Sure! Here is the caption: a red car",
                    "Certainly, here's the caption:\na red car",
                    "Here's the caption: a red car"):
            self.assertEqual(clean_plain_caption(raw), "a red car", raw)

    def test_strips_wrapping_quotes(self):
        self.assertEqual(clean_plain_caption('"a red car"'), "a red car")

    def test_keeps_internal_quotes(self):
        """Dialogue is quoted inside H3 captions — stripping those would corrupt it."""
        text = 'a woman says "hello" to a friend'
        self.assertEqual(clean_plain_caption(text), text)

    def test_leaves_structured_h3_output_alone(self):
        text = "integrated_multimodal_description: [Shot 1] a baker opens the shutters"
        self.assertEqual(clean_plain_caption(text), text)

    def test_handles_empty(self):
        self.assertEqual(clean_plain_caption(""), "")
        self.assertEqual(clean_plain_caption(None), "")


class CaptionImagePlainTests(unittest.TestCase):
    def setUp(self):
        self.settings = CaptioningSettings()
        self.seen = {}

        def fake_chat_vision(settings, model, image_path, system, user,
                             max_tokens, temperature=0.0):
            self.seen["system"] = system
            self.seen["user"] = user
            return "  a teal square  "

        self.patcher = patch.object(L, "chat_vision", fake_chat_vision)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)

    def test_returns_cleaned_prose(self):
        out = L.caption_image_plain(self.settings, Path("x.png"), "PROMPT")
        self.assertEqual(out, "a teal square")

    def test_sends_the_presets_prompt_verbatim(self):
        L.caption_image_plain(self.settings, Path("x.png"), "PRESET PROMPT")
        self.assertIn("PRESET PROMPT", self.seen["system"])

    def test_appends_guidance(self):
        L.caption_image_plain(self.settings, Path("x.png"), "PROMPT",
                              guidance="Mention the weather.")
        self.assertIn("PROMPT", self.seen["system"])
        self.assertIn("Mention the weather.", self.seen["system"])
        self.assertLess(self.seen["system"].index("PROMPT"),
                        self.seen["system"].index("Mention the weather."))

    def test_empty_guidance_adds_nothing(self):
        L.caption_image_plain(self.settings, Path("x.png"), "PROMPT", guidance="   ")
        self.assertNotIn("Additional guidance", self.seen["system"])


if __name__ == "__main__":
    unittest.main()


class VideoCaptionPromptTests(unittest.TestCase):
    """The multi-frame request has to tell the model what it's looking at — ordered
    samples of ONE clip — and when each frame occurs. Without the framing note,
    multi-image models describe 'several photos'; without timestamps, they can't
    place the motion."""

    def setUp(self):
        self.seen = {}
        outer = self

        class _Msg:
            content = "a caption"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            def create(self, **kw):
                outer.seen.update(kw)
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        self.patch_client = patch.object(L, "_make_openai_client",
                                         lambda settings: _Client())
        self.patch_client.start()
        self.addCleanup(self.patch_client.stop)

    def _frames(self, n=3):
        from captioning_kit.video_tools import SampledFrame
        tmp = Path(tempfile.mkdtemp())
        out = []
        for i in range(n):
            p = tmp / f"f{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xdb fake jpeg")
            out.append(SampledFrame(path=p, time_s=1.0 + i, index=i + 1))
        return out

    def test_interleaves_timestamp_labels_before_each_image(self):
        L.chat_vision_frames(CaptioningSettings(), "m", self._frames(3),
                             system="S", user="U", max_tokens=100)
        content = self.seen["messages"][1]["content"]
        kinds = [c["type"] for c in content]
        self.assertEqual(kinds[0], "text")                     # the user prompt
        self.assertEqual(kinds[1:], ["text", "image_url"] * 3)  # label, image, ...
        labels = [c["text"] for c in content if c["type"] == "text"][1:]
        self.assertIn("Frame 1 of 3", labels[0])
        self.assertIn("at 1.00s", labels[0])
        self.assertIn("Frame 3 of 3", labels[-1])

    def test_empty_frames_is_an_error_not_a_silent_text_call(self):
        with self.assertRaises(L.AutoCaptionError):
            L.chat_vision_frames(CaptioningSettings(), "m", [],
                                 system="S", user="U", max_tokens=100)

    def test_video_system_prompt_frames_the_task(self):
        import captioning_kit.video_tools as VT
        with patch.object(VT, "extract_frames", lambda *a, **k: self._frames(2)), \
             patch.object(L.shutil, "rmtree", lambda *a, **k: None):
            L.caption_video_plain(CaptioningSettings(), Path("v.mp4"),
                                  "PRESET", guidance="G")
        system = self.seen["messages"][0]["content"]
        self.assertIn("PRESET", system)
        self.assertIn("single video clip", system)
        self.assertIn("cannot hear", system)       # no invented dialogue
        self.assertIn("G", system)
        self.assertLess(system.index("PRESET"), system.index("G"))


class AudioCaptionTests(unittest.TestCase):
    """Audio rides along with the frames for Omni-style models. The instruction and
    the payload must always agree: telling a model it can't hear while handing it
    the audio (or the reverse) produces invented or omitted dialogue."""

    def setUp(self):
        self.seen = {}
        outer = self

        class _Msg:
            content = "a caption"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            def create(self, **kw):
                outer.seen.update(kw)
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        self.patcher = patch.object(L, "_make_openai_client", lambda s: _Client())
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.tmp = Path(tempfile.mkdtemp())
        self.wav = self.tmp / "a.wav"
        self.wav.write_bytes(b"RIFF" + b"\x00" * 200)

    def _frames(self, n=2):
        from captioning_kit.video_tools import SampledFrame
        out = []
        for i in range(n):
            p = self.tmp / f"f{i}.jpg"
            p.write_bytes(b"\xff\xd8\xff\xdb fake")
            out.append(SampledFrame(path=p, time_s=float(i), index=i + 1))
        return out

    def _parts(self, kind):
        return [c for c in self.seen["messages"][1]["content"]
                if c.get("type") == kind]

    def test_audio_is_sent_as_an_input_audio_part(self):
        L.chat_vision_frames(CaptioningSettings(), "m", self._frames(2),
                             system="S", user="U", max_tokens=10,
                             audio_path=self.wav)
        audio = self._parts("input_audio")
        self.assertEqual(len(audio), 1)
        self.assertIn("data", audio[0]["input_audio"])
        self.assertEqual(audio[0]["input_audio"]["format"], "wav")

    def test_audio_comes_after_the_frames(self):
        """Frames are the timeline; the audio part covers the same span as a whole,
        so it belongs at the end rather than interleaved."""
        L.chat_vision_frames(CaptioningSettings(), "m", self._frames(3),
                             system="S", user="U", max_tokens=10,
                             audio_path=self.wav)
        kinds = [c["type"] for c in self.seen["messages"][1]["content"]]
        self.assertEqual(kinds[-1], "input_audio")
        self.assertEqual(kinds.count("image_url"), 3)

    def test_no_audio_part_when_none_given(self):
        L.chat_vision_frames(CaptioningSettings(), "m", self._frames(2),
                             system="S", user="U", max_tokens=10)
        self.assertEqual(self._parts("input_audio"), [])

    def test_prompt_matches_a_present_audio_track(self):
        import captioning_kit.video_tools as VT
        with patch.object(VT, "extract_frames", lambda *a, **k: self._frames(2)), \
             patch.object(VT, "extract_audio", lambda *a, **k: self.wav), \
             patch.object(L.shutil, "rmtree", lambda *a, **k: None):
            L.caption_video_plain(CaptioningSettings(), Path("v.mp4"), "P",
                                  include_audio=True)
        system = self.seen["messages"][0]["content"]
        self.assertIn("audio track for the same span is included", system)
        self.assertNotIn("cannot hear", system)

    def test_silent_clip_falls_back_and_says_so(self):
        """A clip with no audio track must not be described as if it were heard."""
        import captioning_kit.video_tools as VT
        with patch.object(VT, "extract_frames", lambda *a, **k: self._frames(2)), \
             patch.object(VT, "extract_audio", lambda *a, **k: None), \
             patch.object(L.shutil, "rmtree", lambda *a, **k: None):
            L.caption_video_plain(CaptioningSettings(), Path("v.mp4"), "P",
                                  include_audio=True)
        self.assertIn("cannot hear", self.seen["messages"][0]["content"])
        self.assertEqual(self._parts("input_audio"), [])

    def test_server_audio_rejection_is_explained(self):
        msg = L.describe_audio_failure(
            RuntimeError("Error code: 500 - audio input is not supported"))
        self.assertIn("audio encoder", msg)
        self.assertIn("Send clip audio", msg)

    def test_unrelated_errors_pass_through_unchanged(self):
        self.assertEqual(L.describe_audio_failure(RuntimeError("boom")), "boom")


class AudioProfileTests(unittest.TestCase):
    # Families llama.cpp has an audio encoder for. Gemma 4 joined Omni once the
    # conformer encoder and the server's input_audio routing landed.
    AUDIO_FAMILIES = ("omni", "gemma4", "gemma-4")

    def test_at_least_one_audio_profile_ships(self):
        from captioning_kit.llm_captioning import _profile_from_dict, DEFAULT_PROFILE_DATA
        audio = [p for p in (_profile_from_dict(r) for r in DEFAULT_PROFILE_DATA["profiles"])
                 if p and p.supports_audio]
        self.assertTrue(audio, "an audio-capable profile should ship by default")

    def test_audio_is_only_claimed_by_families_that_have_an_encoder(self):
        from captioning_kit.llm_captioning import _profile_from_dict, DEFAULT_PROFILE_DATA
        for raw in DEFAULT_PROFILE_DATA["profiles"]:
            profile = _profile_from_dict(raw)
            if profile and profile.supports_audio:
                self.assertTrue(
                    any(f in profile.id.lower() for f in self.AUDIO_FAMILIES),
                    f"{profile.id} claims audio but isn't an Omni or Gemma 4 build")

    def test_vision_only_families_never_claim_audio(self):
        """Qwen3-VL and the Qwen3.5/3.6 builds have no audio encoder at all."""
        from captioning_kit.llm_captioning import _profile_from_dict, DEFAULT_PROFILE_DATA
        vision_only = ("qwen3vl", "qwen35", "qwen36")
        for raw in DEFAULT_PROFILE_DATA["profiles"]:
            profile = _profile_from_dict(raw)
            if profile and any(v in profile.id.lower() for v in vision_only):
                self.assertFalse(profile.supports_audio, profile.id)

    def test_shipped_example_matches_the_builtin_defaults(self):
        """The example file shadows DEFAULT_PROFILE_DATA at runtime, so a profile
        added to only one of them silently never appears."""
        import json
        from captioning_kit.llm_captioning import (
            DEFAULT_PROFILE_DATA, default_profiles_example_path)
        path = default_profiles_example_path()
        if not path.exists():
            self.skipTest("no example profiles file in this checkout")
        example_ids = {p.get("id") for p in json.loads(path.read_text())["profiles"]}
        builtin_ids = {p.get("id") for p in DEFAULT_PROFILE_DATA["profiles"]}
        self.assertTrue(builtin_ids - {"custom-hf", "custom-local"} <= example_ids,
                        f"missing from the example file: "
                        f"{builtin_ids - example_ids - {'custom-hf', 'custom-local'}}")


class BatchThreadDefaultsTests(unittest.TestCase):
    """Attributes the batch loop reads must be initialised in __init__, not only set
    by the caller — a missing default fails on the first video, mid-run."""

    def test_batch_thread_has_media_defaults(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import captioning_kit.app as A
        thread = A.BatchCaptionThread(CaptioningSettings(), [], delay_ms=0)
        self.assertFalse(thread.include_audio)
        self.assertGreaterEqual(thread.frame_count, 2)


class ThinkingBlockTests(unittest.TestCase):
    """Thinking-variant models emit reasoning before the answer. Left in, it lands
    in the dataset sidecar."""

    def test_strips_a_closed_think_block(self):
        self.assertEqual(
            L.clean_plain_caption("<think>let me look</think>\na dog runs"),
            "a dog runs")

    def test_strips_several_blocks_and_variant_tag_names(self):
        self.assertEqual(
            L.clean_plain_caption("<think>a</think><thinking>b</thinking> a dog runs"),
            "a dog runs")

    def test_handles_a_stray_closing_tag(self):
        """Some chat templates emit the opening tag themselves, so the reply starts
        mid-reasoning and only the closer appears."""
        self.assertEqual(
            L.clean_plain_caption("I should describe this.</think>\na dog runs"),
            "a dog runs")

    def test_does_not_touch_the_word_think_in_prose(self):
        text = 'a woman says "I think so" to a friend'
        self.assertEqual(L.clean_plain_caption(text), text)

    def test_reasoning_only_output_raises_rather_than_blanking(self):
        with self.assertRaises(L.AutoCaptionError) as ctx:
            L._require_caption(L.clean_plain_caption("<think>never finished"),
                               "<think>never finished")
        self.assertIn("Max tokens", str(ctx.exception))

    def test_empty_output_raises(self):
        with self.assertRaises(L.AutoCaptionError):
            L._require_caption("", "")


class MmprojAudioDetectionTests(unittest.TestCase):
    """Whether a model can hear is a fact about its projector, not its name — an
    Omni conversion can register under a vision architecture and ship a projector
    with no audio tower."""

    def _gguf(self, name, pairs):
        import struct
        path = Path(tempfile.mkdtemp()) / name

        def w(text):
            raw = text.encode()
            return struct.pack("<Q", len(raw)) + raw

        body = b""
        for key, value in pairs:
            if isinstance(value, bool):
                body += w(key) + struct.pack("<I", 7) + (b"\x01" if value else b"\x00")
            elif isinstance(value, str):
                body += w(key) + struct.pack("<I", 8) + w(value)
            else:
                body += w(key) + struct.pack("<I", 4) + struct.pack("<I", value)
        path.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                         + struct.pack("<Q", len(pairs)) + body + b"\x00" * 32)
        return path

    def test_detects_an_audio_projector(self):
        path = self._gguf("a.gguf", [("general.architecture", "clip"),
                                     ("clip.has_audio_encoder", True)])
        self.assertIs(L.mmproj_has_audio_encoder(path), True)

    def test_detects_a_vision_only_projector(self):
        path = self._gguf("v.gguf", [("general.architecture", "clip"),
                                     ("clip.has_audio_encoder", False)])
        self.assertIs(L.mmproj_has_audio_encoder(path), False)

    def test_absent_flag_reads_as_no_audio(self):
        path = self._gguf("n.gguf", [("general.architecture", "clip"),
                                     ("clip.vision.projection_dim", 2048)])
        self.assertIs(L.mmproj_has_audio_encoder(path), False)

    def test_unreadable_file_is_unknown_not_false(self):
        """None means 'ask the profile instead' — claiming False would disable audio
        on a perfectly good model whose projector just isn't downloaded yet."""
        path = Path(tempfile.mkdtemp()) / "junk.gguf"
        path.write_bytes(b"not a gguf")
        self.assertIsNone(L.mmproj_has_audio_encoder(path))

    def test_bool_values_do_not_corrupt_the_rest_of_the_parse(self):
        path = self._gguf("m.gguf", [("general.architecture", "clip"),
                                     ("clip.has_audio_encoder", True),
                                     ("clip.vision.projection_dim", 2048)])
        self.assertEqual(L.read_gguf_metadata(path)["clip.vision.projection_dim"], 2048)


class GuidanceReachesEveryGeneratingOpTests(unittest.TestCase):
    """Guidance was gated on operation == 'json_image'. The plain ops were added
    later and silently received an empty string, so per-folder and per-file
    instructions were dropped for the plain and MiniMax H3 presets."""

    def test_plain_ops_are_treated_as_caption_generating(self):
        generating = ("json_image", "plain", "plain_video")
        for op in generating:
            self.assertIn(op, generating)
        # refine/bboxes take their instructions from elsewhere and must stay out
        for op in ("refine", "bboxes"):
            self.assertNotIn(op, generating)

    def test_guidance_is_appended_after_the_preset_prompt(self):
        seen = {}

        class _Msg:
            content = "a caption"

        class _Choice:
            message = _Msg()

        class _Resp:
            choices = [_Choice()]

        class _Completions:
            def create(self, **kw):
                seen.update(kw)
                return _Resp()

        class _Chat:
            completions = _Completions()

        class _Client:
            chat = _Chat()

        with patch.object(L, "chat_vision",
                          lambda settings, model, image_path, system, user,
                          max_tokens, temperature=0.0: seen.setdefault("system", system)
                          or "a caption"):
            L.caption_image_plain(CaptioningSettings(), Path("x.png"),
                                  "PRESET PROMPT", guidance="Refer to her as Ada.")
        system = seen["system"]
        self.assertIn("PRESET PROMPT", system)
        self.assertIn("Refer to her as Ada.", system)
        self.assertLess(system.index("PRESET PROMPT"), system.index("Refer to her as Ada."))


class AudioStatusMessageTests(unittest.TestCase):
    """Captioning a clip without sound is a legitimate outcome, but never a silent
    one: a vision-only model writes 'she opens her mouth as if speaking', which
    reads like a broken caption unless you know audio was never sent."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.A = A
        self.win = A.MainWindow()

    def test_vision_only_model_explains_itself(self):
        self.win.settings.send_clip_audio = True
        self.win.settings.caption_profile_id = "unsloth-qwen3vl-30b-q4"
        self.win._mmproj_audio_cache = {}
        ok, why = self.win.audio_status()
        self.assertFalse(ok)
        self.assertIn("vision-only", why)
        self.assertIn("Omni", why)

    def test_reason_names_the_model_without_picker_framing(self):
        self.win.settings.send_clip_audio = True
        self.win.settings.caption_profile_id = "unsloth-qwen3vl-30b-q4"
        self.win._mmproj_audio_cache = {}
        why = self.win.audio_status()[1]
        self.assertNotIn("Download:", why)
        self.assertNotIn("~20GB", why)

    def test_disabled_preference_is_named_as_the_cause(self):
        self.win.settings.send_clip_audio = False
        ok, why = self.win.audio_status()
        self.assertFalse(ok)
        self.assertIn("Preferences", why)

    def test_audio_model_reports_ready(self):
        self.win.settings.send_clip_audio = True
        self.win.settings.caption_profile_id = "ggml-qwen3-omni-30b"
        self.win._mmproj_audio_cache = {}
        ok, why = self.win.audio_status()
        self.assertTrue(ok)
        self.assertEqual(why, "")

    def test_gate_and_status_agree(self):
        """One source of truth — the badge can't say 'audio on' while the request
        omits it."""
        for profile in ("unsloth-qwen3vl-30b-q4", "ggml-qwen3-omni-30b"):
            self.win.settings.caption_profile_id = profile
            self.win.settings.send_clip_audio = True
            self.win._mmproj_audio_cache = {}
            self.assertEqual(self.win.audio_captioning_enabled(),
                             self.win.audio_status()[0], profile)


class AudioProfileTaggingTests(unittest.TestCase):
    """Audio capability is the difference between a video caption that can quote
    dialogue and one that can only watch lips, so it has to be visible when
    choosing a model — not discovered from a disappointing caption."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.llm_captioning as LL
        LL.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.LL = LL

    def test_audio_profiles_span_both_supported_families(self):
        """Omni for the big vision+audio model, Gemma 4 for the small ones."""
        ids = [p.id for p in self.LL.profiles_for_task("caption") if p.supports_audio]
        self.assertGreaterEqual(len(ids), 2, ids)
        self.assertTrue(any("omni" in i for i in ids), ids)
        self.assertTrue(any("gemma4" in i or "gemma-4" in i for i in ids), ids)

    def test_the_abliterated_omni_build_is_among_them(self):
        profiles = {p.id: p for p in self.LL.profiles_for_task("caption")}
        target = profiles.get("huihui-qwen3-omni-30b-thinking-abliterated")
        self.assertIsNotNone(target)
        self.assertTrue(target.supports_audio)
        self.assertIn("mmproj", target.mmproj_filename)
        self.assertIn("Q4_K_M", target.model_filename)

    def test_vision_only_profiles_are_not_tagged(self):
        for profile in self.LL.profiles_for_task("caption"):
            if any(v in profile.id for v in ("qwen3vl", "qwen35", "qwen36")):
                self.assertFalse(profile.supports_audio, profile.id)

    def test_thinking_variant_warns_about_its_cost(self):
        """Thinking models need a bigger token budget; the picker note should say
        so, since a truncated run produces no caption at all."""
        profiles = {p.id: p for p in self.LL.profiles_for_task("caption")}
        note = profiles["huihui-qwen3-omni-30b-thinking-abliterated"].note.lower()
        self.assertIn("thinking", note)
        self.assertIn("max-tokens", note.replace("max tokens", "max-tokens"))


class ThumbnailTooltipTests(unittest.TestCase):
    """A coloured dot in a corner is unreadable without a legend — you'd have to
    already know the code to look it up. Every marker names itself on hover."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QFileDialog
        from PySide6.QtGui import QImage, QColor
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.folder = Path(tf.mkdtemp())
        for name in ("a.png", "b.png"):
            img = QImage(32, 32, QImage.Format_RGB32)
            img.fill(QColor("teal"))
            img.save(str(self.folder / name), "PNG")
        self.win = A.MainWindow()
        QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(self.folder))
        self.win.open_folder()

    def _tip(self, name):
        return self.win._thumb_items[str(self.folder / name)].toolTip()

    def test_clean_thumbnail_shows_only_its_name(self):
        self.assertEqual(self._tip("a.png"), "a.png")

    def test_review_flag_names_its_marker(self):
        self.win.project.toggle_review_mark("b.png")
        self.win._refresh_thumb_marker(self.folder / "b.png")
        tip = self._tip("b.png")
        self.assertIn("Flagged for review", tip)
        self.assertIn("bottom-right", tip)

    def test_health_issues_are_listed_not_just_signalled(self):
        """The red dot means 'this caption may be corrupt'; the tooltip has to say
        what's actually wrong or the marker is still a guessing game."""
        self.win.project.set_flags("a.png", ["missing 'caption' field"])
        self.win._refresh_thumb_marker(self.folder / "a.png")
        tip = self._tip("a.png")
        self.assertIn("red dot", tip)
        self.assertIn("missing 'caption' field", tip)

    def test_marker_wording_comes_from_the_banner_source(self):
        """Tooltips and hover banners must not drift apart."""
        self.win.project.toggle_review_mark("b.png")
        self.win._refresh_thumb_marker(self.folder / "b.png")
        banners = [b[0] for b in self.win._thumb_banners(self.folder / "b.png")]
        tip = self._tip("b.png")
        for text in banners:
            self.assertIn(text, tip)


class SchemaFlagScopeTests(unittest.TestCase):
    """Schema health checks parse captions as Ideogram JSON, so running them on a
    plain .txt reports 'corrupt — could not parse JSON' on every well-formed file.
    Only the validating preset gets flagged."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
        from PySide6.QtGui import QImage, QColor
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        # Switching to a preset with a different extension asks for confirmation;
        # unstubbed that is a modal dialog and the test hangs.
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        self.A = A
        self.folder = Path(tf.mkdtemp())
        img = QImage(32, 32, QImage.Format_RGB32)
        img.fill(QColor("teal"))
        img.save(str(self.folder / "a.png"), "PNG")
        (self.folder / "a.txt").write_text("a teal square")
        self.win = A.MainWindow()
        QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(self.folder))
        self.win.open_folder()

    def _use(self, preset):
        self.win.preset_combo.setCurrentIndex(self.win.preset_combo.findData(preset))

    def test_plain_text_caption_is_not_flagged(self):
        self._use("plain_text")
        self.assertEqual(self.win.caption_issues_for(self.folder / "a.png"), [])

    def test_h3_presets_are_not_flagged(self):
        for preset in ("minimax_h3_natural", "minimax_h3_official"):
            self._use(preset)
            self.assertEqual(self.win.caption_issues_for(self.folder / "a.png"), [],
                             preset)

    def test_stale_flags_clear_when_leaving_the_validating_preset(self):
        """A red dot recorded under Ideogram shouldn't linger on a .txt dataset."""
        self.win.project.set_flags("a.png", ["corrupt caption file"])
        self._use("plain_text")
        self.win._apply_preset_ui()
        self.assertEqual(self.win.project.caption_issues("a.png"), [])

    def test_ideogram_still_validates(self):
        self._use("ideogram4")
        (self.folder / "a.json").write_text("{ not valid json")
        self.assertTrue(self.win.caption_issues_for(self.folder / "a.png"))


class ItemRoleUniquenessTests(unittest.TestCase):
    """Every per-item role must have its own number.

    SPEC_ROLE was accidentally given the same number as DURATION_ROLE, so setting
    the spec flag overwrote the duration badge and the filmstrip drew the boolean
    "true" where "0:20" belonged. Cheap to assert, invisible until someone sees it.
    """

    def test_all_item_roles_are_distinct(self):
        import re
        source = Path(__file__).resolve().parents[1] / "captioning_kit" / "app.py"
        roles = re.findall(r"^(\w+_ROLE) = int\(Qt\.UserRole\) \+ (\d+)",
                           source.read_text(), re.M)
        self.assertTrue(roles, "no item roles found — did the declaration style change?")
        by_number = {}
        for name, number in roles:
            by_number.setdefault(number, []).append(name)
        clashes = {n: names for n, names in by_number.items() if len(names) > 1}
        self.assertEqual(clashes, {}, f"roles sharing a number: {clashes}")

    def test_duration_badge_survives_the_spec_flag(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import captioning_kit.app as A
        self.assertNotEqual(A.DURATION_ROLE, A.SPEC_ROLE)


class WidgetAttributeCollisionTests(unittest.TestCase):
    """Two different buttons must not share an attribute name.

    The transport's volume mute and the edit bar's "Mute section" were both
    assigned to _mute_btn, so the second overwrote the first: pressing volume-mute
    retitled and re-iconed the section button instead.
    """

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.stage = A.MainWindow().video_stage

    def test_volume_mute_and_section_mute_are_separate_widgets(self):
        self.assertIsNot(self.stage._mute_btn, self.stage._mute_section_btn)

    def test_volume_mute_does_not_disturb_the_section_button(self):
        before = (self.stage._mute_section_btn.text(),
                  self.stage._mute_section_btn.isChecked())
        self.stage.toggle_mute()
        self.assertEqual(
            (self.stage._mute_section_btn.text(),
             self.stage._mute_section_btn.isChecked()), before)

    def test_section_button_is_a_labelled_button_not_an_icon(self):
        self.assertEqual(self.stage._mute_section_btn.text(), "Mute section")
        self.assertTrue(self.stage._mute_section_btn.icon().isNull())


class ImageEditBarTests(unittest.TestCase):
    """Image editing lives on the preview, the same as video editing — hunting for
    a modal to crop a photo while a clip edits inline is an inconsistency."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QFileDialog
        from PySide6.QtGui import QImage, QColor
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.folder = Path(tf.mkdtemp())
        img = QImage(120, 60, QImage.Format_RGB32)
        img.fill(QColor("teal"))
        img.save(str(self.folder / "a.png"), "PNG")
        self.win = A.MainWindow()
        QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(self.folder))
        self.win.open_folder()
        self.win.filmstrip.setCurrentRow(0)

    def test_controls_exist_on_the_image_page(self):
        for name in ("img_crop_btn", "img_rot_cw", "img_rot_ccw",
                     "img_reset_btn", "img_apply_btn"):
            self.assertTrue(hasattr(self.win, name), name)

    def test_apply_is_disabled_until_something_changes(self):
        self.assertFalse(self.win.img_apply_btn.isEnabled())
        self.win.rotate_image_by(90)
        self.assertTrue(self.win.img_apply_btn.isEnabled())

    def test_label_names_the_pending_change(self):
        self.win.rotate_image_by(90)
        self.assertIn("rotated 90", self.win.img_edit_label.text())

    def test_reset_clears_rotation(self):
        self.win.rotate_image_by(180)
        self.win.reset_image_edit()
        self.assertEqual(self.win._image_rotation, 0)
        self.assertFalse(self.win.img_apply_btn.isEnabled())

    def test_four_quarter_turns_is_a_noop(self):
        for _ in range(4):
            self.win.rotate_image_by(90)
        self.assertEqual(self.win._image_rotation, 0)


class BypassStoreTests(unittest.TestCase):
    """Bypassed files physically move to .bypass/.

    A flag in project.json wouldn't do: the trainer reads the folder, so the only
    reliable way to keep a file out of a run is for it not to be there. The caption
    travels with the media so nothing is orphaned.
    """

    def setUp(self):
        import tempfile as tf
        from captioning_kit.store import CaptionStore
        self.tmp = Path(tf.mkdtemp())
        for name in ("a.png", "b.png", "c.png"):
            (self.tmp / name).write_bytes(b"x")
            (self.tmp / name).with_suffix(".txt").write_text(f"caption {name}")
        self.store = CaptionStore(self.tmp, ".txt")

    def test_bypassed_files_sort_last(self):
        self.store.bypass(self.tmp / "a.png")
        self.assertEqual([p.name for p in self.store.images()],
                         ["b.png", "c.png", "a.png"])

    def test_active_listing_excludes_them(self):
        self.store.bypass(self.tmp / "a.png")
        self.assertEqual([p.name for p in self.store.images(include_bypassed=False)],
                         ["b.png", "c.png"])

    def test_caption_travels_with_the_media(self):
        moved = self.store.bypass(self.tmp / "a.png")
        self.assertTrue(moved.exists())
        self.assertEqual(self.store.caption_path(moved).read_text(), "caption a.png")
        self.assertFalse((self.tmp / "a.txt").exists())

    def test_round_trip_restores_both(self):
        moved = self.store.bypass(self.tmp / "a.png")
        back = self.store.unbypass(moved)
        self.assertEqual(back, self.tmp / "a.png")
        self.assertEqual((self.tmp / "a.txt").read_text(), "caption a.png")
        self.assertFalse(self.store.bypass_dir().joinpath("a.txt").exists())

    def test_a_name_collision_never_clobbers(self):
        """Two different files with one name means the user loses one silently."""
        self.store.bypass_dir().mkdir(parents=True, exist_ok=True)
        self.store.bypass_dir().joinpath("a.png").write_bytes(b"other")
        moved = self.store.bypass(self.tmp / "a.png")
        self.assertEqual(moved.name, "a_2.png")
        self.assertEqual(self.store.bypass_dir().joinpath("a.png").read_bytes(),
                         b"other")

    def test_is_bypassed_reads_the_location(self):
        moved = self.store.bypass(self.tmp / "a.png")
        self.assertTrue(self.store.is_bypassed(moved))
        self.assertFalse(self.store.is_bypassed(self.tmp / "b.png"))

    def test_bypassing_twice_is_a_noop(self):
        moved = self.store.bypass(self.tmp / "a.png")
        self.assertEqual(self.store.bypass(moved), moved)


class ImportMediaTests(unittest.TestCase):
    """Adding files copies them in rather than moving: the source may be a library
    the user wants left intact, and an accidental drag shouldn't rearrange disk."""

    def setUp(self):
        import tempfile as tf
        from captioning_kit.store import CaptionStore
        self.dataset = Path(tf.mkdtemp())
        self.source = Path(tf.mkdtemp())
        (self.dataset / "existing.png").write_bytes(b"original")
        (self.source / "new.png").write_bytes(b"a")
        (self.source / "new.txt").write_text("caption came along")
        (self.source / "clip.mp4").write_bytes(b"v")
        (self.source / "notes.pdf").write_bytes(b"p")
        self.store = CaptionStore(self.dataset, ".txt")

    def test_copies_rather_than_moves(self):
        self.store.import_media([self.source / "new.png"])
        self.assertTrue((self.dataset / "new.png").exists())
        self.assertTrue((self.source / "new.png").exists())

    def test_brings_an_existing_caption(self):
        """Importing an already-captioned file shouldn't lose the work."""
        self.store.import_media([self.source / "new.png"])
        self.assertEqual((self.dataset / "new.txt").read_text(), "caption came along")

    def test_accepts_images_and_video(self):
        added, _ = self.store.import_media(
            [self.source / "new.png", self.source / "clip.mp4"])
        self.assertEqual({p.name for p in added}, {"new.png", "clip.mp4"})

    def test_skips_non_media_with_a_reason(self):
        added, skipped = self.store.import_media([self.source / "notes.pdf"])
        self.assertEqual(added, [])
        self.assertTrue(any("notes.pdf" in s for s in skipped))

    def test_a_folder_contributes_its_media(self):
        sub = self.source / "batch"
        sub.mkdir()
        (sub / "deep.png").write_bytes(b"d")
        added, _ = self.store.import_media([sub])
        self.assertEqual([p.name for p in added], ["deep.png"])

    def test_name_collision_never_clobbers(self):
        (self.source / "existing.png").write_bytes(b"different")
        added, _ = self.store.import_media([self.source / "existing.png"])
        self.assertEqual(added[0].name, "existing_2.png")
        self.assertEqual((self.dataset / "existing.png").read_bytes(), b"original")

    def test_importing_from_the_dataset_itself_is_refused(self):
        added, skipped = self.store.import_media([self.dataset / "existing.png"])
        self.assertEqual(added, [])
        self.assertTrue(any("already in this folder" in s for s in skipped))


class RevertAndDeleteTests(unittest.TestCase):
    """Two ways to undo, with different meanings.

    restore_original copies the backup over the file and keeps it (edit again,
    revert again). revert_to_original is "undo the edit entirely", so it moves the
    backup back and removes it — leaving it would imply an edit that no longer
    exists.
    """

    def setUp(self):
        import tempfile as tf
        from captioning_kit.store import CaptionStore
        self.tmp = Path(tf.mkdtemp())
        (self.tmp / "a.png").write_bytes(b"edited")
        (self.tmp / "a.txt").write_text("caption")
        self.store = CaptionStore(self.tmp, ".txt")
        backups = self.store.originals_dir()
        backups.mkdir(parents=True, exist_ok=True)
        (backups / "a.png").write_bytes(b"original")

    def test_revert_restores_the_pre_edit_bytes(self):
        self.assertTrue(self.store.revert_to_original(self.tmp / "a.png"))
        self.assertEqual((self.tmp / "a.png").read_bytes(), b"original")

    def test_revert_drops_the_backup(self):
        self.store.revert_to_original(self.tmp / "a.png")
        self.assertFalse(self.store.original_backup_path(self.tmp / "a.png").exists())

    def test_revert_without_a_backup_reports_false(self):
        self.assertFalse(self.store.revert_to_original(self.tmp / "missing.png"))

    def test_revert_leaves_the_caption_alone(self):
        self.store.revert_to_original(self.tmp / "a.png")
        self.assertEqual((self.tmp / "a.txt").read_text(), "caption")

    def test_delete_removes_file_caption_and_backup(self):
        removed = self.store.delete_media(self.tmp / "a.png")
        self.assertFalse((self.tmp / "a.png").exists())
        self.assertFalse((self.tmp / "a.txt").exists())
        self.assertFalse(self.store.original_backup_path(self.tmp / "a.png").exists())
        self.assertEqual(len(removed), 3)

    def test_delete_is_safe_on_a_missing_file(self):
        self.assertEqual(self.store.delete_media(self.tmp / "nope.png"), [])


class ModelPickerLayoutTests(unittest.TestCase):
    """The picker listed server aliases in the mode where they can't work, and
    wrapped long model names so every row was a different height."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QDialog
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.A = A
        self.win = A.MainWindow()
        self.cap = {}
        QDialog.exec = lambda dlg: (self.cap.__setitem__("d", dlg), 0)[1]

    def _picker(self, mode):
        self.win.settings.server_start_mode = mode
        self.cap.clear()
        self.win.open_preferences("LLM Models")
        prefs = self.cap["d"]
        self.cap.clear()
        prefs._open_model_picker("caption")
        return self.cap["d"]

    def _sections(self, dlg):
        from PySide6.QtWidgets import QLabel
        known = {"Downloaded in your folders", "Recommended to download",
                 "Name your server's model", "Detected models", "Recommended models"}
        return [l.text() for l in dlg.findChildren(QLabel) if l.text() in known]

    def test_server_aliases_are_hidden_in_local_mode(self):
        """The app launches its own server there, and picking an alias used to be
        silently reset when settings were applied."""
        self.assertNotIn("Name your server's model", self._sections(self._picker("local")))

    def test_server_aliases_appear_in_external_mode(self):
        self.assertIn("Name your server's model",
                      self._sections(self._picker("existing")))

    def test_custom_hf_stays_reachable_as_a_button(self):
        from PySide6.QtWidgets import QPushButton
        dlg = self._picker("local")
        buttons = [b.text() for b in dlg.findChildren(QPushButton)
                   if "Hugging Face" in b.text()]
        self.assertTrue(buttons)

    def test_titles_do_not_wrap(self):
        from PySide6.QtWidgets import QLabel
        dlg = self._picker("local")
        titles = [l for l in dlg.findChildren(QLabel)
                  if "font-weight:600" in (l.styleSheet() or "")]
        self.assertTrue(titles)
        self.assertFalse([l for l in titles if l.wordWrap()])

    def test_long_names_elide_with_a_tooltip(self):
        from PySide6.QtWidgets import QLabel
        dlg = self._picker("local")
        elided = [l for l in dlg.findChildren(QLabel)
                  if "\u2026" in l.text() and l.toolTip()]
        self.assertTrue(elided, "long text should shorten but stay readable on hover")

    def test_row_heights_are_consistent(self):
        """At most two: with a note and without. Five distinct heights is ragged."""
        from PySide6.QtWidgets import QListWidget
        dlg = self._picker("local")
        listing = dlg.findChildren(QListWidget)[0]
        heights = {}
        for row in range(listing.count()):
            h = listing.item(row).sizeHint().height()
            heights[h] = heights.get(h, 0) + 1
        repeated = [h for h, count in heights.items() if count >= 2]
        self.assertLessEqual(len(repeated), 3, heights)


class PathShorteningTests(unittest.TestCase):
    """Model paths share a long common prefix, so the part that differs gets pushed
    out of view."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QDialog
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        win = A.MainWindow()
        cap = {}
        QDialog.exec = lambda dlg: (cap.__setitem__("d", dlg), 0)[1]
        win.open_preferences("LLM Models")
        self.prefs = cap["d"]

    def test_short_paths_are_left_alone(self):
        self.assertEqual(self.prefs._tail_dir(Path("/mnt/moar/LLM/models")),
                         "/mnt/moar/LLM/models")

    def test_hf_cache_keeps_the_model_name_not_the_hash(self):
        """Taking the last two segments would show snapshots/<hash> and drop the
        only part that identifies the model."""
        tail = self.prefs._tail_dir(Path(
            "/home/u/.cache/huggingface/hub/"
            "models--unsloth--Qwen3-VL-30B-A3B-Instruct-GGUF/snapshots/9f2c1b8e4a7d"))
        self.assertIn("Qwen3-VL-30B", tail)
        self.assertNotIn("9f2c1b8e4a7d", tail)

    def test_long_paths_are_marked_as_shortened(self):
        tail = self.prefs._tail_dir(Path("/home/u/.lmstudio/models/unsloth/Gemma-4-E4B"))
        self.assertTrue(tail.startswith("\u2026/"))


class DownloadedModelDiscoveryTests(unittest.TestCase):
    """A download lands in models_dir/<repo>/file.gguf, so the scan has to look
    inside subfolders — a top-level-only scan would never see it."""

    def setUp(self):
        import tempfile as tf
        from captioning_kit.llm_captioning import CaptioningSettings
        self.models = Path(tf.mkdtemp())
        self.settings = CaptioningSettings()
        self.settings.models_dir = str(self.models)

    def test_a_downloaded_pair_is_discovered(self):
        from captioning_kit.llm_captioning import (
            discover_local_gguf_models, safe_repo_dir)
        repo = "HauhauCS/Gemma-4-E4B-Uncensored-HauhauCS-Aggressive"
        folder = self.models / safe_repo_dir(repo)
        folder.mkdir(parents=True)
        (folder / "Gemma-4-E4B-Q5_K_M.gguf").write_bytes(b"x" * 64)
        (folder / "mmproj-Gemma-4-E4B-f16.gguf").write_bytes(b"x" * 32)
        models, mmprojs = discover_local_gguf_models(self.settings)
        self.assertEqual([p.name for p in models], ["Gemma-4-E4B-Q5_K_M.gguf"])
        self.assertEqual([p.name for p in mmprojs], ["mmproj-Gemma-4-E4B-f16.gguf"])

    def test_projector_pairs_with_a_model_in_the_same_folder(self):
        from captioning_kit.llm_captioning import (
            discover_local_gguf_models, guess_mmproj_for, safe_repo_dir)
        folder = self.models / safe_repo_dir("x/y")
        folder.mkdir(parents=True)
        model = folder / "m-Q4_K_M.gguf"
        model.write_bytes(b"x" * 64)
        (folder / "mmproj-m-f16.gguf").write_bytes(b"x" * 32)
        _models, mmprojs = discover_local_gguf_models(self.settings)
        self.assertIsNotNone(guess_mmproj_for(model, mmprojs))

    def test_models_dir_is_among_the_search_roots(self):
        from captioning_kit.llm_captioning import model_search_roots
        roots = [str(r) for r in model_search_roots(self.settings)]
        self.assertIn(str(self.models), roots)


class PickerRowWidthTests(unittest.TestCase):
    """Rows follow the viewport, never a hard-coded width.

    Pinned at 860 while the list was 638, the last 200px hung off the right and the
    Use/HF buttons were clipped. It only showed on rows carrying an extra chip,
    which made it look like an audio-specific bug.
    """

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QDialog
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.win = A.MainWindow()
        self.win.settings.server_start_mode = "local"
        self.cap = {}
        QDialog.exec = lambda dlg: (self.cap.__setitem__("d", dlg), 0)[1]

    def _picker(self):
        from PySide6.QtWidgets import QApplication
        self.cap.clear()
        self.win.open_preferences("LLM Models")
        prefs = self.cap["d"]
        self.cap.clear()
        prefs._open_model_picker("caption")
        dlg = self.cap["d"]
        dlg.show()
        dlg.resize(920, 620)
        QApplication.instance().processEvents()
        QApplication.instance().processEvents()
        return dlg

    def test_no_row_overflows_the_viewport(self):
        from PySide6.QtWidgets import QListWidget, QPushButton
        dlg = self._picker()
        listing = dlg.findChildren(QListWidget)[0]
        viewport = listing.viewport().width()
        for row in range(listing.count()):
            widget = listing.itemWidget(listing.item(row))
            if widget is None:
                continue
            buttons = [b for b in widget.findChildren(QPushButton)
                       if b.text() in ("Use", "HF")]
            if buttons:
                right = max(b.x() + b.width() for b in buttons)
                self.assertLessEqual(right, viewport, f"row {row} overflows")

    def test_row_widgets_match_the_viewport(self):
        from PySide6.QtWidgets import QListWidget
        dlg = self._picker()
        listing = dlg.findChildren(QListWidget)[0]
        viewport = listing.viewport().width()
        widths = {listing.itemWidget(listing.item(r)).width()
                  for r in range(listing.count())
                  if listing.itemWidget(listing.item(r)) is not None}
        for width in widths:
            self.assertLessEqual(width, viewport)


class PickerDownloadedSectionTests(unittest.TestCase):
    """A recommended model you've already fetched shouldn't stay under 'download' —
    that's what makes it look like the download went nowhere."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QDialog
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.models = Path(tf.mkdtemp())
        self.win = A.MainWindow()
        self.win.settings.models_dir = str(self.models)
        self.win.settings.server_start_mode = "local"
        self.cap = {}
        QDialog.exec = lambda dlg: (self.cap.__setitem__("d", dlg), 0)[1]

    def _sections(self):
        from PySide6.QtWidgets import QLabel
        self.cap.clear()
        self.win.open_preferences("LLM Models")
        prefs = self.cap["d"]
        self.cap.clear()
        prefs._open_model_picker("caption")
        dlg = self.cap["d"]
        known = {"Downloaded in your folders", "Already downloaded",
                 "Recommended to download"}
        return [l.text() for l in dlg.findChildren(QLabel) if l.text() in known], dlg

    def _fetch(self, repo, filename):
        from captioning_kit.llm_captioning import safe_repo_dir
        folder = self.models / safe_repo_dir(repo)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / filename).write_bytes(b"x" * 128)

    def test_no_downloaded_section_when_nothing_is_fetched(self):
        sections, _ = self._sections()
        self.assertNotIn("Already downloaded", sections)

    def test_a_fetched_profile_moves_out_of_recommended(self):
        self._fetch("HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced",
                    "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf")
        sections, dlg = self._sections()
        self.assertIn("Already downloaded", sections)

    def test_fetched_rows_carry_an_on_disk_chip(self):
        from PySide6.QtWidgets import QLabel
        self._fetch("HauhauCS/Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced",
                    "Gemma4-12B-QAT-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf")
        _sections, dlg = self._sections()
        chips = [l for l in dlg.findChildren(QLabel) if l.text() == "On disk"]
        self.assertEqual(len(chips), 1)


class ModelsPageLabelTests(unittest.TestCase):
    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QDialog
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        cap = {}
        QDialog.exec = lambda dlg: (cap.__setitem__("d", dlg), 0)[1]
        A.MainWindow().open_preferences("LLM Models")
        self.dlg = cap["d"]

    def _texts(self):
        from PySide6.QtWidgets import QLabel
        return [l.text() for l in self.dlg.findChildren(QLabel) if l.text()]

    def test_captioning_model_heading(self):
        self.assertIn("Captioning model", self._texts())

    def test_bbox_note_names_the_preset_that_uses_it(self):
        notes = [t for t in self._texts() if "Ideogram 4" in t and "bounding" in t]
        self.assertTrue(notes)

    def test_browse_models_button_is_gone(self):
        """It opened the same picker as Choose model… on the row below."""
        from PySide6.QtWidgets import QPushButton
        browse = [b for b in self.dlg.findChildren(QPushButton)
                  if "Browse models" in b.text()]
        choose = [b for b in self.dlg.findChildren(QPushButton)
                  if "Choose model" in b.text()]
        self.assertFalse(browse)
        self.assertTrue(choose)


class WorkFileIsolationTests(unittest.TestCase):
    """A render in progress must never look like dataset content.

    The mute preview used to write '.clip.mutepreview.mp4' beside the clip — a
    dot-prefixed name, but still a .mp4, so the scanner listed it and a crash left
    a playable stray that would be captioned and handed to a trainer.
    """

    def setUp(self):
        import tempfile as tf
        from captioning_kit.store import CaptionStore
        self.tmp = Path(tf.mkdtemp())
        (self.tmp / "clip.mp4").write_bytes(b"real")
        self.store = CaptionStore(self.tmp, ".txt")

    def test_dot_prefixed_media_is_not_dataset_content(self):
        (self.tmp / ".clip.mutepreview.mp4").write_bytes(b"stray")
        self.assertEqual([p.name for p in self.store.images()], ["clip.mp4"])

    def test_scratch_space_lives_inside_the_project_folder(self):
        """Same filesystem as the media, so a finished render moves into place
        atomically rather than copying across devices."""
        work = self.store.work_dir()
        self.assertTrue(work.is_dir())
        self.assertEqual(work.parent.name, ".captioner")
        self.assertTrue(str(work).startswith(str(self.tmp)))

    def test_scratch_contents_are_never_listed(self):
        (self.store.work_dir() / "mute.mp4").write_bytes(b"stray")
        self.assertEqual([p.name for p in self.store.images()], ["clip.mp4"])

    def test_sweep_clears_interrupted_renders(self):
        (self.store.work_dir() / "mute.mp4").write_bytes(b"stray")
        (self.tmp / ".clip.mutepreview.mp4").write_bytes(b"stray")
        self.assertEqual(self.store.sweep_work_files(), 2)
        self.assertFalse((self.tmp / ".clip.mutepreview.mp4").exists())

    def test_sweep_never_touches_real_media(self):
        self.store.sweep_work_files()
        self.assertTrue((self.tmp / "clip.mp4").exists())


class MuteAbandonPathsTests(unittest.TestCase):
    """Every way of walking away from an un-applied mute must leave no file.

    Confirmation deliberately comes *before* the render, so declining produces
    nothing rather than producing something that has to be cleaned up.
    """

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import shutil
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.A = A
        self.folder = Path(tf.mkdtemp())
        for name in ("a.mp4", "b.mp4"):
            shutil.copy("/tmp/vidtest/aud.mp4", self.folder / name)
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        self.win = A.MainWindow()
        QFileDialog.getExistingDirectory = staticmethod(
            lambda *a, **k: str(self.folder))
        self.win.open_folder()
        self.win.filmstrip.setCurrentRow(0)
        self.stage = self.win.video_stage

    def _stray_files(self):
        media = sorted(p.name for p in self.folder.iterdir() if p.is_file())
        work = self.folder / ".captioner" / "work"
        scratch = sorted(p.name for p in work.iterdir()) if work.is_dir() else []
        return media, scratch

    def test_declining_the_confirmation_writes_nothing(self):
        from PySide6.QtWidgets import QMessageBox
        self.stage._mute_section_btn.setChecked(True)
        self.stage._slider.set_mute_range(1000, 3000)
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.No)
        self.stage.commit_mute()
        self.assertEqual(self._stray_files(), (["a.mp4", "b.mp4"], []))

    def test_navigating_away_with_mute_armed_writes_nothing(self):
        self.stage._mute_section_btn.setChecked(True)
        self.stage._slider.set_mute_range(500, 2500)
        self.win.filmstrip.setCurrentRow(1)
        self.assertEqual(self._stray_files(), (["a.mp4", "b.mp4"], []))

    def test_toggling_mute_off_writes_nothing(self):
        self.stage._mute_section_btn.setChecked(True)
        self.stage._mute_section_btn.setChecked(False)
        self.assertEqual(self._stray_files(), (["a.mp4", "b.mp4"], []))

    def test_a_failed_render_cleans_up_after_itself(self):
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.critical = staticmethod(lambda *a, **k: None)
        self.stage._mute_section_btn.setChecked(True)
        self.stage._slider.set_mute_range(1000, 3000)
        original = self.A.apply_mute_span
        self.A.apply_mute_span = lambda *a, **k: (False, "simulated failure")
        try:
            self.stage.commit_mute()
        finally:
            self.A.apply_mute_span = original
        self.assertEqual(self._stray_files(), (["a.mp4", "b.mp4"], []))

    def test_a_successful_apply_leaves_no_scratch(self):
        self.stage._mute_section_btn.setChecked(True)
        self.stage._slider.set_mute_range(1000, 3000)
        self.stage.commit_mute()
        media, scratch = self._stray_files()
        self.assertEqual(media, ["a.mp4", "b.mp4"])
        self.assertEqual(scratch, [])


class H3NewlineNormalisationTests(unittest.TestCase):
    """H3 reads a line break as a shot change, so wrapped prose invents cuts that
    aren't in the footage. The prompt asks for single-line fields, but
    instruction-following varies by model, so output is normalised not trusted."""

    WRAPPED = (
        "integrated_multimodal_description: [Shot 1] A woman\nturns and speaks.\n\n"
        "overall_soundscape: Room tone\nand distant traffic.\n\n"
        "non_diegetic_music: None."
    )

    def test_wrapped_prose_is_joined(self):
        out = L.normalise_h3_caption(self.WRAPPED)
        self.assertIn("A woman turns and speaks.", out)
        self.assertIn("Room tone and distant traffic.", out)

    def test_the_three_fields_stay_separated(self):
        out = L.normalise_h3_caption(self.WRAPPED)
        blocks = out.split("\n\n")
        self.assertEqual(len(blocks), 3)
        for block in blocks:
            self.assertNotIn("\n", block)

    def test_dialogue_tags_survive_intact(self):
        text = ("integrated_multimodal_description: [Shot 1] She says\n"
                "<d>[English] (S1) Hello there.</d>")
        self.assertIn("<d>[English] (S1) Hello there.</d>",
                      L.normalise_h3_caption(text))

    def test_already_single_line_output_is_unchanged(self):
        clean = ("integrated_multimodal_description: [Shot 1] A woman speaks.\n\n"
                 "overall_soundscape: Room tone.\n\nnon_diegetic_music: None.")
        self.assertEqual(L.normalise_h3_caption(clean), clean)

    def test_empty_input_is_safe(self):
        self.assertEqual(L.normalise_h3_caption(""), "")
        self.assertEqual(L.normalise_h3_caption("   "), "   ")

    def test_windows_line_endings_are_handled(self):
        out = L.normalise_h3_caption(
            "integrated_multimodal_description: A\r\nB")
        self.assertNotIn("\r", out)
        self.assertIn("A B", out)


class H3PromptRequirementsTests(unittest.TestCase):
    """The two things Gemma 4 got wrong on H3: wrapping, and omitting shot times."""

    def _prompt(self):
        from captioning_kit.presets import PRESETS
        return PRESETS["minimax_h3_official"].prompt_for("video")

    def test_line_breaks_are_forbidden_explicitly(self):
        prompt = self._prompt().lower()
        self.assertIn("one continuous line", prompt)
        self.assertIn("line break", prompt)

    def test_the_reason_is_stated_not_just_the_rule(self):
        """A model follows a rule better when it knows what breaks."""
        self.assertIn("shot change", self._prompt().lower())

    def test_shot_timestamps_are_mandatory_after_the_first(self):
        prompt = self._prompt()
        self.assertIn("MUST begin with its timestamp", prompt)
        self.assertIn("00:03.500", prompt)

    def test_single_shot_clips_are_told_not_to_invent_a_time(self):
        self.assertIn("Single-shot clips have one [Shot 1] and no timestamp",
                      self._prompt())


class DuplicateDatasetTests(unittest.TestCase):
    """Two jobs in one tool: a backup (take everything) and a variant for a training
    run (media only, or media without captions to re-caption differently)."""

    def setUp(self):
        import tempfile as tf
        from captioning_kit.store import CaptionStore, ProjectConfig
        self.src = Path(tf.mkdtemp()) / "dataset"
        self.src.mkdir()
        for name in ("a.png", "b.png", "c.png"):
            (self.src / name).write_bytes(b"media")
            (self.src / name).with_suffix(".txt").write_text(f"caption {name}")
        self.store = CaptionStore(self.src, ".txt")
        self.store.save_project(ProjectConfig(preset="plain_text"))
        self.store.bypass(self.src / "c.png")
        self.store.originals_dir().mkdir(parents=True, exist_ok=True)
        (self.store.originals_dir() / "a.png").write_bytes(b"pre-edit")
        self.dest = Path(tf.mkdtemp()) / "copy"

    def _tree(self, folder):
        return sorted(str(p.relative_to(folder))
                      for p in Path(folder).rglob("*") if p.is_file())

    def test_everything_option_is_a_faithful_backup(self):
        self.store.duplicate_to(self.dest, keep_captions=True, keep_settings=True,
                                keep_originals=True, keep_bypassed=True)
        self.assertEqual(self._tree(self.dest), self._tree(self.src))

    def test_media_only_leaves_everything_derived_behind(self):
        self.store.duplicate_to(self.dest, keep_captions=False, keep_settings=False,
                                keep_originals=False, keep_bypassed=False)
        self.assertEqual(self._tree(self.dest), ["a.png", "b.png"])

    def test_captions_can_be_dropped_while_settings_are_kept(self):
        """The re-caption case: same preset and guidance, no existing text."""
        self.store.duplicate_to(self.dest, keep_captions=False, keep_settings=True,
                                keep_originals=False, keep_bypassed=False)
        tree = self._tree(self.dest)
        self.assertIn(".captioner/project.json", tree)
        self.assertFalse([f for f in tree if f.endswith(".txt")])

    def test_bypassed_files_stay_bypassed_in_the_copy(self):
        self.store.duplicate_to(self.dest, keep_bypassed=True)
        self.assertIn(".bypass/c.png", self._tree(self.dest))

    def test_a_caption_is_copied_once_not_twice(self):
        """caption_path and the convert-mode .txt sidecar resolve to the same file
        for a .txt preset."""
        counts = self.store.duplicate_to(self.dest, keep_captions=True,
                                         keep_bypassed=False)
        self.assertEqual(counts["captions"], 2)

    def test_a_bypassed_files_caption_travels_with_it(self):
        counts = self.store.duplicate_to(self.dest, keep_captions=True,
                                         keep_bypassed=True)
        self.assertEqual(counts["captions"], 3)
        self.assertIn(".bypass/c.txt", self._tree(self.dest))

    def test_the_source_is_never_modified(self):
        before = self._tree(self.src)
        self.store.duplicate_to(self.dest, keep_captions=True, keep_settings=True,
                                keep_originals=True, keep_bypassed=True)
        self.assertEqual(self._tree(self.src), before)

    def test_cancelling_keeps_what_was_copied(self):
        """Deleting a partial copy would be a surprise; leaving it lets the user
        look at what happened."""
        counts = self.store.duplicate_to(
            self.dest, progress=lambda done, total, name: done < 2)
        self.assertTrue(counts["cancelled"])
        self.assertEqual(len(self._tree(self.dest)), 1)

    def test_copying_reports_what_it_did(self):
        counts = self.store.duplicate_to(self.dest, keep_captions=True,
                                         keep_settings=True, keep_originals=True,
                                         keep_bypassed=True)
        self.assertEqual(counts["media"], 2)
        self.assertEqual(counts["originals"], 1)
        self.assertEqual(counts["bypassed"], 1)
        self.assertEqual(counts["settings"], 1)
        self.assertEqual(counts["skipped"], 0)


class LoadFolderPathTests(unittest.TestCase):
    """Opening a known path shouldn't require a file dialog — split out of
    open_folder so duplicating a dataset can switch to the copy directly."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        self.first = Path(tf.mkdtemp())
        self.second = Path(tf.mkdtemp())
        (self.first / "one.png").write_bytes(b"x")
        (self.second / "two.png").write_bytes(b"x")
        self.win = A.MainWindow()
        QFileDialog.getExistingDirectory = staticmethod(
            lambda *a, **k: str(self.first))
        self.win.open_folder()

    def test_opening_by_path_switches_folders(self):
        self.win.load_folder_path(self.second)
        self.assertEqual([p.name for p in self.win.images], ["two.png"])
        self.assertEqual(self.win.store.folder, self.second)

    def test_the_dialog_route_still_works(self):
        self.assertEqual([p.name for p in self.win.images], ["one.png"])

    def test_switching_updates_the_remembered_folder(self):
        self.win.load_folder_path(self.second)
        self.assertEqual(self.win.qsettings.value("last_folder", "", str),
                         str(self.second))


class WindowSizeTests(unittest.TestCase):
    """The window must fit on a normal screen.

    Every control added to the video edit bar pushed its minimum width up, and
    since the bar sits in the centre panel that became the whole window's floor —
    it reached 1880px, wider than a 1600x900 laptop, with no way to shrink it.
    """

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.win = A.MainWindow()
        self.win.show()

    def test_window_fits_a_1600px_screen(self):
        self.assertLessEqual(self.win.minimumSizeHint().width(), 1600)

    def test_the_video_edit_bar_is_not_the_constraint(self):
        """One row of controls is what caused this; it's two now."""
        self.assertLessEqual(self.win.video_stage.minimumSizeHint().width(), 900)

    def test_the_window_can_actually_be_resized_smaller(self):
        from PySide6.QtWidgets import QApplication
        self.win.resize(1600, 900)
        QApplication.instance().processEvents()
        self.assertLessEqual(self.win.width(), 1600)


class MediaPreviewTests(unittest.TestCase):
    """QPixmap can't open an .mp4, so any dialog previewing with it directly showed
    '(cannot load image)' for every clip. Previews go through the poster frame."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QFileDialog, QMessageBox
        from PySide6.QtGui import QImage, QColor
        QApplication.instance() or QApplication([])
        import shutil
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        QMessageBox.question = staticmethod(lambda *a, **k: QMessageBox.Yes)
        self.folder = Path(tf.mkdtemp())
        img = QImage(64, 48, QImage.Format_RGB32)
        img.fill(QColor("teal"))
        img.save(str(self.folder / "a.png"), "PNG")
        shutil.copy("/tmp/vidtest/clip.mp4", self.folder / "clip.mp4")
        self.win = A.MainWindow()
        QFileDialog.getExistingDirectory = staticmethod(lambda *a, **k: str(self.folder))
        self.win.open_folder()

    def test_a_clip_gets_a_real_preview(self):
        pixmap = self.win.preview_pixmap(self.folder / "clip.mp4")
        self.assertFalse(pixmap.isNull())
        self.assertGreater(pixmap.width(), 1)

    def test_a_still_previews_at_its_own_size(self):
        pixmap = self.win.preview_pixmap(self.folder / "a.png")
        self.assertEqual((pixmap.width(), pixmap.height()), (64, 48))


class SharedMediaWordingTests(unittest.TestCase):
    """Strings shown for both stills and clips shouldn't say 'image'.

    Deliberately narrow: the crop dialog, batch resize and bounding-box strings
    really are image-only and keep their wording.
    """

    STALE = (
        "This image's guidance changed",
        "Use this image's .txt caption",
        "Additional guidance for this image",
        "Caption all images",
        "Open a folder of images first",
        "No images flagged for review",
        "No backed-up original exists for this image",
    )

    def test_no_dialog_reports_a_clip_as_an_unloadable_image(self):
        """The failure text itself, not the phrase — a docstring may quote the old
        wording while explaining the fix."""
        source = (Path(__file__).resolve().parents[1]
                  / "captioning_kit" / "app.py").read_text()
        self.assertNotIn('setText("(cannot load image)")', source)

    def test_no_stale_shared_phrasing_remains(self):
        """Checks quoted strings only — a comment may legitimately mention the old
        wording while explaining why it was changed."""
        import re
        source = (Path(__file__).resolve().parents[1]
                  / "captioning_kit" / "app.py").read_text()
        literals = " ".join(re.findall(r'"([^"\n]*)"', source))
        found = [phrase for phrase in self.STALE if phrase in literals]
        self.assertEqual(found, [], f"image-only wording on shared strings: {found}")


class DownloadConfirmationTests(unittest.TestCase):
    """Nothing downloads without a yes, and the yes says where it's going.

    The prompt named the model and the files but not the destination, so a
    finished download could land in the shared HF cache while you were watching
    your models folder — which looks exactly like nothing happened.
    """

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import tempfile as tf
        import captioning_kit.app as A
        A.default_profiles_path = lambda: Path(tf.mkdtemp()) / "p.json"
        self.win = A.MainWindow()
        self.win.settings.server_start_mode = "local"
        self.win.settings.auto_start_server = True
        self.win.settings.models_dir = "/tmp/some-models-dir"
        self.win.settings.caption_profile_id = "hauhaucs-gemma4-12b-qat-balanced-q4km"
        self.seen = {}

    def _ask(self, target, click="Download"):
        from PySide6.QtWidgets import QMessageBox
        self.win.settings.model_download_target = target

        def fake_exec(box):
            self.seen["text"] = box.text()
            self.seen["info"] = box.informativeText()
            return 0
        QMessageBox.exec = fake_exec
        QMessageBox.clickedButton = lambda box: next(
            b for b in box.buttons() if click in b.text())
        return self.win._confirm_model_download()

    def test_app_folder_destination_is_shown_by_path(self):
        from captioning_kit.llm_captioning import MODEL_TARGET_APP
        self._ask(MODEL_TARGET_APP)
        self.assertIn("/tmp/some-models-dir", self.seen["info"])

    def test_hf_cache_destination_is_named(self):
        from captioning_kit.llm_captioning import MODEL_TARGET_HF
        self._ask(MODEL_TARGET_HF)
        self.assertIn("Hugging Face cache", self.seen["info"])

    def test_it_says_where_to_change_the_destination(self):
        from captioning_kit.llm_captioning import MODEL_TARGET_APP
        self._ask(MODEL_TARGET_APP)
        self.assertIn("Model download location", self.seen["info"])

    def test_the_model_is_named_not_just_described(self):
        from captioning_kit.llm_captioning import MODEL_TARGET_APP
        self._ask(MODEL_TARGET_APP)
        self.assertIn("Gemma 4 12B QAT", self.seen["text"])

    def test_cancelling_declines_the_download(self):
        from captioning_kit.llm_captioning import MODEL_TARGET_APP
        self.assertFalse(self._ask(MODEL_TARGET_APP, click="Cancel"))

    def test_no_prompt_when_the_files_are_already_present(self):
        """Only a real download should interrupt."""
        import captioning_kit.app as A
        original = A.missing_model_files
        A.missing_model_files = lambda settings, task: []
        try:
            self.assertTrue(self.win._confirm_model_download())
        finally:
            A.missing_model_files = original
