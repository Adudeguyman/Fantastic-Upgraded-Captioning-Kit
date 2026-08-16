import json
import tempfile
import unittest
from pathlib import Path

from captioning_kit.presets import (
    DEFAULT_PRESET,
    PRESET_ORDER,
    PRESETS,
    caption_extensions,
    get_preset,
)
from captioning_kit.store import CaptionStore, ProjectConfig


class PresetDefinitionTests(unittest.TestCase):
    def test_plain_text_is_the_default(self):
        self.assertEqual(DEFAULT_PRESET, "plain_text")
        self.assertTrue(get_preset(None).is_plain)
        self.assertEqual(get_preset(None).extension, ".txt")

    def test_unknown_key_falls_back_to_default(self):
        self.assertEqual(get_preset("nope").key, DEFAULT_PRESET)
        self.assertEqual(get_preset("").key, DEFAULT_PRESET)

    def test_ideogram_preset_keeps_structured_features(self):
        ideo = get_preset("ideogram4")
        self.assertFalse(ideo.is_plain)
        self.assertEqual(ideo.extension, ".json")
        self.assertTrue(ideo.has_boxes)
        self.assertTrue(ideo.validates)

    def test_plain_preset_has_no_boxes_or_schema(self):
        plain = get_preset("plain_text")
        self.assertFalse(plain.has_boxes)
        self.assertFalse(plain.validates)
        self.assertIn("caption", plain.system_prompt.lower())

    def test_every_ordered_key_exists(self):
        for key in PRESET_ORDER:
            self.assertIn(key, PRESETS)

    def test_caption_extensions(self):
        self.assertEqual(caption_extensions(), {".txt", ".json"})


class PlainCaptionStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.img = self.tmp / "shot.png"
        self.img.write_bytes(b"x")
        self.store = CaptionStore(self.tmp, ".txt")

    def test_round_trip(self):
        self.store.save_plain_caption(self.img, "  a cat on a sofa  ")
        text, message = self.store.load_plain_caption(self.img)
        self.assertEqual(text, "a cat on a sofa")
        self.assertIsNone(message)

    def test_missing_sidecar_is_empty_not_an_error(self):
        self.assertEqual(self.store.load_plain_caption(self.img), ("", None))

    def test_saves_next_to_the_image(self):
        path = self.store.save_plain_caption(self.img, "hello")
        self.assertEqual(path, self.tmp / "shot.txt")


class ProjectPresetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "a.png").write_bytes(b"x")
        self.store = CaptionStore(self.tmp, ".txt")

    def test_defaults_to_plain_text(self):
        self.assertEqual(ProjectConfig().preset, DEFAULT_PRESET)

    def test_preset_persists_per_folder(self):
        self.store.save_project(ProjectConfig(preset="ideogram4"))
        self.assertEqual(self.store.load_project().preset, "ideogram4")

    def test_unknown_preset_in_file_normalises(self):
        self.store.save_project(ProjectConfig(preset="ideogram4"))
        path = self.store.project_path()
        data = json.loads(path.read_text())
        data["preset"] = "from-the-future"
        path.write_text(json.dumps(data))
        self.assertEqual(self.store.load_project().preset, DEFAULT_PRESET)


if __name__ == "__main__":
    unittest.main()


class LegacyPerImageKeyTests(unittest.TestCase):
    """Datasets captioned before per-image guidance was renamed to per-file still
    carry the old JSON keys; they must keep loading, and migrate on the next save."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "a.png").write_bytes(b"x")
        self.store = CaptionStore(self.tmp, ".txt")
        self.store.project_dir().mkdir(parents=True, exist_ok=True)
        self.store.project_path().write_text(json.dumps({
            "name": "legacy",
            "folder_guidance": "folder note",
            "per_image": {"a.png": "image note"},
            "per_image_enabled": {"a.png": True},
        }))

    def test_legacy_keys_are_read(self):
        cfg = self.store.load_project()
        self.assertEqual(cfg.per_file, {"a.png": "image note"})
        self.assertEqual(cfg.per_file_enabled, {"a.png": True})

    def test_legacy_guidance_still_resolves(self):
        cfg = self.store.load_project()
        resolved = cfg.resolved_for("a.png")
        self.assertIn("folder note", resolved)
        self.assertIn("image note", resolved)

    def test_saving_migrates_to_the_new_key(self):
        self.store.save_project(self.store.load_project())
        data = json.loads(self.store.project_path().read_text())
        self.assertIn("per_file", data)
        self.assertNotIn("per_image", data)


class MiniMaxPresetTests(unittest.TestCase):
    """Two H3 presets: MiniMax's own structured format, and the natural-language
    style. Both target the same model, so both arm the same trim conform."""

    def setUp(self):
        self.official = get_preset("minimax_h3_official")
        self.natural = get_preset("minimax_h3_natural")

    def test_both_are_plain_text_presets(self):
        for preset in (self.official, self.natural):
            self.assertTrue(preset.is_plain, preset.key)
            self.assertEqual(preset.extension, ".txt", preset.key)
            self.assertFalse(preset.has_boxes, preset.key)

    def test_both_are_bound_to_the_h3_model_target(self):
        from captioning_kit.model_targets import builtin_map
        for preset in (self.official, self.natural):
            self.assertEqual(preset.model_target, "minimax_h3", preset.key)
            self.assertIn(preset.model_target, builtin_map(), preset.key)

    def test_official_carries_minimax_tag_syntax(self):
        prompt = self.official.prompt_for("video")
        for token in ("<d>", "</d>", "[English]", "(S1)", "<cutoff>", "<scenetrans>",
                      "integrated_multimodal_description", "overall_soundscape",
                      "non_diegetic_music"):
            self.assertIn(token, prompt, token)

    def test_natural_has_dialogue_without_the_tag_syntax(self):
        """The natural style still has to capture speech — H3 generates audio — it
        just quotes it inline instead of tagging it."""
        prompt = self.natural.prompt_for("video")
        self.assertIn("dialogue", prompt.lower())
        self.assertIn("quote", prompt.lower())
        self.assertNotIn("<d>", prompt)
        self.assertNotIn("integrated_multimodal_description", prompt)

    def test_video_prompts_cover_sound_and_image_prompts_forbid_it(self):
        for preset in (self.official, self.natural):
            video = preset.prompt_for("video").lower()
            image = preset.prompt_for("image").lower()
            self.assertIn("audio", video + " " + video, preset.key)
            self.assertNotEqual(video, image, preset.key)
            # a still has no soundtrack; inventing one poisons the pair
            self.assertTrue("n/a" in image or "never describe audio" in image,
                            preset.key)

    def test_image_prompts_do_not_ask_for_camera_movement(self):
        for preset in (self.official, self.natural):
            image = preset.prompt_for("image").lower()
            self.assertIn("no motion" if preset is self.official else "no camera",
                          image.replace("do not describe camera movement", "no camera"),
                          preset.key)

    def test_natural_style_prompts_stay_prose(self):
        for media in ("image", "video"):
            self.assertIn("flowing paragraph", self.natural.prompt_for(media).lower())

    def test_both_appear_in_the_ordered_list(self):
        self.assertIn("minimax_h3_official", PRESET_ORDER)
        self.assertIn("minimax_h3_natural", PRESET_ORDER)
        self.assertEqual(len(PRESET_ORDER), len(PRESETS))


class MediaSpecificPromptTests(unittest.TestCase):
    """Stills and clips need different guidance, so every preset resolves its prompt
    by media with a fallback to the shared one."""

    def test_plain_text_differs_by_media(self):
        preset = get_preset("plain_text")
        self.assertNotEqual(preset.prompt_for("image"), preset.prompt_for("video"))
        self.assertIn("text-to-image", preset.prompt_for("image"))
        self.assertIn("text-to-video", preset.prompt_for("video"))

    def test_video_prompt_mentions_speech(self):
        self.assertIn("say", get_preset("plain_text").prompt_for("video").lower())

    def test_unset_override_falls_back_to_the_shared_prompt(self):
        preset = get_preset("ideogram4")
        self.assertEqual(preset.prompt_for("image"), preset.system_prompt)
        self.assertEqual(preset.prompt_for("video"), preset.system_prompt)

    def test_unknown_media_falls_back(self):
        preset = get_preset("plain_text")
        self.assertEqual(preset.prompt_for("hologram"), preset.system_prompt)


class MediaModeTests(unittest.TestCase):
    """The Photos/Videos choice is stored per folder alongside the preset, so a
    dataset reopens captioning the kind of media it was built for."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "a.png").write_bytes(b"x")
        self.store = CaptionStore(self.tmp, ".txt")

    def test_defaults_to_auto(self):
        self.assertEqual(ProjectConfig().media_mode, "auto")

    def test_persists_per_folder(self):
        self.store.save_project(ProjectConfig(media_mode="video"))
        self.assertEqual(self.store.load_project().media_mode, "video")

    def test_unknown_value_falls_back_to_auto(self):
        self.store.save_project(ProjectConfig(media_mode="video"))
        path = self.store.project_path()
        data = json.loads(path.read_text())
        data["media_mode"] = "holograms"
        path.write_text(json.dumps(data))
        self.assertEqual(self.store.load_project().media_mode, "auto")

    def test_only_presets_with_distinct_guidance_offer_the_choice(self):
        self.assertTrue(get_preset("plain_text").has_media_variants)
        self.assertTrue(get_preset("minimax_h3_official").has_media_variants)
        self.assertTrue(get_preset("minimax_h3_natural").has_media_variants)
        # Ideogram has one prompt for everything, so the toggle would be noise
        self.assertFalse(get_preset("ideogram4").has_media_variants)
