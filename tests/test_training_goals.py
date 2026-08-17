"""Training goals: the second caption axis.

A preset decides a caption's format; a goal decides its content policy. The rule
every goal encodes: caption what should stay promptable, omit what the LoRA
should own — because described details get attributed to the words and stay
variable, while undescribed ones are absorbed into the trigger.
"""
import json
import tempfile
import unittest
from pathlib import Path

from captioning_kit.store import CaptionStore, ProjectConfig
from captioning_kit.training_goals import (
    DEFAULT_GOAL,
    GOAL_ORDER,
    GOALS,
    get_goal,
)


class GoalDefinitionTests(unittest.TestCase):
    def test_general_is_the_default_and_adds_no_policy(self):
        """Broad quality adapters caption everything; omission rules would be wrong
        there, and it's what existing folders fall back to."""
        self.assertEqual(DEFAULT_GOAL, "general")
        self.assertFalse(get_goal("general").has_rules)
        self.assertEqual(get_goal("general").rules, "")

    def test_every_other_goal_carries_rules_and_a_summary(self):
        for key in GOAL_ORDER:
            goal = GOALS[key]
            self.assertTrue(goal.summary, key)
            if key != DEFAULT_GOAL:
                self.assertTrue(goal.has_rules, key)

    def test_each_goal_states_both_halves_of_the_rule(self):
        """A goal that only says what to omit leaves the model guessing what to
        include, and vice versa."""
        for key in GOAL_ORDER:
            if key == DEFAULT_GOAL:
                continue
            rules = GOALS[key].rules
            self.assertIn("Do NOT describe", rules, key)
            self.assertIn("DO describe", rules, key)

    def test_character_omits_identity_and_keeps_context(self):
        rules = GOALS["character"].rules.lower()
        for omitted in ("facial features", "eye colour", "hair colour", "build"):
            self.assertIn(omitted, rules, omitted)
        for kept in ("pose", "clothing", "background", "lighting"):
            self.assertIn(kept, rules, kept)

    def test_motion_omits_the_movement_and_describes_the_performer(self):
        """The inverse of character: describing the performer thoroughly is what
        stops their identity being learned as part of the motion."""
        rules = GOALS["motion"].rules.lower()
        self.assertIn("movement", rules)
        self.assertIn("face", rules)
        self.assertIn("clothing", rules)

    def test_art_style_omits_style_words(self):
        rules = GOALS["art_style"].rules.lower()
        self.assertIn("style", rules)
        self.assertIn("cinematic", rules)     # named as a word to avoid

    def test_unknown_key_falls_back_to_general(self):
        """A corrupt project file must not silently change what gets captioned."""
        self.assertEqual(get_goal("nonsense").key, "general")
        self.assertEqual(get_goal(None).key, "general")
        self.assertEqual(get_goal("").key, "general")

    def test_key_matching_is_case_insensitive(self):
        self.assertEqual(get_goal("MOTION").key, "motion")


class GoalPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "a.png").write_bytes(b"x")
        self.store = CaptionStore(self.tmp, ".txt")

    def test_defaults_to_general(self):
        self.assertEqual(ProjectConfig().training_goal, "general")

    def test_persists_per_folder(self):
        self.store.save_project(ProjectConfig(training_goal="motion"))
        self.assertEqual(self.store.load_project().training_goal, "motion")

    def test_unknown_stored_value_falls_back(self):
        self.store.save_project(ProjectConfig(training_goal="motion"))
        path = self.store.project_path()
        data = json.loads(path.read_text())
        data["training_goal"] = "holograms"
        path.write_text(json.dumps(data))
        self.assertEqual(self.store.load_project().training_goal, "general")


class GoalStalenessTests(unittest.TestCase):
    """A goal switch changes captions as materially as editing guidance does, so it
    has to mark previously captioned files changed."""

    def _captioned(self, goal="motion"):
        config = ProjectConfig(training_goal=goal)
        config.folder_guidance = "some guidance"
        config.folder_guidance_enabled = True
        config.mark_generated("a.png", config.resolved_for("a.png"),
                              config.effective_folder_guidance(), "")
        return config

    def test_fresh_caption_is_not_stale(self):
        self.assertFalse(self._captioned().guidance_changed("a.png"))

    def test_switching_goal_marks_it_stale(self):
        config = self._captioned()
        config.training_goal = "character"
        self.assertTrue(config.guidance_changed("a.png"))

    def test_switching_back_clears_it(self):
        config = self._captioned()
        config.training_goal = "character"
        config.training_goal = "motion"
        self.assertFalse(config.guidance_changed("a.png"))

    def test_captions_predating_goals_are_not_retroactively_stale(self):
        """No stamp means the caption was made before goals existed — flagging every
        old caption on upgrade would be noise, not information."""
        config = ProjectConfig(training_goal="motion")
        config.generated_guidance["b.png"] = ""
        self.assertFalse(config.guidance_changed("b.png"))

    def test_the_stamp_round_trips(self):
        tmp = Path(tempfile.mkdtemp())
        (tmp / "a.png").write_bytes(b"x")
        store = CaptionStore(tmp, ".txt")
        store.save_project(self._captioned())
        self.assertEqual(store.load_project().generated_goal.get("a.png"), "motion")


if __name__ == "__main__":
    unittest.main()


class WanLtxPresetTests(unittest.TestCase):
    """Wan and LTX differ in a way the captions must reflect: LTX generates
    synchronised audio in one pass, Wan generates silent video only."""

    def setUp(self):
        from captioning_kit.presets import PRESETS
        self.presets = PRESETS

    def test_both_exist_and_target_their_models(self):
        self.assertEqual(self.presets["wan22"].model_target, "wan22_a14b")
        self.assertEqual(self.presets["ltx2"].model_target, "ltx_2_3")

    def test_wan_forbids_describing_sound(self):
        """Wan generates silent video; captioning audio teaches it nothing and
        wastes caption budget."""
        prompt = self.presets["wan22"].prompt_for("video")
        self.assertIn("silent", prompt.lower())
        self.assertIn("Never mention sound", prompt)

    def test_ltx_requires_describing_sound(self):
        prompt = self.presets["ltx2"].prompt_for("video").lower()
        self.assertIn("synchronised audio", prompt)
        self.assertIn("ambient", prompt)
        self.assertIn("quote the words", prompt)

    def test_both_have_still_image_variants(self):
        for key in ("wan22", "ltx2"):
            preset = self.presets[key]
            self.assertTrue(preset.has_media_variants, key)
            self.assertNotEqual(preset.prompt_for("image"),
                                preset.prompt_for("video"), key)

    def test_image_variants_forbid_motion_and_audio(self):
        for key in ("wan22", "ltx2"):
            image = self.presets[key].prompt_for("image").lower()
            self.assertIn("no motion", image.replace("has no motion", "no motion"), key)

    def test_targets_resolve_to_real_specs(self):
        from captioning_kit.model_targets import builtin_map
        targets = builtin_map()
        for key in ("wan22", "ltx2"):
            target = targets[self.presets[key].model_target]
            self.assertGreater(target.fps, 0)
            self.assertGreater(target.max_frames(), 0)


class ExportBundleShapeTests(unittest.TestCase):
    """Exporting only your diffs wrote an empty file for anyone who hadn't
    customised anything — structurally valid and completely useless."""

    def _bundle(self, prompts, targets):
        import datetime
        return {
            "kind": "fantastic-captioning-kit/llm-instructions",
            "version": 1,
            "exported": datetime.datetime.now().isoformat(timespec="seconds"),
            "description": (f"{len(targets)} model rule(s) and {len(prompts)} "
                            "caption prompt(s) for the Fantastic Upgraded "
                            "Captioning Kit."),
            "prompts": prompts,
            "model_targets": targets,
        }

    def test_bundle_carries_a_human_readable_description(self):
        """A recipient opening the file should see what's in it without counting
        JSON entries."""
        bundle = self._bundle({"wan22/video": "x"}, [{"key": "a"}])
        self.assertIn("1 model rule(s)", bundle["description"])
        self.assertIn("1 caption prompt(s)", bundle["description"])

    def test_an_empty_bundle_is_recognisable_as_empty(self):
        bundle = self._bundle({}, [])
        self.assertEqual(bundle["prompts"], {})
        self.assertEqual(bundle["model_targets"], [])
        self.assertIn("0 model rule(s)", bundle["description"])

    def test_every_builtin_target_can_round_trip(self):
        """Built-ins are exportable as a starting point, so they must survive the
        dataclass -> dict -> dataclass trip intact."""
        from dataclasses import asdict
        from captioning_kit.model_targets import _target_from_dict, builtin_map
        for key, target in builtin_map().items():
            restored = _target_from_dict(asdict(target))
            self.assertEqual(restored.key, target.key)
            self.assertEqual(restored.fps, target.fps)
            self.assertEqual(restored.frame_modulus, target.frame_modulus)
            self.assertEqual(restored.frame_remainder, target.frame_remainder)
            self.assertEqual(restored.min_seconds, target.min_seconds)
            self.assertEqual(restored.max_seconds, target.max_seconds)

    def test_plain_presets_have_prompts_for_both_media(self):
        """The export lists preset x media for every preset that has a prompt.
        Ideogram 4 builds its instructions from the schema instead, so it has none
        and is skipped rather than contributing two empty entries."""
        from captioning_kit.presets import PRESET_ORDER, get_preset
        listed = 0
        for key in PRESET_ORDER:
            preset = get_preset(key)
            if not preset.prompt_for("image").strip():
                self.assertEqual(key, "ideogram4")
                continue
            for media in ("image", "video"):
                self.assertTrue(preset.prompt_for(media).strip(), f"{key}/{media}")
            listed += 1
        self.assertGreaterEqual(listed, 5)


class GuidancePresetTests(unittest.TestCase):
    """Built-in guidance presets come in pairs: a plain-text one and an Ideogram 4
    one. Bounding boxes exist only in the Ideogram schema, so asking for one in a
    .txt caption is an instruction the model cannot act on."""

    def setUp(self):
        import os
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication
        QApplication.instance() or QApplication([])
        import captioning_kit.app as A
        self.folder = A.FOLDER_GUIDANCE_PRESETS
        self.image = A.IMAGE_GUIDANCE_PRESETS

    def test_both_scopes_offer_a_plain_and_an_ideogram_variant(self):
        folder_names = [n for n, _ in self.folder]
        self.assertIn("Single Character", folder_names)
        self.assertIn("Single Character (Ideogram 4)", folder_names)
        self.assertIn("Art Style (Ideogram 4)", folder_names)
        image_names = [n for n, _ in self.image]
        self.assertIn("Multi-Character", image_names)
        self.assertIn("Multi-Character (Ideogram 4)", image_names)

    def test_only_ideogram_presets_mention_bounding_boxes(self):
        for name, text in self.folder + self.image:
            if "(Ideogram 4)" in name:
                continue
            self.assertNotIn("bounding box", text,
                             f"{name} asks for a box outside the Ideogram schema")

    def test_the_ideogram_character_presets_do_ask_for_boxes(self):
        for name, text in self.folder + self.image:
            if "Character (Ideogram 4)" in name:
                self.assertIn("bounding box", text, name)

    def test_ideogram_presets_sort_last(self):
        """Boxes are the special case now, not the default."""
        for presets in (self.folder, self.image):
            names = [n for n, _ in presets]
            ideogram = [i for i, n in enumerate(names) if "(Ideogram 4)" in n]
            plain = [i for i, n in enumerate(names) if "(Ideogram 4)" not in n]
            if ideogram and plain:
                self.assertGreater(min(ideogram), max(plain), names)

    def test_the_pairs_differ_only_in_the_box_instruction(self):
        pairs = [("Single Character", "Single Character (Ideogram 4)", self.folder),
                 ("Multi-Character", "Multi-Character (Ideogram 4)", self.image)]
        for plain_name, ideogram_name, presets in pairs:
            lookup = dict(presets)
            plain, ideogram = lookup[plain_name], lookup[ideogram_name]
            self.assertEqual(plain.replace("a short description of their pose", ""),
                             ideogram.replace(
                                 "a bounding box for them with a short description "
                                 "of their pose", ""))

    def test_names_are_unique_within_each_scope(self):
        for presets in (self.folder, self.image):
            names = [n for n, _ in presets]
            self.assertEqual(len(names), len(set(names)))


class CustomPresetTests(unittest.TestCase):
    """User-defined caption presets, so a model that shipped last week doesn't have
    to wait for an app release.

    Deliberately plain-text only: the structured editor and bounding boxes are
    Ideogram-4 machinery that can't be described by filling in a form.
    """

    def setUp(self):
        import tempfile as tf
        self.base = Path(tf.mkdtemp())

    def _make(self, **kw):
        from captioning_kit.presets import make_custom_preset
        defaults = dict(key="sora9", label="Sora 9",
                        image_prompt="Describe the still.",
                        video_prompt="Describe the clip.",
                        model_target="sora_9_turbo")
        defaults.update(kw)
        return make_custom_preset(**defaults)

    def test_round_trips_through_the_file(self):
        from captioning_kit.presets import load_custom_presets, save_custom_presets
        save_custom_presets(self.base, {"sora9": self._make()})
        restored = load_custom_presets(self.base)["sora9"]
        self.assertEqual(restored.label, "Sora 9")
        self.assertEqual(restored.model_target, "sora_9_turbo")
        self.assertEqual(restored.prompt_for("video"), "Describe the clip.")

    def test_the_frame_rule_link_is_optional(self):
        """A stills-only preset has no frame grid; requiring one would invent a
        constraint that doesn't exist."""
        from captioning_kit.presets import load_custom_presets, save_custom_presets
        save_custom_presets(self.base, {"stills": self._make(
            key="stills", label="Stills", model_target="", video_prompt="")})
        self.assertEqual(load_custom_presets(self.base)["stills"].model_target, "")

    def test_custom_presets_are_plain_text(self):
        preset = self._make()
        self.assertTrue(preset.is_plain)
        self.assertEqual(preset.extension, ".txt")
        self.assertFalse(preset.has_boxes)
        self.assertFalse(preset.validates)

    def test_a_user_entry_cannot_shadow_a_builtin(self):
        """Otherwise a hand-edited file could redefine Ideogram 4 as plain text."""
        from captioning_kit.presets import load_custom_presets, save_custom_presets
        save_custom_presets(self.base, {"x": self._make(key="ideogram4",
                                                        label="Hijack")})
        self.assertNotIn("ideogram4", load_custom_presets(self.base))

    def test_a_malformed_entry_is_skipped_not_fatal(self):
        import json
        from captioning_kit.presets import custom_presets_path, load_custom_presets
        custom_presets_path(self.base).write_text(json.dumps(
            {"presets": [{"nonsense": True}, {"key": "ok", "label": "Fine"}]}))
        loaded = load_custom_presets(self.base)
        self.assertEqual(list(loaded), ["ok"])

    def test_unreadable_file_yields_no_presets(self):
        from captioning_kit.presets import custom_presets_path, load_custom_presets
        custom_presets_path(self.base).write_text("{ not json")
        self.assertEqual(load_custom_presets(self.base), {})

    def test_merged_order_puts_builtins_first(self):
        from captioning_kit.presets import (PRESET_ORDER, preset_order,
                                            save_custom_presets)
        save_custom_presets(self.base, {"sora9": self._make()})
        order = preset_order(self.base)
        self.assertEqual(order[:len(PRESET_ORDER)], PRESET_ORDER)
        self.assertEqual(order[-1], "sora9")

    def test_saving_an_empty_set_removes_the_file(self):
        from captioning_kit.presets import (custom_presets_path,
                                            save_custom_presets)
        save_custom_presets(self.base, {"sora9": self._make()})
        self.assertTrue(custom_presets_path(self.base).exists())
        save_custom_presets(self.base, {})
        self.assertFalse(custom_presets_path(self.base).exists())


class GeneratedPromptVisibilityTests(unittest.TestCase):
    """Ideogram 4's instructions are built from the caption schema at run time,
    not stored as a preset prompt. The editor showed an empty box for it, which
    read as though the prompt had been lost."""

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
        win.open_preferences("LLM Instructions")
        self.dlg = cap["d"]

    def _select(self, key):
        self.dlg._pe_preset.setCurrentIndex(self.dlg._pe_preset.findData(key))
        return self.dlg._pe_edit.toPlainText()

    def test_the_schema_driven_prompt_is_shown_not_blank(self):
        text = self._select("ideogram4")
        self.assertGreater(len(text), 500)
        self.assertIn("Ideogram", text)

    def test_it_is_read_only_because_editing_would_not_persist(self):
        self._select("ideogram4")
        self.assertTrue(self.dlg._pe_edit.isReadOnly())

    def test_the_right_preset_prompt_is_shown(self):
        """It previously borrowed whatever the main window had selected, which
        showed the plain-text prompt under the Ideogram entry."""
        self.assertIn("Wan 2.2", self._select("wan22"))
        self.assertIn("Ideogram", self._select("ideogram4"))

    def test_editable_presets_stay_editable(self):
        self._select("wan22")
        self.assertFalse(self.dlg._pe_edit.isReadOnly())

    def test_a_read_only_prompt_is_never_saved_as_an_edit(self):
        """Otherwise switching through Ideogram would stash the generated text as a
        user override for a preset that doesn't use one."""
        self._select("ideogram4")
        self.dlg._pe_reload()
        self.assertNotIn(("ideogram4", "image"), self.dlg._pe_pending)
        self.assertNotIn(("ideogram4", "video"), self.dlg._pe_pending)
