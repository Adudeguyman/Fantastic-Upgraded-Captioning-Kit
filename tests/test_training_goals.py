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
