import json
import tempfile
import unittest
from pathlib import Path

from ideogram_captioner.store import CaptionStore, ProjectConfig


class StoreTests(unittest.TestCase):
    def test_lists_images_and_saves_matching_caption_stem(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.PNG"
            image.write_bytes(b"not actually loaded by the store")
            (folder / "sample.txt").write_text("plain caption", encoding="utf-8")

            store = CaptionStore(folder, ".caption")
            self.assertEqual(store.images(), [image])

            saved_path = store.save_caption(
                image,
                {
                    "high_level_description": "A sign",
                    "compositional_deconstruction": {"background": "wall", "elements": []},
                },
            )

            self.assertEqual(saved_path, folder / "sample.caption")
            raw = saved_path.read_text(encoding="utf-8")
            # captions are written pretty-printed (indented, multi-line) for readability
            self.assertIn("\n  ", raw)
            self.assertIn('"high_level_description": "A sign"', raw)
            self.assertEqual(json.loads(raw)["high_level_description"], "A sign")

    def test_imports_plain_text_caption_files(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.jpg"
            image.write_bytes(b"x")
            (folder / "sample.txt").write_text("a concise plain caption", encoding="utf-8")

            caption, message = CaptionStore(folder, ".txt").load_caption(image)

            self.assertIn("Imported plain text", message)
            self.assertEqual(caption["high_level_description"], "a concise plain caption")

    def test_missing_caption_loads_blank_and_saves_new_json(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.jpg"
            image.write_bytes(b"x")
            store = CaptionStore(folder, ".json")

            caption, message = store.load_caption(image)
            self.assertIn("click Save to create it", message)
            self.assertEqual(caption["high_level_description"], "")

            caption["high_level_description"] = "A new caption"
            saved_path = store.save_caption(image, caption)

            self.assertEqual(saved_path, folder / "sample.json")
            saved = json.loads(saved_path.read_text(encoding="utf-8"))
            self.assertEqual(saved["high_level_description"], "A new caption")
            self.assertIn("style_description", saved)
            self.assertIn("compositional_deconstruction", saved)

    def test_failure_marker_round_trip(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            image = folder / "sample.jpg"
            image.write_bytes(b"x")
            store = CaptionStore(folder, ".json")

            path = store.save_failure_marker(
                image,
                {
                    "operation": "json_image",
                    "reason": "caption_json_parse_failed",
                    "error": "Could not parse model JSON",
                },
            )

            self.assertEqual(path, folder / "sample.caption_failed.json")
            self.assertTrue(store.has_failure_marker(image))
            marker = store.load_failure_marker(image)
            self.assertIsNotNone(marker)
            self.assertEqual(marker["operation"], "json_image")
            self.assertTrue(store.clear_failure_marker(image))
            self.assertFalse(store.has_failure_marker(image))
            self.assertFalse(store.clear_failure_marker(image))

    def test_edit_folder_is_not_listed_as_source_images(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            edit_folder = folder / "edit"
            edit_folder.mkdir()
            (edit_folder / "sample.jpg").write_bytes(b"x")

            self.assertEqual(CaptionStore(edit_folder, ".json").images(), [])

    def test_caption_flags_persist_clear_and_prune(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "a.png").write_bytes(b"x")
            (folder / "b.png").write_bytes(b"x")
            store = CaptionStore(folder, ".json")

            proj = ProjectConfig(name="t")
            proj.set_flags("a.png", ["missing structured sections"])
            proj.set_flags("b.png", [])              # empty list clears
            proj.set_flags("ghost.png", ["orphan"])  # no matching image
            store.save_project(proj)

            reloaded = store.load_project()
            self.assertTrue(reloaded.is_flagged("a.png"))
            self.assertEqual(reloaded.caption_issues("a.png"), ["missing structured sections"])
            self.assertFalse(reloaded.is_flagged("b.png"))
            self.assertFalse(reloaded.is_flagged("ghost.png"))  # pruned on load

            reloaded.clear_flag("a.png")
            self.assertFalse(reloaded.is_flagged("a.png"))

    def test_review_marks_persist_toggle_and_prune(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "a.png").write_bytes(b"x")
            (folder / "b.png").write_bytes(b"x")
            store = CaptionStore(folder, ".json")

            proj = ProjectConfig(name="t")
            self.assertTrue(proj.toggle_review_mark("a.png"))   # now marked
            self.assertFalse(proj.toggle_review_mark("a.png"))  # toggled off
            proj.set_review_mark("a.png", True)
            proj.set_review_mark("ghost.png", True)             # no matching image
            store.save_project(proj)

    def test_convert_flag_round_trips_only_when_on(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder / "a.png").write_bytes(b"x")
            store = CaptionStore(folder, ".json")
            store.save_project(ProjectConfig(convert_txt_to_json=True))
            self.assertTrue(store.load_project().convert_txt_to_json)
            self.assertIs(json.loads(store.project_path().read_text()).get("convert_txt_to_json"), True)
            store.save_project(ProjectConfig(convert_txt_to_json=False))
            self.assertNotIn("convert_txt_to_json", json.loads(store.project_path().read_text()))
            self.assertFalse(store.load_project().convert_txt_to_json)

    def test_source_text_sidecar_detection(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            img = folder / "shot.jpg.webp"
            img.write_bytes(b"x")
            store = CaptionStore(folder, ".json")
            self.assertFalse(store.has_source_text(img))
            self.assertEqual(store.load_source_text(img), "")
            (folder / "shot.jpg.txt").write_text("  four puppies, indoor \n", encoding="utf-8")
            self.assertEqual(store.source_text_path(img).name, "shot.jpg.txt")
            self.assertTrue(store.has_source_text(img))
            self.assertEqual(store.load_source_text(img), "four puppies, indoor")
            # when .txt IS the caption extension, there is no separate source
            txt_store = CaptionStore(folder, ".txt")
            self.assertFalse(txt_store.has_source_text(img))
            self.assertEqual(txt_store.load_source_text(img), "")

    def test_any_source_text_folder_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            imgs = [folder / "a.png", folder / "b.png"]
            for p in imgs:
                p.write_bytes(b"x")
            store = CaptionStore(folder, ".json")
            self.assertFalse(store.any_source_text(imgs))
            (folder / "b.txt").write_text("hello", encoding="utf-8")
            self.assertTrue(store.any_source_text(imgs))


if __name__ == "__main__":
    unittest.main()
