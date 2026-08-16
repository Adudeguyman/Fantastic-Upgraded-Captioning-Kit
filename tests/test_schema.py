import json
import unittest

from captioning_kit.schema import caption_health, normalize_caption, parse_palette_text, serialize_caption


class SchemaTests(unittest.TestCase):
    def test_serializes_compact_json_with_literal_unicode(self):
        text = serialize_caption(
            {
                "high_level_description": "Café sign",
                "style_description": {
                    "aesthetics": "clean",
                    "lighting": "soft",
                    "photo": "50mm",
                    "medium": "photograph",
                },
                "compositional_deconstruction": {"background": "street", "elements": []},
            }
        )

        self.assertNotIn(": ", text)
        self.assertNotIn(", ", text)
        self.assertIn("Café", text)
        self.assertEqual(json.loads(text)["high_level_description"], "Café sign")

    def test_preserves_ideogram_key_order(self):
        text = serialize_caption(
            {
                "style_description": {
                    "aesthetics": "bold",
                    "lighting": "studio",
                    "medium": "graphic_design",
                    "art_style": "flat vector",
                    "color_palette": ["#ffffff", "#123ABC"],
                },
                "compositional_deconstruction": {
                    "background": "paper",
                    "elements": [
                        {
                            "type": "text",
                            "bbox": [100, 200, 300, 400],
                            "text": "SALE",
                            "desc": "large text",
                            "color_palette": ["#ff0000"],
                        }
                    ],
                },
            }
        )

        style_order = [
            '"aesthetics"',
            '"lighting"',
            '"medium"',
            '"art_style"',
            '"color_palette"',
        ]
        self.assertEqual([text.index(key) for key in style_order], sorted(text.index(key) for key in style_order))

        self.assertIn(
            '"elements":[{"type":"text","bbox":[100,200,300,400],"text":"SALE","desc":"large text","color_palette":["#FF0000"]}]',
            text,
        )

    def test_palette_validation_and_bbox_normalization(self):
        colors, invalid = parse_palette_text("#abcDEF, nope #123456 #fff", 2)
        self.assertEqual(colors, ["#ABCDEF", "#123456"])
        self.assertEqual(invalid, ["nope", "#fff"])

        caption = normalize_caption(
            {
                "compositional_deconstruction": {
                    "background": "",
                    "elements": [{"type": "obj", "bbox": [900, -10, 100, 1200], "desc": "box"}],
                }
            }
        )
        self.assertEqual(caption["compositional_deconstruction"]["elements"][0]["bbox"], [100, 0, 900, 1000])


class CaptionHealthTests(unittest.TestCase):
    GOOD = {
        "high_level_description": "A red fox sitting in snow at dusk.",
        "style_description": {"aesthetics": "naturalistic", "lighting": "golden hour", "medium": "photograph"},
        "compositional_deconstruction": {"background": "snowy field", "elements": [{"type": "obj", "desc": "fox"}]},
    }

    def test_healthy_caption_has_no_issues(self):
        self.assertEqual(caption_health(self.GOOD), [])

    def test_sparse_but_structured_passes(self):
        sparse = {
            "high_level_description": "A logo on white.",
            "style_description": {"aesthetics": "minimal", "lighting": "flat", "medium": "graphic_design"},
            "compositional_deconstruction": {"background": "white", "elements": []},
        }
        self.assertEqual(caption_health(sparse), [])

    def test_flat_blob_is_flagged(self):
        blob = normalize_caption({"high_level_description": "the image shows a thing " * 200})
        issues = caption_health(blob)
        self.assertTrue(any("flat text blob" in i for i in issues))

    def test_off_schema_keys_become_empty_and_flag(self):
        off = normalize_caption({"caption": "a fox", "tags": ["fox"]})
        self.assertTrue(caption_health(off))

    def test_empty_caption_flagged(self):
        self.assertTrue(caption_health(normalize_caption({})))

    def test_refusal_flagged(self):
        ref = normalize_caption({"high_level_description": "I'm sorry, but I can't assist with this image."})
        self.assertTrue(any("refusal" in i for i in caption_health(ref)))

    def test_runaway_repetition_flagged(self):
        rep = normalize_caption({
            "high_level_description": "a man a man a man a man a man a man a man a man",
            "style_description": {"aesthetics": "x"},
        })
        self.assertTrue(any("repetit" in i for i in caption_health(rep)))

    def test_non_dict_is_flagged(self):
        self.assertTrue(caption_health("just a string"))


if __name__ == "__main__":
    unittest.main()
