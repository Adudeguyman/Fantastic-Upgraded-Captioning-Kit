"""Caption presets.

A preset describes the *shape* of a dataset's captions: what file they're written
to, how they're edited, and which pipeline features apply. The app is built around
these so the Ideogram-4-specific machinery (structured fields, bounding boxes,
schema health checks) is gated behind the preset that actually needs it, rather
than being assumed everywhere.

Presets are stored per-folder in .captioner/project.json, so opening a dataset
always brings back the format it was captioned in.
"""
from __future__ import annotations

from dataclasses import dataclass


from .prompt_texts import (
    H3_NATURAL_IMAGE,
    H3_NATURAL_VIDEO,
    H3_OFFICIAL_IMAGE,
    H3_OFFICIAL_VIDEO,
    PLAIN_IMAGE_PROMPT,
    PLAIN_VIDEO_PROMPT,
    LTX_IMAGE,
    LTX_VIDEO,
    WAN_IMAGE,
    WAN_VIDEO,
)


@dataclass(frozen=True)
class CaptionPreset:
    """One caption format.

    editor:      'plain' = a single free-text field, 'structured' = the Ideogram
                 field/elements editor.
    extension:   sidecar written next to the image.
    has_boxes:   whether bounding boxes and the grounding pass apply.
    validates:   whether the Ideogram JSON schema health checks apply.
    """
    key: str
    label: str
    extension: str
    editor: str
    has_boxes: bool
    validates: bool
    # Default guidance. Stills and clips need genuinely different instructions — a
    # photo has no timeline, no camera movement over time and no audio — so a preset
    # may override either. Empty overrides fall back to system_prompt.
    system_prompt: str
    image_prompt: str = ""
    video_prompt: str = ""
    blurb: str = ""
    # Key of a ModelTarget (see model_targets.py). Selecting the preset arms that
    # target in the video trim controls, so choosing an H3 preset also conforms clips
    # to H3's frame grid.
    model_target: str = ""

    @property
    def is_plain(self) -> bool:
        return self.editor == "plain"

    @property
    def has_media_variants(self) -> bool:
        """True when stills and clips get genuinely different guidance, which is what
        makes the Photos/Videos choice meaningful."""
        return bool(self.image_prompt and self.video_prompt
                    and self.image_prompt != self.video_prompt)

    def prompt_for(self, media: str) -> str:
        """Guidance for 'image' or 'video', falling back to the shared prompt."""
        if media == "video" and self.video_prompt:
            return self.video_prompt
        if media == "image" and self.image_prompt:
            return self.image_prompt
        return self.system_prompt


PLAIN_TEXT = CaptionPreset(
    key="plain_text",
    label="Plain text",
    extension=".txt",
    editor="plain",
    has_boxes=False,
    validates=False,
    system_prompt=PLAIN_IMAGE_PROMPT,
    image_prompt=PLAIN_IMAGE_PROMPT,
    video_prompt=PLAIN_VIDEO_PROMPT,
    blurb="One free-text caption per file, saved as a .txt sidecar. The common "
          "format for SD/SDXL/Flux-style training.",
)

IDEOGRAM_4 = CaptionPreset(
    key="ideogram4",
    label="Ideogram 4 JSON",
    extension=".json",
    editor="structured",
    has_boxes=True,
    validates=True,
    system_prompt="",  # supplied by the structured prompt files in captioner_prompts/
    blurb="Structured JSON with style fields, compositional elements and bounding "
          "boxes, for Ideogram 4 dataset preparation.",
)

MINIMAX_H3_OFFICIAL = CaptionPreset(
    key="minimax_h3_official",
    label="MiniMax H3 \u2014 Official Prompt Structure",
    extension=".txt",
    editor="plain",
    has_boxes=False,
    validates=False,
    system_prompt=H3_OFFICIAL_VIDEO,
    image_prompt=H3_OFFICIAL_IMAGE,
    video_prompt=H3_OFFICIAL_VIDEO,
    model_target="minimax_h3",
    blurb="MiniMax's own three-field format with <d> dialogue tags and speaker IDs "
          "\u2014 the structure the base model was trained on.",
)

MINIMAX_H3_NATURAL = CaptionPreset(
    key="minimax_h3_natural",
    label="MiniMax H3 \u2014 Natural Language",
    extension=".txt",
    editor="plain",
    has_boxes=False,
    validates=False,
    system_prompt=H3_NATURAL_VIDEO,
    image_prompt=H3_NATURAL_IMAGE,
    video_prompt=H3_NATURAL_VIDEO,
    model_target="minimax_h3",
    blurb="One flowing paragraph per clip, dialogue quoted inline \u2014 the style used "
          "by fal's published H3 LoRA campaign.",
)

WAN_22 = CaptionPreset(
    key="wan22",
    label="Wan 2.2 video",
    extension=".txt",
    editor="plain",
    has_boxes=False,
    validates=False,
    system_prompt=WAN_VIDEO,
    image_prompt=WAN_IMAGE,
    video_prompt=WAN_VIDEO,
    model_target="wan22_a14b",
    blurb="Visual-only captions for Wan 2.2, which generates silent video \u2014 and "
          "clips conformed to its 16fps / 4n+1 grid.",
)

LTX_2 = CaptionPreset(
    key="ltx2",
    label="LTX-2 video",
    extension=".txt",
    editor="plain",
    has_boxes=False,
    validates=False,
    system_prompt=LTX_VIDEO,
    image_prompt=LTX_IMAGE,
    video_prompt=LTX_VIDEO,
    model_target="ltx_2_3",
    blurb="Captions covering picture and sound for LTX-2, which generates "
          "synchronised audio \u2014 and clips conformed to its 24fps / 8n+1 grid.",
)

PRESETS: dict[str, CaptionPreset] = {
    p.key: p for p in (PLAIN_TEXT, IDEOGRAM_4, MINIMAX_H3_OFFICIAL,
                       MINIMAX_H3_NATURAL, WAN_22, LTX_2)
}
PRESET_ORDER: tuple[str, ...] = (
    PLAIN_TEXT.key, IDEOGRAM_4.key, MINIMAX_H3_OFFICIAL.key, MINIMAX_H3_NATURAL.key,
    WAN_22.key, LTX_2.key,
)
DEFAULT_PRESET = PLAIN_TEXT.key


def get_preset(key: str | None) -> CaptionPreset:
    """Look up a preset, falling back to the default for unknown/missing keys so a
    hand-edited or future project file can't break loading a folder."""
    return PRESETS.get(key or "", PRESETS[DEFAULT_PRESET])


def caption_extensions() -> set[str]:
    """Every extension a preset might write — used to keep caption sidecars out of
    the image listing."""
    return {p.extension for p in PRESETS.values()}
