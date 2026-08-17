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


# --- user-defined presets -----------------------------------------------------
#
# Stored alongside the app so a preset for a model that shipped last week doesn't
# have to wait for a release. Deliberately plain-text only: the structured editor
# and bounding boxes are Ideogram-4 machinery that can't be described by filling
# in a form.

CUSTOM_PRESETS_FILENAME = "captioner_custom_presets.json"


def custom_presets_path(base_dir) -> "Path":
    from pathlib import Path
    return Path(base_dir) / CUSTOM_PRESETS_FILENAME


def make_custom_preset(key: str, label: str, image_prompt: str = "",
                       video_prompt: str = "", model_target: str = "",
                       blurb: str = "") -> CaptionPreset:
    """A plain-text preset defined by the user.

    model_target may be empty: a stills-only preset has no frame grid to conform
    to, and forcing a choice would invent a constraint that doesn't exist.
    """
    return CaptionPreset(
        key=key,
        label=label,
        extension=".txt",
        editor="plain",
        has_boxes=False,
        validates=False,
        system_prompt=video_prompt or image_prompt,
        image_prompt=image_prompt,
        video_prompt=video_prompt,
        model_target=model_target or "",
        blurb=blurb or "Added by you.",
    )


def _custom_from_dict(raw: dict) -> CaptionPreset | None:
    key = str(raw.get("key") or "").strip()
    label = str(raw.get("label") or "").strip()
    if not key or not label or key in PRESETS:
        return None          # never let a user entry shadow a built-in
    return make_custom_preset(
        key=key,
        label=label,
        image_prompt=str(raw.get("image_prompt") or ""),
        video_prompt=str(raw.get("video_prompt") or ""),
        model_target=str(raw.get("model_target") or ""),
        blurb=str(raw.get("blurb") or ""),
    )


def load_custom_presets(base_dir) -> dict[str, CaptionPreset]:
    """Read user presets. A malformed entry is skipped, not fatal — one bad hand
    edit shouldn't stop the app opening."""
    import json
    path = custom_presets_path(base_dir)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict[str, CaptionPreset] = {}
    for raw in (data.get("presets") or []):
        if isinstance(raw, dict):
            preset = _custom_from_dict(raw)
            if preset is not None:
                out[preset.key] = preset
    return out


def save_custom_presets(base_dir, presets: dict[str, CaptionPreset]) -> None:
    import json
    path = custom_presets_path(base_dir)
    if not presets:
        path.unlink(missing_ok=True)
        return
    payload = {
        "_format": 1,
        "presets": [
            {
                "key": p.key,
                "label": p.label,
                "image_prompt": p.image_prompt,
                "video_prompt": p.video_prompt,
                "model_target": p.model_target,
                "blurb": p.blurb,
            }
            for p in presets.values()
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def all_presets(base_dir=None) -> dict[str, CaptionPreset]:
    """Built-ins plus user presets, built-ins first."""
    merged = dict(PRESETS)
    if base_dir is not None:
        merged.update(load_custom_presets(base_dir))
    return merged


def preset_order(base_dir=None) -> tuple[str, ...]:
    order = list(PRESET_ORDER)
    if base_dir is not None:
        order += [k for k in load_custom_presets(base_dir) if k not in order]
    return tuple(order)
