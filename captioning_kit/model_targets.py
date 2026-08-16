"""Target specs for video-generation models, used by auto-trim/conform.

Every model here constrains its training clips the same three ways — a native
frame rate, a legal frame-count grid, and a spatial multiple — so one dataclass
covers all of them. What differs is the constants, and those differ more than you
would expect: Wan is 4n+1, LTX is 8n+1, MiniMax H3 is 17n+5.

Two of these snap in *opposite* directions when a clip doesn't fit, which is why
we conform clips ourselves rather than letting the trainer do it:

  * LTX takes only the first N frames of a longer clip and silently ignores the
    rest — an over-long clip loses its ending without warning.
  * MiniMax H3 rounds a frame request *up* to the next legal count, padding
    rather than truncating.

Specs drift, and several of the H3 numbers come from community implementations
(ComfyUI, MLX and Metal engines agreeing with each other) rather than an official
MiniMax spec. So each entry carries a source note and a verified date, and the
whole table can be overridden by a user-editable JSON file — see load_targets().
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Bucket:
    """One legal (width, height, frames) training shape."""
    width: int
    height: int
    frames: int

    @property
    def label(self) -> str:
        return f"{self.width}\u00d7{self.height}\u00d7{self.frames}"


@dataclass(frozen=True)
class ModelTarget:
    """What one model wants its training clips to look like.

    frame_modulus/frame_remainder express the frame-count grid as
    ``frames % modulus == remainder``: Wan 4n+1 is (4, 1), LTX 8n+1 is (8, 1),
    H3 17n+5 is (17, 5). A modulus of 1 with remainder 0 means "no constraint".
    """
    key: str
    label: str
    fps: float
    frame_modulus: int
    frame_remainder: int
    dimension_multiple: int
    max_pixels: int          # 0 = no documented cap
    min_seconds: float
    max_seconds: float
    buckets: tuple[Bucket, ...] = ()
    exact_fps: bool = False        # True when off-rate sources MUST be resampled
    notes: str = ""
    source: str = ""
    verified: str = ""             # ISO date this entry was last checked

    # ---- frame-grid maths ----

    def is_legal_frames(self, frames: int) -> bool:
        if frames <= 0:
            return False
        return frames % self.frame_modulus == self.frame_remainder % self.frame_modulus

    def snap_frames(self, frames: int, mode: str = "down") -> int:
        """Nearest legal frame count. 'down' never exceeds the source clip (our
        default — we can always drop frames, never invent them); 'up' matches what
        H3 itself does; 'nearest' minimises the change."""
        if frames <= 0:
            return self.smallest_legal_frames()
        m, r = self.frame_modulus, self.frame_remainder % self.frame_modulus
        below = frames - ((frames - r) % m)
        above = below if below == frames else below + m
        if below < self.smallest_legal_frames():
            below = self.smallest_legal_frames()
        if mode == "up":
            return above
        if mode == "nearest":
            return below if (frames - below) <= (above - frames) else above
        return below if below <= frames else self.smallest_legal_frames()

    def smallest_legal_frames(self) -> int:
        r = self.frame_remainder % self.frame_modulus
        return r if r > 0 else self.frame_modulus

    def seconds_for_frames(self, frames: int) -> float:
        return frames / self.fps if self.fps else 0.0

    def frames_for_seconds(self, seconds: float) -> int:
        return int(round(seconds * self.fps))

    def max_frames(self) -> int:
        """The frame count the model actually tops out at, which is the legal value
        nearest its max duration rather than the one below it: H3's ceiling is 362
        frames (15.083s), not the 345 you'd get by rounding 15s down, and Wan's is
        the official 81 rather than 77."""
        return self.snap_frames(self.frames_for_seconds(self.max_seconds), "nearest")

    def min_frames(self) -> int:
        return self.snap_frames(self.frames_for_seconds(self.min_seconds), "up")

    def fit_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Nearest legal size for an explicit conform, preserving the source aspect.

        Used only when the user asks to conform a clip — NOT to normalise a dataset.
        Trainers assign each clip to an aspect-ratio bucket and resize during latent
        caching, so forcing everything to one size here would crop away picture the
        trainer would have kept. This just rounds to the multiple the encoder needs
        and respects a documented pixel budget.
        """
        width, height = max(1, int(width)), max(1, int(height))
        if self.max_pixels and width * height > self.max_pixels:
            scale = (self.max_pixels / (width * height)) ** 0.5
            width, height = max(1, int(width * scale)), max(1, int(height * scale))
        width, height = self.snap_dimension(width), self.snap_dimension(height)
        # Snapping rounds down, so it can only reduce area — but guard anyway.
        while self.max_pixels and width * height > self.max_pixels:
            mult = max(1, self.dimension_multiple)
            if width >= height:
                width = max(mult, width - mult)
            else:
                height = max(mult, height - mult)
        return width, height

    def snap_dimension(self, value: int) -> int:
        """Round a pixel dimension down to the model's required multiple."""
        mult = max(1, self.dimension_multiple)
        return max(mult, (int(value) // mult) * mult)

    def describe_frames(self, frames: int) -> str:
        """Human-readable verdict for the trim UI."""
        if self.is_legal_frames(frames):
            return f"{frames} frames \u00b7 {self.seconds_for_frames(frames):.2f}s \u2713"
        nearest = self.snap_frames(frames, "down")
        return (f"{frames} frames \u2014 not on the {self.frame_modulus}n"
                f"{self.frame_remainder:+d} grid (nearest below: {nearest})")


# Frame rules, verified 2026-08-13. See module docstring on confidence.

WAN22_A14B = ModelTarget(
    key="wan22_a14b",
    label="Wan 2.2 A14B (T2V/I2V)",
    fps=16.0,
    frame_modulus=4, frame_remainder=1,        # 4n+1: 9, 41, 81 ...
    dimension_multiple=16,
    max_pixels=1280 * 720,
    min_seconds=1.0, max_seconds=5.0,
    buckets=(Bucket(832, 480, 81), Bucket(480, 832, 81), Bucket(1280, 720, 81)),
    notes="The A14B example code runs at 16 fps even though the 5B variant is 24 — "
          "check which variant you're training. 81 frames (~5s) is the official count.",
    source="Wan2.2 repo + HF discussion on A14B fps; community training guides",
    verified="2026-08-13",
)

WAN22_TI2V_5B = ModelTarget(
    key="wan22_ti2v_5b",
    label="Wan 2.2 TI2V-5B",
    fps=24.0,
    frame_modulus=4, frame_remainder=1,
    dimension_multiple=16,
    max_pixels=1280 * 720,
    min_seconds=1.0, max_seconds=5.0,
    buckets=(Bucket(1280, 720, 121), Bucket(720, 1280, 121)),
    notes="Officially 720P at 24 fps — the fps differs from A14B in the same family.",
    source="Wan-AI/Wan2.2-I2V-A14B model card",
    verified="2026-08-13",
)

LTX_2_3 = ModelTarget(
    key="ltx_2_3",
    label="LTX-2.3",
    fps=24.0,
    frame_modulus=8, frame_remainder=1,        # frames % 8 == 1
    dimension_multiple=32,
    max_pixels=0,
    min_seconds=1.0, max_seconds=10.0,
    buckets=(Bucket(960, 544, 49), Bucket(768, 448, 89), Bucket(512, 512, 121)),
    notes="Preprocessing takes only the FIRST N frames of a longer clip and ignores "
          "the rest, so trim to the bucket length yourself. Spatial dims must be "
          "multiples of 32.",
    source="Lightricks/LTX-2 ltx-trainer dataset-preparation.md",
    verified="2026-08-13",
)

LTX_2_5 = ModelTarget(
    key="ltx_2_5",
    label="LTX-2.5",
    fps=24.0,
    frame_modulus=8, frame_remainder=1,
    dimension_multiple=32,
    max_pixels=0,
    min_seconds=1.0, max_seconds=10.0,
    buckets=(Bucket(960, 544, 49), Bucket(768, 448, 89), Bucket(512, 512, 121)),
    notes="Shares the LTX-2 trainer and its dataset geometry, but checkpoints and "
          "LoRAs are NOT interchangeable with 2.3 — a LoRA only works with the model "
          "it was trained on. 2.5 ships a new video decoder, so re-verify these "
          "constraints before a long training run.",
    source="Lightricks/LTX-2 trainer docs (shared); ltx.io model pages",
    verified="2026-08-13",
)

MINIMAX_H3 = ModelTarget(
    key="minimax_h3",
    label="MiniMax H3 (Hailuo 3.0)",
    fps=24.0,
    frame_modulus=17, frame_remainder=5,       # 17n+5: 5, 22, 39, 56 ... 243, 362
    dimension_multiple=32,
    max_pixels=768 * 1344,
    # 3s floor is the training-dataset minimum (the generation API's own duration
    # enum starts at 4s, but we're preparing training clips here).
    min_seconds=3.0, max_seconds=15.0,
    # Buckets actually used for a trained H3 realism adapter, rather than derived:
    # 16:9, scope, and near-square, all multiples of 32 with comparable areas.
    buckets=(Bucket(1280, 704, 73), Bucket(1280, 544, 73), Bucket(960, 704, 73)),
    exact_fps=True,
    notes="Builds video in 17-frame blocks; the model rounds a request UP to the next "
          "legal count. Exactly 24.000 fps — resample 23.976/25/30 sources. Trim "
          "fade-ins and black frames: clean first/last frames matter. Dims are "
          "multiples of 32, product no more than 768\u00d71344.",
    source="MiniMax's official VIDEO_PROMPT_WRITING_GUIDE_base_en.md on the "
           "MiniMaxAI/MiniMax-H3 repo for the prompt format, dialogue tags and "
           "soundscape fields; fal H3 LoRA trainer docs (number_of_frames must satisfy frames %% 17 == 5, "
           "default 73) and fal's published 16-run training campaign for buckets, fps "
           "and caption structure; grid independently confirmed by ComfyUI, MLX and "
           "Metal implementations",
    verified="2026-08-13",
)

BUILTIN_TARGETS: tuple[ModelTarget, ...] = (
    WAN22_A14B, WAN22_TI2V_5B, LTX_2_3, LTX_2_5, MINIMAX_H3,
)


def builtin_map() -> dict[str, ModelTarget]:
    return {t.key: t for t in BUILTIN_TARGETS}


def _target_from_dict(data: dict) -> ModelTarget:
    buckets = tuple(
        Bucket(int(b["width"]), int(b["height"]), int(b["frames"]))
        for b in data.get("buckets", []) if {"width", "height", "frames"} <= set(b)
    )
    fields = {k: v for k, v in data.items() if k in ModelTarget.__dataclass_fields__}
    fields["buckets"] = buckets
    fields["fps"] = float(fields.get("fps") or 24.0)
    for key in ("frame_modulus", "frame_remainder", "dimension_multiple"):
        fields[key] = int(fields.get(key) or 1)
    fields["max_pixels"] = int(fields.get("max_pixels") or 0)
    fields["frame_modulus"] = max(1, fields["frame_modulus"])
    fields["dimension_multiple"] = max(1, fields["dimension_multiple"])
    return ModelTarget(**fields)


def targets_path(base_dir: Path) -> Path:
    return Path(base_dir) / "model_targets.json"


def load_targets(base_dir: Path | None = None) -> dict[str, ModelTarget]:
    """Built-in targets, overlaid with the user's JSON if present.

    These specs change, and some were pieced together from community sources, so
    the table is data rather than code: a user can correct an entry or add a model
    without waiting for an app update. A malformed file is ignored rather than
    breaking startup.
    """
    targets = builtin_map()
    if base_dir is None:
        return targets
    path = targets_path(base_dir)
    if not path.exists():
        return targets
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return targets
    entries = raw.get("targets", raw) if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return targets
    for item in entries:
        if not isinstance(item, dict) or not item.get("key"):
            continue
        try:
            target = _target_from_dict(item)
        except (TypeError, ValueError, KeyError):
            continue
        targets[target.key] = target
    return targets


def save_targets(base_dir: Path, targets: dict[str, ModelTarget]) -> Path:
    path = targets_path(Path(base_dir))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_comment": "Edit to correct or add model targets. Delete this file to "
                    "restore the built-in defaults.",
        "targets": [asdict(t) for t in targets.values()],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
