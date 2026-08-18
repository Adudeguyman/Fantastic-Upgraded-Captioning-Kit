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

The numbers here describe what TRAINERS accept, not what the models generate —
the two diverge, and badly: H3 generates 362 frames but its LoRA trainers cap at
124, LTX generates ~10s but trains on at most 121 frames. A dataset-prep tool
that uses generation ceilings passes clips every trainer rejects or silently
truncates, which is exactly the failure this table exists to prevent.

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
    # Trainer-side limits. A model's generation ceiling and its trainer's ceiling
    # are different numbers (H3 generates 362 frames but its LoRA trainers take at
    # most 124), and this tool prepares TRAINING clips, so the trainer's number is
    # the one every frame calculation uses. 0 means "derive from max_seconds", the
    # old behaviour, for models where the two ceilings coincide.
    max_train_frames: int = 0
    # When a trainer publishes an exact list of accepted counts (fal's H3 trainers
    # take 22, 39, 56, 73, 90, 107, 124 and nothing else), that list — not the
    # modulus grid — decides legality and snapping. Empty means "use the grid".
    train_frame_choices: tuple[int, ...] = ()
    notes: str = ""
    source: str = ""
    verified: str = ""             # ISO date this entry was last checked

    # ---- frame-grid maths ----

    def is_legal_frames(self, frames: int) -> bool:
        if frames <= 0:
            return False
        if self.train_frame_choices:
            return frames in self.train_frame_choices
        return frames % self.frame_modulus == self.frame_remainder % self.frame_modulus

    def snap_frames(self, frames: int, mode: str = "down") -> int:
        """Nearest legal frame count. 'down' never exceeds the source clip (our
        default — we can always drop frames, never invent them); 'up' matches what
        H3's generator does with a request; 'nearest' minimises the change.

        With an enumerated choice list, snapping picks from the list. fal's own
        adjustment formula (17 * (v // 17) + 5, clamped) can round UP by as many as
        five frames — 123 becomes 124 — which is fine for a generation request but
        wrong for us: we'd be promising the trainer a frame we don't have. Our
        'down' stays a true floor over the list.
        """
        if self.train_frame_choices:
            choices = self.train_frame_choices
            if mode == "down":
                fits = [c for c in choices if c <= frames]
                return fits[-1] if fits else choices[0]
            if mode == "up":
                fits = [c for c in choices if c >= frames]
                return fits[0] if fits else choices[-1]
            return min(choices, key=lambda c: (abs(c - frames), c))
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
        if self.train_frame_choices:
            return self.train_frame_choices[0]
        r = self.frame_remainder % self.frame_modulus
        return r if r > 0 else self.frame_modulus

    def seconds_for_frames(self, frames: int) -> float:
        return frames / self.fps if self.fps else 0.0

    def frames_for_seconds(self, seconds: float) -> int:
        return int(round(seconds * self.fps))

    def max_frames(self) -> int:
        """The longest clip a TRAINER will accept, not the longest the model can
        generate. H3 generates 362 frames but its LoRA trainers cap at 124 — hand
        one a 362-frame clip and it truncates or rejects, so the generation number
        was always the wrong ceiling for a dataset-prep tool. An explicit
        max_train_frames (or the top of the choice list) wins; the seconds-derived
        value remains the fallback for models where the two ceilings coincide,
        snapped nearest so Wan's ceiling is the official 81 rather than 77."""
        if self.train_frame_choices:
            return self.train_frame_choices[-1]
        if self.max_train_frames > 0:
            return self.snap_frames(self.max_train_frames, "down")
        return self.snap_frames(self.frames_for_seconds(self.max_seconds), "nearest")

    def min_frames(self) -> int:
        if self.train_frame_choices:
            return self.train_frame_choices[0]
        return self.snap_frames(self.frames_for_seconds(self.min_seconds), "up")

    def grid_description(self) -> str:
        """The legality rule in the words the user should see: an enumerated
        trainer list when one exists, the modulus grid otherwise."""
        if self.train_frame_choices:
            return "one of " + ", ".join(str(c) for c in self.train_frame_choices)
        return f"on the {self.frame_modulus}n+{self.frame_remainder} grid"

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
        return (f"{frames} frames \u2014 not {self.grid_description()} "
                f"(nearest below: {nearest})")


# Frame rules, re-verified 2026-08-18 against TRAINER documentation. The first
# pass (2026-08-13) sourced these from what the models GENERATE, which for H3 and
# LTX put the ceiling far above what any trainer accepts. See each entry's source.

WAN22_A14B = ModelTarget(
    key="wan22_a14b",
    label="Wan 2.2 A14B (T2V/I2V)",
    fps=16.0,
    frame_modulus=4, frame_remainder=1,        # 4n+1: 9, 41, 81 ...
    dimension_multiple=16,
    max_pixels=1280 * 720,
    # 9 frames (0.5625s) is the smallest grid count trainers actually take;
    # musubi-tuner accepts any 4n+1 and treats stills as 1 frame.
    min_seconds=0.5625, max_seconds=5.0625,
    max_train_frames=81,
    buckets=(Bucket(832, 480, 81), Bucket(480, 832, 81), Bucket(1280, 720, 81)),
    notes="The A14B example code runs at 16 fps even though the 5B variant is 24 — "
          "check which variant you're training. 81 frames (~5s) is both the official "
          "generation count and the ceiling every trainer uses; here generation and "
          "training happen to agree.",
    source="Wan2.2 repo + HF discussion on A14B fps; fal wan-22-trainer "
          "(auto-scale fits clips to 81 frames at 16 fps); musubi-tuner configs "
          "(max_frames = 81, target_frames on the 4n+1 grid)",
    verified="2026-08-18",
)

WAN22_TI2V_5B = ModelTarget(
    key="wan22_ti2v_5b",
    label="Wan 2.2 TI2V-5B",
    fps=24.0,
    frame_modulus=4, frame_remainder=1,
    dimension_multiple=16,
    max_pixels=1280 * 720,
    min_seconds=0.375, max_seconds=5.05,
    max_train_frames=121,
    buckets=(Bucket(1280, 720, 121), Bucket(720, 1280, 121)),
    notes="Officially 720P at 24 fps — the fps differs from A14B in the same family. "
          "121 frames is the official count; trainers accept any 4n+1 up to it, and "
          "VRAM usually forces shorter clips in practice.",
    source="Wan-AI/Wan2.2 model card; musubi-tuner (target_frames on the 4n+1 grid)",
    verified="2026-08-18",
)

LTX_2_3 = ModelTarget(
    key="ltx_2_3",
    label="LTX-2.3",
    fps=24.0,
    frame_modulus=8, frame_remainder=1,        # frames % 8 == 1
    dimension_multiple=32,
    max_pixels=0,
    # The trainers' window, not the model's: fal's LTX-2 trainer enumerates
    # 9..121 as the valid values, and the official ltx-trainer docs top out at
    # 121 (512\u00d7512) even though the model generates ~10s clips. The code itself
    # only enforces the 8n+1 grid, so a longer bucket may work — but nothing
    # documents one, and this tool prepares clips a trainer is known to take.
    min_seconds=0.375, max_seconds=5.0417,
    max_train_frames=121,
    buckets=(Bucket(960, 544, 49), Bucket(768, 448, 89), Bucket(512, 512, 121)),
    notes="Preprocessing takes only the FIRST N frames of a longer clip and ignores "
          "the rest, so trim to the bucket length yourself. Spatial dims must be "
          "multiples of 32. Training tops out at 121 frames (~5s) even though "
          "generation runs to ~10s.",
    source="Lightricks/LTX-Video-Trainer dataset-preparation.md (8n+1, buckets to "
          "121; only first N frames used) and datasets.py (num_frames %% 8 == 1 "
          "enforced); fal LTX-2 trainer (valid number_of_frames: 9, 17 ... 113, 121)",
    verified="2026-08-18",
)

LTX_2_5 = ModelTarget(
    key="ltx_2_5",
    label="LTX-2.5",
    fps=24.0,
    frame_modulus=8, frame_remainder=1,
    dimension_multiple=32,
    max_pixels=0,
    min_seconds=0.375, max_seconds=5.0417,
    max_train_frames=121,
    buckets=(Bucket(960, 544, 49), Bucket(768, 448, 89), Bucket(512, 512, 121)),
    notes="Shares the LTX-2 trainer and its dataset geometry, but checkpoints and "
          "LoRAs are NOT interchangeable with 2.3 — a LoRA only works with the model "
          "it was trained on. 2.5 ships a new video decoder, so re-verify these "
          "constraints before a long training run.",
    source="Lightricks/LTX-Video-Trainer docs (shared with 2.3); fal LTX-2 trainer "
          "(valid number_of_frames: 9, 17 ... 113, 121); ltx.io model pages",
    verified="2026-08-18",
)

MINIMAX_H3 = ModelTarget(
    key="minimax_h3",
    label="MiniMax H3 (Hailuo 3.0)",
    fps=24.0,
    frame_modulus=17, frame_remainder=5,       # 17n+5 — but see the choice list
    dimension_multiple=32,
    max_pixels=768 * 1344,
    # The trainers' exact legal set. 22 frames (0.92s) is the true floor — the
    # VAE's shortest encodable clip — and 124 (5.17s) the true ceiling. The old
    # 3s/15s window came from the GENERATION side (the API's duration enum and
    # the model's 362-frame output ceiling) and let through clips every trainer
    # rejects or truncates.
    min_seconds=0.9167, max_seconds=5.1667,
    max_train_frames=124,
    train_frame_choices=(22, 39, 56, 73, 90, 107, 124),
    # Buckets actually used for a trained H3 realism adapter, rather than derived:
    # 16:9, scope, and near-square, all multiples of 32 with comparable areas.
    buckets=(Bucket(1280, 704, 73), Bucket(1280, 544, 73), Bucket(960, 704, 73)),
    exact_fps=True,
    notes="Trainers accept exactly 22, 39, 56, 73, 90, 107 or 124 frames (fal "
          "adjusts anything else with 17 * (n // 17) + 5, clamped — 100\u2192\u200990, "
          "123\u2192124 — so we cut to a legal count ourselves rather than gamble on "
          "the adjustment). Generation runs to 362 frames but no trainer takes "
          "past 124. Exactly 24.000 fps — resample 23.976/25/30 sources. Trim "
          "fade-ins and black frames: clean first/last frames matter. Dims are "
          "multiples of 32, product no more than 768\u00d71344.",
    source="fal H3 LoRA trainer docs, all four endpoints (frames %% 17 == 5, valid "
           "counts 22\u2013124, default 73, adjustment formula 17*(n//17)+5 with clamp); "
           "MiniMax's official VIDEO_PROMPT_WRITING_GUIDE_base_en.md on the "
           "MiniMaxAI/MiniMax-H3 repo for the prompt format, dialogue tags and "
           "soundscape fields; fal's published 16-run training campaign for buckets, "
           "fps and caption structure; 22-frame VAE floor independently confirmed by "
           "ComfyUI/ai-toolkit-based trainers",
    verified="2026-08-18",
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
    fields["max_train_frames"] = max(0, int(fields.get("max_train_frames") or 0))
    # Sorted and deduplicated so snap_frames can rely on order; junk entries are
    # dropped rather than failing the whole target.
    choices = fields.get("train_frame_choices") or ()
    fields["train_frame_choices"] = tuple(sorted({
        int(c) for c in choices if isinstance(c, (int, float)) and int(c) > 0
    }))
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
