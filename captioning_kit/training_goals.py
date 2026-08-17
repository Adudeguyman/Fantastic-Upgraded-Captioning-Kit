"""Training goals: what a caption should describe, and what it should leave out.

Separate from presets on purpose. A preset decides the caption's *format* — which
file, which shape, H3's tags or Ideogram's JSON. A goal decides its *content
policy* — which details belong in the text and which belong to the adapter. The
two are orthogonal: an H3-structured caption for a motion LoRA is a legitimate
combination, and folding them into one list couldn't express it.

The rule every goal is derived from: caption what should stay promptable, omit
what the LoRA should own. Anything you describe is attributed to your words, so
the model treats it as variable; anything you leave undescribed is absorbed into
the trigger and comes back every time the adapter fires.

Note this is a real trade rather than settled fact. Omission disentangles more
cleanly; describing constant features gives a stronger, more reliable draw at
inference — practitioners on Wan report trigger-only adapters bind appearance
weakly, especially on distilled/lightning checkpoints. Newer text encoders (Flux
and later) are also less sensitive to it than booru-tag-era models were. The goals
below take the omission side, and say so, so the choice is visible rather than
implied.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingGoal:
    key: str
    label: str
    summary: str        # one line, shown next to the picker
    rules: str = ""     # appended to the preset prompt; "" = no policy at all

    @property
    def has_rules(self) -> bool:
        return bool(self.rules.strip())


GENERAL = TrainingGoal(
    key="general",
    label="General",
    summary="Describe everything. No omissions — for broad quality or realism "
            "training runs rather than one isolated concept.",
    rules="",
)

CHARACTER = TrainingGoal(
    key="character",
    label="Character likeness",
    summary="Omit the face and build so they bind to the trigger; describe pose, "
            "clothing and setting so those stay promptable.",
    rules="""\
This dataset trains a character likeness. The training will learn the person's
identity from the pictures themselves, so leave it out of the caption; everything
else should be described so it stays controllable later.

Do NOT describe: facial features, face shape, eye colour or shape, hair colour or
length, skin tone, body build, or permanent marks such as scars, freckles and
tattoos. Leaving these out is what binds them to the trigger.

DO describe fully: pose and gesture, facial expression and gaze direction,
clothing and accessories, the setting and background, lighting, colour, and
camera framing and angle.

If a detail is true of this subject in every shot, leave it out. If it changes
between shots, describe it.""",
)

CONCEPT = TrainingGoal(
    key="concept",
    label="Concept / object",
    summary="Omit the object's defining traits; describe where it sits, how it's "
            "lit and what surrounds it.",
    rules="""\
This dataset trains a specific object or concept. The training will learn the
thing itself from the pictures, so leave its appearance out of the caption;
describe its surroundings so those stay controllable later.

Do NOT describe: the object's own shape, colour, materials, markings, logos or
construction. Describing them would teach the model to treat those words as the
source of the object, rather than learning the object from the pictures.

DO describe fully: where the object sits and how it's placed, its scale relative
to the scene, what surrounds it, the background and setting, lighting and
shadow, and camera framing and angle.

Name the object plainly (its class word) without detailing its appearance.""",
)

MOTION = TrainingGoal(
    key="motion",
    label="Motion / action",
    summary="Omit the movement itself; describe the performer, clothing and "
            "setting in full so identity doesn't bind to the motion.",
    rules="""\
This dataset trains a movement. The training will learn the action from the
clips, so leave it out of the caption; describe whoever and whatever is
performing it so those stay controllable later.

Do NOT describe: the movement, choreography, action or gesture being trained.
Naming it would tie the movement to those words instead of letting it be learned
from the footage.

DO describe fully: the performer's appearance including face, hair, build and
skin tone; their clothing and footwear; the location and every notable object in
it; lighting and time of day; and camera framing and any camera movement.

Describing the performer thoroughly is what stops their identity being learned as
part of the motion. Be specific rather than generic about them.""",
)

ART_STYLE = TrainingGoal(
    key="art_style",
    label="Art style",
    summary="Omit style words entirely; describe subject matter thoroughly so the "
            "rendering treatment is the only unexplained signal.",
    rules="""\
This dataset trains a visual style. The training will learn the rendering
treatment from the pictures, so leave it out of the caption; describe what is
depicted so the subject matter stays controllable later.

Do NOT describe: the style, medium, rendering technique, brushwork, line quality,
grain, palette or era. Naming the style makes the model attach it to those words
instead of learning it implicitly.

DO describe fully: what is actually depicted — subjects, objects, setting,
composition, pose and action. Describe the content thoroughly and plainly.

Avoid style adjectives entirely, including "painterly", "cinematic", "stylised",
"illustrated" and "retro".""",
)

VIDEO_STYLE = TrainingGoal(
    key="video_style",
    label="Video style / look",
    summary="Omit the grade and texture; describe content, action and camera so "
            "the look is what's left unexplained.",
    rules="""\
This dataset trains a cinematographic look. The training will learn the visual
treatment from the clips, so leave it out of the caption; describe the content so
it stays controllable later.

Do NOT describe: colour grade, film stock, grain, contrast, halation, bloom,
lens character, or mood words for the look.

DO describe fully: the subjects and what they do across the clip, the setting,
the action in order, shot size and camera movement, and any dialogue or sound if
the caption format asks for it.

Describe what the camera sees, never how the footage has been treated.""",
)

GOALS: dict[str, TrainingGoal] = {
    g.key: g for g in (GENERAL, CHARACTER, CONCEPT, MOTION, ART_STYLE, VIDEO_STYLE)
}
GOAL_ORDER: tuple[str, ...] = (
    GENERAL.key, CHARACTER.key, CONCEPT.key, MOTION.key, ART_STYLE.key,
    VIDEO_STYLE.key,
)
DEFAULT_GOAL = GENERAL.key


def get_goal(key: str | None) -> TrainingGoal:
    """Never raises: an unknown or missing key falls back to General, which applies
    no policy, so a corrupt project file can't silently change what gets captioned."""
    return GOALS.get((key or "").strip().lower(), GOALS[DEFAULT_GOAL])


# --- user overlay -------------------------------------------------------------
#
# Same treatment as the model frame rules: goals are data, not code. What to
# describe and what to omit is an evolving practice rather than a settled fact
# (see the note at the top of this module), so a user can rewrite a goal or add
# one without waiting for a release.

import json
from dataclasses import asdict
from pathlib import Path

GOALS_FILENAME = "captioner_training_goals.json"


def builtin_goal_map() -> dict[str, TrainingGoal]:
    return dict(GOALS)


def goals_path(base_dir) -> Path:
    return Path(base_dir) / GOALS_FILENAME


def _goal_from_dict(data: dict) -> TrainingGoal | None:
    key = str(data.get("key") or "").strip().lower()
    label = str(data.get("label") or "").strip()
    if not key or not label:
        return None
    return TrainingGoal(
        key=key,
        label=label,
        summary=str(data.get("summary") or ""),
        rules=str(data.get("rules") or ""),
    )


def load_goals(base_dir=None) -> dict[str, TrainingGoal]:
    """Built-in goals overlaid with the user's file. Malformed entries are skipped
    rather than breaking startup."""
    goals = builtin_goal_map()
    if base_dir is None:
        return goals
    try:
        data = json.loads(goals_path(base_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return goals
    for raw in (data.get("goals") or []):
        if isinstance(raw, dict):
            goal = _goal_from_dict(raw)
            if goal is not None:
                goals[goal.key] = goal
    return goals


def save_goals(base_dir, goals: dict[str, TrainingGoal]) -> None:
    """Write only what differs from the built-ins.

    A snapshot of everything would freeze the shipped wording out: a later release
    improving a goal's rules would be silently overridden by a stale copy the user
    never knowingly edited.
    """
    builtins = builtin_goal_map()
    changed = {k: g for k, g in goals.items()
               if k not in builtins or g != builtins[k]}
    path = goals_path(base_dir)
    if not changed:
        path.unlink(missing_ok=True)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "_comment": "Edit to change or add training goals. Delete this file to "
                    "restore the built-in defaults.",
        "goals": [asdict(g) for g in changed.values()],
    }, indent=2), encoding="utf-8")


def goal_order(base_dir=None) -> tuple[str, ...]:
    order = list(GOAL_ORDER)
    if base_dir is not None:
        order += [k for k in load_goals(base_dir) if k not in order]
    return tuple(order)


def make_custom_goal(key: str, label: str, summary: str = "",
                     rules: str = "") -> TrainingGoal:
    return TrainingGoal(key=key, label=label, summary=summary, rules=rules)
