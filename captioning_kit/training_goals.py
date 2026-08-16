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
            "adapters rather than one isolated concept.",
    rules="",
)

CHARACTER = TrainingGoal(
    key="character",
    label="Character likeness",
    summary="Omit the face and build so they bind to the trigger; describe pose, "
            "clothing and setting so those stay promptable.",
    rules="""\
This dataset trains a character likeness, so the adapter must own the person's
identity while everything else stays promptable.

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
This dataset trains a specific object or concept, so the adapter must own the
thing itself while its context stays promptable.

Do NOT describe: the object's own shape, colour, materials, markings, logos or
construction. Those are what the adapter is for.

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
This dataset trains a movement, so the adapter must own the action while
everyone and everything performing it stays promptable.

Do NOT describe: the movement, choreography, action or gesture being trained.
Naming it would attribute it to your words instead of the adapter.

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
This dataset trains a visual style, so the adapter must own the rendering
treatment while the subject matter stays promptable.

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
This dataset trains a cinematographic look, so the adapter must own the visual
treatment while the content stays promptable.

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
