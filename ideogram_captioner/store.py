from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .schema import IMAGE_EXTENSIONS, caption_from_plain_text, default_caption, parse_caption_text, serialize_caption

PROJECT_DIRNAME = ".captioner"
PROJECT_FILENAME = "project.json"


@dataclass
class ProjectConfig:
    """Per-dataset captioning configuration stored alongside the images.

    Lives at ``<folder>/.captioner/project.json`` and travels with the folder.
    Nothing machine-specific (model/server/paths) belongs here.
    """

    name: str = ""
    folder_guidance: str = ""
    folder_guidance_enabled: bool = True
    per_image: dict[str, str] = field(default_factory=dict)
    per_image_enabled: dict[str, bool] = field(default_factory=dict)
    creative_json: bool | None = None  # None = inherit the global setting
    # When True, captioning feeds each image's matching .txt sidecar to the model
    # as a source caption to upgrade into structured JSON (folder-wide mode).
    convert_txt_to_json: bool = False
    # filename -> the effective guidance string that produced its current caption.
    # Lets us flag images whose guidance has changed since they were last run.
    generated_guidance: dict[str, str] = field(default_factory=dict)
    # filename -> list of issue strings from the last health check (corrupt/off-schema output).
    # Empty/absent = no known problems. Surfaced as a review marker; cleared when re-saved.
    caption_flags: dict[str, list[str]] = field(default_factory=dict)
    # filenames the user has manually flagged for review (independent of caption_flags).
    review_marks: set[str] = field(default_factory=set)

    def set_review_mark(self, filename: str, marked: bool) -> None:
        if marked:
            self.review_marks.add(filename)
        else:
            self.review_marks.discard(filename)

    def toggle_review_mark(self, filename: str) -> bool:
        if filename in self.review_marks:
            self.review_marks.discard(filename)
            return False
        self.review_marks.add(filename)
        return True

    def is_review_marked(self, filename: str) -> bool:
        return filename in self.review_marks

    def set_flags(self, filename: str, issues: list[str]) -> None:
        """Record (or clear) the health issues found for an image's caption."""
        if issues:
            self.caption_flags[filename] = list(issues)
        else:
            self.caption_flags.pop(filename, None)

    def clear_flag(self, filename: str) -> None:
        self.caption_flags.pop(filename, None)

    def caption_issues(self, filename: str) -> list[str]:
        return list(self.caption_flags.get(filename, []))

    def is_flagged(self, filename: str) -> bool:
        return bool(self.caption_flags.get(filename))

    def mark_generated(self, filename: str, guidance: str) -> None:
        """Stamp the guidance that produced this image's just-saved caption."""
        self.generated_guidance[filename] = guidance or ""

    def last_run_guidance(self, filename: str) -> str | None:
        """The guidance recorded at the last successful generation, or None."""
        return self.generated_guidance.get(filename)

    def guidance_changed(self, filename: str) -> bool:
        """True when the current effective guidance differs from what produced the
        last generated caption. Images never generated (no stamp) are not flagged."""
        prev = self.generated_guidance.get(filename)
        if prev is None:
            return False
        return prev.strip() != self.resolved_for(filename).strip()

    def per_image_guidance(self, filename: str) -> str:
        return self.per_image.get(filename, "")

    def has_per_image_guidance(self, filename: str) -> bool:
        return bool(self.per_image.get(filename, "").strip())

    def per_image_active(self, filename: str) -> bool:
        """Whether this image's per-image guidance is applied. Default on; an
        explicit False suppresses it without deleting the text."""
        return self.per_image_enabled.get(filename, True)

    def resolved_for(self, filename: str) -> str:
        """Folder guidance (if enabled) with per-image guidance appended."""
        parts: list[str] = []
        if self.folder_guidance_enabled and self.folder_guidance.strip():
            parts.append(self.folder_guidance.strip())
        per_image = self.per_image.get(filename, "")
        if per_image.strip() and self.per_image_active(filename):
            parts.append(per_image.strip())
        return "\n\n".join(parts)


class CaptionStore:
    def __init__(self, folder: str | Path, extension: str) -> None:
        self.folder = Path(folder)
        self.extension = extension

    def images(self) -> list[Path]:
        if self.folder.name.lower() == "edit":
            return []
        return sorted(
            [path for path in self.folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
            key=lambda path: path.name.lower(),
        )

    def caption_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(self.extension)

    def source_text_path(self, image_path: Path) -> Path:
        """The plain-text source caption sidecar for an image (image.jpg -> image.txt),
        following the same last-suffix convention as the JSON caption."""
        return image_path.with_suffix(".txt")

    def load_source_text(self, image_path: Path) -> str:
        """The image's .txt source caption stripped of whitespace, or "" if none.
        Returns "" when .txt is itself the caption extension (no separate source)."""
        path = self.source_text_path(image_path)
        if path == self.caption_path(image_path):
            return ""
        try:
            if path.is_file():
                return path.read_text(encoding="utf-8-sig", errors="replace").strip()
        except OSError:
            pass
        return ""

    def has_source_text(self, image_path: Path) -> bool:
        path = self.source_text_path(image_path)
        if path == self.caption_path(image_path):
            return False
        return path.is_file()

    def any_source_text(self, images) -> bool:
        """True if at least one image in the folder has a matching .txt sidecar.
        Used to gate the convert feature — pointless with no source captions."""
        return any(self.has_source_text(img) for img in images)

    def failure_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(".caption_failed.json")

    def load_failure_marker(self, image_path: Path) -> dict[str, Any] | None:
        path = self.failure_path(image_path)
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def has_failure_marker(self, image_path: Path) -> bool:
        return self.failure_path(image_path).exists()

    def save_failure_marker(self, image_path: Path, marker: dict[str, Any]) -> Path:
        path = self.failure_path(image_path)
        path.write_text(json.dumps(marker, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def clear_failure_marker(self, image_path: Path) -> bool:
        path = self.failure_path(image_path)
        if not path.exists():
            return False
        path.unlink()
        return True

    def load_caption(self, image_path: Path) -> tuple[dict[str, Any], str | None]:
        caption_path = self.caption_path(image_path)
        if not caption_path.exists():
            return default_caption(), f"No {self.extension} caption yet; edit fields or click Save to create it."

        raw = caption_path.read_text(encoding="utf-8-sig")
        if not raw.strip():
            return default_caption(), f"{caption_path.name} is empty."

        try:
            return parse_caption_text(raw), None
        except (json.JSONDecodeError, ValueError) as exc:
            if self.extension in {".txt", ".caption"}:
                return caption_from_plain_text(raw), f"Imported plain text from {caption_path.name}; save will convert it to Ideogram JSON."
            return default_caption(), f"Could not parse {caption_path.name}: {exc}"

    def save_caption(self, image_path: Path, caption: dict[str, Any]) -> Path:
        caption_path = self.caption_path(image_path)
        caption_path.write_text(serialize_caption(caption, indent=2), encoding="utf-8")
        return caption_path

    def project_dir(self) -> Path:
        return self.folder / PROJECT_DIRNAME

    def project_path(self) -> Path:
        return self.project_dir() / PROJECT_FILENAME

    def load_project(self) -> ProjectConfig:
        path = self.project_path()
        if not path.exists():
            return ProjectConfig()
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError):
            return ProjectConfig()
        if not isinstance(data, dict):
            return ProjectConfig()

        per_image: dict[str, str] = {}
        raw_per_image = data.get("per_image", {})
        if isinstance(raw_per_image, dict):
            for key, value in raw_per_image.items():
                if isinstance(key, str) and isinstance(value, str):
                    per_image[key] = value

        # Prune orphans: drop per-image entries whose image no longer exists.
        existing = {path.name for path in self.images()}
        if existing:
            per_image = {name: text for name, text in per_image.items() if name in existing}

        per_image_enabled: dict[str, bool] = {}
        raw_enabled = data.get("per_image_enabled", {})
        if isinstance(raw_enabled, dict):
            for key, value in raw_enabled.items():
                if isinstance(key, str) and isinstance(value, bool):
                    per_image_enabled[key] = value
        if existing:
            per_image_enabled = {n: e for n, e in per_image_enabled.items() if n in existing}

        generated_guidance: dict[str, str] = {}
        raw_gen = data.get("generated_guidance", {})
        if isinstance(raw_gen, dict):
            for key, value in raw_gen.items():
                if isinstance(key, str) and isinstance(value, str):
                    generated_guidance[key] = value
        if existing:
            generated_guidance = {n: g for n, g in generated_guidance.items() if n in existing}

        caption_flags: dict[str, list[str]] = {}
        raw_flags = data.get("caption_flags", {})
        if isinstance(raw_flags, dict):
            for key, value in raw_flags.items():
                if isinstance(key, str) and isinstance(value, list):
                    issues = [str(v) for v in value if isinstance(v, str)]
                    if issues:
                        caption_flags[key] = issues
        if existing:
            caption_flags = {n: v for n, v in caption_flags.items() if n in existing}

        review_marks: set[str] = set()
        raw_marks = data.get("review_marks", [])
        if isinstance(raw_marks, list):
            for name in raw_marks:
                if isinstance(name, str):
                    review_marks.add(name)
        if existing:
            review_marks = {n for n in review_marks if n in existing}

        creative = data.get("creative_json")
        return ProjectConfig(
            name=str(data.get("name", "")),
            folder_guidance=str(data.get("folder_guidance", "")),
            folder_guidance_enabled=bool(data.get("folder_guidance_enabled", True)),
            per_image=per_image,
            per_image_enabled=per_image_enabled,
            creative_json=creative if isinstance(creative, bool) else None,
            convert_txt_to_json=bool(data.get("convert_txt_to_json", False)),
            generated_guidance=generated_guidance,
            caption_flags=caption_flags,
            review_marks=review_marks,
        )

    def save_project(self, config: ProjectConfig) -> Path:
        path = self.project_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "name": config.name or self.folder.name,
            "folder_guidance": config.folder_guidance,
            "folder_guidance_enabled": config.folder_guidance_enabled,
            "per_image": {name: text for name, text in config.per_image.items() if text.strip()},
        }
        enabled = {
            name: flag for name, flag in config.per_image_enabled.items()
            if config.per_image.get(name, "").strip()
        }
        if enabled:
            data["per_image_enabled"] = enabled
        if config.creative_json is not None:
            data["creative_json"] = config.creative_json
        if config.convert_txt_to_json:
            data["convert_txt_to_json"] = True
        # Keep a stamp for every still-present image (empty string is meaningful:
        # "generated with no guidance"), so changes are detected after a restart.
        gen = {name: text for name, text in config.generated_guidance.items()}
        if gen:
            data["generated_guidance"] = gen
        flags = {name: list(v) for name, v in config.caption_flags.items() if v}
        if flags:
            data["caption_flags"] = flags
        if config.review_marks:
            data["review_marks"] = sorted(config.review_marks)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
