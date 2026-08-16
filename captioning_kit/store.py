from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .presets import DEFAULT_PRESET, get_preset
from .training_goals import DEFAULT_GOAL, get_goal
from .video_tools import VIDEO_EXTENSIONS
from .schema import IMAGE_EXTENSIONS, caption_from_plain_text, caption_health, default_caption, parse_caption_text, serialize_caption

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
    per_file: dict[str, str] = field(default_factory=dict)
    per_file_enabled: dict[str, bool] = field(default_factory=dict)
    creative_json: bool | None = None  # None = inherit the global setting
    # When True, captioning feeds each image's matching .txt sidecar to the model
    # as a source caption to upgrade into structured JSON (folder-wide mode).
    convert_txt_to_json: bool = False
    # filename -> the effective guidance string that produced its current caption.
    # Lets us flag images whose guidance has changed since they were last run.
    generated_guidance: dict[str, str] = field(default_factory=dict)
    # The same snapshot split by scope, so a "guidance changed" notice can say whether
    # the folder-wide guidance, this image's guidance, or both changed. Absent for
    # captions made before split-stamping existed (those fall back to a generic notice).
    generated_folder: dict[str, str] = field(default_factory=dict)
    generated_image: dict[str, str] = field(default_factory=dict)
    # filename -> list of issue strings from the last health check (corrupt/off-schema output).
    # Empty/absent = no known problems. Surfaced as a review marker; cleared when re-saved.
    caption_flags: dict[str, list[str]] = field(default_factory=dict)
    # filenames the user has manually flagged for review (independent of caption_flags).
    review_marks: set[str] = field(default_factory=set)
    # filenames where convert mode is overridden OFF — even with a matching .txt, this
    # image is captioned from the image alone. Stored as exceptions (default = use .txt).
    convert_omit: set[str] = field(default_factory=set)
    # Which caption preset this dataset uses (see presets.py). Stored per-folder so a
    # dataset always reopens in the format it was captioned in.
    preset: str = DEFAULT_PRESET
    # "auto" follows each file; "image"/"video" pin the captioning guidance to the
    # kind of dataset the user is actually building.
    media_mode: str = "auto"
    # What this dataset is training. Changes which details the caption omits.
    training_goal: str = DEFAULT_GOAL
    generated_goal: dict = field(default_factory=dict)

    def set_convert_omit(self, filename: str, omit: bool) -> None:
        if omit:
            self.convert_omit.add(filename)
        else:
            self.convert_omit.discard(filename)

    def is_convert_omitted(self, filename: str) -> bool:
        return filename in self.convert_omit

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

    def mark_generated(self, filename: str, guidance: str,
                       folder: str | None = None, image: str | None = None) -> None:
        """Stamp the guidance that produced this image's just-saved caption. The
        folder/per-file parts are stamped too (when given) so a later change can be
        attributed to a scope."""
        self.generated_guidance[filename] = guidance or ""
        if folder is not None:
            self.generated_folder[filename] = folder
        if image is not None:
            self.generated_image[filename] = image
        # The training goal shapes captions as much as the guidance text does, so a
        # goal switch has to read as "changed" too.
        self.generated_goal[filename] = self.training_goal

    def last_run_guidance(self, filename: str) -> str | None:
        """The guidance recorded at the last successful generation, or None."""
        return self.generated_guidance.get(filename)

    def guidance_changed(self, filename: str) -> bool:
        """True when the current effective guidance differs from what produced the
        last generated caption. Images never generated (no stamp) are not flagged."""
        prev = self.generated_guidance.get(filename)
        if prev is None:
            return False
        stamped_goal = self.generated_goal.get(filename)
        # Missing stamp = captioned before goals existed; not a change on its own.
        if stamped_goal is not None and stamped_goal != self.training_goal:
            return True
        return prev.strip() != self.resolved_for(filename).strip()

    def effective_folder_guidance(self) -> str:
        """The folder-wide guidance actually applied right now ("" when disabled/empty)."""
        if self.folder_guidance_enabled and self.folder_guidance.strip():
            return self.folder_guidance.strip()
        return ""

    def effective_image_guidance(self, filename: str) -> str:
        """This image's per-file guidance actually applied right now ("" when off/empty)."""
        per_file = self.per_file.get(filename, "")
        if per_file.strip() and self.per_file_active(filename):
            return per_file.strip()
        return ""

    def folder_guidance_changed(self, filename: str) -> bool:
        """True when the folder-wide guidance differs from the last generation's.
        False when there's no split stamp (caption predates split-stamping)."""
        if filename not in self.generated_folder:
            return False
        return self.generated_folder[filename].strip() != self.effective_folder_guidance()

    def image_guidance_changed(self, filename: str) -> bool:
        """True when this image's per-file guidance differs from the last generation's.
        False when there's no split stamp (caption predates split-stamping)."""
        if filename not in self.generated_image:
            return False
        return self.generated_image[filename].strip() != self.effective_image_guidance(filename)

    def per_file_guidance(self, filename: str) -> str:
        return self.per_file.get(filename, "")

    def has_per_file_guidance(self, filename: str) -> bool:
        return bool(self.per_file.get(filename, "").strip())

    def per_file_active(self, filename: str) -> bool:
        """Whether this image's per-file guidance is applied. Default on; an
        explicit False suppresses it without deleting the text."""
        return self.per_file_enabled.get(filename, True)

    def resolved_for(self, filename: str) -> str:
        """Folder guidance (if enabled) with per-file guidance appended."""
        parts: list[str] = []
        if self.folder_guidance_enabled and self.folder_guidance.strip():
            parts.append(self.folder_guidance.strip())
        per_file = self.per_file.get(filename, "")
        if per_file.strip() and self.per_file_active(filename):
            parts.append(per_file.strip())
        return "\n\n".join(parts)


class CaptionStore:
    def __init__(self, folder: str | Path, extension: str) -> None:
        self.folder = Path(folder)
        self.extension = extension

    BYPASS_DIRNAME = ".bypass"

    def bypass_dir(self) -> Path:
        """Holding pen for files kept out of the dataset.

        A subfolder rather than a flag in project.json: the trainer reads the
        folder, so the only way to reliably exclude a file from a run is for it not
        to be there. A dot-prefixed name keeps it out of most tooling's way, and the
        caption travels with the media so nothing is orphaned.
        """
        return self.folder / self.BYPASS_DIRNAME

    def is_bypassed(self, path: Path) -> bool:
        return Path(path).parent.name == self.BYPASS_DIRNAME

    def _media_in(self, folder: Path) -> list[Path]:
        exts = set(IMAGE_EXTENSIONS) | VIDEO_EXTENSIONS
        if not folder.is_dir():
            return []
        try:
            found = [p for p in folder.iterdir()
                     # A dot-prefixed name is a working file, not dataset content —
                     # a stray render left behind by a crash must never be captioned
                     # or handed to a trainer just because it ends in .mp4.
                     if p.is_file() and not p.name.startswith(".")
                     and p.suffix.lower() in exts]
        except OSError:
            return []
        return sorted(found, key=lambda p: p.name.lower())

    def work_dir(self) -> Path:
        """Scratch space inside the folder, out of the dataset's way.

        Same filesystem as the media, so a finished render can be moved into place
        atomically rather than copied across devices.
        """
        path = self.folder / PROJECT_DIRNAME / "work"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sweep_work_files(self) -> int:
        """Delete leftovers from an interrupted render. Returns how many went."""
        removed = 0
        for candidate in (self.folder / PROJECT_DIRNAME / "work",):
            if not candidate.is_dir():
                continue
            for path in candidate.iterdir():
                try:
                    if path.is_file():
                        path.unlink()
                        removed += 1
                except OSError:
                    continue
        # Older builds wrote previews beside the clip; clear those too.
        try:
            for path in self.folder.glob(".*.mutepreview.*"):
                path.unlink()
                removed += 1
        except OSError:
            pass
        return removed

    def images(self, include_bypassed: bool = True) -> list[Path]:
        """Every captionable media file — images and videos. The name predates video
        support; it's the single listing the filmstrip is built from.

        Bypassed files come last so the filmstrip can separate them visually, and
        callers that shouldn't see them at all (batch captioning) pass False.
        """
        if self.folder.name.lower() == "edit":
            return []
        active = self._media_in(self.folder)
        if not include_bypassed:
            return active
        return active + self.bypassed_images()

    def bypassed_images(self) -> list[Path]:
        return self._media_in(self.bypass_dir())

    def duplicate_to(self, dest: Path, *, keep_captions: bool = True,
                     keep_settings: bool = True, keep_originals: bool = False,
                     keep_bypassed: bool = True,
                     progress=None) -> dict:
        """Copy this dataset somewhere else, choosing what comes along.

        Always copies the media itself; everything derived from it is optional.
        Copies rather than moves, and never writes into the source, so a mistake
        here can't damage the dataset being duplicated.

        progress(done, total, name) is called per file and may return False to
        cancel, in which case the partial copy is left in place for inspection
        rather than silently deleted.
        """
        dest = Path(dest)
        dest.mkdir(parents=True, exist_ok=True)
        counts = {"media": 0, "captions": 0, "originals": 0, "bypassed": 0,
                  "settings": 0, "skipped": 0, "cancelled": False}

        jobs: list[tuple[Path, Path, str]] = []
        queued: set[Path] = set()

        def add(src: Path, target: Path, kind: str) -> None:
            # A .txt-preset caption and the convert-mode source sidecar resolve to
            # the same file, so without this they'd be copied (and counted) twice.
            if src in queued:
                return
            queued.add(src)
            jobs.append((src, target, kind))
        for path in self._media_in(self.folder):
            add(path, dest / path.name, "media")
            if keep_captions:
                for companion in (self.caption_path(path), path.with_suffix(".txt")):
                    if companion.exists() and companion != path:
                        add(companion, dest / companion.name, "captions")
        if keep_bypassed:
            for path in self.bypassed_images():
                target_dir = dest / self.BYPASS_DIRNAME
                add(path, target_dir / path.name, "bypassed")
                if keep_captions:
                    companion = self.caption_path(path)
                    if companion.exists():
                        add(companion, target_dir / companion.name, "captions")
        if keep_originals and self.originals_dir().is_dir():
            for path in sorted(self.originals_dir().iterdir()):
                if path.is_file():
                    add(path, dest / self.ORIGINALS_DIRNAME / path.name, "originals")
        if keep_settings:
            project = self.project_path()
            if project.exists():
                add(project, dest / PROJECT_DIRNAME / project.name, "settings")

        total = len(jobs)
        for index, (src, target, kind) in enumerate(jobs, start=1):
            if progress is not None and progress(index, total, src.name) is False:
                counts["cancelled"] = True
                break
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                counts[kind] += 1
            except OSError:
                counts["skipped"] += 1
        return counts

    def import_media(self, sources: list[Path]) -> tuple[list[Path], list[str]]:
        """Copy media into the dataset folder. Returns (added, skipped_reasons).

        Copies rather than moves: the source may be a library the user wants to keep
        intact, and an accidental drag shouldn't rearrange their disk. A caption
        sitting beside a source file comes with it, so importing an already-captioned
        file doesn't lose the work.
        """
        exts = set(IMAGE_EXTENSIONS) | VIDEO_EXTENSIONS
        added: list[Path] = []
        skipped: list[str] = []
        queue: list[Path] = []
        for raw in sources:
            src = Path(raw)
            if src.is_dir():
                queue.extend(self._media_in(src))
            elif src.is_file():
                queue.append(src)
        for src in queue:
            if src.suffix.lower() not in exts:
                skipped.append(f"{src.name} (not an image or video)")
                continue
            try:
                if src.parent.resolve() == self.folder.resolve():
                    skipped.append(f"{src.name} (already in this folder)")
                    continue
            except OSError:
                pass
            target = self.folder / src.name
            if target.exists():
                stem, suffix = src.stem, src.suffix
                n = 2
                while (self.folder / f"{stem}_{n}{suffix}").exists():
                    n += 1
                target = self.folder / f"{stem}_{n}{suffix}"
            try:
                shutil.copy2(src, target)
            except OSError as exc:
                skipped.append(f"{src.name} ({exc})")
                continue
            # Bring a caption that already sits beside the source.
            for ext in (self.extension, ".txt"):
                companion = src.with_suffix(ext)
                if companion.exists() and companion != src:
                    try:
                        shutil.copy2(companion, target.with_suffix(ext))
                    except OSError:
                        pass
            added.append(target)
        return added, skipped

    def bypass(self, path: Path) -> Path:
        """Move a file and its caption out of the dataset. Returns the new path."""
        path = Path(path)
        if self.is_bypassed(path):
            return path
        dest_dir = self.bypass_dir()
        dest_dir.mkdir(parents=True, exist_ok=True)
        return self._move_with_caption(path, dest_dir)

    def unbypass(self, path: Path) -> Path:
        """Bring a bypassed file and its caption back into the dataset."""
        path = Path(path)
        if not self.is_bypassed(path):
            return path
        return self._move_with_caption(path, self.folder)

    def _move_with_caption(self, path: Path, dest_dir: Path) -> Path:
        target = dest_dir / path.name
        if target.exists():
            # Never clobber: a name collision means two different files, and the
            # user would lose one silently.
            stem, suffix = path.stem, path.suffix
            n = 2
            while (dest_dir / f"{stem}_{n}{suffix}").exists():
                n += 1
            target = dest_dir / f"{stem}_{n}{suffix}"
        caption = self.caption_path(path)
        os.replace(path, target)
        if caption.exists():
            os.replace(caption, self.caption_path(target))
        # A matching .txt source (convert mode) travels too, when it isn't the
        # caption itself.
        source_txt = path.with_suffix(".txt")
        if source_txt.exists() and source_txt != caption:
            os.replace(source_txt, target.with_suffix(".txt"))
        return target

    def caption_path(self, image_path: Path) -> Path:
        return image_path.with_suffix(self.extension)

    # ---- original backups (crop/resize safety net) ----
    # Edits are destructive-in-place so the dataset keeps one file per image with a
    # stable name; the pre-edit file is copied into <folder>/.original/<name> first.
    # First backup wins: repeated edits never overwrite the true original.

    ORIGINALS_DIRNAME = ".original"

    def originals_dir(self) -> Path:
        return self.folder / self.ORIGINALS_DIRNAME

    def original_backup_path(self, image_path: Path) -> Path:
        return self.originals_dir() / Path(image_path).name

    def has_original_backup(self, image_path: Path) -> bool:
        return self.original_backup_path(image_path).exists()

    def backup_original(self, image_path: Path) -> Path:
        """Copy the image into .original/ unless a backup already exists (first
        backup wins). Returns the backup path."""
        import shutil
        dst = self.original_backup_path(image_path)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, dst)
        return dst

    def restore_original(self, image_path: Path) -> bool:
        """Copy the backed-up original over the current file. The backup is kept, so
        the image can be edited and reverted again. False if no backup exists."""
        import shutil
        src = self.original_backup_path(image_path)
        if not src.exists():
            return False
        shutil.copy2(src, image_path)
        return True

    def revert_to_original(self, image_path: Path) -> bool:
        """Put the pre-edit file back and drop the backup.

        Differs from restore_original, which copies the backup over the file and
        keeps it: this is "undo the edit entirely", so leaving the backup behind
        would imply an edit that no longer exists.
        """
        src = self.original_backup_path(image_path)
        if not src.exists():
            return False
        os.replace(src, image_path)
        try:
            backups = self.originals_dir()
            if backups.is_dir() and not any(backups.iterdir()):
                backups.rmdir()
        except OSError:
            pass
        return True

    def delete_media(self, image_path: Path) -> list[Path]:
        """Delete a file and everything derived from it. Returns what was removed."""
        image_path = Path(image_path)
        removed: list[Path] = []
        candidates = [image_path, self.caption_path(image_path),
                      image_path.with_suffix(".txt"),
                      self.original_backup_path(image_path)]
        for path in candidates:
            if path in removed or not path.exists() or not path.is_file():
                continue
            try:
                path.unlink()
                removed.append(path)
            except OSError:
                continue
        return removed

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

    def caption_file_issues(self, image_path: Path) -> list[str]:
        """Health issues for an image's caption file as it sits on disk, including parse
        failures. Empty list means either there is no caption yet (nothing to flag) or
        the caption is healthy. Re-validates existing files (e.g. on folder open) so a
        hand-edited or corrupt caption is flagged, not only freshly generated ones."""
        caption_path = self.caption_path(image_path)
        if not caption_path.exists():
            return []
        try:
            raw = caption_path.read_text(encoding="utf-8-sig")
        except OSError:
            return ["could not read caption file"]
        if not raw.strip():
            return ["caption file is empty"]
        try:
            caption = parse_caption_text(raw)
        except (json.JSONDecodeError, ValueError):
            return ["corrupt caption file — could not parse JSON"]
        return caption_health(caption)

    def save_caption(self, image_path: Path, caption: dict[str, Any]) -> Path:
        caption_path = self.caption_path(image_path)
        caption_path.write_text(serialize_caption(caption, indent=2), encoding="utf-8")
        return caption_path

    # ---- plain-text presets ----
    # A plain preset's caption *is* the sidecar's text, so these skip the schema
    # entirely rather than round-tripping through the Ideogram caption dict.

    def load_plain_caption(self, image_path: Path) -> tuple[str, str | None]:
        """(text, status message). Missing file is normal: empty text, no message."""
        caption_path = self.caption_path(image_path)
        if not caption_path.exists():
            return "", None
        try:
            return caption_path.read_text(encoding="utf-8-sig").strip(), None
        except OSError as exc:
            return "", f"Could not read {caption_path.name}: {exc}"

    def save_plain_caption(self, image_path: Path, text: str) -> Path:
        caption_path = self.caption_path(image_path)
        caption_path.write_text((text or "").strip() + "\n", encoding="utf-8")
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

        per_file: dict[str, str] = {}
        # "per_image" is the pre-rename key: datasets captioned before per-file
        # guidance was generalised beyond images still use it, so read either.
        raw_per_file = data.get("per_file", data.get("per_image", {}))
        if isinstance(raw_per_file, dict):
            for key, value in raw_per_file.items():
                if isinstance(key, str) and isinstance(value, str):
                    per_file[key] = value

        # Prune orphans: drop per-file entries whose image no longer exists.
        existing = {path.name for path in self.images()}
        if existing:
            per_file = {name: text for name, text in per_file.items() if name in existing}

        per_file_enabled: dict[str, bool] = {}
        raw_enabled = data.get("per_file_enabled", data.get("per_image_enabled", {}))
        if isinstance(raw_enabled, dict):
            for key, value in raw_enabled.items():
                if isinstance(key, str) and isinstance(value, bool):
                    per_file_enabled[key] = value
        if existing:
            per_file_enabled = {n: e for n, e in per_file_enabled.items() if n in existing}

        generated_guidance: dict[str, str] = {}
        raw_gen = data.get("generated_guidance", {})
        if isinstance(raw_gen, dict):
            for key, value in raw_gen.items():
                if isinstance(key, str) and isinstance(value, str):
                    generated_guidance[key] = value
        if existing:
            generated_guidance = {n: g for n, g in generated_guidance.items() if n in existing}

        def _load_str_map(field_name: str) -> dict[str, str]:
            out: dict[str, str] = {}
            raw = data.get(field_name, {})
            if isinstance(raw, dict):
                for key, value in raw.items():
                    if isinstance(key, str) and isinstance(value, str):
                        out[key] = value
            if existing:
                out = {n: g for n, g in out.items() if n in existing}
            return out

        generated_folder = _load_str_map("generated_folder")
        generated_image = _load_str_map("generated_image")

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

        convert_omit: set[str] = set()
        raw_omit = data.get("convert_omit", [])
        if isinstance(raw_omit, list):
            for name in raw_omit:
                if isinstance(name, str):
                    convert_omit.add(name)
        if existing:
            convert_omit = {n for n in convert_omit if n in existing}

        creative = data.get("creative_json")
        return ProjectConfig(
            name=str(data.get("name", "")),
            folder_guidance=str(data.get("folder_guidance", "")),
            folder_guidance_enabled=bool(data.get("folder_guidance_enabled", True)),
            per_file=per_file,
            per_file_enabled=per_file_enabled,
            creative_json=creative if isinstance(creative, bool) else None,
            convert_txt_to_json=bool(data.get("convert_txt_to_json", False)),
            generated_guidance=generated_guidance,
            generated_folder=generated_folder,
            generated_image=generated_image,
            caption_flags=caption_flags,
            review_marks=review_marks,
            convert_omit=convert_omit,
            # get_preset() normalises unknown keys back to the default, so a
            # hand-edited or newer project file can still be opened.
            preset=get_preset(data.get("preset")).key,
            training_goal=get_goal(data.get("training_goal")).key,
            generated_goal=dict(data.get("generated_goal") or {}),
            media_mode=(str(data.get("media_mode", "auto")).strip().lower()
                        if str(data.get("media_mode", "auto")).strip().lower()
                        in ("auto", "image", "video") else "auto"),
        )

    def save_project(self, config: ProjectConfig) -> Path:
        path = self.project_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "name": config.name or self.folder.name,
            "folder_guidance": config.folder_guidance,
            "folder_guidance_enabled": config.folder_guidance_enabled,
            "per_file": {name: text for name, text in config.per_file.items() if text.strip()},
        }
        enabled = {
            name: flag for name, flag in config.per_file_enabled.items()
            if config.per_file.get(name, "").strip()
        }
        if enabled:
            data["per_file_enabled"] = enabled
        if config.creative_json is not None:
            data["creative_json"] = config.creative_json
        if config.convert_txt_to_json:
            data["convert_txt_to_json"] = True
        data["preset"] = config.preset
        data["media_mode"] = config.media_mode
        data["training_goal"] = config.training_goal
        data["generated_goal"] = config.generated_goal
        # Keep a stamp for every still-present image (empty string is meaningful:
        # "generated with no guidance"), so changes are detected after a restart.
        gen = {name: text for name, text in config.generated_guidance.items()}
        if gen:
            data["generated_guidance"] = gen
        gen_folder = {name: text for name, text in config.generated_folder.items()}
        if gen_folder:
            data["generated_folder"] = gen_folder
        gen_image = {name: text for name, text in config.generated_image.items()}
        if gen_image:
            data["generated_image"] = gen_image
        flags = {name: list(v) for name, v in config.caption_flags.items() if v}
        if flags:
            data["caption_flags"] = flags
        if config.review_marks:
            data["review_marks"] = sorted(config.review_marks)
        if config.convert_omit:
            data["convert_omit"] = sorted(config.convert_omit)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
