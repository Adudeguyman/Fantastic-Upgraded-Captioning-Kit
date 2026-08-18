"""Video tooling: managed ffmpeg/ffprobe, metadata probing, and poster frames.

ffmpeg is the backbone for all video work (trim, crop, resize, fps, frame
extraction). It's managed the same way as llama-server: the app can download a
pinned static build into <app>/ffmpeg, and prefers that over anything on PATH so
behaviour is identical across machines. BtbN's FFmpeg-Builds provide portable
single-archive builds for Linux and Windows that need no system install.
"""
from __future__ import annotations

import ctypes
import json
import os
import shutil
import re
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .llm_captioning import app_base_dir

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}

_UA = {"User-Agent": "fantastic-captioning-kit"}

# BtbN static builds ("latest" tag is a rolling release of master; gpl builds
# include libx264/x265 which we want for re-encoding).
_FFMPEG_URLS = {
    ("linux", "x86_64"): (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-linux64-gpl.tar.xz"),
    ("linux", "aarch64"): (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-linuxarm64-gpl.tar.xz"),
    ("windows", "x86_64"): (
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/"
        "ffmpeg-master-latest-win64-gpl.zip"),
}


# FFmpeg's own logger writes straight to stderr and does NOT go through Qt's
# logging categories, so QT_LOGGING_RULES can't touch it. Decoding ordinary
# real-world clips (HE-AAC audio especially) emits hundreds of decoder complaints
# per file, which buries genuine errors in the console. Measured on a corrupt-audio
# sample: INFO 11 lines, WARNING 10, ERROR 10, FATAL 0 — the chatter is logged at
# ERROR, so only FATAL is quiet. Real playback failures still reach us through
# QMediaPlayer.errorOccurred, which the UI surfaces properly.
AV_LOG_FATAL = 8

_logs_quieted = False


def quiet_ffmpeg_logs(level: int = AV_LOG_FATAL) -> bool:
    """Lower the log level of the libavutil that Qt's multimedia plugin already
    loaded. Must be called after a QMediaPlayer exists, or the library isn't
    resident yet. Best-effort: returns False and changes nothing if it can't
    attach, since console tidiness must never break playback."""
    global _logs_quieted
    if _logs_quieted:
        return True
    try:
        import PySide6
    except ImportError:
        return False
    qt_dir = Path(PySide6.__file__).resolve().parent / "Qt"
    if os.name == "nt":
        candidates = sorted((qt_dir / "bin").glob("avutil*.dll"), reverse=True)
    else:
        lib_dir = qt_dir / "lib"
        candidates = (sorted(lib_dir.glob("libavutil.so.*"), reverse=True)
                      + sorted(lib_dir.glob("libavutil*.dylib"), reverse=True))
    for cand in candidates:
        try:
            if os.name == "nt":
                lib = ctypes.WinDLL(str(cand))
            else:
                # RTLD_NOLOAD attaches to Qt's already-loaded copy. Loading it
                # fresh fails on unresolved deps (e.g. OpenSSL symbols).
                lib = ctypes.CDLL(str(cand), mode=getattr(os, "RTLD_NOLOAD", 0))
            lib.av_log_set_level(ctypes.c_int(level))
        except (OSError, AttributeError, ValueError):
            continue
        _logs_quieted = True
        return True
    return False


def is_video(path: Path | str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def managed_ffmpeg_dir() -> Path:
    return app_base_dir() / "ffmpeg"


def _exe(name: str) -> str:
    return f"{name}.exe" if os.name == "nt" else name


def _find_tool(name: str) -> Path | None:
    """Managed binary first (consistent across machines), then PATH."""
    root = managed_ffmpeg_dir()
    if root.is_dir():
        direct = root / _exe(name)
        if direct.is_file():
            return direct
        for cand in root.rglob(_exe(name)):
            if cand.is_file():
                return cand
    which = shutil.which(name)
    return Path(which) if which else None


def find_ffmpeg() -> Path | None:
    return _find_tool("ffmpeg")


def find_ffprobe() -> Path | None:
    return _find_tool("ffprobe")


def ffmpeg_available() -> bool:
    return find_ffmpeg() is not None and find_ffprobe() is not None


def _platform_key() -> tuple[str, str]:
    system = "windows" if os.name == "nt" else sys.platform
    if system.startswith("linux"):
        system = "linux"
    machine = os.uname().machine if hasattr(os, "uname") else "x86_64"
    if machine in ("AMD64", "amd64", "x64"):
        machine = "x86_64"
    if machine in ("arm64",):
        machine = "aarch64"
    return system, machine


def ffmpeg_download_url() -> str | None:
    return _FFMPEG_URLS.get(_platform_key())


def install_ffmpeg(progress: Callable[[str], None] | None = None) -> Path:
    """Download and unpack a static ffmpeg build into the managed dir. Returns the
    ffmpeg binary path. Raises RuntimeError on unsupported platforms or failures."""
    url = ffmpeg_download_url()
    if url is None:
        raise RuntimeError(
            "No prebuilt ffmpeg is available for this platform. Install ffmpeg with "
            "your package manager so it's on PATH, then try again.")
    dest_root = managed_ffmpeg_dir()
    dest_root.mkdir(parents=True, exist_ok=True)
    archive = dest_root / url.rsplit("/", 1)[1]
    if progress:
        progress("Downloading ffmpeg \u2026")
    request = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(request, timeout=60) as resp, open(archive, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 512)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(f"Downloading ffmpeg \u2026 {done * 100 // total}%")
    if progress:
        progress("Unpacking ffmpeg \u2026")
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(dest_root)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest_root)
    archive.unlink(missing_ok=True)
    ffmpeg = _find_tool("ffmpeg")
    ffprobe = _find_tool("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise RuntimeError("The ffmpeg archive didn't contain the expected binaries.")
    for tool in (ffmpeg, ffprobe):
        try:
            tool.chmod(tool.stat().st_mode | 0o755)
        except OSError:
            pass
    if progress:
        progress("ffmpeg ready.")
    return ffmpeg


@dataclass(frozen=True)
class VideoInfo:
    """Metadata for one video, from ffprobe."""
    duration_s: float
    width: int
    height: int
    fps: float
    frame_count: int | None
    codec: str
    # Display rotation in degrees from the stream's display matrix (0/90/180/270).
    # width/height above are already the ROTATED (displayed) size — a phone clip
    # stored 1920x1080 with rotation 90 reports 1080x1920, because that's what the
    # viewer, the trainer and the crop rect all see.
    rotation: int = 0

    @property
    def is_rotated(self) -> bool:
        return self.rotation % 180 != 0

    @property
    def duration_label(self) -> str:
        """mm:ss (or h:mm:ss) for the filmstrip badge."""
        total = max(0, int(round(self.duration_s)))
        h, rem = divmod(total, 3600)
        m, s = divmod(rem, 60)
        return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _parse_rate(rate: str) -> float:
    """'30000/1001' -> 29.97; '25/1' -> 25.0. 0.0 if unparsable."""
    try:
        if "/" in rate:
            num, den = rate.split("/", 1)
            den_f = float(den)
            return float(num) / den_f if den_f else 0.0
        return float(rate)
    except (TypeError, ValueError):
        return 0.0


def probe_video(path: Path | str, timeout: float = 20.0) -> VideoInfo | None:
    """Read duration/size/fps for a video. None when ffprobe is missing or the file
    can't be read — callers treat that as 'unknown', never a crash."""
    ffprobe = find_ffprobe()
    if ffprobe is None:
        return None
    cmd = [str(ffprobe), "-v", "error", "-select_streams", "v:0",
           "-show_entries",
           "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,codec_name",
           # side data carries the display matrix; without asking for it, a rotated
           # phone clip reports its stored (landscape) size instead of what plays.
           "-show_entries", "stream_side_data=rotation",
           "-show_entries", "stream_tags=rotate",
           "-show_entries", "format=duration",
           "-of", "json", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0 or not proc.stdout.strip():
        return None
    try:
        data = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    streams = data.get("streams") or []
    if not streams:
        return None
    st = streams[0]
    fps = _parse_rate(str(st.get("avg_frame_rate") or "")) or \
        _parse_rate(str(st.get("r_frame_rate") or ""))
    try:
        duration = float((data.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        duration = 0.0
    # A phone clip is stored landscape with a display matrix telling players to
    # rotate it. Reporting the stored size would mean the spec check, the crop rect
    # and the trainer's bucketing all disagree with what's on screen.
    rotation = 0
    for side in (st.get("side_data_list") or []):
        if isinstance(side, dict) and side.get("rotation") is not None:
            try:
                rotation = int(round(float(side["rotation"]))) % 360
            except (TypeError, ValueError):
                rotation = 0
            break
    else:
        raw_tag = (st.get("tags") or {}).get("rotate")
        if raw_tag is not None:
            try:
                rotation = int(round(float(raw_tag))) % 360
            except (TypeError, ValueError):
                rotation = 0
    stored_w = int(st.get("width") or 0)
    stored_h = int(st.get("height") or 0)
    disp_w, disp_h = ((stored_h, stored_w) if rotation % 180 else (stored_w, stored_h))
    frames = None
    raw_frames = st.get("nb_frames")
    if isinstance(raw_frames, str) and raw_frames.isdigit():
        frames = int(raw_frames)
    elif duration and fps:
        frames = int(round(duration * fps))
    return VideoInfo(
        duration_s=duration,
        width=disp_w,
        height=disp_h,
        rotation=rotation,
        fps=round(fps, 3),
        frame_count=frames,
        codec=str(st.get("codec_name") or ""),
    )


def extract_poster(path: Path | str, out_path: Path | str,
                   at_s: float | None = None, timeout: float = 30.0) -> bool:
    """Write a single poster frame (JPEG/PNG by out_path suffix). Defaults to ~10%
    into the clip so black lead-ins don't produce black thumbnails."""
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    if at_s is None:
        info = probe_video(path)
        at_s = min(info.duration_s * 0.1, 3.0) if info and info.duration_s else 0.0
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ffmpeg), "-v", "error", "-y", "-ss", f"{max(0.0, at_s):.3f}",
           "-i", str(path), "-frames:v", "1", str(out)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and out.exists() and out.stat().st_size > 0


# ---------------------------------------------------------------------------
# Editing: trim / fps conform / resize
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VideoEditPlan:
    """A single re-encode: trim to [start_s, end_s), optionally conform fps and
    scale. Everything happens in one ffmpeg pass so a clip is only ever
    re-compressed once, however many things you change at the same time."""
    start_s: float
    end_s: float
    fps: float | None = None            # None keeps the source rate
    width: int | None = None            # None keeps the source size
    height: int | None = None
    frame_limit: int | None = None      # emit exactly this many frames
    crop: tuple[int, int, int, int] | None = None   # (x, y, w, h) in SOURCE pixels
    rotate: int = 0                 # clockwise degrees: 0, 90, 180, 270
    keep_audio: bool = True

    @property
    def duration_s(self) -> float:
        return max(0.0, self.end_s - self.start_s)

    def changes(self, info: "VideoInfo | None") -> list[str]:
        """Human-readable list of what this plan actually alters."""
        out: list[str] = []
        if self.rotate % 360:
            out.append(f"rotate {self.rotate % 360}\u00b0 clockwise")
        if self.crop:
            out.append(f"crop to {self.crop[2]}\u00d7{self.crop[3]}")
        if info is not None:
            trimmed = self.start_s > 0.01 or self.end_s < info.duration_s - 0.01
            if trimmed:
                out.append(f"trim to {self.duration_s:.2f}s")
            if self.fps and abs(self.fps - info.fps) > 0.01:
                out.append(f"{info.fps:g} \u2192 {self.fps:g} fps")
            if self.width and self.height and (self.width, self.height) != (info.width, info.height):
                out.append(f"{info.width}\u00d7{info.height} \u2192 {self.width}\u00d7{self.height}")
        if self.frame_limit:
            out.append(f"exactly {self.frame_limit} frames")
        return out

    def is_noop(self, info: "VideoInfo | None") -> bool:
        return not self.changes(info)


def plan_for_target(info: "VideoInfo", target, start_s: float, end_s: float,
                    bucket=None, snap: str = "down",
                    crop: tuple[int, int, int, int] | None = None) -> VideoEditPlan:
    """Conform a chosen span to a model's requirements.

    The frame count is snapped *down* by default: we can always drop frames from
    the source but never invent them, and both failure modes we know about are
    silent (LTX truncates a long clip, H3 pads a short one), so it's better to hand
    the trainer an exactly-legal clip than to rely on either behaviour.
    """
    fps = target.fps
    span = max(0.0, end_s - start_s)
    # A span is only ever meant to hold a whole number of frames, but it arrives via
    # milliseconds: 107 frames at 24fps is 4458ms, and 4.458 * 24 == 106.992, so a
    # plain int() would silently drop a frame — and for a 17n+5 grid, dropping one
    # frame costs a whole 17-frame block.
    available = int(round(span * fps))
    frames = target.snap_frames(min(available, target.max_frames()), snap)
    # A source shorter than the trainer's smallest accepted count (H3's is 22)
    # can't be conformed by cutting: snapping up would promise frames that don't
    # exist, ffmpeg would emit a short file, and the trainer would silently skip
    # it. Report what's actually there — an illegal count — so the caller's
    # below-minimum check refuses loudly instead.
    frames = max(1, min(frames, available))
    # Target dimensions follow the cropped frame, not the original one.
    width, height = (crop[2], crop[3]) if crop else (info.width, info.height)
    if bucket is not None:
        width, height = bucket.width, bucket.height
    else:
        width, height = target.fit_dimensions(width, height)
    # The emitted duration follows the frame count, not the other way round.
    return VideoEditPlan(
        start_s=start_s,
        end_s=start_s + (frames / fps if fps else span),
        fps=fps,
        width=width,
        height=height,
        frame_limit=frames,
        crop=crop,
    )


def ffmpeg_edit_command(src: Path | str, dst: Path | str, plan: VideoEditPlan,
                        ffmpeg: Path | str | None = None) -> list[str]:
    """Build the one-pass command.

    ffmpeg auto-rotates on decode by default, so the filter chain already sees the
    displayed orientation — which is why crop and scale coordinates match what the
    user drew. The output then carries no rotation metadata.
    """
    # Kept separate from execution so the exact arguments can be asserted in tests
    # without running anything.
    exe = str(ffmpeg or find_ffmpeg() or "ffmpeg")
    # -ss before -i seeks fast; with a re-encode it is still frame-accurate.
    cmd = [exe, "-v", "error", "-y", "-ss", f"{max(0.0, plan.start_s):.3f}"]
    cmd += ["-i", str(src)]
    # Bake any display rotation into the pixels and clear the metadata, so the
    # output can't be re-rotated by a downstream player or trainer. Without this a
    # conformed portrait clip ships as landscape pixels plus a "rotate me" flag.
    cmd += ["-metadata:s:v:0", "rotate=0"]
    if plan.duration_s > 0:
        cmd += ["-t", f"{plan.duration_s:.3f}"]
    filters: list[str] = []
    rotate = plan.rotate % 360
    if rotate:
        # Rotate first: the crop rect and target dimensions are expressed in the
        # orientation the user is looking at, so rotating afterwards would swap the
        # axes underneath them.
        filters.extend({
            90: ["transpose=1"],
            180: ["transpose=1", "transpose=1"],
            270: ["transpose=2"],
        }[rotate])
    if plan.crop:
        # First in the chain: the rect was drawn on the source frame, so cropping
        # after a scale would apply the coordinates to the wrong pixel grid.
        x, y, w, h = plan.crop
        filters.append(f"crop={w}:{h}:{x}:{y}")
    if plan.width and plan.height:
        # Scale to fill, then centre-crop, so the output is exactly the requested
        # size without distorting the picture.
        filters.append(
            f"scale={plan.width}:{plan.height}:force_original_aspect_ratio=increase")
        filters.append(f"crop={plan.width}:{plan.height}")
    if plan.fps:
        filters.append(f"fps={plan.fps:g}")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    if plan.frame_limit:
        cmd += ["-frames:v", str(int(plan.frame_limit))]
    cmd += ["-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p"]
    if plan.keep_audio:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    else:
        cmd += ["-an"]
    cmd += [str(dst)]
    return cmd


def apply_video_edit(src: Path | str, dst: Path | str, plan: VideoEditPlan,
                     timeout: float = 900.0) -> tuple[bool, str]:
    """Run the edit. Returns (ok, message); never raises for ordinary failures."""
    if find_ffmpeg() is None:
        return False, "ffmpeg isn't installed."
    cmd = ffmpeg_edit_command(src, dst, plan)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out."
    except OSError as exc:
        return False, f"Could not run ffmpeg: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        return False, detail[-1] if detail else "ffmpeg failed."
    out = Path(dst)
    if not out.exists() or out.stat().st_size == 0:
        return False, "ffmpeg produced no output."
    return True, ""


# ---------------------------------------------------------------------------
# Frame sampling for captioning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampledFrame:
    path: Path
    time_s: float       # position in the source clip
    index: int          # 1-based, for "Frame 2 of 6" labels


def extract_frames(video_path: Path | str, out_dir: Path | str, count: int,
                   start_s: float = 0.0, end_s: float | None = None,
                   max_edge: int = 768) -> list[SampledFrame]:
    """Sample `count` frames evenly across [start_s, end_s) for captioning.

    Sampling is *centred*: t = start + (i + 0.5) * span / count. The obvious
    ``i * span / count`` skews early and never reaches the final stretch of the
    clip (with 8 frames the last ~12% is unsampled), and endings matter — a
    caption should know how the action resolves. Centred sampling covers the span
    symmetrically without duplicating the exact first/last frame.

    Frames are decoded at most `max_edge` px on the long side: captioning doesn't
    need full resolution, and every pixel is paid for again in prompt tokens.

    Honours the caller's span so a trimmed clip is captioned as it will ship,
    not as it sits on disk.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise RuntimeError("ffmpeg isn't installed, so frames can't be sampled.")
    video_path = Path(video_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if end_s is None:
        info = probe_video(video_path)
        end_s = info.duration_s if info else 0.0
    span = max(0.0, float(end_s) - float(start_s))
    if span <= 0 or count <= 0:
        return []
    frames: list[SampledFrame] = []
    scale = f"scale='min({max_edge},iw)':'min({max_edge},ih)':force_original_aspect_ratio=decrease"
    for i in range(count):
        t = float(start_s) + (i + 0.5) * span / count
        dest = out_dir / f"frame_{i + 1:02d}.jpg"
        cmd = [str(ffmpeg), "-v", "error", "-y", "-ss", f"{t:.3f}",
               "-i", str(video_path), "-frames:v", "1", "-vf", scale,
               "-q:v", "3", str(dest)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        except (subprocess.TimeoutExpired, OSError):
            continue
        if proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0:
            frames.append(SampledFrame(path=dest, time_s=t, index=i + 1))
    return frames


def has_audio_stream(path: Path | str) -> bool:
    """True when the file carries at least one audio track."""
    ffprobe = find_ffprobe()
    if ffprobe is None:
        return False
    cmd = [str(ffprobe), "-v", "error", "-select_streams", "a:0",
           "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return proc.returncode == 0 and "audio" in (proc.stdout or "")


def extract_audio(video_path: Path | str, out_wav: Path | str,
                  start_s: float = 0.0, end_s: float | None = None) -> Path | None:
    """Pull the (trimmed) audio out as 16kHz mono WAV, or None if there's no audio.

    16k mono is what the speech/audio encoders want, and it keeps the base64
    payload small — a 15s clip lands around 500KB rather than several MB.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None or not has_audio_stream(video_path):
        return None
    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ffmpeg), "-v", "error", "-y", "-ss", f"{max(0.0, start_s):.3f}",
           "-i", str(video_path)]
    if end_s is not None and end_s > start_s:
        cmd += ["-t", f"{end_s - start_s:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(out_wav)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size < 1000:
        return None
    return out_wav


def mute_span_command(src: Path | str, dst: Path | str, start_s: float, end_s: float,
                      ffmpeg: Path | str | None = None) -> list[str]:
    """Silence the audio between start_s and end_s, leaving the video untouched.

    The video stream is copied rather than re-encoded: only the audio changes, so
    re-compressing the picture would cost quality and time for nothing. Useful for
    clipping a half-spoken word off the head or tail of a trimmed clip without
    losing the frames it sits under.
    """
    exe = str(ffmpeg or find_ffmpeg() or "ffmpeg")
    lo, hi = min(start_s, end_s), max(start_s, end_s)
    return [
        exe, "-v", "error", "-y", "-i", str(src),
        "-af", f"volume=enable='between(t,{lo:.3f},{hi:.3f})':volume=0",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "160k",
        str(dst),
    ]


def apply_mute_span(src: Path | str, dst: Path | str, start_s: float, end_s: float,
                    timeout: float = 600.0) -> tuple[bool, str]:
    """Run the mute. Returns (ok, message); never raises for ordinary failures."""
    if find_ffmpeg() is None:
        return False, "ffmpeg isn't installed."
    if not has_audio_stream(src):
        return False, "This clip has no audio track, so there's nothing to mute."
    try:
        proc = subprocess.run(mute_span_command(src, dst, start_s, end_s),
                              capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out."
    except OSError as exc:
        return False, f"Could not run ffmpeg: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        return False, detail[-1] if detail else "ffmpeg failed."
    out = Path(dst)
    if not out.exists() or out.stat().st_size == 0:
        return False, "ffmpeg produced no output."
    return True, ""


def mean_volume_db(path: Path | str, start_s: float = 0.0,
                   end_s: float | None = None) -> float | None:
    """Mean volume of a span in dBFS, or None if it can't be measured.

    Used to prove a mute actually took: -91dB is digital silence, so a span that
    was loud before and is silent after is measurable rather than assumed.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return None
    cmd = [str(ffmpeg), "-v", "info", "-ss", f"{start_s:.3f}"]
    if end_s is not None and end_s > start_s:
        cmd += ["-t", f"{end_s - start_s:.3f}"]
    cmd += ["-i", str(path), "-af", "volumedetect", "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        return None
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?) dB", proc.stderr or "")
    return float(match.group(1)) if match else None


def audio_peaks(path: Path | str, buckets: int = 240,
                start_s: float = 0.0, end_s: float | None = None) -> list[float]:
    """Normalised 0..1 loudness per time bucket, for drawing a waveform.

    Decodes to low-rate mono PCM and takes the peak per bucket. 8kHz is plenty for
    seeing *where* sound is — the point is placing a cut on a word boundary, not
    inspecting the signal — and it keeps a 15s clip under a quarter-million samples,
    cheap enough to compute in pure Python without pulling in numpy.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None or buckets <= 0 or not has_audio_stream(path):
        return []
    rate = 8000
    cmd = [str(ffmpeg), "-v", "error"]
    if start_s:
        cmd += ["-ss", f"{start_s:.3f}"]
    cmd += ["-i", str(path)]
    if end_s is not None and end_s > start_s:
        cmd += ["-t", f"{end_s - start_s:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", str(rate), "-f", "s16le", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=120)
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0 or not proc.stdout:
        return []
    import array
    samples = array.array("h")
    data = proc.stdout
    samples.frombytes(data[:len(data) - (len(data) % 2)])
    if not samples:
        return []
    if sys.byteorder == "big":
        samples.byteswap()
    total = len(samples)
    per = max(1, total // buckets)
    peaks: list[float] = []
    for i in range(0, total, per):
        window = samples[i:i + per]
        if not window:
            continue
        peaks.append(max(abs(min(window)), abs(max(window))) / 32768.0)
    if not peaks:
        return []
    # Normalise to the clip's own loudest point: an absolute scale would render a
    # quiet dialogue clip as a flat line.
    loudest = max(peaks) or 1.0
    return [min(1.0, p / loudest) for p in peaks[:buckets]]


def extract_single_frame(video: Path | str, out_path: Path | str, time_s: float,
                         timeout: float = 120.0) -> tuple[bool, str]:
    """Write the frame at time_s as a still. Returns (ok, message).

    Seeks before -i for speed, then takes exactly one frame. The output format
    follows the destination's extension, so saving as .png keeps it lossless while
    .jpg goes through the usual encoder.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False, "ffmpeg isn't installed."
    out = Path(out_path)
    cmd = [str(ffmpeg), "-v", "error", "-y", "-ss", f"{max(0.0, time_s):.3f}",
           "-i", str(video), "-frames:v", "1"]
    if out.suffix.lower() in (".jpg", ".jpeg"):
        cmd += ["-q:v", "2"]
    cmd.append(str(out))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out."
    except OSError as exc:
        return False, f"Could not run ffmpeg: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        return False, detail[-1] if detail else "ffmpeg failed."
    if not out.exists() or out.stat().st_size == 0:
        return False, "No frame at that position."
    return True, ""


# Container extension -> encoder. Anything not listed falls back to re-encoding
# as AAC, which every player handles.
_AUDIO_ENCODERS = {
    ".wav": ["-c:a", "pcm_s16le"],
    ".flac": ["-c:a", "flac"],
    ".mp3": ["-c:a", "libmp3lame", "-q:a", "2"],
    ".m4a": ["-c:a", "aac", "-b:a", "192k"],
    ".aac": ["-c:a", "aac", "-b:a", "192k"],
    ".ogg": ["-c:a", "libvorbis", "-q:a", "5"],
    ".opus": ["-c:a", "libopus", "-b:a", "128k"],
}


def export_audio(video_path: Path | str, out_path: Path | str,
                 start_s: float = 0.0, end_s: float | None = None,
                 timeout: float = 300.0) -> tuple[bool, str]:
    """Save a clip's audio at full quality. Returns (ok, message).

    Distinct from extract_audio, which downsamples to 16kHz mono because that's
    what the speech encoders want — fine for a model, wrong for a file you intend
    to listen to or edit. This keeps the source rate and channels, and picks an
    encoder from the destination's extension.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False, "ffmpeg isn't installed."
    if not has_audio_stream(video_path):
        return False, "This clip has no audio track."
    out = Path(out_path)
    cmd = [str(ffmpeg), "-v", "error", "-y", "-ss", f"{max(0.0, start_s):.3f}",
           "-i", str(video_path)]
    if end_s is not None and end_s > start_s:
        cmd += ["-t", f"{end_s - start_s:.3f}"]
    suffix = out.suffix.lower()
    if suffix not in _AUDIO_ENCODERS:
        # ffmpeg infers the container from the extension, so an unknown one fails
        # with "Invalid argument" and no clue what to do about it.
        return False, ("Unsupported audio format '" + (suffix or "(none)") + "'. "
                       "Use one of: " + ", ".join(sorted(_AUDIO_ENCODERS)) + ".")
    cmd += ["-vn", "-map", "0:a:0"]
    cmd += _AUDIO_ENCODERS[suffix]
    cmd.append(str(out))
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "ffmpeg timed out."
    except OSError as exc:
        return False, f"Could not run ffmpeg: {exc}"
    if proc.returncode != 0:
        detail = (proc.stderr or "").strip().splitlines()
        return False, detail[-1] if detail else "ffmpeg failed."
    if not out.exists() or out.stat().st_size == 0:
        return False, "No audio was written."
    return True, ""
