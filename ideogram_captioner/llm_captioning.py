from __future__ import annotations

import base64
import hashlib
import io
import json
import mimetypes
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from datetime import datetime, timezone
from urllib.parse import urlparse
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable

from PIL import Image

from .schema import normalize_bbox, normalize_caption


ProgressCallback = Callable[[str], None]


class AutoCaptionError(RuntimeError):
    """Raised for user-actionable captioning setup or model failures."""


class ModelJsonError(AutoCaptionError):
    """Raised when a model response cannot be parsed as the expected JSON object."""

    def __init__(
        self,
        message: str,
        raw_output: str = "",
        candidate: str = "",
        repair_output: str = "",
    ) -> None:
        super().__init__(message)
        self.raw_output = raw_output
        self.candidate = candidate
        self.repair_output = repair_output


@dataclass(frozen=True)
class ModelProfile:
    id: str
    label: str
    api_model: str
    kind: str = "hf"
    hf_repo: str = ""
    mmproj_repo: str = ""
    model_filename: str = ""
    mmproj_filename: str = ""
    local_model_path: str = ""
    local_mmproj_path: str = ""
    vram_gb: float = 0.0          # approx VRAM needed at default context (0 = unknown)
    note: str = ""                # optional one-line annotation shown in the picker


@dataclass(frozen=True)
class ModelRuntimeConfig:
    label: str
    api_model: str
    kind: str = "hf"
    hf_repo: str = ""
    mmproj_repo: str = ""
    model_filename: str = ""
    mmproj_filename: str = ""
    local_model_path: str = ""
    local_mmproj_path: str = ""


@dataclass(frozen=True)
class ModelAssets:
    model_path: Path | None = None
    mmproj_path: Path | None = None


DEFAULT_JSON_REFINE_INSTRUCTIONS = """
Improve the existing structured JSON caption while preserving the image's actual content.
Add useful detail where the current JSON is vague, but do not invent subjects, text, brands,
or identities that are not visible in the image.
""".strip()


def app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def default_models_dir() -> Path:
    return app_base_dir() / "models"


def default_llama_dir() -> Path:
    """Stable home for the managed llama.cpp binary. Deliberately independent of
    the models dir: the user can relocate large model files to another drive, but
    the small server binary stays put so discovery and teardown keep working."""
    return app_base_dir() / "llama"


def default_settings_path() -> Path:
    return app_base_dir() / "captioner_settings.json"


def default_profiles_path() -> Path:
    return app_base_dir() / "captioner_model_profiles.json"


def default_profiles_example_path() -> Path:
    return app_base_dir() / "captioner_model_profiles.example.json"


def default_prompts_path() -> Path:
    return app_base_dir() / "captioner_prompts"


def find_llama_server() -> Path | None:
    """Locate a llama-server binary the app can launch, in priority order:
    app-relative install spots, the dedicated managed llama dir (where downloaded
    builds land), then anything on the system PATH. Returns None if nothing fits."""
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    # a binary we downloaded/installed (recorded in installed.json) wins
    record = read_installed_llama()
    if record and record.binary:
        recorded = Path(record.binary)
        if recorded.exists():
            return recorded
    base = app_base_dir()
    llama = default_llama_dir()
    candidates: list[Path] = [
        base / executable,
        base / "tools" / executable,
        base / "llama.cpp" / executable,
        base / "llama.cpp" / "build" / "bin" / executable,
        base / "llama.cpp-cuda" / executable,
        # dedicated managed location (downloaded prebuilts unpack here)
        llama / executable,
        llama / "bin" / executable,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # finally, fall back to the system PATH (e.g. a user-installed llama-server)
    on_path = shutil.which(executable)
    if on_path:
        return Path(on_path)
    return None


@dataclass(frozen=True)
class GpuInfo:
    """Best-effort GPU description used to pick a llama.cpp backend/asset.

    `vendor`/`name`/`sm` are whatever detection could read off the machine — never
    hardcoded — and `backend` is the recommended default ('cuda'|'vulkan'|'cpu').
    """
    vendor: str = "none"          # "nvidia" | "none"
    name: str = ""                # e.g. "NVIDIA GeForce RTX 5090" (as reported)
    compute_cap: str = ""         # e.g. "12.0" (NVIDIA only)
    sm: str = ""                  # e.g. "120" (NVIDIA only)
    backend: str = "vulkan"       # recommended backend
    vram_total_gb: float | None = None   # total VRAM in GB (NVIDIA only)

    @property
    def summary(self) -> str:
        if self.vendor == "nvidia" and self.name:
            bits = []
            if self.sm:
                bits.append(f"sm{self.sm}")
            if self.vram_total_gb:
                bits.append(f"{self.vram_total_gb:.0f}GB")
            paren = f" ({', '.join(bits)})" if bits else ""
            return f"{self.name}{paren} \u2192 CUDA"
        if self.vendor == "nvidia":
            return "NVIDIA GPU \u2192 CUDA"
        return f"No NVIDIA GPU detected \u2192 {self.backend.upper()}"


def _sm_from_compute_cap(compute_cap: str) -> str:
    """'12.0' -> '120', '8.6' -> '86'. Empty string if it doesn't parse."""
    match = re.match(r"\s*(\d+)\.(\d+)\s*$", compute_cap)
    if not match:
        return ""
    return str(int(match.group(1)) * 10 + int(match.group(2)))


def detect_gpu() -> GpuInfo:
    """Probe the machine for a GPU. Never raises.

    NVIDIA is detected via nvidia-smi (giving us the exact compute capability we
    need to match a CUDA build). Anything else falls back to a Vulkan
    recommendation — broad GPU support — with CPU left as a manual override.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,compute_cap,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            first = result.stdout.strip().splitlines()[0]
            parts = [p.strip() for p in first.split(",")]
            name = parts[0] if parts else ""
            compute_cap = parts[1] if len(parts) > 1 else ""
            vram_gb = None
            if len(parts) > 2:
                try:
                    vram_gb = round(float(parts[2]) / 1024.0, 1)
                except ValueError:
                    vram_gb = None
            return GpuInfo(
                vendor="nvidia",
                name=name,
                compute_cap=compute_cap,
                sm=_sm_from_compute_cap(compute_cap),
                backend="cuda",
                vram_total_gb=vram_gb,
            )
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    # No NVIDIA GPU found (or no nvidia-smi): Vulkan covers most other hardware.
    return GpuInfo(vendor="none", backend="vulkan")


# ---- lightweight resource monitor (RAM always; VRAM/GPU% on NVIDIA) -----------

@dataclass
class ResourceSample:
    ram_percent: float | None = None
    ram_used_gb: float | None = None
    ram_total_gb: float | None = None
    vram_used_gb: float | None = None
    vram_total_gb: float | None = None
    gpu_percent: float | None = None


def _read_ram() -> tuple[float | None, float | None]:
    """(used_gb, total_gb) without a third-party dependency. Linux reads
    /proc/meminfo; Windows uses GlobalMemoryStatusEx. Anything else -> (None, None)."""
    try:
        if sys.platform.startswith("linux"):
            info = {}
            for line in Path("/proc/meminfo").read_text().splitlines():
                key, _, rest = line.partition(":")
                info[key.strip()] = rest.strip().split()[0] if rest.strip() else "0"
            total_kb = float(info["MemTotal"])
            avail_kb = float(info.get("MemAvailable", info.get("MemFree", "0")))
            gb = 1024.0 * 1024.0
            return (total_kb - avail_kb) / gb, total_kb / gb
        if os.name == "nt":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = _MemStatus()
            stat.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            gb = 1000.0 ** 3
            return (stat.ullTotalPhys - stat.ullAvailPhys) / gb, stat.ullTotalPhys / gb
    except Exception:
        return None, None
    return None, None


def _parse_gpu_usage(text: str) -> tuple[float | None, float | None, float | None]:
    """Parse one CSV row of `memory.used,memory.total,utilization.gpu` (in MiB, MiB,
    %) into (vram_used_gb, vram_total_gb, gpu_percent)."""
    try:
        row = text.strip().splitlines()[0]
        used_mb, total_mb, util = [p.strip() for p in row.split(",")]
        return float(used_mb) / 1024.0, float(total_mb) / 1024.0, float(util)
    except (ValueError, IndexError):
        return None, None, None


def _read_gpu_usage() -> tuple[float | None, float | None, float | None]:
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            return _parse_gpu_usage(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return None, None, None


def sample_resources() -> ResourceSample:
    used, total = _read_ram()
    percent = (used / total * 100.0) if (used is not None and total) else None
    vram_used, vram_total, gpu_pct = _read_gpu_usage()
    return ResourceSample(
        ram_percent=percent, ram_used_gb=used, ram_total_gb=total,
        vram_used_gb=vram_used, vram_total_gb=vram_total, gpu_percent=gpu_pct,
    )


def format_resources(sample: ResourceSample) -> str:
    """Compact one-line readout, omitting any part we couldn't measure."""
    parts = []
    if sample.ram_percent is not None:
        parts.append(f"RAM {sample.ram_percent:.0f}%")
    if sample.vram_used_gb is not None and sample.vram_total_gb:
        parts.append(f"VRAM {sample.vram_used_gb:.1f}/{sample.vram_total_gb:.0f} GB")
    if sample.gpu_percent is not None:
        parts.append(f"GPU {sample.gpu_percent:.0f}%")
    return "  \u00b7  ".join(parts)


# ---- managed llama.cpp binary: version model + asset resolution ---------------

# Default/floor build baked into the app. llama.cpp releases continuously (a new
# build every few hours), so a freshly-pinned number goes "stale" almost
# immediately — the update flag means "a newer build exists", not "you're behind".
PINNED_LLAMA_BUILD = 9828

# Where each backend's prebuilt binaries come from. Official llama.cpp ships
# Windows (CUDA/Vulkan/CPU), macOS (Metal) and Linux (Vulkan/CPU) — but NOT Linux
# CUDA, which comes from a community builder.
LLAMA_REPO_OFFICIAL = "ggml-org/llama.cpp"
LLAMA_REPO_LINUX_CUDA = "keypaa/llamaup"


def llama_repo_for(system: str, backend: str) -> str:
    if system.lower().startswith("linux") and backend == "cuda":
        return LLAMA_REPO_LINUX_CUDA
    return LLAMA_REPO_OFFICIAL


def current_platform() -> tuple[str, str]:
    """(system, arch) normalised to the tokens release assets use."""
    system = platform.system()  # 'Windows' | 'Linux' | 'Darwin'
    machine = platform.machine().lower()
    if machine in ("x86_64", "amd64", "x64"):
        arch = "x64"
    elif machine in ("arm64", "aarch64"):
        arch = "arm64"
    else:
        arch = machine
    return system, arch


def parse_build_number(text: str) -> int | None:
    """Pull the bNNNN build number out of a tag or asset name."""
    match = re.search(r"b(\d{3,})", text or "")
    return int(match.group(1)) if match else None


def is_update_available(installed_build: int | None, latest_build: int | None) -> bool:
    if installed_build is None or latest_build is None:
        return False
    return latest_build > installed_build


def _asset_name(asset) -> str:
    name = asset.get("name") if isinstance(asset, dict) else getattr(asset, "name", "")
    return (name or "").lower()


def _highest_cuda(candidates: list) -> object | None:
    """Pick the asset with the newest CUDA version token (newest works on the
    widest range of GPUs, including the latest architectures)."""
    best = None
    best_ver = (-1, -1)
    for asset in candidates:
        match = re.search(r"cuda-?(\d+)\.(\d+)", _asset_name(asset))
        ver = (int(match.group(1)), int(match.group(2))) if match else (0, 0)
        if ver > best_ver:
            best_ver = ver
            best = asset
    return best


def select_llama_assets(assets: list, *, system: str, arch: str, backend: str, sm: str = "") -> tuple:
    """From a release's asset list, choose the binary (and any runtime companion)
    for this platform/backend. Token-matched against the live names so it survives
    filename drift. Returns a tuple of matching assets, or () if nothing fits."""
    arch = arch.lower()
    sysl = system.lower()

    def hit(*needles, exclude=()):
        out = []
        for asset in assets:
            n = _asset_name(asset)
            if all(x in n for x in needles) and not any(x in n for x in exclude):
                out.append(asset)
        return out

    if sysl.startswith("win"):
        if backend == "cuda":
            main = _highest_cuda(hit("win", "cuda", arch, ".zip", exclude=("cudart",)))
            runtime = _highest_cuda(hit("cudart", "win", "cuda", arch, ".zip"))
            return tuple(a for a in (main, runtime) if a is not None)
        if backend == "vulkan":
            found = hit("win", "vulkan", arch, ".zip")
            return (found[0],) if found else ()
        found = hit("win", "cpu", arch, ".zip") or hit(
            "win", arch, ".zip", exclude=("cuda", "vulkan", "hip", "cudart")
        )
        return (found[0],) if found else ()

    if sysl.startswith("darwin") or sysl == "macos":
        found = hit("macos", arch, ".tar.gz")
        return (found[0],) if found else ()

    # Linux
    if backend == "cuda" and sm:
        found = hit("linux", "cuda", f"sm{sm}", arch, ".tar.gz")
        return (found[0],) if found else ()
    found = hit("ubuntu", arch, ".tar.gz")  # official Vulkan/CPU build
    return (found[0],) if found else ()


@dataclass
class InstalledLlama:
    """What's actually on disk in the managed llama dir (installed.json)."""
    source: str = ""        # repo the binary came from
    build: int = 0          # bNNNN as int
    backend: str = ""       # cuda | vulkan | cpu | metal
    sm: str = ""            # GPU sm it was matched for (cuda only)
    asset: str = ""         # primary asset filename
    sha256: str = ""        # verified digest of the primary asset
    binary: str = ""        # path to the llama-server executable
    published_at: str = ""  # release date of the build we installed (ISO)
    installed_at: str = ""  # when we installed it (ISO)


# A build older than this many days earns a gentle "update recommended". Age, not
# build count, is the meaningful unit: llama.cpp ships ~6 builds/day, so a build
# delta is noise, but "your binary is a month old" is a real signal.
LLAMA_UPDATE_RECOMMENDED_DAYS = 30

# llama.cpp errors loudly when a model needs a newer build than you have; this is
# the strongest update signal (it blocks work), so we watch for it on job failures.
_ARCH_ERROR_MARKERS = (
    "unknown model architecture",
    "unsupported model architecture",
    "unknown architecture",
    "unsupported architecture",
)


def is_model_arch_error(message: str) -> bool:
    text = (message or "").lower()
    return any(marker in text for marker in _ARCH_ERROR_MARKERS)


def _parse_iso(timestamp: str) -> datetime | None:
    if not timestamp:
        return None
    cleaned = timestamp.strip().replace("Z", "+00:00")
    for candidate in (cleaned, cleaned[:10]):
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            continue
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    return None


def build_age_days(record: InstalledLlama | None, now: datetime | None = None) -> int | None:
    if record is None:
        return None
    when = _parse_iso(record.published_at)
    if when is None:
        return None
    now = now or datetime.now(timezone.utc)
    return max(0, (now - when).days)


def update_state(installed: InstalledLlama | None, latest_build: int | None,
                 now: datetime | None = None) -> dict:
    """Classify the managed binary: 'none' (not installed), 'up_to_date',
    'available' (a newer build exists — informational), or 'recommended' (old
    enough to nudge). Age drives 'recommended' even if 'latest' can't be fetched."""
    if installed is None:
        return {"state": "none", "age_days": None, "installed_build": None, "latest_build": latest_build}
    age = build_age_days(installed, now=now)
    newer = is_update_available(installed.build, latest_build)
    if age is not None and age >= LLAMA_UPDATE_RECOMMENDED_DAYS:
        state = "recommended"
    elif newer:
        state = "available"
    else:
        state = "up_to_date"
    return {"state": state, "age_days": age, "installed_build": installed.build, "latest_build": latest_build}


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def installed_record_path() -> Path:
    return default_llama_dir() / "installed.json"


def read_installed_llama() -> InstalledLlama | None:
    path = installed_record_path()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    valid = {f.name for f in fields(InstalledLlama)}
    return InstalledLlama(**{k: v for k, v in data.items() if k in valid})


def write_installed_llama(record: InstalledLlama) -> Path:
    path = installed_record_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(record), indent=2), encoding="utf-8")
    return path


# ---- release fetch + download/verify/unpack/swap ------------------------------

GITHUB_API = "https://api.github.com"


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    sha256: str = ""   # from the asset's API digest, when present
    size: int = 0


@dataclass(frozen=True)
class ReleaseInfo:
    repo: str
    tag: str
    build: int | None
    published_at: str
    assets: tuple


def _github_get(url: str, timeout: float = 10.0):
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "ideogram-captioner"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_release(repo: str, tag: str | None = None, timeout: float = 10.0) -> ReleaseInfo | None:
    """Metadata-only fetch of a release (a pinned tag, or latest). Best-effort:
    any failure returns None so callers stay silent rather than erroring."""
    suffix = f"tags/{tag}" if tag else "latest"
    try:
        data = _github_get(f"{GITHUB_API}/repos/{repo}/releases/{suffix}", timeout=timeout)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    assets = []
    for asset in data.get("assets", []) or []:
        if not isinstance(asset, dict):
            continue
        digest = asset.get("digest") or ""
        sha = digest.split(":", 1)[1] if isinstance(digest, str) and digest.startswith("sha256:") else ""
        assets.append(ReleaseAsset(
            name=asset.get("name", "") or "",
            url=asset.get("browser_download_url", "") or "",
            sha256=sha,
            size=int(asset.get("size", 0) or 0),
        ))
    tag_name = data.get("tag_name", "") or ""
    return ReleaseInfo(
        repo=repo,
        tag=tag_name,
        build=parse_build_number(tag_name),
        published_at=data.get("published_at", "") or "",
        assets=tuple(assets),
    )


def verify_sha256(path: Path, expected_hex: str) -> bool:
    if not expected_hex:
        return False
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest().lower() == expected_hex.strip().lower()


def download_file(url: str, dest: Path, progress: ProgressCallback | None = None, timeout: float = 120.0) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": "ideogram-captioner"})
    with urllib.request.urlopen(request, timeout=timeout) as response, open(dest, "wb") as out:
        total = int(response.headers.get("Content-Length", 0) or 0)
        done = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if progress and total:
                progress(f"Downloading {dest.name}\u2026 {done * 100 // total}%")
    return dest


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def extract_archive(archive_path: Path, dest_dir: Path) -> None:
    """Extract a .zip or .tar.gz into dest_dir, refusing entries that would
    escape it (path-traversal guard) even though the source is a trusted release."""
    archive_path = Path(archive_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    name = archive_path.name.lower()
    if name.endswith(".zip"):
        with zipfile.ZipFile(archive_path) as zf:
            for member in zf.namelist():
                if not _is_within(dest_dir, dest_dir / member):
                    raise AutoCaptionError(f"Unsafe path in archive: {member}")
            zf.extractall(dest_dir)
    elif name.endswith(".tar.gz") or name.endswith(".tgz"):
        with tarfile.open(archive_path, "r:gz") as tf:
            for member in tf.getmembers():
                if not _is_within(dest_dir, dest_dir / member.name):
                    raise AutoCaptionError(f"Unsafe path in archive: {member.name}")
            try:
                tf.extractall(dest_dir, filter="data")   # py3.12+: strips unsafe metadata
            except TypeError:
                tf.extractall(dest_dir)                   # py3.10/3.11: manual guard above
    else:
        raise AutoCaptionError(f"Unsupported archive type: {archive_path.name}")


def find_server_in_dir(dirpath: Path) -> Path | None:
    executable = "llama-server.exe" if os.name == "nt" else "llama-server"
    for found in Path(dirpath).rglob(executable):
        return found
    return None


def _managed_llama_paths() -> tuple[Path, Path, Path, Path]:
    base = default_llama_dir()
    return base, base / "bin", base / ".staging", base / ".backup"


def install_llama_release(
    release: ReleaseInfo,
    assets: tuple,
    *,
    backend: str,
    sm: str = "",
    progress: ProgressCallback | None = None,
    downloader: Callable[..., Path] = download_file,
) -> InstalledLlama:
    """Download + verify + unpack the chosen assets, then swap them into place
    keeping the previous binary in .backup for rollback. The current binary is
    only touched after every download has verified, so a failure never clobbers
    a working install."""
    if not assets:
        raise AutoCaptionError("No matching llama.cpp asset to install.")
    base, bindir, staging, backup = _managed_llama_paths()
    shutil.rmtree(staging, ignore_errors=True)
    extracted = staging / "extract"
    extracted.mkdir(parents=True, exist_ok=True)

    for asset in assets:
        archive = staging / asset.name
        if progress:
            progress(f"Downloading {asset.name}\u2026")
        downloader(asset.url, archive, progress=progress)
        if asset.sha256:
            if progress:
                progress(f"Verifying {asset.name}\u2026")
            if not verify_sha256(archive, asset.sha256):
                shutil.rmtree(staging, ignore_errors=True)
                raise AutoCaptionError(f"Checksum mismatch for {asset.name}; install aborted.")
        extract_archive(archive, extracted)

    server = find_server_in_dir(extracted)
    if server is None:
        shutil.rmtree(staging, ignore_errors=True)
        raise AutoCaptionError("Downloaded archive did not contain llama-server.")

    # Swap: back up the current bin (single backup kept for rollback), move the new
    # tree into place. Done only after downloads verified, so it can't half-clobber.
    if bindir.exists():
        shutil.rmtree(backup, ignore_errors=True)
        shutil.move(str(bindir), str(backup))
    shutil.move(str(extracted), str(bindir))
    final_server = find_server_in_dir(bindir)
    if final_server is None:
        raise AutoCaptionError("llama-server vanished during install.")
    try:
        os.chmod(final_server, 0o755)
    except OSError:
        pass

    record = InstalledLlama(
        source=release.repo,
        build=release.build or 0,
        backend=backend,
        sm=sm or "",
        asset=assets[0].name,
        sha256=assets[0].sha256,
        binary=str(final_server),
        published_at=release.published_at,
        installed_at=_now_iso(),
    )
    write_installed_llama(record)
    shutil.rmtree(staging, ignore_errors=True)
    return record


def has_llama_backup() -> bool:
    _base, _bindir, _staging, backup = _managed_llama_paths()
    return backup.exists() and find_server_in_dir(backup) is not None


def rollback_llama() -> bool:
    """Restore the previous binary from .backup (used when a fresh install fails
    its first launch). Returns True if a backup was restored."""
    base, bindir, staging, backup = _managed_llama_paths()
    if not (backup.exists() and find_server_in_dir(backup) is not None):
        return False
    shutil.rmtree(bindir, ignore_errors=True)
    shutil.move(str(backup), str(bindir))
    server = find_server_in_dir(bindir)
    if server is not None:
        existing = read_installed_llama()
        if existing is not None:
            existing.binary = str(server)
            write_installed_llama(existing)
    return True


@dataclass(frozen=True)
class LlamaPlan:
    release: ReleaseInfo
    assets: tuple
    backend: str
    sm: str
    gpu: GpuInfo
    system: str
    arch: str
    repo: str

    @property
    def total_size(self) -> int:
        return sum(getattr(a, "size", 0) for a in self.assets)

    @property
    def description(self) -> str:
        size_mb = self.total_size / (1024 * 1024) if self.total_size else 0
        size = f"~{size_mb:.0f} MB " if size_mb else ""
        who = self.gpu.name if (self.gpu.vendor == "nvidia" and self.gpu.name) else "your system"
        build = f"b{self.release.build}" if self.release.build else (self.release.tag or "latest")
        return f"{size}{self.backend.upper()} build ({build}) for {who}"


def resolve_backend(settings, gpu: GpuInfo) -> str:
    hint = (getattr(settings, "llama_backend_hint", "auto") or "auto").lower()
    if hint in ("cuda", "vulkan", "cpu"):
        return hint
    return gpu.backend


def plan_llama_acquisition(settings, *, latest: bool = False, fetch=None):
    """Detect GPU, choose a backend + source repo, fetch a release (the pinned
    build, or the latest), and resolve matching assets. Returns None when offline
    or when no prebuilt fits this platform (caller falls back to manual/build)."""
    fetch = fetch or fetch_release
    gpu = detect_gpu()
    system, arch = current_platform()
    backend = resolve_backend(settings, gpu)
    repo = llama_repo_for(system, backend)
    release = None
    if not latest:
        release = fetch(repo, f"b{PINNED_LLAMA_BUILD}")
    if release is None:
        release = fetch(repo, None)   # latest — also the pinned-not-found fallback
    if release is None:
        return None
    sm = gpu.sm if backend == "cuda" else ""
    assets = select_llama_assets(release.assets, system=system, arch=arch, backend=backend, sm=sm)
    if not assets:
        return None
    return LlamaPlan(release=release, assets=assets, backend=backend, sm=sm,
                     gpu=gpu, system=system, arch=arch, repo=repo)


DEFAULT_PROFILE_DATA: dict[str, Any] = {
    "profiles": [
        {
            "id": "unsloth-qwen3vl-30b-q4",
            "label": "Download: Unsloth Qwen3-VL 30B-A3B Q4 (recommended, ~20GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "unsloth-qwen3vl-30b",
            "hf_repo": "unsloth/Qwen3-VL-30B-A3B-Instruct-GGUF",
            "model_filename": "Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
            "mmproj_filename": "mmproj-BF16.gguf",
            "vram_gb": 20,
        },
        {
            "id": "unsloth-qwen3vl-8b-q4",
            "label": "Download: Qwen3-VL 8B Q4 \u2014 Ideogram 4 text encoder (~8GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "unsloth-qwen3vl-8b",
            "hf_repo": "unsloth/Qwen3-VL-8B-Instruct-GGUF",
            "model_filename": "Qwen3-VL-8B-Instruct-UD-Q4_K_XL.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
            "vram_gb": 8,
            "note": "Ideogram 4 uses Qwen3-VL-8B as its text encoder. This is a 'native' model choice, however higher parameter models from Qwen or Gemma may result in better captioning accuracy.",
        },
        {
            "id": "llmfan46-gemma4-31b-qat-heretic-q4_0",
            "label": "Download: Gemma 4 31B IT QAT Uncensored Heretic Q4_0 (~22GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "gemma4-31b-heretic",
            "hf_repo": "llmfan46/gemma-4-31B-it-qat-q4_0-uncensored-heretic-GGUF",
            "model_filename": "gemma-4-31B-it-qat-Q4_0.gguf",
            "mmproj_filename": "gemma-4-31B-it-uncensored-heretic-BF16.gguf",
            "vram_gb": 22,
        },
        {
            "id": "llmfan46-gemma4-12b-qat-heretic-q4_0",
            "label": "Download: Gemma 4 12B IT QAT Uncensored Heretic Q4_0 (~11GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "gemma4-12b-heretic",
            "hf_repo": "llmfan46/gemma-4-12B-it-qat-q4_0-uncensored-heretic-GGUF",
            "model_filename": "gemma-4-12B-it-qat-q4_0-uncensored-heretic-Q4_0.gguf",
            "mmproj_filename": "gemma-4-12B-it-qat-q4_0-uncensored-heretic-mmproj-BF16.gguf",
            "vram_gb": 11,
        },
        {
            "id": "hauhaucs-qwen35-9b-aggressive-q6k",
            "label": "Download: Qwen3.5-9B Uncensored HauhauCS Aggressive Q6_K (~10GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "hauhaucs-qwen35-9b",
            "hf_repo": "HauhauCS/Qwen3.5-9B-Uncensored-HauhauCS-Aggressive",
            "model_filename": "Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q6_K.gguf",
            "mmproj_filename": "mmproj-Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-BF16.gguf",
            "vram_gb": 10,
        },
        {
            "id": "hauhaucs-gemma4-26b-balanced-q4km",
            "label": "Download: Gemma4-26B A4B Uncensored HauhauCS-Balanced Q4_K_M (~20GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "gemma4-26b-balanced",
            "hf_repo": "HauhauCS/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced",
            "model_filename": "Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf",
            "mmproj_filename": "mmproj-Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-f16.gguf",
            "vram_gb": 20,
        },
        {
            "id": "huihui-qwen3vl-30b-abliterated-i1-q4ks",
            "label": "Download: Huihui Qwen3-VL 30B abliterated i1 Q4_K_S (~20GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "huihui-qwen3vl-30b",
            "hf_repo": "mradermacher/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-i1-GGUF",
            "mmproj_repo": "mradermacher/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-GGUF",
            "model_filename": "Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated.i1-Q4_K_S.gguf",
            "mmproj_filename": "Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated.mmproj-f16.gguf",
            "vram_gb": 20,
        },
        {
            "id": "davidau-qwen36-27b-heretic-q6k",
            "label": "Download: DavidAU Qwen3.6 27B Heretic Q6_K (~26GB)",
            "tasks": ["caption", "bbox"],
            "kind": "hf",
            "api_model": "davidau-qwen36-27b-heretic",
            "hf_repo": "DavidAU/Qwen3.6-27B-Heretic-Uncensored-FINETUNE-NEO-CODE-Di-IMatrix-MAX-GGUF",
            "model_filename": "Qwen3.6-27B-NEO-CODE-HERE-2T-OT-Q6_K.gguf",
            "mmproj_filename": "mmproj-F16.gguf",
            "vram_gb": 26,
        },
        {
            "id": "server-qwen3vl",
            "label": "Existing server alias: qwen3vl",
            "tasks": ["caption", "bbox"],
            "kind": "server",
            "api_model": "qwen3vl",
        },
        {
            "id": "server-gemma-vl",
            "label": "Existing server alias: gemma-vl",
            "tasks": ["caption"],
            "kind": "server",
            "api_model": "gemma-vl",
        },
    ]
}

CUSTOM_HF_PROFILE = ModelProfile("custom-hf", "Custom Hugging Face GGUF", "", kind="custom_hf")
CUSTOM_LOCAL_PROFILE = ModelProfile("custom-local", "Custom local GGUF files", "local-model", kind="custom_local")


def _profile_from_dict(raw: dict[str, Any]) -> ModelProfile | None:
    profile_id = str(raw.get("id", "")).strip()
    label = str(raw.get("label", "")).strip()
    if not profile_id or not label:
        return None

    kind = str(raw.get("kind", "")).strip().lower()
    if not kind:
        kind = "hf" if raw.get("hf_repo") else "server"
    if kind not in {"hf", "server", "local"}:
        return None

    try:
        vram_gb = float(raw.get("vram_gb", 0) or 0)
    except (TypeError, ValueError):
        vram_gb = 0.0

    return ModelProfile(
        id=profile_id,
        label=label,
        api_model=str(raw.get("api_model", "")).strip(),
        kind=kind,
        hf_repo=str(raw.get("hf_repo", "")).strip(),
        mmproj_repo=str(raw.get("mmproj_repo", "")).strip(),
        model_filename=str(raw.get("model_filename", "")).strip(),
        mmproj_filename=str(raw.get("mmproj_filename", "")).strip(),
        local_model_path=str(raw.get("local_model_path", "")).strip(),
        local_mmproj_path=str(raw.get("local_mmproj_path", "")).strip(),
        vram_gb=vram_gb,
        note=str(raw.get("note", "")).strip(),
    )


def _profile_tasks(raw: dict[str, Any]) -> set[str]:
    tasks = raw.get("tasks", ["caption", "bbox"])
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, list):
        return {"caption", "bbox"}
    out = {str(task).strip().lower() for task in tasks}
    if "all" in out:
        return {"caption", "bbox"}
    return {task for task in out if task in {"caption", "bbox"}}


def _read_profile_data(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None
    if isinstance(loaded, dict) and isinstance(loaded.get("profiles"), list):
        return loaded
    return None


def profile_seed_data() -> dict[str, Any]:
    return _read_profile_data(default_profiles_example_path()) or DEFAULT_PROFILE_DATA


def load_model_profiles(path: Path | None = None) -> dict[str, tuple[ModelProfile, ...]]:
    if path is not None:
        data = _read_profile_data(path) or DEFAULT_PROFILE_DATA
    else:
        data = _read_profile_data(default_profiles_path()) or profile_seed_data()

    profiles_by_task: dict[str, list[ModelProfile]] = {"caption": [], "bbox": []}
    seen: dict[str, set[str]] = {"caption": set(), "bbox": set()}
    for raw in data.get("profiles", []):
        if not isinstance(raw, dict):
            continue
        profile = _profile_from_dict(raw)
        if profile is None:
            continue
        for task in _profile_tasks(raw):
            if profile.id in seen[task]:
                continue
            profiles_by_task[task].append(profile)
            seen[task].add(profile.id)

    for task in ("caption", "bbox"):
        if not profiles_by_task[task]:
            for raw in DEFAULT_PROFILE_DATA["profiles"]:
                if task in _profile_tasks(raw):
                    profile = _profile_from_dict(raw)
                    if profile is not None:
                        profiles_by_task[task].append(profile)
        profiles_by_task[task].extend([CUSTOM_HF_PROFILE, CUSTOM_LOCAL_PROFILE])

    return {task: tuple(profiles) for task, profiles in profiles_by_task.items()}


# Where downloaded model files land. The shared HF cache is the location other
# Hugging Face tools (ai-toolkit, transformers, etc.) read from and write to.
MODEL_TARGET_APP = "App models folder"
MODEL_TARGET_HF = "Shared Hugging Face cache"


@dataclass
class CaptioningSettings:
    base_url: str = "http://127.0.0.1:8000/v1"
    api_key: str = "dummy"
    hf_token: str = ""
    models_dir: str = ""
    model_download_target: str = MODEL_TARGET_APP
    extra_model_dirs: str = ""   # newline/`;`-separated extra folders to search

    caption_profile_id: str = "unsloth-qwen3vl-30b-q4"
    caption_model: str = "unsloth-qwen3vl-30b"
    caption_hf_repo: str = ""
    caption_model_filename: str = ""
    caption_mmproj_filename: str = ""
    caption_local_model_path: str = ""
    caption_local_mmproj_path: str = ""

    bbox_profile_id: str = "unsloth-qwen3vl-30b-q4"
    bbox_model: str = "unsloth-qwen3vl-30b"
    bbox_hf_repo: str = ""
    bbox_model_filename: str = ""
    bbox_mmproj_filename: str = ""
    bbox_local_model_path: str = ""
    bbox_local_mmproj_path: str = ""

    add_bboxes_after_json: bool = True
    overwrite_bboxes: bool = True
    filter_bbox_targets: bool = False
    use_caption_model_for_bboxes: bool = False
    creative_json: bool = True
    disable_thinking: bool = True
    vision_image_format: str = "auto"
    json_refine_instructions: str = DEFAULT_JSON_REFINE_INSTRUCTIONS

    max_tokens_caption: int = 2000
    max_tokens_json: int = 12000
    max_tokens_bboxes: int = 3000
    context_chars: int = 1200
    max_targets_per_call: int = 0

    server_start_mode: str = "local"
    auto_start_server: bool = True
    llama_server_path: str = ""
    llama_context: int = 16384
    llama_gpu_layers: int = -1
    llama_batch: int = 2048
    llama_ubatch: int = 512
    llama_parallel: int = 1
    llama_threads: int = 0
    llama_extra_args: str = "-fa on"
    llama_reasoning_budget: int = 2048
    caption_server_command: str = ""
    bbox_server_command: str = ""
    server_startup_timeout: float = 120.0
    stop_server_after_job: bool = False
    # Managed llama.cpp binary acquisition.
    llama_backend_hint: str = "auto"        # auto | cuda | vulkan | cpu
    llama_auto_update_check: bool = True     # background, once-a-day, metadata-only

    # Appearance (applied at startup). Empty font family = auto-detect an installed font.
    ui_font_family: str = ""
    mono_font_family: str = ""
    ui_font_size: int = 10
    color_window: str = "#111318"
    color_toolbar: str = "#0d0f14"
    color_panel: str = "#171a21"
    color_text: str = "#d9dee9"
    color_accent: str = "#2f6fed"
    color_selection: str = "#315fbd"
    color_field: str = "#222733"
    color_field_text: str = "#f2f5fb"
    color_editor_bg: str = "#10131a"
    color_list_bg: str = "#1f2430"
    color_canvas: str = "#05070a"

    def __post_init__(self) -> None:
        if not self.models_dir:
            self.models_dir = str(default_models_dir())
        if self.caption_profile_id == "custom":
            self.caption_profile_id = "custom-hf"
        if self.bbox_profile_id == "custom":
            self.bbox_profile_id = "custom-hf"
        # Legacy setting kept only so older settings files still load.
        self.use_caption_model_for_bboxes = False


def profile_labels(task: str) -> list[str]:
    return [profile.label for profile in profiles_for_task(task)]


def model_size_tier(vram_gb: float) -> str:
    """Coarse size class for a model's VRAM need (independent of the user's card)."""
    if vram_gb <= 0:
        return ""
    if vram_gb <= 8:
        return "Small"
    if vram_gb <= 16:
        return "Medium"
    if vram_gb <= 26:
        return "Large"
    return "XL"


def estimate_gguf_vram_gb(model_path, mmproj_path=None) -> float:
    """Rough VRAM estimate for a local GGUF, read from file size(s) only (no model
    load, no metadata parse). On-disk weights dominate VRAM use; we add the vision
    projector when present, then a small headroom for KV cache and compute buffers.
    This is a byte-size approximation — actual VRAM also depends on context length.
    Returns 0.0 if the size can't be read."""
    try:
        total = Path(model_path).stat().st_size
    except (OSError, ValueError, TypeError):
        return 0.0
    if mmproj_path:
        try:
            total += Path(mmproj_path).stat().st_size
        except (OSError, ValueError, TypeError):
            pass
    gb = total / (1024 ** 3)
    if gb <= 0:
        return 0.0
    return round(gb * 1.12 + 0.8, 1)   # ~12% buffers + ~0.8GB fixed headroom


def vram_fit(estimate_gb: float, total_gb: float | None, reserve_gb: float = 2.5) -> str:
    """Verdict for a model's estimated need against a card's total VRAM:
    'fits' (comfortable), 'tight' (will load but little headroom), 'too_big'
    (likely OOM without CPU offload), or 'unknown' (no estimate / no VRAM read).
    A reserve is held back for the OS/desktop/other apps."""
    if not estimate_gb or estimate_gb <= 0 or not total_gb or total_gb <= 0:
        return "unknown"
    usable = total_gb - reserve_gb
    if usable <= 0:
        return "too_big"
    if estimate_gb <= usable * 0.85:
        return "fits"
    if estimate_gb <= usable:
        return "tight"
    return "too_big"


def recommend_profile_for_vram(task: str, total_gb: float | None) -> "ModelProfile | None":
    """Pick a downloadable model suited to the card. Prefers the curated default
    (first profile) when it comfortably fits; otherwise the largest that fits; if
    none fit, the smallest (least likely to OOM). With VRAM unknown, returns the
    smallest as a safe default. Server/custom profiles are skipped."""
    candidates = [p for p in profiles_for_task(task) if p.kind == "hf" and p.vram_gb > 0]
    if not candidates:
        return None
    if not total_gb or total_gb <= 0:
        return min(candidates, key=lambda p: p.vram_gb)
    fitting = [p for p in candidates if vram_fit(p.vram_gb, total_gb) == "fits"]
    if fitting:
        if candidates[0] in fitting:   # the curated flagship fits — prefer it
            return candidates[0]
        return max(fitting, key=lambda p: p.vram_gb)
    return min(candidates, key=lambda p: p.vram_gb)


def profiles_for_task(task: str) -> tuple[ModelProfile, ...]:
    profiles = load_model_profiles()
    return profiles["bbox"] if task == "bbox" else profiles["caption"]


def profile_id_from_label(task: str, label: str) -> str:
    for profile in profiles_for_task(task):
        if profile.label == label:
            return profile.id
    return profiles_for_task(task)[0].id


def profile_label_from_id(task: str, profile_id: str) -> str:
    if profile_id == "custom":
        profile_id = "custom-hf"
    for profile in profiles_for_task(task):
        if profile.id == profile_id:
            return profile.label
    return profiles_for_task(task)[0].label


def _profile_by_id(task: str, profile_id: str) -> ModelProfile:
    if profile_id == "custom":
        profile_id = "custom-hf"
    for profile in profiles_for_task(task):
        if profile.id == profile_id:
            return profile
    return profiles_for_task(task)[0]


def _custom_runtime_config(settings: CaptioningSettings, task: str, profile: ModelProfile) -> ModelRuntimeConfig:
    if task == "bbox":
        api_model = settings.bbox_model.strip()
        if profile.kind == "custom_local":
            return ModelRuntimeConfig(
                label=profile.label,
                api_model=api_model or profile.api_model,
                kind="local",
                local_model_path=settings.bbox_local_model_path.strip(),
                local_mmproj_path=settings.bbox_local_mmproj_path.strip(),
            )
        return ModelRuntimeConfig(
            label=profile.label,
            api_model=api_model,
            kind="hf",
            hf_repo=settings.bbox_hf_repo.strip(),
            model_filename=settings.bbox_model_filename.strip(),
            mmproj_filename=settings.bbox_mmproj_filename.strip(),
        )

    api_model = settings.caption_model.strip()
    if profile.kind == "custom_local":
        return ModelRuntimeConfig(
            label=profile.label,
            api_model=api_model or profile.api_model,
            kind="local",
            local_model_path=settings.caption_local_model_path.strip(),
            local_mmproj_path=settings.caption_local_mmproj_path.strip(),
        )
    return ModelRuntimeConfig(
        label=profile.label,
        api_model=api_model,
        kind="hf",
        hf_repo=settings.caption_hf_repo.strip(),
        model_filename=settings.caption_model_filename.strip(),
        mmproj_filename=settings.caption_mmproj_filename.strip(),
    )


def runtime_config_for_task(settings: CaptioningSettings, task: str) -> ModelRuntimeConfig:
    profile_id = settings.bbox_profile_id if task == "bbox" else settings.caption_profile_id
    profile = _profile_by_id(task, profile_id)
    if profile.kind in {"custom_hf", "custom_local"}:
        return _custom_runtime_config(settings, task, profile)

    api_model = settings.bbox_model.strip() if task == "bbox" else settings.caption_model.strip()
    return ModelRuntimeConfig(
        label=profile.label,
        api_model=api_model or profile.api_model,
        kind=profile.kind,
        hf_repo=profile.hf_repo,
        mmproj_repo=profile.mmproj_repo,
        model_filename=profile.model_filename,
        mmproj_filename=profile.mmproj_filename,
        local_model_path=profile.local_model_path,
        local_mmproj_path=profile.local_mmproj_path,
    )


def has_model_config(settings: CaptioningSettings, task: str = "caption") -> bool:
    """True if a model is at least specified for `task` (a local GGUF path or an HF
    repo) — used to decide whether the local server can be started, or whether the
    user should be sent to the Models page first."""
    try:
        config = runtime_config_for_task(settings, task)
    except Exception:
        return False
    return bool((config.local_model_path or "").strip() or (config.hf_repo or "").strip())


def _looks_foreign_home(path_str: str) -> bool:
    """True if an absolute path lives under some *other* user's home directory
    (e.g. a build-machine '/home/claude/...' path on a machine whose home is
    '/home/ace'). Used to discard paths that leaked into a shipped or stale
    settings file so they don't crash the app with permission errors."""
    if not path_str or not path_str.strip():
        return False
    try:
        p = Path(path_str.strip()).expanduser()
        home = Path.home()
    except Exception:
        return False
    if not p.is_absolute():
        return False
    home_parent = home.parent                      # /home  ·  /Users  ·  C:\Users
    if home_parent.name.lower() not in ("home", "users"):
        return False                               # unusual home (e.g. /root); don't guess
    try:
        rel = p.relative_to(home_parent)
    except ValueError:
        return False                               # not under the home container at all
    first = rel.parts[0] if rel.parts else ""
    return bool(first) and first.lower() != home.name.lower()


def _sanitize_foreign_paths(s: "CaptioningSettings") -> "CaptioningSettings":
    """Drop build-machine / other-user paths that leaked into settings, replacing
    the models dir with this machine's default and clearing foreign file paths."""
    if _looks_foreign_home(s.models_dir):
        s.models_dir = str(default_models_dir())
    for attr in ("llama_server_path", "caption_local_model_path", "caption_local_mmproj_path",
                 "bbox_local_model_path", "bbox_local_mmproj_path"):
        if _looks_foreign_home(getattr(s, attr, "")):
            setattr(s, attr, "")
    if s.extra_model_dirs:
        kept = [ln for ln in re.split(r"[\r\n;]+", s.extra_model_dirs)
                if ln.strip() and not _looks_foreign_home(ln)]
        s.extra_model_dirs = "\n".join(kept)
    return s


def load_settings(path: Path | None = None) -> CaptioningSettings:
    path = path or default_settings_path()
    defaults = CaptioningSettings()
    if not path.exists():
        return defaults

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults

    allowed = {field.name for field in fields(CaptioningSettings)}
    values = asdict(defaults)
    if isinstance(raw, dict):
        for key, value in raw.items():
            if key in allowed:
                values[key] = value
    return _sanitize_foreign_paths(CaptioningSettings(**values))


def save_settings(settings: CaptioningSettings, path: Path | None = None) -> Path:
    path = path or default_settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def safe_repo_dir(repo_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "__", repo_id.strip())
    return cleaned.strip("._") or "custom_model"


def hf_hub_cache_dir() -> Path:
    """Resolved Hugging Face hub cache dir, honoring HF_HUB_CACHE / HF_HOME, else
    the standard ~/.cache/huggingface/hub. This is the shared cache other HF tools
    (ai-toolkit, transformers, etc.) read from and write to."""
    env = os.environ.get("HF_HUB_CACHE")
    if env:
        return Path(env).expanduser()
    home = os.environ.get("HF_HOME")
    if home:
        return Path(home).expanduser() / "hub"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE as _hub_cache
        return Path(_hub_cache)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def lmstudio_models_dir() -> Path:
    """Default LM Studio models folder. LM Studio stores GGUFs under
    ~/.lmstudio/models/<author>/<model>/<file>.gguf."""
    return Path.home() / ".lmstudio" / "models"


def known_server_model_dirs() -> list[Path]:
    """Default model folders for the local servers this app supports, across
    Linux/macOS/Windows. Only existing dirs are returned (de-duplicated).

    Notes on what actually yields loadable GGUFs:
    - LM Studio: ~/.lmstudio/models holds .gguf files (findable on all OSes; on
      Windows Path.home() resolves to C:\\Users\\<user>).
    - llama.cpp: the legacy -hf cache (~/.cache/llama.cpp or $LLAMA_CACHE /
      $XDG_CACHE_HOME) holds flat <repo>_<file>.gguf files; recent builds migrate
      these into the shared HF cache, which is searched separately.
    - vLLM: downloads into the shared HF cache (searched separately), no own dir.
    - Ollama: stores sha256 blobs + manifests (no .gguf extension), so a *.gguf
      scan won't surface loadable models there; the folder is still listed for
      completeness and in case a user dropped raw GGUFs in.
    """
    home = Path.home()
    cands: list[Path] = [home / ".lmstudio" / "models"]            # LM Studio

    env_llama = os.environ.get("LLAMA_CACHE")                       # llama.cpp -hf
    if env_llama:
        cands.append(Path(env_llama).expanduser())
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        cands.append(Path(xdg).expanduser() / "llama.cpp")
    cands.append(home / ".cache" / "llama.cpp")

    env_ollama = os.environ.get("OLLAMA_MODELS")                    # Ollama
    if env_ollama:
        cands.append(Path(env_ollama).expanduser())
    cands.append(home / ".ollama" / "models")
    cands.append(Path("/usr/share/ollama/.ollama/models"))         # Linux service

    out: list[Path] = []
    seen: set[str] = set()
    for p in cands:
        try:
            if not p.exists():
                continue
            key = str(p.resolve())
        except (OSError, ValueError):
            continue
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def model_search_roots(settings: "CaptioningSettings") -> list[Path]:
    """All folders to search for local GGUF models, de-duplicated. Order: the app
    models_dir, the user's Extra folders, the shared Hugging Face cache (also used
    by vLLM and recent llama.cpp), then the other built-in servers' default
    folders. Non-existent roots are kept here and skipped by the callers."""
    roots: list[Path] = [Path(settings.models_dir).expanduser()]
    roots.extend(_parse_dir_lines(getattr(settings, "extra_model_dirs", "")))
    roots.append(hf_hub_cache_dir())
    roots.extend(known_server_model_dirs())
    out: list[Path] = []
    seen: set[str] = set()
    for r in roots:
        try:
            key = str(r.resolve())
        except (OSError, ValueError):
            key = str(r)
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def _parse_dir_lines(value: str) -> list[Path]:
    out: list[Path] = []
    for line in re.split(r"[\r\n;]+", value or ""):
        line = line.strip()
        if line:
            out.append(Path(line).expanduser())
    return out


def _find_in_dir(root: Path, filename: str) -> Path | None:
    """First file named `filename` under root (checked directly, then recursively)."""
    try:
        if not root.exists():
            return None
        direct = root / filename
        if direct.is_file():
            return direct
        for hit in root.rglob(filename):
            if hit.is_file():
                return hit
    except (OSError, ValueError):
        return None
    return None


def locate_existing_model_file(settings: "CaptioningSettings", repo: str, filename: str) -> Path | None:
    """Find an already-downloaded copy of (repo, filename) so we can skip the
    download. Checks, in order: the app's per-repo folder, the shared Hugging Face
    cache (try_to_load_from_cache), then any Extra model folders (recursive match
    by filename — covers LM Studio's author/model/file layout)."""
    if not filename:
        return None
    if repo:
        flat = Path(settings.models_dir).expanduser() / safe_repo_dir(repo) / filename
        if flat.is_file():
            return flat
        try:
            from huggingface_hub import try_to_load_from_cache
            cached = try_to_load_from_cache(repo_id=repo, filename=filename)
            if isinstance(cached, str) and Path(cached).is_file():
                return Path(cached)
        except Exception:
            pass
    extra = _parse_dir_lines(getattr(settings, "extra_model_dirs", ""))
    for root in extra + known_server_model_dirs():
        hit = _find_in_dir(root, filename)
        if hit is not None:
            return hit
    return None


def discover_local_gguf_models(settings: "CaptioningSettings") -> tuple[list[Path], list[Path]]:
    """Scan the model folders for .gguf files: the app models_dir, the user's Extra
    folders, the shared HF cache (also vLLM + recent llama.cpp), and the other
    built-in servers' default folders (LM Studio, llama.cpp legacy cache, Ollama),
    across Linux/macOS/Windows. Returns (models, mmprojs): files whose name
    contains 'mmproj' are treated as vision projectors and kept separate so the
    model list stays clean and we can auto-pair a projector on selection."""
    roots = model_search_roots(settings)
    seen: set[str] = set()
    models: list[Path] = []
    mmprojs: list[Path] = []
    for root in roots:
        try:
            if not root.exists():
                continue
            for path in root.rglob("*.gguf"):
                if not path.is_file():
                    continue
                key = str(path.resolve())
                if key in seen:
                    continue
                seen.add(key)
                (mmprojs if "mmproj" in path.name.lower() else models).append(path)
        except (OSError, ValueError):
            continue
    models.sort(key=lambda p: p.name.lower())
    mmprojs.sort(key=lambda p: p.name.lower())
    return models, mmprojs


def guess_mmproj_for(model_path: Path, mmprojs: list[Path]) -> Path | None:
    """Best-guess vision projector for a chosen model file: prefer one sitting in
    the same folder (LM Studio and HF snapshots keep the pair together)."""
    same_dir = [m for m in mmprojs if m.parent == model_path.parent]
    if same_dir:
        return same_dir[0]
    return None


def server_host_port(base_url: str) -> tuple[str, int]:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port
    if port is None:
        port = 443 if parsed.scheme == "https" else 8000
    return host, port


def _split_filenames(value: str) -> list[str]:
    return [part.strip() for part in re.split(r"[;,]", value) if part.strip()]


def ensure_model_assets(
    settings: CaptioningSettings,
    task: str,
    progress: ProgressCallback | None = None,
) -> ModelAssets:
    config = runtime_config_for_task(settings, task)
    if config.local_model_path:
        model_path = Path(config.local_model_path).expanduser()
        if not model_path.exists():
            raise AutoCaptionError(f"Local model file does not exist: {model_path}")
        mmproj_path: Path | None = None
        if config.local_mmproj_path:
            mmproj_path = Path(config.local_mmproj_path).expanduser()
            if not mmproj_path.exists():
                raise AutoCaptionError(f"Local mmproj file does not exist: {mmproj_path}")
        if progress:
            progress(f"Using local model file: {model_path.name}")
        return ModelAssets(model_path=model_path, mmproj_path=mmproj_path)

    filenames = _split_filenames(config.model_filename)
    mmproj_filenames = _split_filenames(config.mmproj_filename)

    if not config.hf_repo or not filenames:
        return ModelAssets()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise AutoCaptionError("Install huggingface_hub to download Hugging Face model files.") from exc

    models_root = Path(settings.models_dir).expanduser().resolve()
    model_repo = config.hf_repo
    mmproj_repo = config.mmproj_repo or config.hf_repo
    model_dir = models_root / safe_repo_dir(model_repo)
    mmproj_dir = models_root / safe_repo_dir(mmproj_repo)
    use_hf_cache = (getattr(settings, "model_download_target", MODEL_TARGET_APP) == MODEL_TARGET_HF)
    if not use_hf_cache:
        model_dir.mkdir(parents=True, exist_ok=True)
        mmproj_dir.mkdir(parents=True, exist_ok=True)
    token = settings.hf_token.strip() or None

    def download_file(repo_id: str, filename: str, local_dir: Path) -> Path:
        # Reuse an existing copy anywhere we know to look (app folder, the shared
        # HF cache, or an Extra model folder like LM Studio) before downloading.
        existing = locate_existing_model_file(settings, repo_id, filename)
        if existing is not None:
            if progress:
                progress(f"Using existing model file: {existing.name}")
            return existing
        if progress:
            dest = "Hugging Face cache" if use_hf_cache else "models folder"
            progress(f"Downloading {filename} from {repo_id} to the {dest}...")
        try:
            if use_hf_cache:
                # No local_dir → lands in the shared cache, returns the resolved path.
                path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
            else:
                path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=str(local_dir),
                    token=token,
                )
        except Exception as exc:  # pragma: no cover - depends on network/HF
            raise AutoCaptionError(f"Could not download {filename} from {repo_id}: {exc}") from exc
        return Path(path)

    downloaded_models = [download_file(model_repo, filename, model_dir) for filename in filenames]
    downloaded_mmproj = [download_file(mmproj_repo, filename, mmproj_dir) for filename in mmproj_filenames]

    model_path = downloaded_models[0] if downloaded_models else None
    mmproj_path = downloaded_mmproj[0] if downloaded_mmproj else None
    return ModelAssets(model_path=model_path, mmproj_path=mmproj_path)


def missing_model_files(settings: CaptioningSettings, task: str = "caption") -> list[str]:
    """Filenames the launcher would download from Hugging Face because they aren't
    present locally yet — used for the pre-download confirmation. Empty when
    nothing needs downloading (user-provided local files, or already cached)."""
    try:
        config = runtime_config_for_task(settings, task)
    except Exception:
        return []
    if config.local_model_path:
        return []  # user pointed at local files; nothing for us to download
    filenames = _split_filenames(config.model_filename)
    mmproj_filenames = _split_filenames(config.mmproj_filename)
    if not config.hf_repo or not filenames:
        return []
    mmproj_repo = config.mmproj_repo or config.hf_repo
    missing: list[str] = []
    for filename in filenames:
        if locate_existing_model_file(settings, config.hf_repo, filename) is None:
            missing.append(filename)
    for filename in mmproj_filenames:
        if locate_existing_model_file(settings, mmproj_repo, filename) is None:
            missing.append(filename)
    return missing


def format_server_command(
    template: str,
    settings: CaptioningSettings,
    task: str,
    assets: ModelAssets,
) -> str:
    config = runtime_config_for_task(settings, task)
    values = {
        "base_url": settings.base_url,
        "api_model": config.api_model,
        "models_dir": str(Path(settings.models_dir).expanduser().resolve()),
        "model_path": str(assets.model_path or ""),
        "mmproj_path": str(assets.mmproj_path or ""),
    }
    try:
        return template.format(**values)
    except KeyError as exc:
        raise AutoCaptionError(f"Unknown server command placeholder: {exc}") from exc


def _split_extra_args(value: str) -> list[str]:
    if not value.strip():
        return []
    if os.name == "nt":
        # Keep this field simple: users can put switches and values separated by spaces.
        return [part for part in value.split() if part]

    import shlex

    return shlex.split(value)


def resolve_llama_server_path(settings: CaptioningSettings) -> Path | None:
    """The llama-server we'll launch: an explicit user path wins, else auto-detect
    (which prefers the managed binary)."""
    raw = settings.llama_server_path.strip()
    if raw:
        return Path(raw).expanduser()
    return find_llama_server()


def find_nccl_lib_dir() -> Path | None:
    """Locate a directory containing libnccl.so.2 (Linux only).

    From build b8738 onward, llama.cpp's CUDA prebuilt binaries dynamically link
    NCCL but the release archive doesn't bundle libnccl.so.2, so on a machine with
    no system NCCL the server fails to even load. PyTorch (and most CUDA Python
    envs) ship NCCL via the nvidia-nccl-cu12 wheel or in torch/lib, so we find that
    copy and put its folder on the loader path — no system install needed."""
    if os.name == "nt" or sys.platform == "darwin":
        return None
    soname = "libnccl.so.2"
    # 1) the nvidia-nccl wheel, located without importing it (cheap, no CUDA init)
    try:
        import importlib.util
        spec = importlib.util.find_spec("nvidia.nccl")
        for loc in (spec.submodule_search_locations or []) if spec else []:
            lib = Path(loc) / "lib"
            if (lib / soname).exists():
                return lib
    except Exception:
        pass
    # 2) scan site-packages for the wheel's or torch's bundled copy
    try:
        import site
        import sysconfig
        roots: set[Path] = set()
        try:
            roots.update(Path(p) for p in site.getsitepackages())
        except Exception:
            pass
        user = getattr(site, "getusersitepackages", lambda: None)()
        if user:
            roots.add(Path(user))
        for key in ("purelib", "platlib"):
            try:
                roots.add(Path(sysconfig.get_paths()[key]))
            except Exception:
                pass
        for root in roots:
            for rel in ("nvidia/nccl/lib", "torch/lib"):
                lib = root / rel
                if (lib / soname).exists():
                    return lib
    except Exception:
        pass
    return None


def server_launch_env(binary_path: Path | None) -> dict:
    """A child-process environment that lets a launched server find shared libs
    sitting beside its binary (and in sibling lib/ dirs).

    llama.cpp prebuilts bundle their ggml/CUDA libraries next to the executable,
    but the system linker won't always look there (only if the binary was built
    with an $ORIGIN rpath). Prepending those dirs to the loader path makes the
    launch robust regardless of the user's working directory or linker config —
    and keeps any bundled runtime entirely inside the managed folder, so nothing
    in the system, conda, or venv environment has to be touched.

    One exception: recent CUDA builds need libnccl.so.2, which the release does
    not ship. We locate the copy already present in the Python env (PyTorch's) and
    add its folder too, so single-GPU users aren't blocked by a missing NCCL.
    """
    env = os.environ.copy()
    if binary_path is None:
        return env
    try:
        bin_dir = Path(binary_path).expanduser().resolve().parent
    except (OSError, RuntimeError):
        return env
    candidates = [bin_dir]
    for base in (bin_dir, bin_dir.parent):
        for sub in ("lib", "lib64"):
            sibling = base / sub
            if sibling.is_dir():
                candidates.append(sibling)
    nccl_dir = find_nccl_lib_dir()
    if nccl_dir is not None:
        candidates.append(nccl_dir)
    seen: set[str] = set()
    dirs: list[str] = []
    for cand in candidates:
        text = str(cand)
        if text not in seen:
            seen.add(text)
            dirs.append(text)
    if os.name == "nt":
        var = "PATH"
    elif sys.platform == "darwin":
        var = "DYLD_LIBRARY_PATH"
    else:
        var = "LD_LIBRARY_PATH"
    existing = env.get(var, "")
    env[var] = os.pathsep.join(dirs + ([existing] if existing else []))
    return env


_ROUTER_SUPPORT: dict[str, bool] = {}


def llama_server_supports_router(binary_path: Path | None) -> bool:
    """Whether this llama-server build can start model-less (router mode), detected
    from --help and cached per binary. Router flag names have shifted across
    versions, so we look for any of the known ones. Never raises."""
    if binary_path is None:
        return False
    key = str(binary_path)
    if key in _ROUTER_SUPPORT:
        return _ROUTER_SUPPORT[key]
    supported = False
    try:
        result = subprocess.run(
            [str(binary_path), "--help"], capture_output=True, text=True, timeout=10
        )
        text = (result.stdout or "") + (result.stderr or "")
        supported = any(flag in text for flag in ("--models-dir", "--models-preset", "--model-dir"))
        if not supported and re.search(r"--models(\s|,|\b)", text):
            supported = True
    except (OSError, ValueError, subprocess.SubprocessError):
        supported = False
    _ROUTER_SUPPORT[key] = supported
    return supported


def build_llama_server_command(settings: CaptioningSettings, task: str, assets: ModelAssets,
                               *, model_less: bool = False) -> str:
    server_path = resolve_llama_server_path(settings)
    if server_path is None or not server_path.exists():
        exe = "llama-server.exe" if os.name == "nt" else "llama-server"
        raise AutoCaptionError(
            f"No {exe} found. Use \u201cGet llama.cpp\u201d in Preferences to install "
            "one, or set a llama-server path."
        )

    host, port = server_host_port(settings.base_url)

    if model_less:
        # Start the server with no model resident (router mode) so it can be
        # health-checked before committing to a download/load. Build-dependent.
        if not llama_server_supports_router(server_path):
            raise AutoCaptionError(
                "This llama-server build can't start without a model (no router "
                "mode). Update llama.cpp, or load a model to start the server."
            )
        models_dir = Path(settings.models_dir).expanduser()
        try:
            models_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        args = [str(server_path), "--host", host, "--port", str(port),
                "--models-dir", str(models_dir)]
        args.extend(_split_extra_args(settings.llama_extra_args))
        if os.name == "nt":
            return subprocess.list2cmdline(args)
        import shlex
        return shlex.join(args)

    if assets.model_path is None:
        raise AutoCaptionError("Local llama.cpp mode needs a downloadable or local GGUF model profile.")

    config = runtime_config_for_task(settings, task)
    args = [
        str(server_path),
        "-m",
        str(assets.model_path),
        "--host",
        host,
        "--port",
        str(port),
        "--alias",
        config.api_model or "captioner-model",
        "-c",
        str(max(512, int(settings.llama_context))),
        "-b",
        str(max(1, int(settings.llama_batch))),
        "-ub",
        str(max(1, int(settings.llama_ubatch))),
        "-np",
        str(max(1, int(settings.llama_parallel))),
    ]
    # Negative = auto: omit -ngl so llama.cpp's fitter places as many layers as fit
    # in free VRAM (and spills the rest to CPU) instead of OOM-aborting. A concrete
    # value (incl. 0 for CPU-only) is passed through and disables the fitter.
    if int(settings.llama_gpu_layers) >= 0:
        args.extend(["-ngl", str(int(settings.llama_gpu_layers))])
    if assets.mmproj_path is not None:
        args.extend(["--mmproj", str(assets.mmproj_path)])
    if settings.llama_threads > 0:
        args.extend(["-t", str(settings.llama_threads)])
    if settings.disable_thinking:
        args.extend(["--reasoning", "off"])
    elif settings.llama_reasoning_budget >= 0:
        args.extend(["--reasoning-budget", str(max(0, int(settings.llama_reasoning_budget)))])
    args.extend(_split_extra_args(settings.llama_extra_args))
    if os.name == "nt":
        return subprocess.list2cmdline(args)

    import shlex

    return shlex.join(args)


def api_models_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1"):
        return base + "/models"
    return base + "/v1/models"


def server_model_ids(base_url: str, api_key: str = "", timeout: float = 3.0) -> set[str]:
    request = urllib.request.Request(api_models_url(base_url))
    if api_key:
        request.add_header("Authorization", f"Bearer {api_key}")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        if not 200 <= getattr(response, "status", 200) < 300:
            raise AutoCaptionError(f"Server /models returned HTTP {getattr(response, 'status', 'unknown')}.")
        payload = json.loads(response.read().decode("utf-8"))

    if not isinstance(payload, dict):
        return set()
    models = payload.get("data", [])
    if not isinstance(models, list):
        return set()

    ids: set[str] = set()
    for model in models:
        if isinstance(model, dict):
            model_id = model.get("id")
            if isinstance(model_id, str) and model_id.strip():
                ids.add(model_id.strip())
    return ids


def is_server_ready(base_url: str, api_key: str = "", timeout: float = 3.0) -> bool:
    try:
        server_model_ids(base_url, api_key, timeout)
        return True
    except Exception:
        return False


def start_server_process(
    command: str,
    base_url: str,
    api_key: str,
    log_dir: Path,
    name: str,
    startup_timeout: float,
    progress: ProgressCallback | None = None,
    env: dict | None = None,
) -> subprocess.Popen:
    if not command.strip():
        raise AutoCaptionError("Server auto-start is enabled, but no server command is configured.")

    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    log_file = log_path.open("a", encoding="utf-8", errors="replace")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0

    if progress:
        progress(f"Starting {name} server...")
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        creationflags=creationflags,
        env=env,
        # shell=True means our handle is the shell; llama-server is its child. Put
        # the shell in its own session/group so stop_server_process can signal the
        # whole group and actually kill llama-server (and free its VRAM) instead of
        # orphaning it. Windows uses CREATE_NEW_PROCESS_GROUP + taskkill /T instead.
        start_new_session=(os.name != "nt"),
    )
    process._captioner_log_file = log_file  # type: ignore[attr-defined]

    deadline = time.time() + startup_timeout
    while time.time() < deadline:
        if process.poll() is not None:
            close_process_log(process)
            hint = _server_startup_hint(log_path)
            raise AutoCaptionError(f"{name} server exited during startup.{hint} See {log_path}.")
        if is_server_ready(base_url, api_key=api_key, timeout=3.0):
            if progress:
                progress(f"{name} server is ready.")
            return process
        time.sleep(1.0)

    stop_server_process(process)
    raise AutoCaptionError(f"{name} server did not become ready within {startup_timeout:.0f} seconds.")


def _server_startup_hint(log_path: Path) -> str:
    """Turn a known fatal startup log line into an actionable hint. Today that's
    mainly the missing-shared-library case (libnccl.so.2 on recent CUDA builds)."""
    try:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
    except OSError:
        return ""
    m = re.search(r"error while loading shared libraries:\s*([^\s:]+)", tail)
    if not m:
        return ""
    lib = m.group(1)
    if "nccl" in lib.lower():
        return (f" The CUDA build can't find {lib}. Recent llama.cpp CUDA releases "
                "need NVIDIA NCCL, which the download doesn't bundle. Install it in "
                "this environment with:  pip install nvidia-nccl-cu12  (or "
                "conda install -c conda-forge nccl), then start the server again.")
    return (f" The server can't find the shared library {lib}. Install it or add its "
            "folder to your library path, then try again.")


def close_process_log(process: subprocess.Popen) -> None:
    try:
        log_file = getattr(process, "_captioner_log_file", None)
        if log_file is not None:
            log_file.close()
    except Exception:
        pass


def _posix_signal_group(process: subprocess.Popen, sig: int) -> bool:
    """Signal the process group led by the launched shell so its llama-server child
    is hit too. Returns False if the group couldn't be resolved/signalled."""
    try:
        pgid = os.getpgid(process.pid)
    except (ProcessLookupError, OSError):
        return False
    try:
        os.killpg(pgid, sig)
        return True
    except ProcessLookupError:
        return True  # already gone
    except OSError:
        return False


def stop_server_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    try:
        if process.poll() is None:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
            else:
                # Terminate the whole group (shell + llama-server). Fall back to the
                # single process if the group can't be resolved.
                if not _posix_signal_group(process, signal.SIGTERM):
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    if not _posix_signal_group(process, signal.SIGKILL):
                        process.kill()
        try:
            process.wait(timeout=10)
        except Exception:
            pass
    finally:
        close_process_log(process)


def ensure_server_running(
    settings: CaptioningSettings,
    task: str,
    progress: ProgressCallback | None = None,
    *,
    model_less: bool = False,
) -> subprocess.Popen | None:
    """Make sure a server is available for `task` when configured to run one locally.

    Only acts when start mode is 'local' and auto-start is on. Downloads the
    model/mmproj if needed, launches llama-server, and waits for it to answer.
    With model_less=True it launches in router mode with no model resident (for a
    quick 'is the server up?' check). Returns the launched process, or None when
    nothing was started — existing/custom mode, auto-start off, or already up.
    """
    if settings.server_start_mode != "local" or not settings.auto_start_server:
        return None
    # already answering (a server we launched earlier, or one the user started)
    if is_server_ready(settings.base_url, api_key=settings.api_key, timeout=3.0):
        return None

    if model_less:
        command = build_llama_server_command(settings, task, ModelAssets(), model_less=True)
    else:
        assets = ensure_model_assets(settings, task, progress=progress)
        command = build_llama_server_command(settings, task, assets)
    launch_env = server_launch_env(resolve_llama_server_path(settings))
    log_dir = Path(settings.models_dir).expanduser().resolve() / "server_logs"
    return start_server_process(
        command,
        base_url=settings.base_url,
        api_key=settings.api_key,
        log_dir=log_dir,
        name="llama-server",
        startup_timeout=settings.server_startup_timeout,
        progress=progress,
        env=launch_env,
    )


def image_to_data_url(path: Path, vision_image_format: str = "auto") -> str:
    fmt = vision_image_format.lower().strip()
    if fmt not in {"auto", "original", "png", "jpeg", "jpg"}:
        raise ValueError(f"Invalid vision image format: {vision_image_format}")

    suffix = path.suffix.lower()
    convert_to: str | None = None
    if fmt == "png":
        convert_to = "PNG"
    elif fmt in {"jpeg", "jpg"}:
        convert_to = "JPEG"
    elif fmt == "auto" and suffix == ".webp":
        convert_to = "PNG"

    if convert_to is None:
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "application/octet-stream"
        b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{b64}"

    with Image.open(path) as image:
        if convert_to == "JPEG":
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            mime = "image/jpeg"
        else:
            if image.mode not in {"RGB", "RGBA"}:
                image = image.convert("RGBA" if "A" in image.getbands() else "RGB")
            mime = "image/png"

        buffer = io.BytesIO()
        image.save(buffer, format=convert_to)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:{mime};base64,{b64}"


def extract_json(text: str) -> dict[str, Any]:
    raw_output = text
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        raise ModelJsonError(f"No JSON object found in model output: {text[:500]!r}", raw_output=raw_output)

    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ModelJsonError(f"Could not parse model JSON: {exc}", raw_output=raw_output, candidate=candidate) from exc
    if not isinstance(parsed, dict):
        raise ModelJsonError("Model output JSON root was not an object.", raw_output=raw_output, candidate=candidate)
    return parsed


def _make_openai_client(settings: CaptioningSettings):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AutoCaptionError("Install openai to use auto captioning.") from exc
    return OpenAI(base_url=settings.base_url.rstrip("/") + "/", api_key=settings.api_key or "dummy")


def request_user_prompt(settings: CaptioningSettings, user: str) -> str:
    if not settings.disable_thinking:
        return user
    stripped = user.lstrip()
    if stripped.startswith("/no_think"):
        return user
    return "/no_think\n\n" + user


def strip_thinking_output(content: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL)
    return cleaned.strip()


def empty_response_detail(response: Any, choice: Any, content: str | None = None) -> str:
    finish_reason = getattr(choice, "finish_reason", None)
    response_model = getattr(response, "model", "")
    detail = f"finish_reason={finish_reason}, response_model={response_model or 'unknown'}"
    if content and not strip_thinking_output(content):
        detail += ", content contained only thinking output"
    return detail


def empty_response_hint(settings: CaptioningSettings) -> str:
    if settings.disable_thinking:
        return (
            "The server returned a completion object but no assistant text. Check the llama-server log for "
            "template/mmproj errors."
        )
    return (
        "Thinking/reasoning is enabled and the server returned no visible assistant text. If finish_reason=length, "
        "the model likely used the response budget before producing the final answer. Increase the task max-token "
        "budget and Context size, lower the Thinking token budget, or turn Disable thinking/reasoning back on."
    )


def request_failure_message(kind: str, exc: Exception) -> str:
    message = str(exc)
    hint = ""
    if "connection error" in message.lower():
        hint = (
            " The local server may have crashed or closed the connection during generation. "
            "Check models/server_logs for llama-server assertions or out-of-memory errors."
        )
    return f"{kind} model request failed: {exc}.{hint}"


def chat_text(
    settings: CaptioningSettings,
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    if not model:
        raise AutoCaptionError("No caption model name is configured.")
    client = _make_openai_client(settings)
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": request_user_prompt(settings, user)},
            ],
        )
    except Exception as exc:
        raise AutoCaptionError(request_failure_message("Text", exc)) from exc
    choice = response.choices[0] if response.choices else None
    content = choice.message.content if choice is not None else None
    visible_content = strip_thinking_output(content) if content else ""
    if visible_content:
        return visible_content
    detail = empty_response_detail(response, choice, content)
    raise AutoCaptionError(f"Text model '{model}' returned no visible response. {detail}. {empty_response_hint(settings)}")


def chat_vision(
    settings: CaptioningSettings,
    model: str,
    image_path: Path,
    system: str,
    user: str,
    max_tokens: int,
    temperature: float = 0.0,
) -> str:
    if not model:
        raise AutoCaptionError("No vision model name is configured.")
    client = _make_openai_client(settings)
    image_url = image_to_data_url(image_path, settings.vision_image_format)

    def request(image_first: bool):
        content_parts = [
            {"type": "text", "text": request_user_prompt(settings, user)},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        if image_first:
            content_parts.reverse()
        return client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content_parts},
            ],
        )

    errors: list[str] = []
    for image_first in (True, False):
        try:
            response = request(image_first=image_first)
        except Exception as exc:
            raise AutoCaptionError(request_failure_message("Vision", exc)) from exc
        choice = response.choices[0] if response.choices else None
        content = choice.message.content if choice is not None else None
        visible_content = strip_thinking_output(content) if content else ""
        if visible_content:
            return visible_content
        errors.append(f"image_first={image_first}, {empty_response_detail(response, choice, content)}")

    raise AutoCaptionError(
        f"Vision model '{model}' returned no visible response after image-order retry. {'; '.join(errors)}. "
        f"{empty_response_hint(settings)}"
    )


JSON_REPAIR_SYSTEM = """
You repair malformed JSON emitted by another model.
Return exactly one compact valid JSON object. No markdown. No commentary.
Preserve the original content and field names whenever possible.
If the response contains extra prose, remove the prose.
If the response is truncated or impossible to fully repair, return the most complete valid object that preserves the usable content.
""".strip()

JSON_REPAIR_USER = """
The model response below was supposed to be {expected}.

Parser error:
{error}

Repair the response into one valid JSON object.

Model response:
{raw_output}
""".strip()


def _repair_prompt_text(raw_output: str, limit: int = 24000) -> str:
    raw_output = raw_output.strip()
    if len(raw_output) <= limit:
        return raw_output
    half = limit // 2
    return raw_output[:half].rstrip() + "\n\n... [middle omitted for JSON repair] ...\n\n" + raw_output[-half:].lstrip()


def parse_json_with_repair(
    settings: CaptioningSettings,
    task: str,
    raw_output: str,
    expected: str,
    max_tokens: int,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    first_error: ModelJsonError
    try:
        return extract_json(raw_output)
    except ModelJsonError as exc:
        first_error = exc
        if progress is not None:
            progress("Model returned invalid JSON; retrying with a JSON repair prompt.")

    config = runtime_config_for_task(settings, task)
    repair_raw = ""
    try:
        repair_raw = chat_text(
            settings=settings,
            model=config.api_model,
            system=JSON_REPAIR_SYSTEM,
            user=format_prompt(
                JSON_REPAIR_USER,
                expected=expected,
                error=str(first_error),
                raw_output=_repair_prompt_text(first_error.raw_output or raw_output),
            ),
            max_tokens=max(2000, max_tokens),
            temperature=0.0,
        )
        parsed = extract_json(repair_raw)
    except Exception as repair_error:
        message = f"{first_error}; repair retry failed: {repair_error}"
        raise ModelJsonError(
            message,
            raw_output=first_error.raw_output or raw_output,
            candidate=first_error.candidate,
            repair_output=repair_raw,
        ) from repair_error

    if progress is not None:
        progress("JSON repair retry succeeded.")
    return parsed


PLAIN_CAPTION_SYSTEM = """
You write factual image captions for image-generation datasets.
Return one polished plain-text caption only. No markdown, no JSON, no bullet points.
Preserve any visible text exactly. Describe the main subjects, setting, style,
lighting, camera/viewpoint, and notable objects without guessing identities.
""".strip()

PLAIN_CAPTION_USER = """
Write a detailed but clean text-to-image caption for this image. Keep it useful
for recreating the image, but avoid unsupported proper names or speculation.
""".strip()

CREATIVE_DIRECTIVE = """
Expansion policy:
- Preserve the source caption's idea.
- Add useful visual detail when it helps the caption.
- Add only supportive background and scene details that do not replace or contradict the source caption.
- Never introduce a different main subject.
- Preserve trigger tokens/names/styles exactly.
- Do not invent appearance details for named people or trigger identities.
""".strip()

FAITHFUL_DIRECTIVE = """
Fidelity policy:
- Fill in only what the structured schema needs.
- Do not add new subjects, props, setting, style details, colors, brands, text, or atmosphere not present in the source caption.
- If the source caption is sparse, the JSON stays sparse.
""".strip()

JSON_SCHEMA_INSTRUCTIONS = """
Return exactly one compact valid JSON object. No markdown. No commentary.

Schema:
{
  "high_level_description": "...",
  "style_description": {
    "aesthetics": "...",
    "lighting": "...",
    "photo": "...",
    "medium": "photograph"
  },
  "compositional_deconstruction": {
    "background": "...",
    "elements": [
      {"type": "obj", "desc": "..."},
      {"type": "text", "text": "...", "desc": "..."}
    ]
  }
}

Field guidance:
- high_level_description: one or two sentences summarizing the whole image.
- aesthetics: concise visual style keywords, e.g. "moody, cinematic, desaturated" or "warm, playful, vibrant".
- lighting: concrete light quality, source, and shadow behavior, e.g. "golden hour, rim light, dramatic shadows" or "bright afternoon sunlight, long soft shadows".
- photo: camera, lens, viewpoint, focus, and photographic traits for photos, e.g. "35mm, f/1.4, bokeh", "shallow depth of field, eye-level, 85mm lens", or "wide angle, f/8, long exposure".
- medium: use a compact medium label such as "photograph", "illustration", "3d_render", "painting", or "graphic_design".
- art_style: style and medium traits for non-photo captions, e.g. "flat vector illustration, bold outlines" or "flat vector design, generous whitespace, sans-serif typography".
- background: describe the environment, setting, distant scenery, surfaces, and atmosphere.
- elements desc: describe each subject/object with its visible appearance, clothing/materials, pose/action, and important props.

Rules:
- Include high_level_description, style_description, and compositional_deconstruction.
- compositional_deconstruction must contain background first, then elements.
- Use "photo" for photographic images, or replace it with "art_style" for non-photo artwork.
- Use exactly one of "photo" or "art_style".
- Do not include bbox values. Bboxes are added in a separate pass.
- Do not include color_palette fields.
- type is "obj" for normal subjects/objects and "text" only for literal visible text.
- Text elements must preserve the literal visible text exactly.
- A coherent subject is one element; do not split people, vehicles, plants, buildings, or products into parts.
- Put ground, sky, walls, distant scenery, and ambient environment into background.
- Put people, animals, vehicles, products, furniture, props, signs, and visible text into elements.
- Keep trigger tokens, names, identifiers, and stylized spelling exactly.
""".strip()

TEXT_TO_JSON_SYSTEM = """
You convert an existing vetted sidecar caption into an Ideogram 4 structured JSON caption.
The source caption is authoritative; organize it into the schema without recaptioning the image.
""".strip()

TEXT_TO_JSON_USER = """
Existing vetted caption:
{caption}

Convert this caption into Ideogram 4 structured JSON.
""".strip()

IMAGE_TO_JSON_SYSTEM = """
You inspect an image and produce an Ideogram 4 structured JSON caption.
The image is authoritative. Describe only what is visible.
""".strip()

IMAGE_TO_JSON_USER = """
Create an Ideogram 4 structured JSON caption for this image.
Do not reference any existing sidecar caption.
""".strip()

JSON_REFINE_SYSTEM = """
You revise an existing Ideogram 4 structured JSON caption for an image dataset.
Use the image as the visual authority, the current JSON as the structure to improve,
the sidecar caption as supporting context, and the user's edit instructions as the task.

Return exactly one compact valid JSON object. No markdown. No commentary.

Schema:
{
  "high_level_description": "...",
  "style_description": {
    "aesthetics": "...",
    "lighting": "...",
    "photo": "...",
    "medium": "photograph"
  },
  "compositional_deconstruction": {
    "background": "...",
    "elements": [
      {"type": "obj", "bbox": [y1,x1,y2,x2], "desc": "..."},
      {"type": "text", "bbox": [y1,x1,y2,x2], "text": "...", "desc": "..."}
    ]
  }
}

Field guidance:
- high_level_description: one or two sentences summarizing the whole image.
- aesthetics: concise visual style keywords, e.g. "moody, cinematic, desaturated" or "warm, playful, vibrant".
- lighting: concrete light quality, source, and shadow behavior, e.g. "golden hour, rim light, dramatic shadows" or "bright afternoon sunlight, long soft shadows".
- photo: camera, lens, viewpoint, focus, and photographic traits for photos, e.g. "35mm, f/1.4, bokeh", "shallow depth of field, eye-level, 85mm lens", or "wide angle, f/8, long exposure".
- medium: use a compact medium label such as "photograph", "illustration", "3d_render", "painting", or "graphic_design".
- art_style: style and medium traits for non-photo captions, e.g. "flat vector illustration, bold outlines" or "flat vector design, generous whitespace, sans-serif typography".
- background: describe the environment, setting, distant scenery, surfaces, and atmosphere.
- elements desc: describe each subject/object with its visible appearance, clothing/materials, pose/action, and important props.
- bbox: when present, use [y_min,x_min,y_max,x_max] normalized 0..1000 with origin at top-left.

Rules:
- Preserve trigger tokens, names, identifiers, and stylized spelling exactly.
- Preserve literal visible text exactly.
- Preserve existing bbox values for unchanged elements. Do not invent bboxes for new elements.
- Do not remove real visible elements unless the user's instructions explicitly say to.
- A coherent subject is one element; do not split people, vehicles, plants, buildings, or products into parts.
- Put people, animals, vehicles, products, furniture, props, signs, and visible text into elements.
- Put ground, sky, walls, distant scenery, and ambient environment into background.
- For photographic images use "photo"; for non-photo artwork use "art_style". Use exactly one of those keys.
""".strip()

JSON_REFINE_USER = """
User edit instructions:
{instructions}

Existing sidecar caption:
{source_caption}

Current structured JSON:
{caption_json}

Revise the structured JSON according to the instructions.
""".strip()

BATCH_GROUND_SYSTEM = """
Locate multiple existing target elements in the image.
The targets already exist in a structured JSON caption. Your job is only to supply coordinates.
Do not invent new elements. Do not split or merge elements. Do not reinterpret targets.

Return only valid compact JSON in exactly this shape:
{"bboxes":{"0":[x1,y1,x2,y2],"1":null}}

Rules:
- Include every requested target id exactly once.
- Use null if the target is not visible or you are not confident.
- bbox values are normalized 0..1000.
- bbox origin is top-left.
- bbox format is [x1,y1,x2,y2].
- bbox should tightly cover the visible extent of that target only.
""".strip()

BATCH_GROUND_USER = """
Supporting structured JSON context:
{context_json}

Targets to locate:
{targets_json}

Return only:
{{"bboxes":{{"0":[x1,y1,x2,y2],"1":null}}}}
""".strip()


GUIDANCE_PREAMBLE = """
PER-IMAGE USER GUIDANCE (authoritative).
The instructions below are supplied by the user for THIS specific image. Follow them exactly. Where they conflict with the general rules above, the user guidance takes precedence. In particular, when the guidance calls for it, you MAY:
- Emit a part, prop, held object, or weapon as its OWN separate element, even though the general rule prefers one element per subject. Anything the user wants individually located must be its own element.
- Omit attributes the user tells you to skip (for example eye color, hair color, skin tone, or specific garments), even though the general rules ask you to name them.
- Use exact trigger tokens or names in place of generic descriptions, and append or prepend fixed text to specific fields, exactly as instructed.
Bounding boxes are added automatically in a later step, so you do not need to output coordinates yourself — just make sure anything that needs its own box is emitted as its own element. Do not let the general rules override an explicit user instruction.

USER GUIDANCE:
{guidance}
""".strip()


DEFAULT_PROMPT_TEXTS: dict[str, str] = {
    "plain_caption_system": PLAIN_CAPTION_SYSTEM,
    "plain_caption_user": PLAIN_CAPTION_USER,
    "creative_directive": CREATIVE_DIRECTIVE,
    "faithful_directive": FAITHFUL_DIRECTIVE,
    "json_schema_instructions": JSON_SCHEMA_INSTRUCTIONS,
    "guidance_preamble": GUIDANCE_PREAMBLE,
    "text_to_json_system": TEXT_TO_JSON_SYSTEM,
    "text_to_json_user": TEXT_TO_JSON_USER,
    "image_to_json_system": IMAGE_TO_JSON_SYSTEM,
    "image_to_json_user": IMAGE_TO_JSON_USER,
    "json_refine_system": JSON_REFINE_SYSTEM,
    "json_refine_user": JSON_REFINE_USER,
    "bbox_system": BATCH_GROUND_SYSTEM,
    "bbox_user": BATCH_GROUND_USER,
}


def load_prompts(path: Path | None = None) -> dict[str, Any]:
    folder = path or default_prompts_path()
    prompts = dict(DEFAULT_PROMPT_TEXTS)
    if not folder.exists() or not folder.is_dir():
        return prompts
    for name in DEFAULT_PROMPT_TEXTS:
        prompt_path = folder / f"{name}.txt"
        if not prompt_path.exists():
            continue
        try:
            prompts[name] = prompt_path.read_text(encoding="utf-8-sig").strip()
        except OSError:
            continue
    return prompts


def write_default_prompts(path: Path | None = None) -> Path:
    folder = path or default_prompts_path()
    folder.mkdir(parents=True, exist_ok=True)
    for name, text in DEFAULT_PROMPT_TEXTS.items():
        prompt_path = folder / f"{name}.txt"
        if not prompt_path.exists():
            prompt_path.write_text(text, encoding="utf-8")
    return folder


def format_prompt(template: str, **values: Any) -> str:
    try:
        return template.format(**values)
    except KeyError as exc:
        raise AutoCaptionError(f"Prompt is missing required placeholder {{{exc.args[0]}}}.") from exc


def generate_plain_caption(settings: CaptioningSettings, image_path: Path) -> str:
    config = runtime_config_for_task(settings, "caption")
    prompts = load_prompts()
    raw = chat_vision(
        settings=settings,
        model=config.api_model,
        image_path=image_path,
        system=prompts["plain_caption_system"],
        user=prompts["plain_caption_user"],
        max_tokens=settings.max_tokens_caption,
        temperature=0.2,
    )
    return raw.strip().strip('"').strip()


def _directive(settings: CaptioningSettings) -> str:
    prompts = load_prompts()
    key = "creative_directive" if settings.creative_json else "faithful_directive"
    return str(prompts[key])


def json_system_prompt(
    prompts: dict[str, Any],
    task_system_key: str,
    settings: CaptioningSettings,
    guidance: str = "",
) -> str:
    parts = [
        str(prompts[task_system_key]).strip(),
        str(prompts["json_schema_instructions"]).strip(),
        _directive(settings).strip(),
    ]
    guidance = (guidance or "").strip()
    if guidance:
        parts.append(format_prompt(str(prompts["guidance_preamble"]), guidance=guidance).strip())
    return "\n\n".join(part for part in parts if part)


def generate_json_from_text(
    settings: CaptioningSettings,
    caption_text: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    caption_text = caption_text.strip()
    if not caption_text:
        raise AutoCaptionError("No source text caption was found.")
    config = runtime_config_for_task(settings, "caption")
    prompts = load_prompts()
    raw = chat_text(
        settings=settings,
        model=config.api_model,
        system=json_system_prompt(prompts, "text_to_json_system", settings),
        user=format_prompt(
            prompts["text_to_json_user"],
            caption=caption_text,
            directive="",
        ),
        max_tokens=settings.max_tokens_json,
        temperature=0.0,
    )
    parsed = parse_json_with_repair(
        settings=settings,
        task="caption",
        raw_output=raw,
        expected="an Ideogram 4 structured caption JSON object",
        max_tokens=settings.max_tokens_json,
        progress=progress,
    )
    return normalize_caption(parsed)


def generate_json_from_image(
    settings: CaptioningSettings,
    image_path: Path,
    progress: ProgressCallback | None = None,
    guidance: str = "",
) -> dict[str, Any]:
    config = runtime_config_for_task(settings, "caption")
    prompts = load_prompts()
    raw = chat_vision(
        settings=settings,
        model=config.api_model,
        image_path=image_path,
        system=json_system_prompt(prompts, "image_to_json_system", settings, guidance=guidance),
        user=format_prompt(prompts["image_to_json_user"], directive=""),
        max_tokens=settings.max_tokens_json,
        temperature=0.0,
    )
    parsed = parse_json_with_repair(
        settings=settings,
        task="caption",
        raw_output=raw,
        expected="an Ideogram 4 structured caption JSON object",
        max_tokens=settings.max_tokens_json,
        progress=progress,
    )
    return normalize_caption(parsed)


def _preserve_missing_refined_bboxes(original: dict[str, Any], refined: dict[str, Any]) -> dict[str, Any]:
    original = normalize_caption(original)
    refined = normalize_caption(refined)
    original_elements = original.get("compositional_deconstruction", {}).get("elements", [])
    refined_elements = refined.get("compositional_deconstruction", {}).get("elements", [])
    for index, refined_element in enumerate(refined_elements):
        if normalize_bbox(refined_element.get("bbox")) is not None or index >= len(original_elements):
            continue
        original_element = original_elements[index]
        if refined_element.get("type") != original_element.get("type"):
            continue
        bbox = normalize_bbox(original_element.get("bbox"))
        if bbox is not None:
            refined_element["bbox"] = bbox
    return normalize_caption(refined)


def generate_json_refinement(
    settings: CaptioningSettings,
    image_path: Path,
    caption: dict[str, Any],
    source_caption: str,
    instructions: str,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    instructions = instructions.strip()
    if not instructions:
        raise AutoCaptionError("No JSON refinement instructions were provided.")
    config = runtime_config_for_task(settings, "caption")
    prompts = load_prompts()
    current_caption = normalize_caption(caption)
    raw = chat_vision(
        settings=settings,
        model=config.api_model,
        image_path=image_path,
        system=prompts["json_refine_system"],
        user=format_prompt(
            prompts["json_refine_user"],
            instructions=instructions,
            source_caption=source_caption.strip() or "(none)",
            caption_json=json.dumps(current_caption, ensure_ascii=False, indent=2),
        ),
        max_tokens=settings.max_tokens_json,
        temperature=0.0,
    )
    parsed = parse_json_with_repair(
        settings=settings,
        task="caption",
        raw_output=raw,
        expected="a refined Ideogram 4 structured caption JSON object",
        max_tokens=settings.max_tokens_json,
        progress=progress,
    )
    return _preserve_missing_refined_bboxes(current_caption, parsed)


def bbox_xyxy_to_yxyx(bbox: Any) -> list[int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox]
    except (TypeError, ValueError):
        return None
    return normalize_bbox([y1, x1, y2, x2])


def parse_batch_bboxes_with_reasons(
    raw: str,
    settings: CaptioningSettings | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, list[int] | None], dict[str, str]]:
    if settings is None:
        parsed = extract_json(raw)
    else:
        parsed = parse_json_with_repair(
            settings=settings,
            task="bbox",
            raw_output=raw,
            expected='a compact JSON object shaped like {"bboxes":{"0":[x1,y1,x2,y2],"1":null}}',
            max_tokens=settings.max_tokens_bboxes,
            progress=progress,
        )
    if "bboxes" in parsed and isinstance(parsed["bboxes"], dict):
        raw_map = parsed["bboxes"]
    elif "targets" in parsed and isinstance(parsed["targets"], list):
        raw_map = {}
        for item in parsed["targets"]:
            if not isinstance(item, dict):
                continue
            target_id = item.get("id")
            if target_id is not None:
                raw_map[str(target_id)] = item.get("bbox") if item.get("found", True) else None
    else:
        raise AutoCaptionError(f"Response must contain a bboxes object. Got keys: {list(parsed.keys())}")

    out: dict[str, list[int] | None] = {}
    reasons: dict[str, str] = {}
    for key, value in raw_map.items():
        key_text = str(key)
        if value is None:
            out[key_text] = None
            reasons[key_text] = "model returned null"
            continue
        bbox = bbox_xyxy_to_yxyx(value)
        out[key_text] = bbox
        if bbox is None:
            reasons[key_text] = "model returned invalid bbox"
    return out, reasons


def parse_batch_bboxes(raw: str) -> dict[str, list[int] | None]:
    return parse_batch_bboxes_with_reasons(raw)[0]


def should_try_bbox(element: dict[str, Any]) -> bool:
    element_type = element.get("type")
    if element_type not in {"obj", "text"}:
        return False

    desc = str(element.get("desc", "")).lower()
    words = re.findall(r"[a-z0-9]+", desc)
    dense_terms = {
        "crowd",
        "crowds",
        "starfield",
        "stars",
        "particles",
        "confetti",
        "field of",
        "background",
        "sky",
        "clouds",
        "grass field",
        "water surface",
    }
    if any(term in desc for term in dense_terms):
        return False

    vague_dense_terms = {"pattern", "patterns", "texture", "textures"}
    if any(term in words for term in vague_dense_terms):
        concrete_terms = {
            "animal",
            "arm",
            "body",
            "boy",
            "car",
            "cat",
            "chair",
            "child",
            "dog",
            "door",
            "face",
            "frame",
            "girl",
            "hand",
            "head",
            "headband",
            "holder",
            "leg",
            "man",
            "pants",
            "panties",
            "person",
            "roll",
            "shirt",
            "shelf",
            "sign",
            "sink",
            "table",
            "tattoo",
            "toilet",
            "top",
            "vehicle",
            "woman",
        }
        return any(term in words for term in concrete_terms)

    return True


def bbox_target_indices_with_reasons(
    elements: list[dict[str, Any]],
    settings: CaptioningSettings,
) -> tuple[list[int], dict[int, str]]:
    to_locate: list[int] = []
    skipped: dict[int, str] = {}
    for index, element in enumerate(elements):
        if element.get("type") not in {"obj", "text"}:
            skipped[index] = "not an obj/text element"
            continue
        has_bbox = normalize_bbox(element.get("bbox")) is not None
        if has_bbox and not settings.overwrite_bboxes:
            skipped[index] = "existing bbox kept"
            continue
        if settings.filter_bbox_targets and not should_try_bbox(element):
            skipped[index] = "filtered as vague/ambient"
            continue
        to_locate.append(index)
    return to_locate, skipped


def bbox_target_indices(elements: list[dict[str, Any]], settings: CaptioningSettings) -> list[int]:
    to_locate, _skipped = bbox_target_indices_with_reasons(elements, settings)
    return to_locate


def make_localization_context(data: dict[str, Any], max_chars: int) -> str:
    context: dict[str, Any] = {}
    high = data.get("high_level_description")
    if isinstance(high, str) and high.strip():
        context["high_level_description"] = high.strip()
    comp = data.get("compositional_deconstruction")
    if isinstance(comp, dict):
        background = comp.get("background")
        if isinstance(background, str) and background.strip():
            context["background"] = background.strip()

    text = json.dumps(context, ensure_ascii=False, separators=(",", ":"))
    if max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "..."
    return text


def chunk_list(items: list[Any], chunk_size: int) -> list[list[Any]]:
    if chunk_size <= 0:
        return [items]
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def build_targets_for_chunk(elements: list[dict[str, Any]], indices: list[int]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for index in indices:
        element = elements[index]
        element_type = element.get("type", "obj")
        target: dict[str, Any] = {
            "id": str(index),
            "type": element_type if element_type in {"obj", "text"} else "obj",
            "desc": str(element.get("desc", "")).strip(),
        }
        if element_type == "text":
            text = str(element.get("text", "")).strip()
            if text:
                target["text"] = text
        targets.append(target)
    return targets


def ordered_element_with_bbox(
    element: dict[str, Any],
    bbox: list[int] | None,
    keep_existing_if_no_new: bool,
) -> dict[str, Any]:
    element_type = element.get("type", "obj")
    if element_type not in {"obj", "text"}:
        element_type = "obj"

    existing_bbox = normalize_bbox(element.get("bbox")) if "bbox" in element else None
    final_bbox = bbox if bbox is not None else (existing_bbox if keep_existing_if_no_new else None)
    desc = str(element.get("desc", "")).strip()

    if element_type == "text":
        out: dict[str, Any] = {"type": "text"}
        if final_bbox is not None:
            out["bbox"] = final_bbox
        out["text"] = str(element.get("text", "")).strip()
        out["desc"] = desc
        return out

    out = {"type": "obj"}
    if final_bbox is not None:
        out["bbox"] = final_bbox
    out["desc"] = desc
    return out


def add_bboxes_to_caption(
    settings: CaptioningSettings,
    image_path: Path,
    caption: dict[str, Any],
    progress: ProgressCallback | None = None,
) -> tuple[dict[str, Any], int, int, dict[str, int]]:
    data = normalize_caption(caption)
    elements = data.get("compositional_deconstruction", {}).get("elements", [])
    elements = [element for element in elements if isinstance(element, dict)]

    to_locate, skipped_before = bbox_target_indices_with_reasons(elements, settings)

    config = runtime_config_for_task(settings, "bbox")
    prompts = load_prompts()
    context_json = make_localization_context(data, settings.context_chars)
    located: dict[int, list[int] | None] = {}
    skipped_reasons: dict[int, str] = dict(skipped_before)
    attempted = 0
    added = 0

    for chunk in chunk_list(to_locate, settings.max_targets_per_call):
        if not chunk:
            continue
        prompt = format_prompt(
            prompts["bbox_user"],
            context_json=context_json,
            targets_json=json.dumps(build_targets_for_chunk(elements, chunk), ensure_ascii=False, separators=(",", ":")),
        )
        attempted += len(chunk)
        raw = chat_vision(
            settings=settings,
            model=config.api_model,
            image_path=image_path,
            system=prompts["bbox_system"],
            user=prompt,
            max_tokens=settings.max_tokens_bboxes,
            temperature=0.0,
        )
        bbox_map, response_reasons = parse_batch_bboxes_with_reasons(raw, settings=settings, progress=progress)
        for index in chunk:
            key = str(index)
            if key not in bbox_map:
                located[index] = None
                skipped_reasons[index] = "model omitted target id"
                continue
            bbox = bbox_map.get(key)
            located[index] = bbox
            if bbox is not None:
                added += 1
            else:
                skipped_reasons[index] = response_reasons.get(key, "model returned no bbox")

    new_elements = [
        ordered_element_with_bbox(
            element,
            bbox=located.get(index),
            keep_existing_if_no_new=not settings.overwrite_bboxes,
        )
        for index, element in enumerate(elements)
    ]
    data["compositional_deconstruction"]["elements"] = new_elements
    reason_counts: dict[str, int] = {}
    for reason in skipped_reasons.values():
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return normalize_caption(data), attempted, added, reason_counts
