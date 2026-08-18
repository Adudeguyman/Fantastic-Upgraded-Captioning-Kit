"""Stage 0/1 of the PySide6 port: a launchable window with the locked layout
bones (toolbar, collapsible guidance left, image view center, tabbed
Caption/Elements panel right, thumbnail filmstrip bottom), wired to real
folder-open and image display. Editing/AI behavior arrives in later stages.
"""

from __future__ import annotations

import copy
import difflib
import html
import datetime
import json
import math
import os
import re
import tempfile
import subprocess
import sys
import time
import webbrowser
from dataclasses import asdict, replace
from pathlib import Path

try:
    from PySide6.QtCore import QObject, Qt, QSize, QSettings, QRect, QRectF, QPoint, QPointF, QMimeData, QEvent, QThread, Signal, QByteArray, QTimer, QPropertyAnimation, QEasingCurve, Property, QParallelAnimationGroup, QAbstractAnimation, QVariantAnimation, QUrl
    from PySide6.QtGui import QAction, QBrush, QColor, QFont, QFontDatabase, QFontMetrics, QKeySequence, QPainter, QPainterPath, QPen, QPixmap, QIcon, QShortcut, QTextCharFormat, QTextCursor, QTextFormat, QDrag, QPolygonF
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QGridLayout,
        QFrame,
        QLayout,
        QGraphicsPixmapItem,
        QGraphicsOpacityEffect,
        QGraphicsDropShadowEffect,
        QGraphicsScene,
        QGraphicsView,
        QGraphicsItem,
        QGraphicsRectItem,
        QGraphicsPathItem,
        QHBoxLayout,
        QButtonGroup,
        QAbstractButton,
        QCheckBox,
        QRadioButton,
        QColorDialog,
        QComboBox,
        QDialog,
        QDialogButtonBox,
        QDoubleSpinBox,
        QFormLayout,
        QInputDialog,
        QLabel,
        QLineEdit,
        QProgressBar,
        QProgressDialog,
        QSlider,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMenu,
        QMessageBox,
        QPlainTextEdit,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QStackedWidget,
        QStyle,
        QStyledItemDelegate,
        QStyleOptionGraphicsItem,
        QTabWidget,
        QToolBar,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # Tier 2: clear message instead of a raw traceback.
    sys.stderr.write(
        "\nPySide6 is not installed in this environment.\n"
        "Run the installer for your setup:\n\n"
        "    ./install_venv.sh     (or install_venv.bat on Windows)\n"
        "    ./install_conda.sh    (or install_conda.bat on Windows)\n\n"
        f"(original import error: {exc})\n"
    )
    raise SystemExit(1)

# Shared backend — imported unchanged from the existing Tkinter app.
from .store import CaptionStore, ProjectConfig
from .llm_captioning import (
    json_system_prompt,
    load_prompts,
    CaptioningSettings,
    AutoCaptionError,
    add_bboxes_to_caption,
    generate_json_from_image,
    generate_json_refinement,
    load_settings,
    save_settings,
    profiles_for_task,
    profile_labels,
    profile_id_from_label,
    _split_filenames,
    discover_local_gguf_models,
    normalise_h3_caption,
    locate_existing_model_file,
    model_search_roots,
    estimate_gguf_vram_gb,
    guess_mmproj_for,
    CUSTOM_LOCAL_PROFILE,
    profile_label_from_id,
    is_server_ready,
    server_model_ids,
    server_log_path,
    diagnose_server_log,
    BUILTIN_OOM_HINT,
    ensure_server_running,
    stop_server_process,
    find_llama_server,
    detect_gpus,
    recommend_profile_for_vram,
    vram_fit,
    model_size_tier,
    has_model_config,
    missing_model_files,
    lmstudio_models_dir,
    known_server_model_dirs,
    hf_hub_cache_dir,
    MODEL_TARGET_APP,
    MODEL_TARGET_HF,
    llama_server_supports_router,
    sample_resources,
    format_resources,
    caption_image_plain,
    caption_video_plain,
    runtime_config_for_task,
    mmproj_has_audio_encoder,
    existing_mmproj_path,
    plan_llama_acquisition,
    install_llama_release,
    rollback_llama,
    has_llama_backup,
    read_installed_llama,
    update_state,
    fetch_release,
    is_model_arch_error,
    default_profiles_path,
    default_models_dir,
    profile_seed_data,
)
from .presets import (
    PRESET_ORDER,
    PRESETS,
    CaptionPreset,
    all_presets,
    get_preset,
    load_custom_presets,
    make_custom_preset,
    save_custom_presets,
)
from .training_goals import (
    DEFAULT_GOAL,
    GOAL_ORDER,
    GOALS,
    TrainingGoal,
    builtin_goal_map,
    get_goal,
    goal_order,
    load_goals,
    make_custom_goal,
    save_goals,
)
from .video_tools import (
    VideoInfo,
    extract_poster,
    ffmpeg_available,
    find_ffmpeg,
    install_ffmpeg,
    has_audio_stream,
    is_video,
    quiet_ffmpeg_logs,
    VIDEO_EXTENSIONS,
    probe_video,
    apply_mute_span,
    audio_peaks,
    export_audio,
    extract_single_frame,
    VideoEditPlan,
    apply_video_edit as run_video_edit,
    plan_for_target,
)
from .llm_captioning import app_base_dir
from .model_targets import (
    ModelTarget,
    _target_from_dict,
    builtin_map,
    load_targets,
    save_targets,
    targets_path,
)
from .schema import IMAGE_EXTENSIONS, caption_health, default_caption, serialize_caption


# User-facing product name, used for the window title and About box.
APP_TITLE = "Fantastic Upgraded Captioning Kit"

THUMB = 64
# Filmstrip hover-preview popup (designed spec, dark theme). Shows instantly on
# hover — no dwell or fade (it felt laggy), so only layout constants remain.
PREVIEW_PAD = 6            # popup inner padding
PREVIEW_IMG_W = 196        # image area (4:3)
PREVIEW_IMG_H = 147
PREVIEW_W = PREVIEW_IMG_W + 2 * PREVIEW_PAD   # 208 popup width
PREVIEW_GAP = 8            # gap between popup and thumbnail
PREVIEW_ARROW = 13         # diamond pointer size
# Filmstrip unsaved indicator (amber corner dot, replaces the red glow).
DOT_APPEAR = 120           # ms scale+fade in, OutCubic
DOT_DISAPPEAR = 90         # ms scale+fade out, OutQuad


# Lucide icons (MIT License, lucide.dev) — inner SVG, recolored at render time.
_LUCIDE_ICONS = {
    "braces": "<path d='M8 3H7a2 2 0 0 0-2 2v5a2 2 0 0 1-2 2 2 2 0 0 1 2 2v5c0 1.1.9 2 2 2h1' /> <path d='M16 21h1a2 2 0 0 0 2-2v-5c0-1.1.9-2 2-2a2 2 0 0 1-2-2V5a2 2 0 0 0-2-2h-1' />",
    "check": "<path d='M20 6 9 17l-5-5' />",
    "chevron-down": "<path d='m6 9 6 6 6-6' />",
    "chevron-left": "<path d='m15 18-6-6 6-6' />",
    "chevron-right": "<path d='m9 18 6-6-6-6' />",
    "chevron-up": "<path d='m18 15-6-6-6 6' />",
    "chevrons-left": "<path d='m11 17-5-5 5-5' /> <path d='m18 17-5-5 5-5' />",
    "crop": "<path d='M6 2v14a2 2 0 0 0 2 2h14' /> <path d='M18 22V8a2 2 0 0 0-2-2H2' />",
    "link": ("<path d='M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71' />"
             " <path d='M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71' />"),
    "mouse-pointer": ("<path d='M12.586 12.586 19 19' />"
                      " <path d='M3.688 3.037a.497.497 0 0 0-.651.651l6.5 15.999a.501.501"
                      " 0 0 0 .947-.062l1.569-6.083a2 2 0 0 1 1.448-1.479l6.124-1.579a.5.5"
                      " 0 0 0 .063-.947z' />"),
    "hand": ("<path d='M18 11V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2' />"
             " <path d='M14 10V4a2 2 0 0 0-2-2a2 2 0 0 0-2 2v2' />"
             " <path d='M10 10.5V6a2 2 0 0 0-2-2a2 2 0 0 0-2 2v8' />"
             " <path d='M18 8a2 2 0 1 1 4 0v6a8 8 0 0 1-8 8h-2c-2.8 0-4.5-.86-5.99-2.34"
             "l-3.6-3.6a2 2 0 0 1 2.83-2.82L7 15' />"),
    "copy": ("<rect width='14' height='14' x='8' y='8' rx='2' ry='2' />"
             " <path d='M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2' />"),
    "image-plus": ("<path d='M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h7' />"
                   " <path d='M16 5h6' /> <path d='M19 2v6' />"
                   " <circle cx='9' cy='9' r='2' />"
                   " <path d='m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21' />"),
    "eye-off": ("<path d='M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68' />"
                " <path d='M6.61 6.61A13.5 13.5 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61' />"
                " <path d='M14.12 14.12a3 3 0 1 1-4.24-4.24' /> <path d='m2 2 20 20' />"),
    "rotate-cw": ("<path d='M21 12a9 9 0 1 1-9-9c2.52 0 4.93 1 6.74 2.74L21 8' />"
                  " <path d='M21 3v5h-5' />"),
    "rotate-ccw": ("<path d='M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8' />"
                   " <path d='M3 3v5h5' />"),
    "ellipsis": "<circle cx='12' cy='12' r='1' /> <circle cx='19' cy='12' r='1' /> <circle cx='5' cy='12' r='1' />",
    "film": "<rect width='18' height='18' x='3' y='3' rx='2' /> <path d='M7 3v18' /> <path d='M17 3v18' /> <path d='M3 7.5h4' /> <path d='M3 12h18' /> <path d='M3 16.5h4' /> <path d='M17 7.5h4' /> <path d='M17 16.5h4' />",
    "flag": "<path d='M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z' /> <path d='M4 22v-7' />",
    "folder-open": "<path d='m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2' />",
    "info": "<circle cx='12' cy='12' r='10' /> <path d='M12 16v-4' /> <path d='M12 8h.01' />",
    "lock": "<rect width='18' height='11' x='3' y='11' rx='2' ry='2' /> <path d='M7 11V7a5 5 0 0 1 10 0v4' />",
    "lock-open": "<rect width='18' height='11' x='3' y='11' rx='2' ry='2' /> <path d='M7 11V7a5 5 0 0 1 9.9-1' />",
    "maximize": "<path d='M8 3H5a2 2 0 0 0-2 2v3' /> <path d='M21 8V5a2 2 0 0 0-2-2h-3' /> <path d='M3 16v3a2 2 0 0 0 2 2h3' /> <path d='M16 21h3a2 2 0 0 0 2-2v-3' />",
    "maximize-2": "<path d='M15 3h6v6' /> <path d='m21 3-7 7' /> <path d='m3 21 7-7' /> <path d='M9 21H3v-6' />",
    "pause": "<rect x='14' y='4' width='4' height='16' rx='1' /> <rect x='6' y='4' width='4' height='16' rx='1' />",
    "play": "<path fill='{color}' d='M6 3.5v17a1 1 0 0 0 1.5.86l13-8.5a1 1 0 0 0 0-1.72l-13-8.5A1 1 0 0 0 6 3.5z' />",
    "volume-2": "<path d='M11 4.7a.7.7 0 0 0-1.2-.5L6 8H4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2l3.8 3.8a.7.7 0 0 0 1.2-.5z' /> <path d='M16 9a5 5 0 0 1 0 6' /> <path d='M19.4 6.6a9 9 0 0 1 0 10.8' />",
    "volume-x": "<path d='M11 4.7a.7.7 0 0 0-1.2-.5L6 8H4a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h2l3.8 3.8a.7.7 0 0 0 1.2-.5z' /> <line x1='22' x2='16' y1='9' y2='15' /> <line x1='16' x2='22' y1='9' y2='15' />",
    "mouse-pointer-2": "<path d='M4.037 4.688a.495.495 0 0 1 .651-.651l16 6.5a.5.5 0 0 1-.063.947l-6.124 1.58a2 2 0 0 0-1.438 1.435l-1.579 6.126a.5.5 0 0 1-.947.063z' />",
    "move": "<path d='M12 2v20' /> <path d='m15 19-3 3-3-3' /> <path d='m19 9 3 3-3 3' /> <path d='M2 12h20' /> <path d='m5 9-3 3 3 3' /> <path d='m9 5 3-3 3 3' />",
    "panel-left-close": "<rect width='18' height='18' x='3' y='3' rx='2' /> <path d='M9 3v18' /> <path d='m16 15-3-3 3-3' />",
    "panel-left-open": "<rect width='18' height='18' x='3' y='3' rx='2' /> <path d='M9 3v18' /> <path d='m14 9 3 3-3 3' />",
    "panel-right-close": "<rect width='18' height='18' x='3' y='3' rx='2' /> <path d='M15 3v18' /> <path d='m8 9 3 3-3 3' />",
    "panel-right-open": "<rect width='18' height='18' x='3' y='3' rx='2' /> <path d='M15 3v18' /> <path d='m10 15-3-3 3-3' />",
    "pencil": "<path d='M21.174 6.812a1 1 0 0 0-3.986-3.987L3.842 16.174a2 2 0 0 0-.5.83l-1.321 4.352a.5.5 0 0 0 .623.622l4.353-1.32a2 2 0 0 0 .83-.497z' /> <path d='m15 5 4 4' />",
    "plus": "<path d='M5 12h14' /> <path d='M12 5v14' />",
    "save": "<path d='M15.2 3a2 2 0 0 1 1.4.6l3.8 3.8a2 2 0 0 1 .6 1.4V19a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2z' /> <path d='M17 21v-7a1 1 0 0 0-1-1H8a1 1 0 0 0-1 1v7' /> <path d='M7 3v4a1 1 0 0 0 1 1h7' />",
    "save-all": "<path d='M10 2v3a1 1 0 0 0 1 1h5' /> <path d='M18 18v-6a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6' /> <path d='M18 22H4a2 2 0 0 1-2-2V6' /> <path d='M8 18a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9.172a2 2 0 0 1 1.414.586l2.828 2.828A2 2 0 0 1 22 6.828V16a2 2 0 0 1-2.01 2z' />",
    "scaling": "<path d='M12 3H5a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7' /> <path d='M16 3h5v5' /> <path d='m21 3-6.5 6.5' /> <path d='M7 17h.01' /> <path d='M11 17h.01' /> <path d='M7 13h.01' />",
    "settings": "<path d='M9.671 4.136a2.34 2.34 0 0 1 4.659 0 2.34 2.34 0 0 0 3.319 1.915 2.34 2.34 0 0 1 2.33 4.033 2.34 2.34 0 0 0 0 3.831 2.34 2.34 0 0 1-2.33 4.033 2.34 2.34 0 0 0-3.319 1.915 2.34 2.34 0 0 1-4.659 0 2.34 2.34 0 0 0-3.32-1.915 2.34 2.34 0 0 1-2.33-4.033 2.34 2.34 0 0 0 0-3.831A2.34 2.34 0 0 1 6.35 6.051a2.34 2.34 0 0 0 3.319-1.915' /> <circle cx='12' cy='12' r='3' />",
    "square-dashed": "<path d='M5 3a2 2 0 0 0-2 2' /> <path d='M19 3a2 2 0 0 1 2 2' /> <path d='M21 19a2 2 0 0 1-2 2' /> <path d='M5 21a2 2 0 0 1-2-2' /> <path d='M9 3h1' /> <path d='M9 21h1' /> <path d='M14 3h1' /> <path d='M14 21h1' /> <path d='M3 9v1' /> <path d='M21 9v1' /> <path d='M3 14v1' /> <path d='M21 14v1' />",
    "square-plus": "<rect width='18' height='18' x='3' y='3' rx='2' /> <path d='M8 12h8' /> <path d='M12 8v8' />",
    "trash-2": "<path d='M10 11v6' /> <path d='M14 11v6' /> <path d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6' /> <path d='M3 6h18' /> <path d='M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2' />",
    "x": "<path d='M18 6 6 18' /> <path d='m6 6 12 12' />",
}

_LUCIDE_TPL = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" '
    'stroke="{color}" stroke-width="{sw}" stroke-linecap="round" stroke-linejoin="round">{inner}</svg>'
)
_LUCIDE_PIXMAP_CACHE: dict = {}


def lucide_pixmap(name: str, color: str = "#A6ADB6", size: int = 18, stroke: float = 1.8) -> QPixmap:
    """Render a Lucide glyph to a crisp (2x) recolored pixmap. Cached by params."""
    key = (name, color, size, stroke)
    cached = _LUCIDE_PIXMAP_CACHE.get(key)
    if cached is not None:
        return cached
    inner = _LUCIDE_ICONS.get(name, "")
    if not inner:
        # A missing glyph silently renders an empty button, which is very easy to
        # ship by accident — make it obvious in dev instead.
        print(f"[icons] unknown lucide glyph: {name!r}", file=sys.stderr)
    # Glyph bodies may carry their own {color} (e.g. filled shapes), so expand the
    # body first — placeholders inside `inner` aren't seen by the outer format().
    inner = inner.replace("{color}", color)
    svg = _LUCIDE_TPL.format(color=color, sw=stroke, inner=inner)
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    dpr = 2.0
    pm = QPixmap(int(size * dpr), int(size * dpr))
    pm.fill(Qt.transparent)
    pm.setDevicePixelRatio(dpr)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.Antialiasing, True)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    _LUCIDE_PIXMAP_CACHE[key] = pm
    return pm


_ARROW_FILE_CACHE: dict = {}


def lucide_arrow_url(name: str, color: str, size: int = 16,
                     stroke: float = 2.6) -> str:
    """Render an icon to a PNG on disk and return a stylesheet-ready URL.

    Qt stylesheets can only point sub-control arrows at an image, and the CSS
    border-triangle trick renders as small blocks rather than arrows in this
    theme — so the spin buttons get the same chevrons the rest of the app uses.
    Cached per (name, colour, size) because the sheet is rebuilt on theme change.
    """
    key = (name, color, size, stroke)
    hit = _ARROW_FILE_CACHE.get(key)
    if hit is not None and Path(hit).exists():
        return hit
    folder = Path(tempfile.gettempdir()) / "captioning_kit_icons"
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^a-z0-9]+", "_", f"{name}_{color}_{size}_{stroke}".lower())
    path = folder / f"{safe}.png"
    if not path.exists():
        pixmap = lucide_pixmap(name, color, size, stroke)
        if not pixmap.save(str(path), "PNG"):
            return ""
    url = path.as_posix()
    _ARROW_FILE_CACHE[key] = url
    return url


def lucide_icon(name: str, color: str = "#A6ADB6", size: int = 18, stroke: float = 1.8) -> QIcon:
    return QIcon(lucide_pixmap(name, color, size, stroke))


# App/taskbar icon: a rounded accent tile with a white image glyph. Rendered from
# SVG at several sizes so the window manager always has a crisp one. No asset file
# needed, so it survives packaging/path changes.
_APP_ICON_TILE = "#0f848a"
_APP_ICON_SVG = (
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24'>"
    "<rect x='1.5' y='1.5' width='21' height='21' rx='5.5' fill='{tile}'/>"
    "<g fill='none' stroke='#FFFFFF' stroke-width='1.8' "
    "stroke-linecap='round' stroke-linejoin='round'>"
    "<path d='M6.6 10 L6.6 7 L9.6 7'/>"
    "<path d='M17.4 10 L17.4 7 L14.4 7'/>"
    "<path d='M6.6 14 L6.6 17 L9.6 17'/>"
    "<path d='M17.4 14 L17.4 17 L14.4 17'/>"
    "</g>"
    "<circle cx='12' cy='12' r='1.7' fill='#FFFFFF'/>"
    "</svg>"
)


def app_icon(tile: str = _APP_ICON_TILE) -> QIcon:
    data = QByteArray(_APP_ICON_SVG.format(tile=tile).encode("utf-8"))
    icon = QIcon()
    for sz in (16, 20, 24, 32, 48, 64, 128, 256):
        renderer = QSvgRenderer(data)
        pm = QPixmap(sz, sz)
        pm.fill(Qt.transparent)
        painter = QPainter(pm)
        painter.setRenderHint(QPainter.Antialiasing, True)
        renderer.render(painter, QRectF(0, 0, sz, sz))
        painter.end()
        icon.addPixmap(pm)
    return icon


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    return QColor(
        round(a.red() * t + b.red() * (1 - t)),
        round(a.green() * t + b.green() * (1 - t)),
        round(a.blue() * t + b.blue() * (1 - t)),
    )


class Theme:
    """Resolved dark-theme palette. Fixed token roles + a derived accent ramp.

    Only the accent (and fonts) are user-editable; everything else is fixed so the
    theme stays coherent. Phase-1 token source of truth — QSS and the few painted
    widgets both read from here.
    """

    surface_0 = "#0F1115"
    surface_1 = "#171A1F"
    surface_2 = "#1E2227"
    surface_3 = "#262B31"
    surface_hover = "#2E343B"
    border = "#2A2F37"
    border_strong = "#373D46"
    border_strong_hover = "#454C56"
    text_primary = "#ECEEF1"
    text_secondary = "#A6ADB6"
    text_muted = "#6C737C"
    text_disabled = "#4A5158"
    success = "#45B964"
    warning = "#E0A33B"
    error = "#E5594B"
    tooltip_bg = "#22262C"
    tooltip_text = "#C9CFD7"

    def __init__(self, settings: CaptioningSettings) -> None:
        base = QColor(settings.color_accent) if settings.color_accent else QColor("#4C8DFF")
        if not base.isValid():
            base = QColor("#4C8DFF")
        self.accent = base.name()
        self.accent_hover = base.lighter(118).name()
        self.accent_pressed = base.darker(115).name()
        self.accent_on_subtle = base.lighter(140).name()
        self.accent_subtle = _mix(base, QColor(self.surface_0), 0.18).name()
        self.accent_subtle_border = _mix(base, QColor(self.surface_0), 0.42).name()


def build_stylesheet(s: CaptioningSettings) -> str:
    """Dark QSS theme built from the token palette. Applies live (no restart)."""
    t = Theme(s)
    # Spin-button arrows are images: Qt gives sub-controls no way to draw a shape,
    # and the CSS border-triangle fallback renders as blocks in this theme.
    up_arrow = lucide_arrow_url("chevron-up", t.text_secondary)
    down_arrow = lucide_arrow_url("chevron-down", t.text_secondary)
    up_arrow_hi = lucide_arrow_url("chevron-up", t.text_primary)
    down_arrow_hi = lucide_arrow_url("chevron-down", t.text_primary)
    up_arrow_off = lucide_arrow_url("chevron-up", t.border)
    down_arrow_off = lucide_arrow_url("chevron-down", t.border)
    return f"""
    QWidget {{ background: {t.surface_0}; color: {t.text_primary}; }}
    QMainWindow, QDialog {{ background: {t.surface_0}; }}
    QToolBar {{ background: {t.surface_1}; border: none; padding: 6px; spacing: 6px; }}
    QToolBar QToolButton {{ color: {t.text_secondary}; padding: 6px 10px; border-radius: 6px; background: transparent; border: none; }}
    QToolBar QToolButton:hover {{ background: {t.surface_hover}; color: {t.text_primary}; }}
    QToolBar QToolButton:checked {{ background: {t.accent}; color: #FFFFFF; }}
    QSplitter::handle {{ background: {t.border}; }}
    #Panel {{ background: {t.surface_1}; }}
    #Stage {{ background: {t.surface_0}; }}
    QStatusBar {{ background: {t.surface_1}; color: {t.text_secondary}; }}
    QStatusBar::item {{ border: none; }}
    QLabel {{ background: transparent; color: {t.text_primary}; }}
    QLabel#Hint {{ color: {t.text_muted}; }}
    QLabel#FieldHead {{ color: #0f848a; font-weight: 500; }}
    #PanelDivider {{ border: none; background: {t.border}; max-height: 1px; min-height: 1px; margin: 6px 0; }}
    QLabel#SectionLabel {{ color: {t.text_primary}; font-weight: 600; }}
    QLabel#CountStatus {{ color: {t.text_secondary}; }}

    QPushButton {{ background: {t.surface_3}; color: {t.text_primary}; border: 1px solid {t.border_strong}; border-radius: 6px; padding: 6px 14px; font-weight: 500; }}
    QPushButton:hover {{ background: {t.surface_hover}; border-color: {t.border_strong_hover}; }}
    QPushButton:pressed {{ background: {t.surface_1}; }}
    QPushButton:disabled {{ background: {t.surface_2}; border-color: {t.border}; color: {t.text_disabled}; }}
    QPushButton#Primary {{ background: {t.accent}; color: #FFFFFF; border: none; font-weight: 600; }}
    QPushButton#Primary:hover {{ background: {t.accent_hover}; }}
    QPushButton#Primary:pressed {{ background: {t.accent_pressed}; color: #DCE8FF; }}
    QPushButton#Primary:disabled {{ background: {t.surface_2}; color: {t.text_disabled}; }}
    QPushButton#Danger {{ background: transparent; border: 1px solid #4A3437; color: {t.error}; }}
    QPushButton#Danger:hover {{ background: rgba(229,89,75,0.12); }}

    QToolButton {{ background: {t.surface_2}; color: {t.text_secondary}; border: 1px solid {t.border}; border-radius: 6px; padding: 4px; }}
    QToolButton:hover {{ background: {t.surface_hover}; border-color: {t.border_strong_hover}; color: {t.text_primary}; }}
    QToolButton:checked {{ background: {t.accent}; color: #FFFFFF; border-color: {t.accent}; }}
    /* Checkable QPushButtons (Snap, Crop, Mute section) had no checked state at
       all, so they looked identical on and off. */
    QPushButton:checked {{ background: {t.accent}; color: #FFFFFF; border-color: {t.accent}; }}
    QPushButton:checked:hover {{ background: {t.accent_hover}; border-color: {t.accent_hover}; }}

    QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {t.surface_2}; color: {t.text_primary};
        border: 1px solid {t.border_strong}; border-radius: 6px; padding: 5px 10px;
        selection-background-color: {t.accent_subtle}; selection-color: {t.text_primary};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {t.accent}; }}

    /* Giving QSpinBox a border and background drops Qt's native step buttons, and
       the fallback arrows render blank on a dark surface. Draw them with the CSS
       triangle trick so every spinbox in the app has visible steppers. */
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border; subcontrol-position: top right;
        width: 18px; border-left: 1px solid {t.border};
        border-top-right-radius: 6px; background: {t.surface_2};
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border; subcontrol-position: bottom right;
        width: 18px; border-left: 1px solid {t.border};
        border-bottom-right-radius: 6px; background: {t.surface_2};
    }}
    QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
    QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
        background: {t.surface_hover};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url({up_arrow}); width: 11px; height: 11px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url({down_arrow}); width: 11px; height: 11px;
    }}
    QSpinBox::up-arrow:hover, QDoubleSpinBox::up-arrow:hover {{ image: url({up_arrow_hi}); }}
    QSpinBox::down-arrow:hover, QDoubleSpinBox::down-arrow:hover {{ image: url({down_arrow_hi}); }}
    QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{ image: url({up_arrow_off}); }}
    QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{ image: url({down_arrow_off}); }}
    QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled {{ background: {t.surface_1}; border-color: {t.border}; color: {t.text_disabled}; }}
    QComboBox QAbstractItemView {{ background: {t.surface_2}; color: {t.text_primary}; border: 1px solid {t.border_strong}; selection-background-color: {t.accent_subtle}; selection-color: {t.text_primary}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}

    QCheckBox {{ background: transparent; color: {t.text_primary}; spacing: 8px; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {t.border_strong}; border-radius: 4px; background: {t.surface_2}; }}
    QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
    QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}

    QRadioButton {{ background: transparent; color: {t.text_primary}; spacing: 8px; }}
    QRadioButton::indicator {{ width: 16px; height: 16px; border: 1px solid {t.border_strong}; border-radius: 9px; background: {t.surface_2}; }}
    QRadioButton::indicator:hover {{ border-color: {t.accent}; }}
    QRadioButton::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}

    QTabWidget::pane {{ border: none; background: {t.surface_1}; }}
    QTabBar::tab {{ background: transparent; color: {t.text_muted}; padding: 6px 14px; border: none; border-radius: 4px; margin: 2px; }}
    QTabBar::tab:selected {{ background: {t.surface_3}; color: {t.text_primary}; }}
    QTabBar::tab:hover:!selected {{ color: {t.text_secondary}; }}

    QListWidget {{ background: {t.surface_1}; border: none; }}
    QListWidget::item {{ padding: 3px 4px; border-radius: 6px; }}
    QListWidget::item:selected {{ background: {t.accent_subtle}; color: {t.text_primary}; }}
    QScrollArea {{ background: transparent; border: none; }}

    #GuidanceBox {{ background: {t.surface_2}; color: {t.text_primary}; border: 1px solid {t.border_strong}; border-radius: 6px; }}
    #GuidanceBoxRO {{ background: {t.surface_1}; color: {t.text_secondary}; border: 1px solid {t.border}; border-radius: 6px; }}
    #ElementRow {{ background: {t.surface_2}; border: 1px solid {t.border}; border-radius: 6px; }}
    #TypePill {{ background: {t.surface_3}; color: {t.text_secondary}; border-radius: 7px; padding: 1px 0; font-size: 10px; }}
    #ElementRow QToolButton {{ background: transparent; border: none; color: {t.text_secondary}; font-size: 11px; }}
    #ElementRow QToolButton:hover {{ color: {t.text_primary}; }}
    #ExpandBtn {{ background: transparent; border: 1px solid {t.border_strong}; border-radius: 4px; }}
    #ExpandBtn:hover {{ border-color: {t.accent}; }}

    #CustomPill {{ background: {t.accent_subtle}; border: 1px solid {t.accent_subtle_border}; border-radius: 13px; }}
    #GrayPill {{ background: {t.surface_2}; border: 1px solid {t.border_strong}; border-radius: 13px; }}
    #PillText {{ background: transparent; border: none; color: {t.text_primary}; padding: 3px 8px; }}
    #PillText:hover {{ color: {t.accent_on_subtle}; }}
    #PillX {{ background: transparent; border: none; color: {t.text_muted}; padding-right: 4px; }}
    #PillX:hover {{ color: {t.error}; }}
    #TriggerDel {{ background: {t.surface_3}; border: 1px solid {t.border_strong}; border-radius: 7px; color: {t.text_secondary}; font-weight: 700; padding: 0; }}
    #TriggerDel:hover {{ background: {t.error}; border-color: {t.error}; color: #FFFFFF; }}
    #UsedPill {{ background: {t.accent_subtle}; border: 1px solid {t.accent_subtle_border}; border-radius: 12px; color: {t.accent_on_subtle}; padding: 3px 10px; }}

    #Rail {{ background: {t.surface_1}; border-right: 1px solid {t.border}; }}
    #RailButton {{ background: transparent; border: none; border-radius: 8px; }}
    #RailButton:hover {{ background: {t.surface_hover}; }}
    #RailButton:checked {{ background: {t.accent_subtle}; }}
    #TopBar {{ background: {t.surface_1}; border-bottom: 1px solid {t.border}; }}
    #TitleLabel {{ color: {t.text_secondary}; }}
    #ToolStrip {{ background: {t.surface_2}; border: 1px solid {t.border_strong}; border-radius: 12px; }}
    #ToolStrip QToolButton {{ background: transparent; border: none; border-radius: 8px; }}
    #ToolStrip QToolButton:hover {{ background: {t.surface_hover}; }}
    #ToolStrip QToolButton:checked {{ background: {t.accent_subtle}; }}
    #NavBar {{ background: {t.surface_1}; border-top: 1px solid {t.border}; }}
    #NavPill {{ background: {t.surface_2}; border: 1px solid {t.border_strong}; border-radius: 14px; }}
    #NavPill QToolButton#NavBtn {{ background: transparent; border: none; border-radius: 10px; padding: 2px; }}
    #NavPill QToolButton#NavBtn:hover {{ background: {t.surface_hover}; }}
    #NavCount {{ color: {t.text_secondary}; }}
    #JsonTab {{ background: {t.surface_1}; border-left: 1px solid {t.border}; }}
    #JsonTab:hover {{ background: {t.surface_hover}; }}
    #JsonSlideOver {{ background: {t.surface_1}; border-left: 1px solid {t.border_strong}; }}
    #PanelGhost {{ background: transparent; border: none; }}
    #CollapseChevron {{ background: transparent; border: none; border-radius: 6px; padding: 2px; }}
    #CollapseChevron:hover {{ background: {t.surface_hover}; }}

    QToolTip {{ background: {t.tooltip_bg}; color: {t.tooltip_text}; border: 1px solid {t.border_strong_hover}; border-radius: 6px; padding: 6px 10px; }}
    QProgressBar {{ background: {t.surface_2}; border: none; border-radius: 6px; }}
    QProgressBar::chunk {{ background: {t.accent}; border-radius: 6px; }}

    QScrollBar:horizontal {{ height: 12px; background: {t.surface_0}; margin: 0; border: none; }}
    QScrollBar:vertical {{ width: 12px; background: {t.surface_0}; margin: 0; border: none; }}
    QScrollBar::handle:horizontal {{ background: {t.accent}; min-width: 28px; border-radius: 5px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {t.accent}; min-height: 28px; border-radius: 5px; margin: 2px; }}
    QScrollBar::handle:hover {{ background: {t.accent_hover}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; background: none; border: none; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
    """


HANDLES = ("nw", "n", "ne", "e", "se", "s", "sw", "w")
BOX_COLOR = "#e8a13c"
HANDLE_COLOR = "#3d7bf2"
MIN_BOX_PX = 2.0

# Phase 3 motion — shared timing so animated widgets feel consistent.
# Fast = small state flips (toggle knob); Med = larger surfaces (slide-over,
# panel collapse). Enter eases with OutCubic, exits use a quicker curve.
MOTION_FAST = 140
MOTION_MED = 180

# Distinct per-box colors (cycled by element index) so every box is
# individually identifiable on the canvas. Spec "box palette" — not accent blue.
BOX_PALETTE = (
    "#E8A13C",  # amber
    "#2FC6B3",  # teal
    "#E5594B",  # red
    "#B07CF0",  # violet
    "#5BC85B",  # green
    "#F06FB0",  # pink
    "#E8D44C",  # yellow
    "#4FB0E0",  # cyan
)


def box_color_for(index: int) -> str:
    return BOX_PALETTE[index % len(BOX_PALETTE)]


GUIDANCE_MODES = ("Inherit", "Faithful", "Creative")
MODE_TO_CREATIVE = {"Inherit": None, "Faithful": False, "Creative": True}
CREATIVE_TO_MODE = {None: "Inherit", False: "Faithful", True: "Creative"}

# Guidance presets. These built-ins live in code (always current); the presets
# file stores only user-added presets, which are merged in after these.
_PRESET_ART_STYLE_IDEOGRAM = (
    "For the high_level_description section append a suffix of  in the style of "
    "my_art_style.\n"
    "For the art_style prepend my_art_style,  in front of the regular art style "
    "description."
)
# The plain-text pair: same intent, no bounding boxes. Boxes only exist in the
# Ideogram 4 schema, so asking for one in a .txt caption is an instruction the
# model can't act on.
_PRESET_SINGLE_CHARACTER = (
    "Describe the image, the character is named:\n"
    "\n"
    "MyKnownCharacter\n"
    "\n"
    "use these triggers words exactly with no spaces instead of the full names.\n"
    "\n"
    "Do not describe the known character's features (eye color, hair color, skin "
    "color) but add a short description of their pose."
)
_PRESET_SINGLE_CHARACTER_IDEOGRAM = (
    "Describe the image, the character is named:\n"
    "\n"
    "MyKnownCharacter\n"
    "\n"
    "use these triggers words exactly with no spaces instead of the full names.\n"
    "\n"
    "Do not describe the known character's features (eye color, hair color, skin "
    "color) but add a bounding box for them with a short description of their pose."
)
_PRESET_MULTI_CHARACTER = (
    "Describe the image, from left to right the characters are:\n"
    "\n"
    "MyKnownCharacter,\n"
    "man,\n"
    "woman,\n"
    "MyOtherKnownCharacter\n"
    "\n"
    "use these triggers words exactly with no spaces instead of the full names.\n"
    "\n"
    "Do not describe their outfits.\n"
    "Do not describe the known character's features (eye color, hair color, skin "
    "color) but add a short description of their pose."
)
_PRESET_MULTI_CHARACTER_IDEOGRAM = (
    "Describe the image, from left to right the characters are:\n"
    "\n"
    "MyKnownCharacter,\n"
    "man,\n"
    "woman,\n"
    "MyOtherKnownCharacter\n"
    "\n"
    "use these triggers words exactly with no spaces instead of the full names.\n"
    "\n"
    "Do not describe their outfits.\n"
    "Do not describe the known character's features (eye color, hair color, skin "
    "color) but add a bounding box for them with a short description of their pose."
)

# Plain-text presets first, Ideogram 4 ones last: boxes are the special case now,
# not the default.
FOLDER_GUIDANCE_PRESETS: list[tuple[str, str]] = [
    ("Single Character", _PRESET_SINGLE_CHARACTER),
    ("Art Style (Ideogram 4)", _PRESET_ART_STYLE_IDEOGRAM),
    ("Single Character (Ideogram 4)", _PRESET_SINGLE_CHARACTER_IDEOGRAM),
]
IMAGE_GUIDANCE_PRESETS: list[tuple[str, str]] = [
    ("Multi-Character", _PRESET_MULTI_CHARACTER),
    ("Multi-Character (Ideogram 4)", _PRESET_MULTI_CHARACTER_IDEOGRAM),
]

GUIDANCE_PRESETS_FILENAME = "captioner_guidance_presets.json"

# Folder-wide tag palette (persists per dataset, in .captioner/). The general
# tags are always available as gray pills beneath any user-added custom ones.
FOLDER_TAGS_FILENAME = "captioner_tags.json"
GENERAL_TAGS = ("man", "woman", "person")

UNSAVED_GLOW = "#ff3b30"  # red glow on filmstrip thumbnails with unsaved edits
UNSAVED_ROLE = int(Qt.UserRole) + 1  # per-item flag: has uncommitted edits
STALE_ROLE = int(Qt.UserRole) + 2    # per-item flag: guidance changed since last caption
STALE_COLOR = "#A78BFA"              # violet — "guidance changed since last run"
REVIEW_ROLE = int(Qt.UserRole) + 3   # per-item flag: caption failed a health check (corrupt/off-schema)
# Orange, not the accent: the trim brackets already use the accent, and a playhead
# in the same colour was hard to pick out from the handle it was sitting next to.
PLAYHEAD_COLOR = "#FF8A3D"
MUTE_COLOR = "#E24B4A"               # red — the span whose audio will be silenced
REVIEW_COLOR = "#E24B4A"             # red — "needs review: caption may be corrupt"
FLAG_ROLE = int(Qt.UserRole) + 4     # per-item flag: user manually flagged for review
FLAG_COLOR = "#E5484D"               # red flag — "you flagged this for manual review"
OMIT_ROLE = int(Qt.UserRole) + 5     # per-item flag: convert mode on but this image's .txt is omitted
DURATION_ROLE = int(Qt.UserRole) + 6  # per-item str: video duration badge ('0:05'); None for images
BYPASS_ROLE = int(Qt.UserRole) + 9      # per-item flag: file is in .bypass/
SEPARATOR_ROLE = int(Qt.UserRole) + 10  # per-item flag: the bypass divider
VIDEO_EDIT_ROLE = int(Qt.UserRole) + 8  # per-item flag: clip has unapplied edits
SPEC_ROLE = int(Qt.UserRole) + 7     # per-item flag: clip doesn't meet the target's specs
SPEC_COLOR = "#E0A33B"               # amber — "won't train as-is on the selected model"
OMIT_COLOR = "#A78BFA"               # violet (guidance family) — "source .txt omitted for this image"

SERVER_PING_INTERVAL_MS = 2000  # how often the background monitor re-checks the server
RESOURCE_SAMPLE_INTERVAL_MS = 2000  # how often the status-bar resource readout refreshes
SERVER_PING_TIMEOUT = 1.0  # per-check network timeout (short so the loop stays responsive)


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


class BBoxItem(QGraphicsRectItem):
    """A draggable, resizable bounding box bound to one element index.

    Geometry lives in scene (image-pixel) coordinates: the item is positioned
    at the box top-left and its local rect runs (0,0)-(w,h). Conversion to the
    schema's 0-1000 space is handled by the controller (MainWindow).
    """

    def __init__(self, scene_rect: QRectF, element_index: int, controller, color: str = BOX_COLOR) -> None:
        super().__init__(0, 0, scene_rect.width(), scene_rect.height())
        self.setPos(scene_rect.topLeft())
        self.element_index = element_index
        self.controller = controller
        self.color = QColor(color)
        self.label = ""
        self._resize_handle: str | None = None
        self.setFlags(
            QGraphicsItem.ItemIsMovable
            | QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        pen = QPen(self.color)
        pen.setCosmetic(True)
        pen.setWidth(2)
        self.setPen(pen)
        self._update_z()

    def set_label(self, text: str) -> None:
        self.label = text
        self.update()

    def set_scene_rect(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        self.setPos(rect.topLeft())
        self.setRect(0, 0, rect.width(), rect.height())
        self._update_z()
        self.update()

    def _update_z(self) -> None:
        # Smaller boxes sit on top, so a box nested inside a larger one is always
        # reachable instead of being swallowed by the bigger box's hit area.
        r = self.rect()
        area = max(1.0, r.width() * r.height())
        self.setZValue(10.0 + min(4000.0, 1.0e7 / area))

    def shape(self) -> QPainterPath:
        # Default QGraphicsRectItem.shape() is just the rect, so the label pill
        # (drawn above the box) isn't clickable. Include the pill — and the resize
        # handles when selected — so both are hittable.
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        path.addRect(self.rect())
        pill = self._pill_geom()
        if pill is not None:
            path.addRect(pill[2])
        if self.isSelected():
            for hr in self._handle_rects(self._hit_size()).values():
                path.addRect(hr)
        return path

    def header_scene_rect(self) -> QRectF | None:
        """The label-pill rectangle in scene coordinates, or None if unlabeled."""
        pill = self._pill_geom()
        if pill is None:
            return None
        return self.mapRectToScene(pill[2])

    def _scale(self) -> float:
        if self.scene() and self.scene().views():
            m = self.scene().views()[0].transform().m11()
            if m:
                return abs(m)
        return 1.0

    def _hit_size(self) -> float:
        return 12.0 / self._scale()

    def _draw_size(self) -> float:
        return 7.0 / self._scale()

    def _handle_points(self) -> dict:
        r = self.rect()
        cx, cy = r.center().x(), r.center().y()
        return {
            "nw": (r.left(), r.top()), "n": (cx, r.top()), "ne": (r.right(), r.top()),
            "e": (r.right(), cy), "se": (r.right(), r.bottom()), "s": (cx, r.bottom()),
            "sw": (r.left(), r.bottom()), "w": (r.left(), cy),
        }

    def _handle_rects(self, size: float) -> dict:
        half = size / 2.0
        return {k: QRectF(x - half, y - half, size, size) for k, (x, y) in self._handle_points().items()}

    def _pill_geom(self):
        """Returns (font, text, QRectF) for the top-left label pill, or None."""
        if not self.label:
            return None
        scale = self._scale()
        font = QFont()
        font.setPixelSize(max(1, int(round(11.0 / scale))))
        fm = QFontMetrics(font)
        pad_x = 6.0 / scale
        cap = max(self.rect().width(), 90.0 / scale)
        text = fm.elidedText(self.label, Qt.ElideRight, int(max(10.0, cap - 2 * pad_x)))
        pill_w = fm.horizontalAdvance(text) + 2 * pad_x
        pill_h = fm.height() + 6.0 / scale
        r = self.rect()
        rect = QRectF(r.left(), r.top() - pill_h, pill_w, pill_h)
        return font, text, rect

    def boundingRect(self) -> QRectF:
        m = self._hit_size()
        rect = self.rect().adjusted(-m, -m, m, m)
        pill = self._pill_geom()
        if pill is not None:
            rect = rect.united(pill[2].adjusted(-1, -1, 1, 1))
        return rect

    def paint(self, painter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        r = self.rect()
        scale = self._scale()
        selected = self.isSelected()
        radius = 2.0 / scale

        painter.save()
        if not selected:
            painter.setOpacity(0.70)
        painter.setBrush(Qt.NoBrush)
        # contrast outline under the colored border (keeps the box visible on light images)
        contrast = QPen(QColor(0, 0, 0, 115))
        contrast.setCosmetic(True)
        contrast.setWidthF(3.0)
        painter.setPen(contrast)
        painter.drawRoundedRect(r, radius, radius)
        # element-colored border
        border = QPen(self.color)
        border.setCosmetic(True)
        border.setWidthF(1.5)
        painter.setPen(border)
        painter.drawRoundedRect(r, radius, radius)
        # top-left label pill (radius 3 3 3 0 — square bottom-left so it tucks into the corner)
        pill = self._pill_geom()
        if pill is not None:
            font, text, prect = pill
            rad = 3.0 / scale
            path = QPainterPath()
            path.moveTo(prect.left(), prect.bottom())
            path.lineTo(prect.left(), prect.top() + rad)
            path.quadTo(prect.left(), prect.top(), prect.left() + rad, prect.top())
            path.lineTo(prect.right() - rad, prect.top())
            path.quadTo(prect.right(), prect.top(), prect.right(), prect.top() + rad)
            path.lineTo(prect.right(), prect.bottom() - rad)
            path.quadTo(prect.right(), prect.bottom(), prect.right() - rad, prect.bottom())
            path.closeSubpath()
            painter.setPen(Qt.NoPen)
            painter.fillPath(path, QBrush(self.color))
            painter.setFont(font)
            painter.setPen(QColor("#15171A"))
            painter.drawText(prect.adjusted(6.0 / scale, 0, -2.0 / scale, 0),
                             int(Qt.AlignVCenter | Qt.AlignLeft), text)
        painter.restore()

        # resize handles: selected only, accent fill + white border, at full opacity
        if selected:
            accent = QColor(getattr(self.controller, "theme", None).accent
                            if getattr(self.controller, "theme", None) else HANDLE_COLOR)
            pen = QPen(QColor("#ffffff"))
            pen.setCosmetic(True)
            pen.setWidthF(1.0)
            painter.setPen(pen)
            painter.setBrush(accent)
            for hr in self._handle_rects(self._draw_size()).values():
                painter.drawRect(hr)

    def _handle_at(self, pos: QPointF) -> str | None:
        for key, hr in self._handle_rects(self._hit_size()).items():
            if hr.contains(pos):
                return key
        return None

    def mousePressEvent(self, event) -> None:
        self.controller.on_box_pressed(self)
        # Read-only (job in progress): allow selecting to view, but no resize. Moves are
        # already prevented by clearing ItemIsMovable in the controller's canvas lock.
        if not getattr(self.controller, "_read_only", False):
            handle = self._handle_at(event.pos())
            if handle and self.isSelected():
                self._resize_handle = handle
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._resize_handle:
            self._resize_to(event.scenePos())
            self.controller.on_box_geometry_live(self)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        resizing = self._resize_handle is not None
        self._resize_handle = None
        super().mouseReleaseEvent(event)
        if resizing:
            self.controller.on_box_geometry_live(self)

    def _resize_to(self, scene_pos: QPointF) -> None:
        sr = self.scene().sceneRect()
        cur = self.mapRectToScene(self.rect())
        left, top, right, bottom = cur.left(), cur.top(), cur.right(), cur.bottom()
        x = _clamp(scene_pos.x(), 0, sr.width())
        y = _clamp(scene_pos.y(), 0, sr.height())
        h = self._resize_handle
        # Each moving edge clamps against the fixed opposite edge: it can approach
        # but never cross it, stopping MIN_BOX_PX short. Midpoints move one edge,
        # corners move two; no edge ever flips past its partner.
        if "n" in h:
            top = min(y, bottom - MIN_BOX_PX)
        if "s" in h:
            bottom = max(y, top + MIN_BOX_PX)
        if "w" in h:
            left = min(x, right - MIN_BOX_PX)
        if "e" in h:
            right = max(x, left + MIN_BOX_PX)
        self.set_scene_rect(QRectF(left, top, right - left, bottom - top))

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange and self.scene():
            sr = self.scene().sceneRect()
            w = self.rect().width()
            h = self.rect().height()
            nx = _clamp(value.x(), 0, max(0, sr.width() - w))
            ny = _clamp(value.y(), 0, max(0, sr.height() - h))
            return QPointF(nx, ny)
        if change == QGraphicsItem.ItemPositionHasChanged:
            self.controller.on_box_geometry_live(self)
        return super().itemChange(change, value)


class CanvasView(QGraphicsView):
    """Graphics view hosting the image and boxes, with draw/delete/pan modes."""

    def __init__(self, scene, controller) -> None:
        super().__init__(scene)
        self.controller = controller
        self.mode = "select"
        self._draw_start: QPointF | None = None
        self._draw_item: QGraphicsRectItem | None = None
        self._space_held = False
        self._panning = False
        self._pan_last = QPoint()
        self._header_drag_item = None
        self.setRenderHints(self.renderHints())
        self.setMouseTracking(True)
        # Needed so Space-to-pan key events reach the canvas when it has focus.
        self.setFocusPolicy(Qt.StrongFocus)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        # Manual panning handles the temporary grab gestures; the explicit pan tool
        # still uses Qt's ScrollHandDrag for left-drag.
        self.setDragMode(QGraphicsView.ScrollHandDrag if mode == "pan" else QGraphicsView.NoDrag)
        if not self._panning:
            self._apply_idle_cursor()

    def _apply_idle_cursor(self) -> None:
        if self._space_held or self.mode == "pan":
            self.viewport().setCursor(Qt.OpenHandCursor)
        else:
            self.viewport().setCursor(Qt.ArrowCursor)

    def _should_start_pan(self, event) -> bool:
        return (event.button() == Qt.MiddleButton
                or (event.button() == Qt.LeftButton and self._space_held))

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = True
            if not self._panning:
                self.viewport().setCursor(Qt.OpenHandCursor)
            event.accept()
            return
        # Delete / Backspace removes the selected box (same path as the delete tool).
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.controller.delete_selected_box():
                event.accept()
            return
        # Arrow keys (and WASD) nudge the selected box (Shift = ×10). Falls through
        # to the default view behaviour (scroll) when no box is selected.
        arrows = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0),
                  Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1),
                  Qt.Key_A: (-1, 0), Qt.Key_D: (1, 0),
                  Qt.Key_W: (0, -1), Qt.Key_S: (0, 1)}
        if event.key() in arrows:
            step = 10 if (event.modifiers() & Qt.ShiftModifier) else 1
            ux, uy = arrows[event.key()]
            if self.controller.nudge_selected_box(ux * step, uy * step):
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_held = False
            if not self._panning:
                self._apply_idle_cursor()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _pan_by(self, delta) -> None:
        h = self.horizontalScrollBar()
        v = self.verticalScrollBar()
        h.setValue(h.value() - delta.x())
        v.setValue(v.value() - delta.y())

    def wheelEvent(self, event) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)
        self.controller._user_zoomed = True
        update_zoom = getattr(self.controller, "_update_zoom_label", None)
        if update_zoom is not None:
            update_zoom()
        event.accept()

    def _scene_pos(self, event) -> QPointF:
        return self.mapToScene(event.position().toPoint())

    def mousePressEvent(self, event) -> None:
        # Temporary pan (middle-button, or Space-held + left-drag) wins over every
        # mode so it never collides with selecting/moving boxes or drawing.
        if self._should_start_pan(event):
            self._panning = True
            self._pan_last = event.position().toPoint()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self.mode == "draw" and event.button() == Qt.LeftButton and not getattr(self.controller, "_read_only", False):
            self._draw_start = self._scene_pos(event)
            self._draw_item = QGraphicsRectItem(QRectF(self._draw_start, self._draw_start))
            accent = getattr(self.controller, "theme", None)
            pen = QPen(QColor(accent.accent if accent else HANDLE_COLOR))
            pen.setCosmetic(True)
            pen.setWidth(2)
            self._draw_item.setPen(pen)
            self.scene().addItem(self._draw_item)
            event.accept()
            return
        if self.mode == "delete" and event.button() == Qt.LeftButton:
            # One-shot action on the current selection (not a hit-test at the click,
            # which would grab a larger overlapping box). Delete the selected box,
            # then drop back to the select tool so the next click selects again.
            self.controller.delete_selected_box()
            revert = getattr(self.controller, "_activate_tool", None)
            if revert is not None:
                revert("select")
            event.accept()
            return
        # A click on a box's header pill should focus AND let you drag that box,
        # even when it sits under a larger box. Raise the header's box so the press
        # lands on it, then fall through to the normal handler (select + move).
        if self.mode == "select" and event.button() == Qt.LeftButton:
            sp = self._scene_pos(event)
            hit = None
            for it in getattr(self.controller, "box_items", []):
                hr = it.header_scene_rect()
                if hr is not None and hr.contains(sp) and (hit is None or it.zValue() >= hit.zValue()):
                    hit = it
            if hit is not None:
                self._header_drag_item = hit
                hit.setZValue(100000.0)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            pos = event.position().toPoint()
            self._pan_by(pos - self._pan_last)
            self._pan_last = pos
            event.accept()
            return
        if self._draw_item is not None:
            rect = QRectF(self._draw_start, self._scene_pos(event)).normalized()
            self._draw_item.setRect(rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._header_drag_item is not None:
            # Restore area-based stacking so small boxes stay reachable on top.
            self._header_drag_item._update_z()
            self._header_drag_item = None
        if self._panning and event.button() in (Qt.LeftButton, Qt.MiddleButton):
            self._panning = False
            self._apply_idle_cursor()
            event.accept()
            return
        if self._draw_item is not None:
            rect = self._draw_item.rect()
            self.scene().removeItem(self._draw_item)
            self._draw_item = None
            self._draw_start = None
            self.controller.apply_drawn_box(rect)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        reposition = getattr(self.controller, "_reposition_toolstrip", None)
        if reposition is not None:
            reposition()


class AiJobThread(QThread):
    """Runs one AI operation off the UI thread.

    Operations: 'json_image' (generate, then optional bbox pass), 'refine',
    'bboxes', and 'plain' (free-text caption for plain/H3 presets). Progress
    strings and the result/error come back as signals so the main thread can update
    widgets safely.
    """

    progress = Signal(str)
    done = Signal(object)
    error = Signal(str)
    server_started = Signal(object)

    def __init__(self, operation, settings, image_path, caption, guidance,
                 source_caption, instructions, system_prompt=""):
        super().__init__()
        self.operation = operation
        # Plain presets supply their own system prompt (plain text, H3 natural,
        # H3 official) — the operation is the same, only the instructions differ.
        self.system_prompt = system_prompt
        # For 'plain_video': caption this span of the clip (the user's trim).
        self.span: tuple[float, float] | None = None
        self.frame_count = 6
        self.include_audio = False
        self.settings = settings
        self.image_path = image_path
        self.caption = caption
        self.guidance = guidance
        self.source_caption = source_caption
        self.instructions = instructions

    def run(self) -> None:
        def prog(msg: str) -> None:
            self.progress.emit(msg)

        try:
            # In local mode this downloads the model (first time) and launches
            # llama-server; a no-op for existing/custom servers or if one is up.
            task = "bbox" if self.operation == "bboxes" else "caption"
            proc = ensure_server_running(self.settings, task, progress=prog)
            if proc is not None:
                self.server_started.emit(proc)
            op = self.operation
            if op == "json_image":
                caption = generate_json_from_image(
                    self.settings, self.image_path, progress=prog,
                    guidance=self.guidance, source_caption=self.source_caption,
                )
                if self.settings.add_bboxes_after_json and not self.isInterruptionRequested():
                    caption, _att, _add, _reasons = add_bboxes_to_caption(
                        self.settings, self.image_path, caption, progress=prog
                    )
            elif op == "refine":
                caption = generate_json_refinement(
                    self.settings,
                    self.image_path,
                    self.caption,
                    self.source_caption,
                    self.instructions,
                    progress=prog,
                )
            elif op == "bboxes":
                caption, _att, _add, _reasons = add_bboxes_to_caption(
                    self.settings, self.image_path, self.caption, progress=prog
                )
            elif op == "plain":
                caption = caption_image_plain(
                    self.settings, self.image_path, self.system_prompt,
                    guidance=self.guidance, progress=prog,
                )
            elif op == "plain_video":
                start_s, end_s = self.span if self.span else (0.0, None)
                caption = caption_video_plain(
                    self.settings, self.image_path, self.system_prompt,
                    guidance=self.guidance, start_s=start_s, end_s=end_s,
                    frame_count=self.frame_count,
                    include_audio=self.include_audio, progress=prog,
                )
            else:
                raise AutoCaptionError(f"Unknown operation: {op}")
            self.done.emit(caption)
        except Exception as exc:  # surfaced to the UI as a readable message
            self.error.emit(str(exc))


class BatchCaptionThread(QThread):
    """Captions every image in the folder sequentially.

    Local servers (e.g. LM Studio) handle one request at a time, so the folder
    is processed image-by-image rather than as a single batched call. Each image
    runs the same path as the single-image job (generate JSON, then the optional
    bbox pass). Results are emitted one at a time so the main thread can save and
    update markers; cancellation is honoured between images.
    """

    item_progress = Signal(int, int, str)   # index (1-based), total, message
    item_done = Signal(str, object)         # image path str, caption
    item_error = Signal(str, str)           # image path str, error message
    batch_finished = Signal(int, int, bool)  # success, fail, cancelled
    server_started = Signal(object)         # launched llama-server process

    def __init__(self, settings, items, delay_ms: int = 0, system_prompt: str = ""):
        super().__init__()
        self.settings = settings
        self.items = items  # list of (Path, guidance, source_caption)
        self.delay_ms = max(0, int(delay_ms))
        # Non-empty for plain presets: batch then produces free text with this
        # prompt instead of the Ideogram JSON structure.
        self.system_prompt = system_prompt
        self.frame_count = 6
        # Defaulted here as well as set by the caller: without it, constructing the
        # thread directly raises AttributeError mid-batch on the first video.
        self.include_audio = False

    def _interruptible_sleep(self, ms: int) -> None:
        waited = 0
        while waited < ms and not self.isInterruptionRequested():
            step = min(50, ms - waited)
            self.msleep(step)
            waited += step

    def run(self) -> None:
        success = 0
        fail = 0
        cancelled = False
        total = len(self.items)
        # Bring up a local server once for the whole run (download + launch on the
        # first run only). If this fails, fail the batch cleanly rather than erroring
        # on every image.
        try:
            proc = ensure_server_running(
                self.settings, "caption",
                progress=lambda m: self.item_progress.emit(0, total, m),
            )
            if proc is not None:
                self.server_started.emit(proc)
        except Exception as exc:
            self.item_error.emit("", f"Could not start the server: {exc}")
            self.batch_finished.emit(0, 0, False)
            return
        for i, (image_path, guidance, source_caption) in enumerate(self.items, start=1):
            if self.isInterruptionRequested():
                cancelled = True
                break
            self.item_progress.emit(i, total, f"Captioning {i}/{total}: {image_path.name}")

            def prog(msg: str, _i=i, _t=total) -> None:
                self.item_progress.emit(_i, _t, f"[{_i}/{_t}] {msg}")

            try:
                if self.system_prompt and is_video(image_path):
                    # Batch has no per-clip trim; the whole clip is captioned. Trim
                    # first (Apply edit) if only part of a clip should ship.
                    caption = caption_video_plain(
                        self.settings, image_path, self.system_prompt,
                        guidance=guidance, frame_count=self.frame_count,
                        include_audio=self.include_audio, progress=prog,
                    )
                elif self.system_prompt:
                    caption = caption_image_plain(
                        self.settings, image_path, self.system_prompt,
                        guidance=guidance, progress=prog,
                    )
                else:
                    caption = generate_json_from_image(
                        self.settings, image_path, progress=prog,
                        guidance=guidance, source_caption=source_caption,
                    )
                    if (self.settings.add_bboxes_after_json
                            and not self.isInterruptionRequested()):
                        caption, _att, _add, _reasons = add_bboxes_to_caption(
                            self.settings, image_path, caption, progress=prog
                        )
                self.item_done.emit(str(image_path), caption)
                success += 1
            except Exception as exc:
                self.item_error.emit(str(image_path), str(exc))
                fail += 1
            # Optional breather between images (also a clean cancellation checkpoint).
            if self.delay_ms and i < total:
                self._interruptible_sleep(self.delay_ms)
        self.batch_finished.emit(success, fail, cancelled)


class ClickableLabel(QLabel):
    """A QLabel that emits `clicked` on left-press — used for the status-bar server
    indicator so it can open the connection settings."""

    clicked = Signal()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class LlamaInstallThread(QThread):
    """Downloads + verifies + installs a llama.cpp build off the GUI thread.
    The plan (assets, backend, sm) is resolved on the main thread first."""

    progress = Signal(str)
    done = Signal(object)   # InstalledLlama
    error = Signal(str)

    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def run(self) -> None:
        try:
            record = install_llama_release(
                self.plan.release,
                self.plan.assets,
                backend=self.plan.backend,
                sm=self.plan.sm,
                progress=lambda m: self.progress.emit(m),
            )
            self.done.emit(record)
        except Exception as exc:  # surfaced to the UI as a readable message
            self.error.emit(str(exc))


class LlamaUpdateCheckThread(QThread):
    """Metadata-only 'is there a newer build?' check, off the GUI thread."""

    result = Signal(int)   # latest build number, or -1 on failure

    def __init__(self, repo: str):
        super().__init__()
        self.repo = repo

    def run(self) -> None:
        try:
            info = fetch_release(self.repo, None)
            self.result.emit(info.build if info and info.build else -1)
        except Exception:
            self.result.emit(-1)


class ServerStatusMonitor(QThread):
    """Polls the OpenAI-compatible endpoint on a background thread.

    The check is a network call, so it must never run on the GUI thread. We loop
    sequentially (ping → wait → ping) rather than on a fixed timer, so a slow or
    unreachable server can't stack up overlapping checks.
    """

    status = Signal(bool)

    def __init__(self, base_url: str, api_key: str, parent=None) -> None:
        super().__init__(parent)
        self.base_url = base_url
        self.api_key = api_key
        self.interval_ms = SERVER_PING_INTERVAL_MS
        self.timeout = SERVER_PING_TIMEOUT

    def update_target(self, base_url: str, api_key: str) -> None:
        # str assignment is atomic under the GIL; a one-cycle-stale value is harmless
        self.base_url = base_url
        self.api_key = api_key

    def run(self) -> None:
        while not self.isInterruptionRequested():
            try:
                ok = is_server_ready(self.base_url, self.api_key, timeout=self.timeout)
            except Exception:
                ok = False
            self.status.emit(ok)
            waited = 0
            while waited < self.interval_ms and not self.isInterruptionRequested():
                self.msleep(50)
                waited += 50


class ResourceMonitor(QThread):
    """Samples RAM (+ VRAM/GPU% on NVIDIA) off the GUI thread, ~every 2s, for the
    status-bar readout. Loops sequentially like the server monitor so a slow
    nvidia-smi can't stack up overlapping samples."""

    sampled = Signal(str)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.interval_ms = RESOURCE_SAMPLE_INTERVAL_MS

    def run(self) -> None:
        while not self.isInterruptionRequested():
            try:
                text = format_resources(sample_resources())
            except Exception:
                text = ""
            self.sampled.emit(text)
            waited = 0
            while waited < self.interval_ms and not self.isInterruptionRequested():
                self.msleep(50)
                waited += 50


class LlamaServerThread(QThread):
    """Brings a local llama-server up off the GUI thread (resolve model, launch,
    wait for readiness). Emits the launched process, or None if one was already up."""

    progress = Signal(str)
    started_proc = Signal(object)   # subprocess.Popen | None
    error = Signal(str)

    def __init__(self, settings, model_less: bool = False):
        super().__init__()
        self.settings = settings
        self.model_less = model_less

    def run(self) -> None:
        try:
            proc = ensure_server_running(self.settings, "caption",
                                         progress=lambda m: self.progress.emit(m),
                                         model_less=self.model_less)
            self.started_proc.emit(proc)
        except Exception as exc:
            self.error.emit(str(exc))


class ServerPopover(QWidget):
    """Small popover above the status-bar server indicator: a status line, a
    'Server settings…' link, and — in local mode — a Start/Stop button. Matches
    the filmstrip preview's rounded dark card, with the pointer beneath."""

    _BORDER = "#0f848a"
    _ARROW = 12

    def __init__(self, theme, *, on_settings, on_start, on_stop, on_start_nomodel=None, parent=None) -> None:
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._t = theme
        self._margin = 16
        self._on_start = on_start
        self._on_stop = on_stop
        self._on_start_nomodel = on_start_nomodel
        self._running = False

        self.card = QWidget(self)
        self.card.setObjectName("ServerPopCard")
        self.card.setMinimumWidth(208)
        self.card.setStyleSheet(
            f"#ServerPopCard {{ background: {theme.surface_2};"
            f" border: 1px solid {self._BORDER}; border-radius: 8px; }}"
        )
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(28)
        shadow.setXOffset(0)
        shadow.setYOffset(10)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.card.setGraphicsEffect(shadow)

        lay = QVBoxLayout(self.card)
        lay.setContentsMargins(14, 12, 14, 12)
        lay.setSpacing(9)

        self.status = QLabel(self.card)
        self.status.setTextFormat(Qt.RichText)
        lay.addWidget(self.status)

        self.startstop = QPushButton(self.card)
        self.startstop.setObjectName("Primary")
        self.startstop.setCursor(Qt.PointingHandCursor)
        self.startstop.clicked.connect(self._toggle)
        lay.addWidget(self.startstop)

        self.startnomodel = QPushButton("Start without model", self.card)
        self.startnomodel.setCursor(Qt.PointingHandCursor)
        self.startnomodel.setToolTip("Launch the server with no model loaded, just to check it runs.")
        self.startnomodel.clicked.connect(self._do_nomodel)
        lay.addWidget(self.startnomodel)

        link = QLabel(
            f'<a href="#" style="color:{self._BORDER}; text-decoration:none;">Server settings…</a>',
            self.card,
        )
        link.setCursor(Qt.PointingHandCursor)
        link.linkActivated.connect(lambda _=None: (self.hide(), on_settings()))
        lay.addWidget(link)

        self.card.move(self._margin, self._margin)

    def _toggle(self) -> None:
        self.hide()
        (self._on_stop if self._running else self._on_start)()

    def _do_nomodel(self) -> None:
        self.hide()
        if callable(self._on_start_nomodel):
            self._on_start_nomodel()

    def configure(self, *, status_html: str, show_startstop: bool, running: bool,
                  show_nomodel: bool = False) -> None:
        self.status.setText(status_html)
        self._running = running
        self.startstop.setVisible(show_startstop)
        if show_startstop:
            self.startstop.setText("Stop server" if running else "Start server")
        self.startnomodel.setVisible(show_nomodel)
        self.card.adjustSize()
        self.resize(self.card.width() + 2 * self._margin,
                    self.card.height() + 2 * self._margin + self._ARROW)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        half = self._ARROW / 2
        cx = self.width() / 2
        cy = float(self._margin + self.card.height())   # bottom edge of the card
        top = QPointF(cx, cy - half)
        right = QPointF(cx + half, cy)
        bottom = QPointF(cx, cy + half)
        left = QPointF(cx - half, cy)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self._t.surface_2))
        p.drawPolygon(QPolygonF([top, right, bottom, left]))
        pen = QPen(QColor(self._BORDER))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(right, bottom)
        p.drawLine(bottom, left)

    def show_above(self, anchor) -> None:
        center = anchor.mapToGlobal(QPoint(anchor.width() // 2, 0))
        self.move(center.x() - self.width() // 2, center.y() - self.height())
        self.show()


class PreferencesDialog(QDialog):
    """Settings dialog with left-sidebar category navigation (groups) and the
    matching parameters on the right. Config-driven from the field spec below.
    """

    # Quick presets: (base_url, api_key placeholder, server_start_mode). The keys
    # are ignored by these servers but the OpenAI client needs a non-empty value.
    SERVER_PRESETS = {
        "LM Studio": ("http://localhost:1234/v1", "lm-studio", "existing"),
        "vLLM": ("http://localhost:8000/v1", "vllm", "existing"),
        "Ollama": ("http://localhost:11434/v1", "ollama", "existing"),
        "Local (llama.cpp)": ("http://127.0.0.1:8231/v1", "llama-cpp", "local"),
    }

    GROUPS = (
        ("Connection/Server", (
            (None, "Local server", "_serverpanel", None),
            (None, "Quick preset", "_preset", None),
            (None, "Connection", "_section", None),
            ("base_url", "Server URL", "text", None),
            ("api_key", "API key", "text", None),
            ("hf_token", "Hugging Face token", "password", None),
            (None, "", "_testserver", None),
            (None, "", "_divider", None),
            (None, "Server", "_section", None),
            ("llama_devices", "Detected GPU", "_gpupicker", None),
            (None, "llama.cpp", "_llamastatus", None),
            ("llama_backend_hint", "Backend (auto-detect override)", "choice", ("auto", "cuda", "vulkan", "cpu")),
            ("llama_auto_update_check", "Auto-check for llama.cpp updates", "bool", None),
            ("server_start_mode", "Server start mode", "choice", ("local", "existing", "custom")),
            ("auto_start_server", "Auto-start server", "bool", None),
            ("llama_server_path", "llama-server path (optional)", "browse_file", None),
            ("llama_context", "llama context size", "int", (0, 2000000)),
            ("llama_gpu_layers", "GPU layers (-1 = auto)", "int", (-1, 10000)),
            ("llama_batch", "Batch size", "int", (0, 1000000)),
            ("llama_ubatch", "Micro-batch size", "int", (0, 1000000)),
            ("llama_parallel", "Parallel slots", "int", (1, 64)),
            ("llama_threads", "Threads (0 = auto)", "int", (0, 1024)),
            ("llama_extra_args", "Extra llama args", "text", None),
            ("llama_reasoning_budget", "Reasoning budget", "int", (0, 1000000)),
            ("caption_server_command", "Caption server command", "text", None),
            ("bbox_server_command", "BBox server command", "text", None),
            ("server_startup_timeout", "Startup timeout (s)", "float", (0.0, 100000.0)),
            ("stop_server_after_job", "Stop server after job", "bool", None),
        )),
        ("LLM Models", ()),
        ("Pipeline", (
            ("creative_json", "Creative JSON (off = faithful)", "bool", None),
            ("disable_thinking", "Disable thinking", "bool", None),
            ("add_bboxes_after_json", "Auto-locate boxes after JSON", "bool", None),
            ("overwrite_bboxes", "Overwrite existing boxes", "bool", None),
            ("filter_bbox_targets", "Filter bbox targets", "bool", None),
            ("vision_image_format", "Vision image format", "choice", ("auto", "jpeg", "png")),
            ("max_tokens_caption", "Max tokens — caption", "int", (1, 200000)),
            ("max_tokens_json", "Max tokens — JSON", "int", (1, 200000)),
            ("video_caption_frames", "Video caption frames", "int", (2, 16)),
            ("send_clip_audio", "Send clip audio (audio-capable models)", "bool", None),
            ("max_tokens_bboxes", "Max tokens — bboxes", "int", (1, 200000)),
            ("context_chars", "Context chars", "int", (0, 100000)),
            ("max_targets_per_call", "Max bbox targets / call (0 = all)", "int", (0, 10000)),
            ("json_refine_instructions", "Refine instructions", "multiline", None),
        )),
        ("Tags", ()),
        # Prompts and per-model frame rules. Both are data the app ships defaults
        # for but can't keep current — new video models appear faster than releases
        # do — so they're editable and shareable here.
        ("LLM Instructions", ()),
        ("Appearance", (
            ("ui_font_family", "UI font", "font", None),
            ("mono_font_family", "Monospace font", "font", None),
            ("ui_font_size", "Font size", "int", (6, 72)),
            ("color_accent", "Accent", "color", None),
        )),
    )

    # Hover help for each setting (shown on the label and the field).
    FIELD_HELP = {
        # Connection
        "base_url": "Base URL of the OpenAI-compatible server requests are sent to (e.g. http://localhost:1234/v1).",
        "api_key": "API key sent with each request. Local servers like LM Studio usually accept any value.",
        "hf_token": "Hugging Face access token, used when downloading gated models.",
        "models_dir": "Folder where this app downloads GGUF models (when download location is the app folder). Files already in the Hugging Face cache or your Extra model folders are discovered and reused.",
        "model_download_target": "Where new downloads land. \u201cShared Hugging Face cache\u201d puts them where ai-toolkit and other HF tools read/write, so models are shared across tools.",
        "extra_model_dirs": "Extra read-only folders to scan for already-downloaded GGUFs (e.g. your LM Studio models folder). One per line.",
        # Pipeline
        "creative_json": "On: JSON may interpret and embellish freely. Off (faithful): stays close to what is literally visible.",
        "disable_thinking": "Suppress the model's chain-of-thought, returning only the final answer. Faster / fewer tokens on models that support it.",
        "add_bboxes_after_json": "After generating JSON, automatically run the box-location pass over the described elements.",
        "overwrite_bboxes": "On: the locate pass replaces every element's box. Off: existing boxes are kept and only missing ones are filled.",
        "filter_bbox_targets": "Skip vague or ambient elements (e.g. 'atmosphere', 'background') when locating boxes, so only concrete objects/text get boxed.",
        "vision_image_format": "Image encoding sent to the vision model. 'auto' chooses per image; force jpeg/png if your server prefers one.",
        "max_tokens_caption": "Maximum tokens the model may generate during the prose caption step.",
        "max_tokens_json": "Maximum tokens the model may generate during the JSON step.",
        "send_clip_audio": "When the selected model can hear (an Omni-style profile), "
                           "send the clip's audio with the frames so captions can "
                           "include real dialogue and sound. Ignored by vision-only "
                           "models.",
        "video_caption_frames": "How many frames are sampled across a clip when "
                                "captioning a video. More frames capture motion "
                                "better but cost context; 4–8 is typical.",
        "max_tokens_bboxes": "Maximum tokens the model may generate during the bounding-box step.",
        "context_chars": "How many characters of the existing caption are passed as context when locating boxes. Larger = more context and more tokens.",
        "max_targets_per_call": "Elements sent per box-location request. 0 = all in one call; a small number splits long lists across requests.",
        "json_refine_instructions": "Standing instructions used by the Refine JSON button when adjusting an existing caption.",
        # Server
        "server_start_mode": "How a server is obtained: 'local' launches llama-server, 'existing' connects to one you already run (e.g. LM Studio), 'custom' uses your command.",
        "auto_start_server": "Automatically start the server (per the start mode) when the app launches or a job needs it.",
        "llama_server_path": "Optional. Leave blank to auto-detect the managed install (Get llama.cpp) or a llama-server on your PATH. Set this only to force a specific binary.",
        "llama_context": "Context window (tokens) llama-server is launched with. 0 uses the model/server default.",
        "llama_gpu_layers": "Model layers offloaded to the GPU. -1 = auto: llama.cpp fits as many layers as your free VRAM allows (and spills the rest to CPU) instead of failing if a model is slightly too big. A set value forces exactly that many (0 = CPU-only).",
        "llama_devices": "Which GPU llama.cpp runs on. Pick one (only shown when there's more than one). The chosen GPU's VRAM is what model size recommendations and fit badges are measured against. Detection uses your installed llama.cpp build, so install it first to see non-NVIDIA cards. The captioner uses a single GPU \u2014 it doesn't split models across cards.",
        "llama_batch": "llama-server batch size (prompt tokens processed per pass). 0 = default.",
        "llama_ubatch": "llama-server micro-batch size; the main driver of compute-buffer VRAM. 512 is plenty for captioning.",
        "llama_parallel": "Concurrent request slots (-np). Captioning runs one image at a time, so 1 keeps VRAM lowest; raise it only if you drive the server from elsewhere too.",
        "llama_threads": "CPU threads for llama-server. 0 = auto-detect.",
        "llama_extra_args": "Extra command-line arguments appended when launching llama-server.",
        "llama_reasoning_budget": "Token budget allotted to model reasoning/thinking when supported. 0 = default.",
        "caption_server_command": "Custom command used to start the caption server (custom start mode).",
        "bbox_server_command": "Custom command used to start the bbox server (custom start mode).",
        "server_startup_timeout": "Seconds to wait for a launched server to become ready before giving up.",
        "stop_server_after_job": "Shut the launched server down after a job finishes, freeing VRAM.",
        # Appearance
        "ui_font_family": "Font family for the interface. '(auto)' uses a cross-platform default.",
        "mono_font_family": "Monospace font for JSON and editor text.",
        "ui_font_size": "Base interface font size, in points.",
        "color_accent": "Accent color for highlights, selection, and trigger chips. Other colors follow a fixed theme.",
    }

    def __init__(self, parent, settings, bbox_same_as_caption: bool = False, default_tags=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(960, 640)
        self.setMinimumWidth(920)
        self.settings = settings
        self.result = None
        self.bbox_same_as_caption = bbox_same_as_caption
        self.tags_result = list(default_tags) if default_tags is not None else None
        self.widgets: dict = {}
        self._qsettings = QSettings("FantasticCaptioningKit", "QtApp")
        self._custom_presets = self._load_custom_presets()
        self._preset_combo = None

        outer = QVBoxLayout(self)
        body = QHBoxLayout()
        self.nav = QListWidget()
        self.nav.setFixedWidth(160)
        self.stack = QStackedWidget()
        body.addWidget(self.nav)
        body.addWidget(self.stack, 1)
        outer.addLayout(body, 1)

        families = ["(auto)"] + sorted(set(QFontDatabase.families()))
        for name, fields in self.GROUPS:
            self.nav.addItem(name)
            if name == "LLM Models":
                page = self._build_models_page()
            elif name == "Tags":
                page = self._build_tags_page()
            elif name == "LLM Instructions":
                page = self._build_llm_page()
            else:
                page = QWidget()
                form = QFormLayout(page)
                form.setContentsMargins(14, 14, 14, 14)
                for key, label, kind, extra in fields:
                    if kind == "_preset":
                        combo = QComboBox()
                        combo.setToolTip(
                            "Fill the fields below with the known defaults for a popular "
                            "server. You can still edit them afterward."
                        )
                        combo.textActivated.connect(self._apply_preset)
                        self._preset_combo = combo
                        self._populate_preset_combo()
                        manage = QPushButton("Manage…")
                        manage.setToolTip("Save the current settings as a preset, or delete custom presets.")
                        manage.clicked.connect(self._manage_presets)
                        row = QHBoxLayout()
                        row.setContentsMargins(0, 0, 0, 0)
                        row.addWidget(combo, 1)
                        row.addWidget(manage, 0)
                        holder = QWidget()
                        holder.setLayout(row)
                        form.addRow(QLabel(label), holder)
                        continue
                    if kind == "_section":
                        head = QLabel(label)
                        head.setObjectName("SectionLabel")
                        head.setStyleSheet("margin-top: 4px;")
                        form.addRow(head)
                        continue
                    if kind == "_divider":
                        line = QFrame()
                        line.setObjectName("PanelDivider")
                        line.setFrameShape(QFrame.HLine)
                        form.addRow(line)
                        continue
                    if kind == "_testserver":
                        test_btn = QPushButton("Test server")
                        test_btn.clicked.connect(self._test_server)
                        form.addRow("", test_btn)
                        continue
                    if kind == "_serverpanel":
                        self._srv_panel_label = QLabel()
                        self._srv_panel_btn = QPushButton()
                        self._srv_panel_btn.setObjectName("Primary")
                        self._srv_panel_btn.setCursor(Qt.PointingHandCursor)
                        self._srv_panel_btn.clicked.connect(self._toggle_local_server_from_prefs)
                        self._srv_panel_nomodel_btn = QPushButton("Start without model")
                        self._srv_panel_nomodel_btn.setCursor(Qt.PointingHandCursor)
                        self._srv_panel_nomodel_btn.setToolTip(
                            "Launch the server with no model loaded, just to check it runs.")
                        self._srv_panel_nomodel_btn.clicked.connect(self._start_nomodel_from_prefs)
                        prow = QHBoxLayout()
                        prow.setContentsMargins(0, 0, 0, 0)
                        prow.addWidget(self._srv_panel_label, 1)
                        prow.addWidget(self._srv_panel_nomodel_btn, 0)
                        prow.addWidget(self._srv_panel_btn, 0)
                        pholder = QWidget()
                        pholder.setLayout(prow)
                        form.addRow(QLabel(label), pholder)
                        self._refresh_server_panel()
                        continue
                    if kind == "_llamastatus":
                        self._llama_status_label = QLabel("…")
                        self._llama_status_label.setObjectName("LlamaStatus")
                        self._llama_status_label.setWordWrap(True)
                        self._llama_action_btn = QPushButton("Get llama.cpp")
                        self._llama_action_btn.clicked.connect(self._acquire_llama)
                        lrow = QHBoxLayout()
                        lrow.setContentsMargins(0, 0, 0, 0)
                        lrow.addWidget(self._llama_status_label, 1)
                        lrow.addWidget(self._llama_action_btn, 0)
                        self._llama_progress = QProgressBar()
                        self._llama_progress.setTextVisible(True)
                        self._llama_progress.setVisible(False)
                        lcol = QVBoxLayout()
                        lcol.setContentsMargins(0, 0, 0, 0)
                        lcol.setSpacing(6)
                        lcol.addLayout(lrow)
                        lcol.addWidget(self._llama_progress)
                        lholder = QWidget()
                        lholder.setLayout(lcol)
                        form.addRow(QLabel(label), lholder)
                        self._refresh_llama_status()
                        continue
                    field = self._make_field(key, kind, extra, families)
                    lbl = QLabel(label)
                    help_text = self.FIELD_HELP.get(key)
                    if help_text:
                        lbl.setToolTip(help_text)
                        field.setToolTip(help_text)
                    form.addRow(lbl, field)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QFrame.NoFrame)
            scroll.setWidget(page)
            self.stack.addWidget(scroll)
        self.nav.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.nav.currentRowChanged.connect(self._on_page_changed)
        self.nav.setCurrentRow(0)
        # Keep the Local server panel in sync with the live start-mode selection.
        if "server_start_mode" in self.widgets:
            self.widgets["server_start_mode"][1].currentTextChanged.connect(
                lambda *_: self._refresh_server_panel()
            )
        self._refresh_llama_path_placeholder()

        buttons = QDialogButtonBox(
            QDialogButtonBox.Save | QDialogButtonBox.Apply | QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        self._apply_btn = buttons.button(QDialogButtonBox.Apply)
        self._apply_btn.setToolTip("Apply these settings now without closing this window.")
        self._apply_btn.clicked.connect(self._apply)
        outer.addWidget(buttons)
        self.setStyleSheet(build_stylesheet(settings))

    def _make_field(self, key, kind, extra, families):
        value = getattr(self.settings, key)
        if kind in ("text", "password"):
            w = QLineEdit(str(value))
            if kind == "password":
                w.setEchoMode(QLineEdit.Password)
            self.widgets[key] = ("text", w)
            return w
        if kind in ("browse_dir", "browse_file"):
            cont = QWidget()
            h = QHBoxLayout(cont)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            edit = QLineEdit(str(value))
            btn = QPushButton("Browse…")
            is_dir = kind == "browse_dir"
            btn.clicked.connect(lambda _c, e=edit, d=is_dir: self._browse_into(e, d))
            h.addWidget(edit, 1)
            h.addWidget(btn)
            self.widgets[key] = ("text", edit)
            return cont
        if kind == "bool":
            w = QCheckBox()
            w.setChecked(bool(value))
            self.widgets[key] = ("bool", w)
            return w
        if kind == "int":
            w = QSpinBox()
            lo, hi = extra or (0, 1000000)
            w.setRange(lo, hi)
            w.setValue(int(value))
            self.widgets[key] = ("int", w)
            return w
        if kind == "float":
            w = QDoubleSpinBox()
            lo, hi = extra or (0.0, 100000.0)
            w.setRange(lo, hi)
            w.setDecimals(1)
            w.setValue(float(value))
            self.widgets[key] = ("float", w)
            return w
        if kind == "multiline":
            w = QPlainTextEdit(str(value))
            w.setFixedHeight(90)
            self.widgets[key] = ("multiline", w)
            return w
        if kind == "dirlist":
            cont = QWidget()
            v = QVBoxLayout(cont)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(6)
            edit = QPlainTextEdit(str(value))
            edit.setPlaceholderText("One folder per line \u2014 e.g. your LM Studio models folder")
            edit.setFixedHeight(64)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(6)
            browse = QPushButton("Browse\u2026")
            browse.clicked.connect(lambda _c, e=edit: self._append_model_dir(e))
            detect = QPushButton("Detect model folders")
            detect.setToolTip("Add the default model folders for LM Studio, llama.cpp, "
                              "and Ollama (whichever exist on this machine).")
            detect.clicked.connect(lambda _c, e=edit: self._detect_server_dirs(e))
            row.addWidget(browse)
            row.addWidget(detect)
            row.addStretch(1)
            v.addWidget(edit)
            v.addLayout(row)
            # Register under "multiline" so _save reads it back with toPlainText().
            self.widgets[key] = ("multiline", edit)
            return cont
        if kind == "_gpupicker":
            cont = QWidget()
            v = QVBoxLayout(cont)
            v.setContentsMargins(0, 0, 0, 0)
            v.setSpacing(6)
            summary = QLabel("")
            summary.setStyleSheet("color: #9aa3ad; font-style: italic;")
            summary.setWordWrap(True)
            v.addWidget(summary)
            checks_holder = QWidget()
            checks_lay = QVBoxLayout(checks_holder)
            checks_lay.setContentsMargins(0, 0, 0, 0)
            checks_lay.setSpacing(4)
            v.addWidget(checks_holder)
            refresh = QPushButton("Re-detect GPUs")
            rrow = QHBoxLayout()
            rrow.setContentsMargins(0, 0, 0, 0)
            rrow.addWidget(refresh)
            rrow.addStretch(1)
            v.addLayout(rrow)
            cont._device_checks = []  # list[(device_token:str, radio)]
            cont._device_group = QButtonGroup(cont)
            cont._device_group.setExclusive(True)
            first = [True]

            def _gpu_label(g):
                vram = f", {g.vram_total_gb:.0f}GB" if g.vram_total_gb else ""
                tags = []
                if g.is_integrated:
                    tags.append("integrated")
                if g.sm:
                    tags.append(f"sm{g.sm}")
                extra = f", {', '.join(tags)}" if tags else ""
                return f"{g.device} \u2014 {g.name}{vram}{extra}"

            def rebuild():
                if first[0]:
                    saved = next((x.strip() for x in str(value).split(",") if x.strip()), "")
                    first[0] = False
                else:
                    saved = next((dev for dev, rb in cont._device_checks if rb.isChecked()), "")
                for _dev, rb in cont._device_checks:
                    cont._device_group.removeButton(rb)
                while checks_lay.count():
                    item = checks_lay.takeAt(0)
                    wdg = item.widget()
                    if wdg is not None:
                        wdg.deleteLater()
                cont._device_checks = []
                gpus = self._detected_gpus(refresh=not first[0])
                if not gpus:
                    summary.setText("No GPU detected for llama.cpp. Install llama.cpp "
                                    "(Get llama.cpp below) so it can enumerate your "
                                    "card \u2014 including AMD/Intel integrated GPUs via "
                                    "Vulkan. Model sizes can't be fit-checked until then.")
                    checks_holder.setVisible(False)
                    return
                if len(gpus) == 1:
                    # Nothing to pick — just report it (still drives recommendations).
                    summary.setText(gpus[0].summary)
                    checks_holder.setVisible(False)
                    return
                checks_holder.setVisible(True)
                summary.setText("Pick the GPU to run captioning on. Its VRAM is what "
                                "model recommendations are sized against. The captioner "
                                "uses one GPU \u2014 it won't split a model across cards.")
                # Default to the saved pick, else the largest-VRAM card.
                if saved not in {g.device for g in gpus}:
                    saved = max(gpus, key=lambda g: g.vram_total_gb or 0).device
                for g in gpus:
                    rb = QRadioButton(_gpu_label(g))
                    rb.setChecked(g.device == saved)
                    cont._device_group.addButton(rb)
                    checks_lay.addWidget(rb)
                    cont._device_checks.append((g.device, rb))
                for _dev, rb in cont._device_checks:
                    rb.toggled.connect(lambda *_: self._on_gpu_selection_changed())
                self._on_gpu_selection_changed()  # sync the default pick into settings

            refresh.clicked.connect(lambda: (rebuild(), self._on_gpu_selection_changed()))
            self.widgets[key] = ("_gpupicker", cont)
            rebuild()
            return cont
        if kind == "choice":
            w = QComboBox()
            w.addItems(list(extra))
            idx = w.findText(str(value))
            w.setCurrentIndex(idx if idx >= 0 else 0)
            self.widgets[key] = ("choice", w)
            return w
        if kind == "font":
            w = QComboBox()
            w.addItems(families)
            cur = str(value) if value else "(auto)"
            idx = w.findText(cur)
            w.setCurrentIndex(idx if idx >= 0 else 0)
            self.widgets[key] = ("font", w)
            return w
        if kind == "color":
            cont = QWidget()
            h = QHBoxLayout(cont)
            h.setContentsMargins(0, 0, 0, 0)
            h.setSpacing(6)
            edit = QLineEdit(str(value))
            edit.setFixedWidth(90)
            swatch = QLabel()
            swatch.setFixedSize(22, 22)

            def update_swatch(*_a):
                swatch.setStyleSheet(f"background:{edit.text().strip() or '#000'}; border:1px solid #888;")

            edit.textChanged.connect(update_swatch)
            update_swatch()
            btn = QPushButton("Pick")

            def pick():
                c = QColorDialog.getColor(QColor(edit.text().strip() or "#000000"), self)
                if c.isValid():
                    edit.setText(c.name())

            btn.clicked.connect(pick)
            h.addWidget(edit)
            h.addWidget(swatch)
            h.addWidget(btn)
            h.addStretch(1)
            self.widgets[key] = ("color", edit)
            return cont
        raise ValueError(f"unknown field kind: {kind}")

    def _build_models_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(6)
        self._profile_combos = {}
        self._hf_group = {}
        self._local_group = {}
        self._profile_host = {}
        self._picker_host = {}
        self._server_picker = {}
        self._picker_hint = {}
        self._model_sel_label = {}
        self._model_row = {}
        self._discovered_mmprojs = []

        # Model files & folders — where models download to and where we look for
        # already-downloaded GGUFs (the HF cache, your LM Studio folder, etc.).
        folders_head = QLabel("Model files & folders")
        folders_head.setObjectName("SectionLabel")
        lay.addWidget(folders_head)
        folders_form = QFormLayout()
        folders_form.setContentsMargins(0, 0, 0, 0)
        self._add_models_field(folders_form, "models_dir", "Models directory", "browse_dir", None)
        self._add_models_field(folders_form, "model_download_target", "Model download location",
                               "choice", (MODEL_TARGET_APP, MODEL_TARGET_HF))
        self._add_models_field(folders_form, "extra_model_dirs", "Extra model folders", "dirlist", None)
        lay.addLayout(folders_form)
        lay.addSpacing(10)

        for task, title in (("caption", "Captioning model"),
                            ("bbox", "BBox VLM \u2014 only used by bbox presets")):
            head_row = QHBoxLayout()
            head_row.setContentsMargins(0, 0, 0, 0)
            section = QLabel(title)
            section.setObjectName("SectionLabel")
            head_row.addWidget(section)
            head_row.addStretch(1)
            # No "Browse models…" button here: it opened the very same picker as
            # "Choose model…" on the row below.
            lay.addLayout(head_row)

            if task == "bbox":
                why = QLabel(
                    "Only presets that draw bounding boxes use this \u2014 Ideogram 4 "
                    "JSON. Plain-text presets (MiniMax H3, Wan, LTX) ignore it "
                    "entirely.")
                why.setObjectName("Hint")
                why.setWordWrap(True)
                lay.addWidget(why)
                self._bbox_lock_btn = QToolButton()
                self._bbox_lock_btn.setObjectName("LockToggle")
                self._bbox_lock_btn.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
                self._bbox_lock_btn.setText("Use the same model as captioning")
                self._bbox_lock_btn.setIconSize(QSize(18, 18))
                self._bbox_lock_btn.setCheckable(True)
                self._bbox_lock_btn.setAutoRaise(True)
                self._bbox_lock_btn.setCursor(Qt.PointingHandCursor)
                self._bbox_lock_btn.setChecked(self.bbox_same_as_caption)
                self._bbox_lock_btn.toggled.connect(self._on_bbox_same_toggled)
                lay.addWidget(self._bbox_lock_btn, 0, Qt.AlignLeft)

            same_note = " With a single local server hosting one VLM, this usually matches the captioning model above." if task == "bbox" else ""
            models_help = {
                "Profile": "Pick a predefined model profile for this task, or a custom one to set the repo/files below." + same_note,
                "API model name": "Model name string sent to the server for this task. Must match what the server exposes." + same_note,
                "Server model": "Pick from the models your existing server reports, or type into the field below. Refresh re-queries /v1/models.",
                "Custom HF repo": "Hugging Face repo to download this task's model from.",
                "Custom model file": "GGUF model filename within the HF repo.",
                "Custom mmproj file": "Vision projector (mmproj) filename. Required for vision models — without it, image input is rejected.",
                "Local model GGUF": "Path to a local GGUF model file for this task instead of downloading.",
                "Local mmproj file": "Path to a local mmproj (vision projector) paired with the local model.",
            }

            def _row(form_layout, text, widget):
                lbl = QLabel(text)
                if text in models_help:
                    lbl.setToolTip(models_help[text])
                    widget.setToolTip(models_help[text])
                form_layout.addRow(lbl, widget)

            # Model picker — a current-selection display (with VRAM fit badge) plus
            # a "Choose model…" button that opens the VRAM-aware picker pop-up. The
            # combo is kept as the canonical selection state but hidden.
            profile_host = QWidget()
            pf = QFormLayout(profile_host)
            pf.setContentsMargins(0, 0, 0, 0)
            combo = QComboBox(profile_host)
            combo.addItems(profile_labels(task))
            cur = profile_label_from_id(task, getattr(self.settings, f"{task}_profile_id"))
            idx = combo.findText(cur)
            combo.setCurrentIndex(idx if idx >= 0 else 0)
            combo.currentTextChanged.connect(lambda _t, tk=task: self._on_profile_changed(tk))
            combo.currentTextChanged.connect(lambda _t, tk=task: self._refresh_model_label(tk))
            combo.hide()
            self._profile_combos[task] = combo

            sel_row = QWidget()
            srl = QHBoxLayout(sel_row)
            srl.setContentsMargins(0, 0, 0, 0)
            srl.setSpacing(8)
            sel_label = QLabel()
            sel_label.setTextFormat(Qt.RichText)
            sel_label.setWordWrap(True)
            self._model_sel_label[task] = sel_label
            choose = QPushButton("Choose model\u2026")
            choose.setCursor(Qt.PointingHandCursor)
            choose.clicked.connect(lambda _c, tk=task: self._open_model_picker(tk))
            srl.addWidget(sel_label, 1)
            srl.addWidget(choose, 0)
            self._model_row[task] = sel_row
            _row(pf, "Model", sel_row)
            lay.addWidget(profile_host)
            self._profile_host[task] = profile_host
            self._refresh_model_label(task)

            # Server-model picker — existing-server mode only
            picker_host = QWidget()
            pk = QFormLayout(picker_host)
            pk.setContentsMargins(0, 0, 0, 0)
            picker = QComboBox()
            picker.activated.connect(lambda i, tk=task: self._apply_server_model(tk, i))
            self._server_picker[task] = picker
            refresh = QPushButton("Refresh")
            refresh.setToolTip("Query the server's /v1/models and list what it reports.")
            refresh.clicked.connect(lambda _c, tk=task: self._refresh_server_models(tk))
            pcont = QWidget()
            pch = QHBoxLayout(pcont)
            pch.setContentsMargins(0, 0, 0, 0)
            pch.setSpacing(6)
            pch.addWidget(picker, 1)
            pch.addWidget(refresh)
            _row(pk, "Server model", pcont)
            hint = QLabel("Click refresh to list the models loaded on your server.")
            hint.setObjectName("Hint")
            hint.setWordWrap(True)
            pk.addRow("", hint)
            self._picker_hint[task] = hint
            warn = QLabel("\u26a0 The model name must match a model currently loaded in your "
                          "external server, or requests will fail.")
            warn.setWordWrap(True)
            warn.setStyleSheet("color: #E0A33B;")
            pk.addRow("", warn)
            lay.addWidget(picker_host)
            self._picker_host[task] = picker_host

            # Exact model string sent to the server — always present, always editable
            form = QFormLayout()
            form.setContentsMargins(0, 0, 0, 0)
            api_edit = QLineEdit(getattr(self.settings, f"{task}_model"))
            self.widgets[f"{task}_model"] = ("text", api_edit)
            _row(form, "API model name", api_edit)
            lay.addLayout(form)

            hf = QWidget()
            hf_form = QFormLayout(hf)
            hf_form.setContentsMargins(0, 0, 0, 0)
            for key, label in (
                (f"{task}_hf_repo", "Custom HF repo"),
                (f"{task}_model_filename", "Custom model file"),
                (f"{task}_mmproj_filename", "Custom mmproj file"),
            ):
                e = QLineEdit(getattr(self.settings, key))
                self.widgets[key] = ("text", e)
                _row(hf_form, label, e)
            lay.addWidget(hf)
            self._hf_group[task] = hf

            loc = QWidget()
            loc_form = QFormLayout(loc)
            loc_form.setContentsMargins(0, 0, 0, 0)
            for key, label in (
                (f"{task}_local_model_path", "Local model GGUF"),
                (f"{task}_local_mmproj_path", "Local mmproj file"),
            ):
                e = QLineEdit(getattr(self.settings, key))
                self.widgets[key] = ("text", e)
                cont = QWidget()
                ch = QHBoxLayout(cont)
                ch.setContentsMargins(0, 0, 0, 0)
                ch.setSpacing(6)
                btn = QPushButton("Browse…")
                btn.clicked.connect(lambda _c, ed=e: self._browse_model_file(ed))
                ch.addWidget(e, 1)
                ch.addWidget(btn)
                _row(loc_form, label, cont)
            lay.addWidget(loc)
            self._local_group[task] = loc
            lay.addSpacing(8)
            self._update_profile_visibility(task)

        # Apply the "same as captioning" state now that all bbox widgets exist.
        self._set_bbox_fields_enabled(not self.bbox_same_as_caption)
        if self.bbox_same_as_caption:
            self._mirror_caption_to_bbox()

        # React to Connection/Server start-mode: server picker vs download UI.
        if "server_start_mode" in self.widgets:
            self.widgets["server_start_mode"][1].currentTextChanged.connect(
                lambda *_: self._on_server_mode_changed()
            )
        self._apply_models_mode()

        open_btn = QPushButton("Open profiles file…")
        open_btn.clicked.connect(self._open_profiles_file)
        lay.addWidget(open_btn, 0, Qt.AlignLeft)
        lay.addStretch(1)
        return page

    def _build_llm_page(self) -> QWidget:
        """Prompts and per-model frame rules, editable and shareable.

        Both are things the app ships defaults for but can't keep current: video
        models arrive faster than releases do, and their frame grids come from
        community reverse-engineering as often as from published specs. Rather than
        making every new model wait for an update, the rules are data — editable
        here, exported as one JSON, and importable by anyone else.
        """
        page = QWidget()
        outer = QVBoxLayout(page)
        outer.setContentsMargins(14, 14, 14, 14)
        outer.setSpacing(10)
        tabs = QTabWidget()
        tabs.addTab(self._build_prompt_editor(), "Caption prompts")
        tabs.addTab(self._build_goals_editor(), "Training goals")
        tabs.addTab(self._build_targets_editor(), "Model frame rules")
        outer.addWidget(tabs, 1)

        row = QHBoxLayout()
        exp = QPushButton("Export\u2026")
        exp.setToolTip("Save prompts and frame rules to one JSON file to share")
        exp.clicked.connect(self._export_llm_bundle)
        row.addWidget(exp)
        imp = QPushButton("Import\u2026")
        imp.setToolTip("Load prompts and frame rules from a shared JSON file")
        imp.clicked.connect(self._import_llm_bundle)
        row.addWidget(imp)
        row.addStretch(1)
        self._llm_status = QLabel("")
        self._llm_status.setObjectName("Hint")
        row.addWidget(self._llm_status)
        outer.addLayout(row)
        return page

    def _build_prompt_editor(self) -> QWidget:
        """Per preset, per media. Same store the caption run reads, so an edit here
        changes what actually gets sent rather than a copy of it."""
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)
        self._pe_custom = load_custom_presets(app_base_dir())
        picker = QHBoxLayout()
        picker.addWidget(QLabel("Preset:"))
        self._pe_preset = QComboBox()
        self._pe_repopulate()
        picker.addWidget(self._pe_preset, 1)
        add_btn = QPushButton("Add\u2026")
        add_btn.setToolTip("Create a caption preset for a model the app doesn't ship")
        add_btn.clicked.connect(self._pe_add)
        picker.addWidget(add_btn)
        self._pe_remove_btn = QPushButton("Remove")
        self._pe_remove_btn.setToolTip("Delete a preset you added")
        self._pe_remove_btn.clicked.connect(self._pe_remove)
        picker.addWidget(self._pe_remove_btn)
        picker.addWidget(QLabel("For:"))
        self._pe_media = QComboBox()
        self._pe_media.addItem("Photos", "image")
        self._pe_media.addItem("Videos", "video")
        picker.addWidget(self._pe_media)
        lay.addLayout(picker)

        # Which model's frame rules this preset conforms clips to. Optional: a
        # stills-only preset has no frame grid, and forcing a choice would invent a
        # constraint that doesn't exist.
        target_row = QHBoxLayout()
        target_row.addWidget(QLabel("Frame rules:"))
        self._pe_target = QComboBox()
        self._pe_target.setToolTip(
            "Conform clips captioned with this preset to a model's fps and "
            "frame-count grid. Leave as None for a photos-only preset.")
        self._pe_target.currentIndexChanged.connect(self._pe_target_changed)
        target_row.addWidget(self._pe_target, 1)
        lay.addLayout(target_row)

        self._pe_generated = QLabel(
            "Built from the caption schema at run time, so it isn't editable here. "
            "Shown for reference \u2014 adjust it with folder or per-file guidance "
            "instead.")
        self._pe_generated.setObjectName("Hint")
        self._pe_generated.setWordWrap(True)
        self._pe_generated.setVisible(False)
        lay.addWidget(self._pe_generated)

        self._pe_edit = QPlainTextEdit()
        self._pe_edit.setLineWrapMode(QPlainTextEdit.NoWrap)
        lay.addWidget(self._pe_edit, 1)

        self._pe_pending: dict[tuple[str, str], str] = {}
        self._pe_current: tuple[str, str] | None = None
        self._pe_preset.currentIndexChanged.connect(self._pe_reload)
        self._pe_media.currentIndexChanged.connect(self._pe_reload)

        row = QHBoxLayout()
        reset = QPushButton("Reset this prompt")
        reset.clicked.connect(self._pe_reset)
        row.addWidget(reset)
        row.addStretch(1)
        hint = QLabel("Edits are saved when you press Save.")
        hint.setObjectName("Hint")
        row.addWidget(hint)
        lay.addLayout(row)
        self._pe_reload()
        return w

    def _generated_prompt_preview(self, preset_key: str) -> str:
        """The instructions a schema-driven preset actually sends, for reading."""
        # Built the same way the run does, rather than borrowing whatever the main
        # window currently has selected — that showed the wrong preset's prompt.
        try:
            prompts = load_prompts()
            key = "image_to_json_system"
            if key not in prompts:
                key = next(iter(prompts), "")
            text = json_system_prompt(prompts, key, self.settings)
            if text.strip():
                return text
        except Exception:
            pass
        return ("These instructions are generated from the caption schema when a "
                "run starts, so there's nothing stored to edit here.")

    def _pe_all(self) -> dict:
        merged = dict(PRESETS)
        merged.update(self._pe_custom)
        return merged

    def _pe_repopulate(self, select: str | None = None) -> None:
        keep = select or self._pe_preset.currentData()
        self._pe_preset.blockSignals(True)
        self._pe_preset.clear()
        for key in PRESET_ORDER:
            self._pe_preset.addItem(PRESETS[key].label, key)
        for key, preset in self._pe_custom.items():
            self._pe_preset.addItem(f"{preset.label}  (added)", key)
        idx = self._pe_preset.findData(keep)
        self._pe_preset.setCurrentIndex(max(0, idx))
        self._pe_preset.blockSignals(False)

    def _pe_refresh_target(self) -> None:
        """Offer every model the app has rules for, built-in and user-added alike —
        a preset for a new model is useless if it can't point at that model."""
        key = self._pe_preset.currentData()
        preset = self._pe_all().get(key)
        self._pe_target.blockSignals(True)
        self._pe_target.clear()
        self._pe_target.addItem("None (photos, or no conforming)", "")
        targets = load_targets(app_base_dir())
        for target_key, target in targets.items():
            self._pe_target.addItem(target.label, target_key)
        current = (preset.model_target if preset else "") or ""
        idx = self._pe_target.findData(current)
        if idx < 0 and current:
            # A preset pointing at rules that have since been deleted.
            self._pe_target.addItem(f"{current} (missing)", current)
            idx = self._pe_target.count() - 1
        self._pe_target.setCurrentIndex(max(0, idx))
        # Built-in presets ship with the right target; changing it would be a
        # silent contradiction of the format they encode.
        editable = key in self._pe_custom
        self._pe_target.setEnabled(editable)
        self._pe_remove_btn.setEnabled(editable)
        self._pe_target.blockSignals(False)

    def _pe_target_changed(self, _idx: int) -> None:
        key = self._pe_preset.currentData()
        if key not in self._pe_custom:
            return
        preset = self._pe_custom[key]
        self._pe_custom[key] = make_custom_preset(
            key=preset.key, label=preset.label,
            image_prompt=preset.image_prompt, video_prompt=preset.video_prompt,
            model_target=self._pe_target.currentData() or "",
            blurb=preset.blurb)

    def _pe_add(self) -> None:
        label, ok = QInputDialog.getText(self, "Add caption preset", "Preset name:")
        label = (label or "").strip()
        if not ok or not label:
            return
        key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "custom_preset"
        if key in self._pe_all():
            QMessageBox.information(self, "Add caption preset",
                                    f"'{label}' already exists.")
            return
        # Seeded from the plain-text prompts rather than blank: an empty prompt
        # produces an empty caption, and starting from something that works is
        # easier than starting from nothing.
        base = get_preset("plain_text")
        self._pe_custom[key] = make_custom_preset(
            key=key, label=label,
            image_prompt=base.prompt_for("image"),
            video_prompt=base.prompt_for("video"),
            model_target="")
        self._pe_repopulate(select=key)
        self._pe_current = None
        self._pe_reload()

    def _pe_remove(self) -> None:
        key = self._pe_preset.currentData()
        if key not in self._pe_custom:
            return
        label = self._pe_custom[key].label
        if QMessageBox.question(
            self, "Remove caption preset",
            f"Delete the '{label}' preset?\n\nFolders currently using it will fall "
            "back to Plain text. Captions already written are untouched.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        self._pe_custom.pop(key, None)
        for media in ("image", "video"):
            self._pe_pending.pop((key, media), None)
            self._qsettings.remove(self._pe_key(key, media))
        self._pe_repopulate()
        self._pe_current = None
        self._pe_reload()

    def _pe_key(self, preset: str, media: str) -> str:
        return f"system_prompt/{preset}/{media}"

    def _pe_reload(self) -> None:
        """Keep unsaved edits per (preset, media) while switching between them."""
        if self._pe_current is not None and not self._pe_edit.isReadOnly():
            self._pe_pending[self._pe_current] = self._pe_edit.toPlainText()
        preset = self._pe_preset.currentData()
        media = self._pe_media.currentData()
        self._pe_current = (preset, media)
        self._pe_refresh_target()
        if self._pe_current in self._pe_pending:
            self._pe_edit.setPlainText(self._pe_pending[self._pe_current])
            return
        source = self._pe_all().get(preset, get_preset(preset))
        default = source.prompt_for(media)
        if not default.strip() and source.editor == "structured":
            # Ideogram 4's instructions are generated from the schema at run time,
            # not stored as a prompt. Showing an empty box made it look as though
            # the prompt had vanished, so show the real thing and say it's built.
            self._pe_edit.setPlainText(self._generated_prompt_preview(preset))
            self._pe_edit.setReadOnly(True)
            self._pe_generated.setVisible(True)
            return
        self._pe_edit.setReadOnly(False)
        self._pe_generated.setVisible(False)
        stored = self._qsettings.value(self._pe_key(preset, media), default, str)
        self._pe_edit.setPlainText(stored)

    def _pe_reset(self) -> None:
        preset, media = self._pe_current or ("", "")
        source = self._pe_all().get(preset, get_preset(preset))
        self._pe_edit.setPlainText(source.prompt_for(media))

    def _pe_commit(self) -> None:
        """Write every touched prompt, and the user's presets. Called from Save."""
        if self._pe_current is not None and not self._pe_edit.isReadOnly():
            self._pe_pending[self._pe_current] = self._pe_edit.toPlainText()
        # A custom preset's prompts live in the preset file, not QSettings, so the
        # preset is self-contained and travels with an export.
        for (preset_key, media), text in list(self._pe_pending.items()):
            if preset_key in self._pe_custom:
                existing = self._pe_custom[preset_key]
                self._pe_custom[preset_key] = make_custom_preset(
                    key=existing.key, label=existing.label,
                    image_prompt=text if media == "image" else existing.image_prompt,
                    video_prompt=text if media == "video" else existing.video_prompt,
                    model_target=existing.model_target, blurb=existing.blurb)
                self._pe_pending.pop((preset_key, media), None)
        save_custom_presets(app_base_dir(), self._pe_custom)
        for (preset, media), text in self._pe_pending.items():
            default = get_preset(preset).prompt_for(media)
            key = self._pe_key(preset, media)
            if text.strip() == default.strip():
                self._qsettings.remove(key)   # back to shipped default, not a copy
            else:
                self._qsettings.setValue(key, text)

    def _build_goals_editor(self) -> QWidget:
        """What a caption should describe and what it should leave out.

        Editable for the same reason the frame rules are: this is an evolving
        practice rather than settled fact, and the omission-vs-description trade
        differs by model and by trainer.
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Goal:"))
        self._tg_combo = QComboBox()
        top.addWidget(self._tg_combo, 1)
        add = QPushButton("Add\u2026")
        add.setToolTip("Define a training goal of your own")
        add.clicked.connect(self._tg_add)
        top.addWidget(add)
        self._tg_remove_btn = QPushButton("Remove")
        self._tg_remove_btn.setToolTip("Delete a goal you added")
        self._tg_remove_btn.clicked.connect(self._tg_remove)
        top.addWidget(self._tg_remove_btn)
        lay.addLayout(top)

        form = QFormLayout()
        form.setContentsMargins(0, 6, 0, 6)
        self._tg_label = QLineEdit()
        self._tg_label.setToolTip("Name shown in the Training dropdown")
        form.addRow("Name", self._tg_label)
        self._tg_summary = QPlainTextEdit()
        self._tg_summary.setMaximumHeight(60)
        self._tg_summary.setToolTip(
            "One line, shown under the dropdown as a quick reference")
        form.addRow("Summary", self._tg_summary)
        lay.addLayout(form)

        lay.addWidget(QLabel("Rules sent to the model"))
        self._tg_rules = QPlainTextEdit()
        self._tg_rules.setToolTip(
            "Appended after the preset's prompt and before your guidance. "
            "Say both what to omit and what to describe \u2014 half the rule leaves "
            "the model guessing about the other half.")
        lay.addWidget(self._tg_rules, 1)

        row = QHBoxLayout()
        self._tg_reset_btn = QPushButton("Reset to built-in")
        self._tg_reset_btn.clicked.connect(self._tg_reset)
        row.addWidget(self._tg_reset_btn)
        row.addStretch(1)
        hint = QLabel("Edits are saved when you press Save.")
        hint.setObjectName("Hint")
        row.addWidget(hint)
        lay.addLayout(row)

        self._tg_working = dict(load_goals(app_base_dir()))
        self._tg_builtin = builtin_goal_map()
        self._tg_current: str | None = None
        for key in goal_order(app_base_dir()):
            goal = self._tg_working.get(key)
            if goal is not None:
                self._tg_combo.addItem(goal.label, key)
        self._tg_combo.currentIndexChanged.connect(self._tg_reload)
        for widget in (self._tg_label,):
            widget.textChanged.connect(self._tg_capture)
        for widget in (self._tg_summary, self._tg_rules):
            widget.textChanged.connect(self._tg_capture)
        self._tg_reload()
        return w

    def _tg_reload(self) -> None:
        key = self._tg_combo.currentData()
        if not key or key not in self._tg_working:
            return
        self._tg_current = None          # suppress capture while repopulating
        goal = self._tg_working[key]
        self._tg_label.setText(goal.label)
        self._tg_summary.setPlainText(goal.summary)
        self._tg_rules.setPlainText(goal.rules)
        self._tg_current = key
        builtin = key in self._tg_builtin
        self._tg_remove_btn.setEnabled(not builtin)
        self._tg_reset_btn.setEnabled(builtin)

    def _tg_capture(self, *_args) -> None:
        key = self._tg_current
        if not key:
            return
        self._tg_working[key] = make_custom_goal(
            key=key,
            label=self._tg_label.text().strip() or key,
            summary=self._tg_summary.toPlainText(),
            rules=self._tg_rules.toPlainText(),
        )
        idx = self._tg_combo.findData(key)
        if idx >= 0 and self._tg_combo.itemText(idx) != self._tg_working[key].label:
            self._tg_combo.setItemText(idx, self._tg_working[key].label)

    def _tg_add(self) -> None:
        label, ok = QInputDialog.getText(self, "Add training goal", "Goal name:")
        label = (label or "").strip()
        if not ok or not label:
            return
        key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "custom_goal"
        if key in self._tg_working:
            QMessageBox.information(self, "Add training goal",
                                    f"'{label}' already exists.")
            return
        self._tg_working[key] = make_custom_goal(
            key=key, label=label,
            summary="Describe what this goal is for, in one line.",
            rules="Do NOT describe: \n\nDO describe fully: ")
        self._tg_combo.addItem(label, key)
        self._tg_combo.setCurrentIndex(self._tg_combo.count() - 1)

    def _tg_remove(self) -> None:
        key = self._tg_current
        if not key or key in self._tg_builtin:
            return
        label = self._tg_working[key].label
        if QMessageBox.question(
            self, "Remove training goal",
            f"Delete the '{label}' goal?\n\nFolders using it fall back to General. "
            "Captions already written are untouched.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        self._tg_working.pop(key, None)
        idx = self._tg_combo.findData(key)
        if idx >= 0:
            self._tg_combo.removeItem(idx)

    def _tg_reset(self) -> None:
        key = self._tg_current
        if not key or key not in self._tg_builtin:
            return
        self._tg_working[key] = self._tg_builtin[key]
        self._tg_reload()

    def _tg_commit(self) -> None:
        save_goals(app_base_dir(), self._tg_working)

    def _build_targets_editor(self) -> QWidget:
        """Frame rules per model: fps, the legal frame-count grid, dimension
        multiple, pixel budget and duration range.

        Editable because these specs drift and several were pieced together from
        community implementations rather than published docs — waiting on an app
        release to correct one, or to add a model that shipped last week, is the
        wrong dependency.
        """
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        top = QHBoxLayout()
        top.addWidget(QLabel("Model:"))
        self._mt_combo = QComboBox()
        top.addWidget(self._mt_combo, 1)
        add = QPushButton("Add\u2026")
        add.setToolTip("Define a model the app doesn't ship rules for")
        add.clicked.connect(self._mt_add)
        top.addWidget(add)
        self._mt_remove = QPushButton("Remove")
        self._mt_remove.setToolTip("Delete a model you added")
        self._mt_remove.clicked.connect(self._mt_remove_current)
        top.addWidget(self._mt_remove)
        lay.addLayout(top)

        form = QFormLayout()
        form.setContentsMargins(0, 6, 0, 6)
        self._mt_fields: dict[str, QWidget] = {}

        def _num(key: str, label: str, lo: float, hi: float, decimals: int = 0,
                 tip: str = "") -> None:
            box = QDoubleSpinBox() if decimals else QSpinBox()
            box.setRange(lo, hi)
            if decimals:
                box.setDecimals(decimals)
            box.setToolTip(tip)
            box.setMaximumWidth(160)
            self._mt_fields[key] = box
            form.addRow(label, box)

        _num("fps", "Frames per second", 1, 240, 3,
             "The rate the model trains at. 16 for Wan 2.2 A14B, 24 for most others.")
        _num("frame_modulus", "Frame grid — every", 1, 512, 0,
             "Legal frame counts follow frames %% modulus == remainder. Wan is 4n+1, "
             "LTX 8n+1, MiniMax H3 17n+5.")
        _num("frame_remainder", "Frame grid — plus", 0, 511, 0,
             "The +N part of the grid. Wan/LTX use 1, H3 uses 5.")
        _num("dimension_multiple", "Dimensions multiple of", 1, 256, 0,
             "Width and height must divide by this. 16 for Wan, 32 for LTX and H3.")
        _num("max_pixels", "Max pixels (0 = no cap)", 0, 100_000_000, 0,
             "Area budget, if the model documents one. H3 caps at 768\u00d71344.")
        _num("min_seconds", "Minimum length (s)", 0, 600, 2,
             "Shorter clips are flagged as unusable for this model.")
        _num("max_seconds", "Maximum length (s)", 0, 600, 2,
             "Longer clips are flagged; trim to fit.")
        self._mt_exact = QCheckBox("Requires exactly this frame rate")
        self._mt_exact.setToolTip(
            "On for models that reject off-rate sources outright (H3 needs 24.000). "
            "Off means the rate is a target, not a hard requirement.")
        self._mt_fields["exact_fps"] = self._mt_exact
        form.addRow("", self._mt_exact)
        self._mt_notes = QPlainTextEdit()
        self._mt_notes.setMaximumHeight(70)
        self._mt_notes.setToolTip("Anything worth remembering about this model's "
                                  "quirks — shown nowhere else, kept with the rules.")
        self._mt_fields["notes"] = self._mt_notes
        form.addRow("Notes", self._mt_notes)
        lay.addLayout(form)

        self._mt_summary = QLabel("")
        self._mt_summary.setObjectName("Hint")
        self._mt_summary.setWordWrap(True)
        lay.addWidget(self._mt_summary)
        lay.addStretch(1)

        row = QHBoxLayout()
        reset = QPushButton("Reset to built-in")
        reset.setToolTip("Discard your edits to this model and restore the shipped "
                         "rules")
        reset.clicked.connect(self._mt_reset_current)
        row.addWidget(reset)
        row.addStretch(1)
        lay.addLayout(row)

        self._mt_working = {k: replace(t) for k, t in load_targets(app_base_dir()).items()}
        self._mt_builtin = builtin_map()
        self._mt_current: str | None = None
        for key, target in self._mt_working.items():
            self._mt_combo.addItem(target.label, key)
        self._mt_combo.currentIndexChanged.connect(self._mt_reload)
        for box in self._mt_fields.values():
            if isinstance(box, (QSpinBox, QDoubleSpinBox)):
                box.valueChanged.connect(self._mt_capture)
            elif isinstance(box, QCheckBox):
                box.toggled.connect(self._mt_capture)
            else:
                box.textChanged.connect(self._mt_capture)
        self._mt_reload()
        return w

    def _mt_reload(self) -> None:
        key = self._mt_combo.currentData()
        if not key or key not in self._mt_working:
            return
        self._mt_current = None          # suppress capture while repopulating
        target = self._mt_working[key]
        f = self._mt_fields
        f["fps"].setValue(target.fps)
        f["frame_modulus"].setValue(target.frame_modulus)
        f["frame_remainder"].setValue(target.frame_remainder)
        f["dimension_multiple"].setValue(target.dimension_multiple)
        f["max_pixels"].setValue(target.max_pixels)
        f["min_seconds"].setValue(target.min_seconds)
        f["max_seconds"].setValue(target.max_seconds)
        f["exact_fps"].setChecked(target.exact_fps)
        f["notes"].setPlainText(target.notes)
        self._mt_current = key
        self._mt_remove.setEnabled(key not in self._mt_builtin)
        self._mt_refresh_summary()

    def _mt_capture(self, *_args) -> None:
        """Fold edits into the working copy as they're made, so switching models
        doesn't lose them."""
        key = self._mt_current
        if not key:
            return
        f = self._mt_fields
        self._mt_working[key] = replace(
            self._mt_working[key],
            fps=float(f["fps"].value()),
            frame_modulus=max(1, int(f["frame_modulus"].value())),
            frame_remainder=int(f["frame_remainder"].value()),
            dimension_multiple=max(1, int(f["dimension_multiple"].value())),
            max_pixels=int(f["max_pixels"].value()),
            min_seconds=float(f["min_seconds"].value()),
            max_seconds=float(f["max_seconds"].value()),
            exact_fps=bool(f["exact_fps"].isChecked()),
            notes=f["notes"].toPlainText(),
        )
        self._mt_refresh_summary()

    def _mt_refresh_summary(self) -> None:
        """Show the rules as legal frame counts, which is the form they're used in
        — a modulus and remainder are hard to sanity-check in the abstract."""
        key = self._mt_current
        if not key:
            return
        t = self._mt_working[key]
        ladder = []
        n = t.smallest_legal_frames()
        while len(ladder) < 6 and n <= t.max_frames():
            ladder.append(str(n))
            n += t.frame_modulus
        self._mt_summary.setText(
            f"Legal frame counts: {', '.join(ladder)} \u2026 up to {t.max_frames()} "
            f"({t.seconds_for_frames(t.max_frames()):.2f}s at {t.fps:g}fps). "
            f"Shortest usable: {t.min_frames()} frames "
            f"({t.seconds_for_frames(t.min_frames()):.2f}s)."
        )

    def _mt_add(self) -> None:
        label, ok = QInputDialog.getText(self, "Add model", "Model name:")
        label = (label or "").strip()
        if not ok or not label:
            return
        key = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") or "custom_model"
        if key in self._mt_working:
            QMessageBox.information(self, "Add model",
                                    f"'{label}' already exists in the list.")
            return
        # Seeded from the most common shape rather than zeros, so a new entry is
        # immediately plausible and only needs the bits that differ.
        self._mt_working[key] = ModelTarget(
            key=key, label=label, fps=24.0, frame_modulus=1, frame_remainder=0,
            dimension_multiple=32, max_pixels=0, min_seconds=1.0, max_seconds=10.0,
            source="added by hand", verified=datetime.date.today().isoformat())
        self._mt_combo.addItem(label, key)
        self._mt_combo.setCurrentIndex(self._mt_combo.count() - 1)

    def _mt_remove_current(self) -> None:
        key = self._mt_current
        if not key or key in self._mt_builtin:
            return
        self._mt_working.pop(key, None)
        idx = self._mt_combo.findData(key)
        if idx >= 0:
            self._mt_combo.removeItem(idx)

    def _mt_reset_current(self) -> None:
        key = self._mt_current
        if not key or key not in self._mt_builtin:
            return
        self._mt_working[key] = replace(self._mt_builtin[key])
        self._mt_reload()

    def _mt_commit(self) -> None:
        """Persist to model_targets.json. Only entries that differ from the shipped
        defaults are written, so the file stays a diff rather than a snapshot that
        would freeze future corrections out."""
        changed = {k: t for k, t in self._mt_working.items()
                   if k not in self._mt_builtin or t != self._mt_builtin[k]}
        path = targets_path(app_base_dir())
        if not changed:
            path.unlink(missing_ok=True)
            return
        save_targets(app_base_dir(), changed)

    BUNDLE_VERSION = 1

    def _export_llm_bundle(self) -> None:
        """Pick what to share, then write it.

        Everything is offered, not just what you've edited: exporting only your
        diffs produced an empty file for anyone who hadn't customised anything, and
        the built-in rules are a perfectly good starting point for someone writing
        rules for a new model.
        """
        self._pe_commit()
        # With nothing customised there'd be nothing ticked, so pressing Export
        # would just say "nothing selected". In that case the only useful export is
        # the built-ins, so tick everything and let the user narrow it down.
        customised = any(t != self._mt_builtin.get(k)
                         for k, t in self._mt_working.items())
        if not customised:
            for key in PRESET_ORDER:
                preset = get_preset(key)
                if not preset.prompt_for("image").strip():
                    continue
                for media in ("image", "video"):
                    stored = self._qsettings.value(self._pe_key(key, media), "", str)
                    if stored and stored.strip() != preset.prompt_for(media).strip():
                        customised = True
                        break

        dlg = QDialog(self)
        dlg.setWindowTitle("Export LLM instructions")
        dlg.setMinimumWidth(560)
        lay = QVBoxLayout(dlg)
        intro = QLabel(
            "Choose what to include. Anything you've changed is ticked; the rest is "
            "offered too, since the built-in rules make a useful starting point for "
            "someone writing rules for a new model."
            if customised else
            "Nothing has been customised yet, so everything is ticked \u2014 the "
            "built-in rules and prompts make a useful starting point for writing "
            "rules for a new model. Untick anything you don't want to share.")
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        listing = QListWidget()
        listing.setSelectionMode(QListWidget.NoSelection)
        lay.addWidget(listing, 1)

        def section(title: str) -> None:
            item = QListWidgetItem(title)
            item.setFlags(Qt.NoItemFlags)
            font = item.font()
            font.setBold(True)
            item.setFont(font)
            listing.addItem(item)

        entries: list[tuple[QListWidgetItem, str, object]] = []
        section("Model frame rules")
        for key, target in self._mt_working.items():
            builtin = self._mt_builtin.get(key)
            if builtin is None:
                mark, ticked = "added", True
            elif target != builtin:
                mark, ticked = "edited", True
            else:
                mark, ticked = "built-in", False
            item = QListWidgetItem(f"    {target.label}  ({mark})")
            item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            item.setCheckState(Qt.Checked if (ticked or not customised)
                               else Qt.Unchecked)
            listing.addItem(item)
            entries.append((item, "target", key))

        section("Caption prompts")
        for key in PRESET_ORDER:
            preset = get_preset(key)
            # Ideogram 4 builds its instructions from the schema rather than a
            # plain system prompt, so it has nothing to list here — including it
            # would put two empty entries in every bundle.
            if not preset.prompt_for("image").strip():
                continue
            for media in ("image", "video"):
                default = preset.prompt_for(media)
                stored = self._qsettings.value(self._pe_key(key, media), "", str)
                edited = bool(stored) and stored.strip() != default.strip()
                label = f"    {preset.label} \u2014 {'videos' if media == 'video' else 'photos'}"
                item = QListWidgetItem(f"{label}  ({'edited' if edited else 'built-in'})")
                item.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
                item.setCheckState(Qt.Checked if (edited or not customised)
                                   else Qt.Unchecked)
                listing.addItem(item)
                entries.append((item, "prompt", (key, media)))

        row = QHBoxLayout()
        for text, state in (("Select all", Qt.Checked), ("Select none", Qt.Unchecked)):
            btn = QPushButton(text)
            btn.clicked.connect(
                lambda _c, st=state: [i.setCheckState(st) for i, _k, _r in entries])
            row.addWidget(btn)
        row.addStretch(1)
        lay.addLayout(row)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        buttons.addButton("Export\u2026", QDialogButtonBox.AcceptRole)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        if dlg.exec() != QDialog.Accepted:
            return

        prompts: dict[str, str] = {}
        targets: list[dict] = []
        for item, kind, ref in entries:
            if item.checkState() != Qt.Checked:
                continue
            if kind == "target":
                targets.append(asdict(self._mt_working[ref]))
            else:
                preset_key, media = ref
                default = get_preset(preset_key).prompt_for(media)
                stored = self._qsettings.value(
                    self._pe_key(preset_key, media), "", str) or default
                prompts[f"{preset_key}/{media}"] = stored
        if not prompts and not targets:
            QMessageBox.information(
                self, "Nothing selected",
                "Tick at least one rule or prompt to export.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export LLM instructions", "captioner-llm-instructions.json",
            "JSON (*.json)")
        if not path:
            return
        bundle = {
            "kind": "fantastic-captioning-kit/llm-instructions",
            "version": self.BUNDLE_VERSION,
            "exported": datetime.datetime.now().isoformat(timespec="seconds"),
            "description": (f"{len(targets)} model rule(s) and {len(prompts)} "
                            "caption prompt(s) for the Fantastic Upgraded "
                            "Captioning Kit."),
            "prompts": prompts,
            "model_targets": targets,
        }
        try:
            Path(path).write_text(json.dumps(bundle, indent=2), encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Export failed", str(exc))
            return
        self._llm_status.setText(
            f"Exported {len(prompts)} prompt(s) and {len(targets)} model rule(s).")

    def _import_llm_bundle(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Import LLM instructions", "", "JSON (*.json)")
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Import failed",
                                 f"Couldn't read that file:\n{exc}")
            return
        if not isinstance(data, dict) or "prompts" not in data and "model_targets" not in data:
            QMessageBox.critical(
                self, "Import failed",
                "That doesn't look like an exported instructions file.")
            return
        prompts = data.get("prompts") or {}
        targets = data.get("model_targets") or []
        # Name what's coming: "3 model rules" doesn't tell you that one of them is
        # about to replace the H3 rules you tuned yourself.
        names = [str(t.get("label") or t.get("key")) for t in targets
                 if isinstance(t, dict)]
        replacing = [n for n, t in zip(names, targets)
                     if isinstance(t, dict) and t.get("key") in self._mt_working]
        detail = ""
        if names:
            detail += "\n\nModels: " + ", ".join(names[:8])
            if len(names) > 8:
                detail += f" (+{len(names) - 8} more)"
        if replacing:
            detail += "\n\nReplaces your current rules for: " + ", ".join(replacing[:8])
        if QMessageBox.question(
            self, "Import LLM instructions",
            f"Load {len(prompts)} prompt(s) and {len(targets)} model rule(s)?"
            f"{detail}\n\nAnything not covered is left alone. Nothing is written "
            "until you press Save.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return
        loaded_prompts = 0
        for key, text in prompts.items():
            if not isinstance(text, str) or "/" not in key:
                continue
            preset, _, media = key.partition("/")
            if preset in PRESETS and media in ("image", "video"):
                self._pe_pending[(preset, media)] = text
                loaded_prompts += 1
        loaded_targets = 0
        for raw in targets:
            if not isinstance(raw, dict) or not raw.get("key"):
                continue
            try:
                target = _target_from_dict(raw)
            except (TypeError, ValueError, KeyError):
                continue     # skip the bad entry, keep the rest
            self._mt_working[target.key] = target
            if self._mt_combo.findData(target.key) < 0:
                self._mt_combo.addItem(target.label, target.key)
            loaded_targets += 1
        self._pe_current = None
        self._pe_reload()
        self._mt_reload()
        self._llm_status.setText(
            f"Loaded {loaded_prompts} prompt(s) and {loaded_targets} model rule(s) "
            "\u2014 press Save to keep them.")

    def _build_tags_page(self) -> QWidget:
        page = QWidget()
        lay = QVBoxLayout(page)
        lay.setContentsMargins(14, 14, 14, 14)
        lay.setSpacing(8)

        head = QLabel("Default tags")
        head.setObjectName("SectionLabel")
        lay.addWidget(head)
        note = QLabel(
            "These tags appear on every folder you open, on top of any folder-specific "
            "tags. Use them for triggers you reuse across datasets (e.g. man, woman, person)."
        )
        note.setObjectName("Hint")
        note.setWordWrap(True)
        lay.addWidget(note)

        self._tags_list = QListWidget()
        self._tags_list.setSelectionMode(QListWidget.ExtendedSelection)
        for t in (self.tags_result or []):
            self._tags_list.addItem(t)
        lay.addWidget(self._tags_list, 1)

        add_row = QHBoxLayout()
        self._tags_input = QLineEdit()
        self._tags_input.setPlaceholderText("New default tag…")
        add_btn = QPushButton("+ Add")
        remove_btn = QPushButton("Remove selected")
        add_row.addWidget(self._tags_input, 1)
        add_row.addWidget(add_btn)
        add_row.addWidget(remove_btn)
        lay.addLayout(add_row)

        def add_tag() -> None:
            text = self._tags_input.text().strip()
            if not text:
                return
            existing = {self._tags_list.item(i).text() for i in range(self._tags_list.count())}
            if text not in existing:
                self._tags_list.addItem(text)
            self._tags_input.clear()

        def remove_selected() -> None:
            for item in self._tags_list.selectedItems():
                self._tags_list.takeItem(self._tags_list.row(item))

        add_btn.clicked.connect(add_tag)
        self._tags_input.returnPressed.connect(add_tag)
        remove_btn.clicked.connect(remove_selected)
        return page

    def _lock_icon(self, locked: bool) -> QPixmap:
        # Gold closed padlock when linked, gray open padlock when independent.
        if locked:
            return lucide_pixmap("lock", "#f5c518", 16)
        return lucide_pixmap("lock-open", "#7f8694", 16)

    def _set_bbox_fields_enabled(self, enabled: bool) -> None:
        self._profile_combos["bbox"].setEnabled(enabled)
        self.widgets["bbox_model"][1].setEnabled(enabled)
        self._hf_group["bbox"].setEnabled(enabled)
        self._local_group["bbox"].setEnabled(enabled)
        if "bbox" in self._picker_host:
            self._picker_host["bbox"].setEnabled(enabled)
        if "bbox" in self._model_row:
            self._model_row["bbox"].setEnabled(enabled)
        # Dim the text of the locked fields so it reads as inactive.
        dim = "" if enabled else "color: #7f8694;"
        for key in ("bbox_model", "bbox_hf_repo", "bbox_model_filename", "bbox_mmproj_filename",
                    "bbox_local_model_path", "bbox_local_mmproj_path"):
            self.widgets[key][1].setStyleSheet(dim)
        self._profile_combos["bbox"].setStyleSheet(dim)
        if hasattr(self, "_bbox_lock_btn"):
            self._bbox_lock_btn.setIcon(QIcon(self._lock_icon(not enabled)))
            self._bbox_lock_btn.setToolTip(
                "Locked: box location uses the captioning model above. Click to set it separately."
                if not enabled else
                "Unlocked: box location uses its own model. Click to use the captioning model."
            )

    def _mirror_caption_to_bbox(self) -> None:
        """Copy the caption model widgets into the bbox widgets (display sync)."""
        self._profile_combos["bbox"].setCurrentText(self._profile_combos["caption"].currentText())
        for suffix in ("model", "hf_repo", "model_filename", "mmproj_filename",
                       "local_model_path", "local_mmproj_path"):
            self.widgets[f"bbox_{suffix}"][1].setText(self.widgets[f"caption_{suffix}"][1].text())
        self._update_profile_visibility("bbox")

    def _on_bbox_same_toggled(self, checked: bool) -> None:
        self.bbox_same_as_caption = checked
        self._set_bbox_fields_enabled(not checked)
        if checked:
            self._mirror_caption_to_bbox()

    def _profile_for_label(self, task: str, label: str):
        for profile in profiles_for_task(task):
            if profile.label == label:
                return profile
        return profiles_for_task(task)[0]

    def _update_profile_visibility(self, task: str) -> None:
        if self._current_server_mode() == "existing":
            self._hf_group[task].setVisible(False)
            self._local_group[task].setVisible(False)
            return
        profile = self._profile_for_label(task, self._profile_combos[task].currentText())
        self._hf_group[task].setVisible(profile.kind == "custom_hf")
        self._local_group[task].setVisible(profile.kind == "custom_local")

    def _on_profile_changed(self, task: str) -> None:
        profile = self._profile_for_label(task, self._profile_combos[task].currentText())
        # named/server/local profiles auto-fill the API model name; custom_hf is user-typed
        if profile.kind != "custom_hf":
            self.widgets[f"{task}_model"][1].setText(profile.api_model)
        self._update_profile_visibility(task)
        if task == "caption" and getattr(self, "bbox_same_as_caption", False):
            self._mirror_caption_to_bbox()

    def _on_page_changed(self, row: int) -> None:
        """Stage cross-page changes live: when arriving on the Models page, re-sync
        its server-mode-dependent UI from the current Connection/Server selections,
        so the user never has to Apply just to see the right model fields."""
        item = self.nav.item(row) if (row is not None and row >= 0) else None
        if item is not None and item.text() == "LLM Models" and self._profile_combos:
            self._apply_models_mode()

    def _on_server_mode_changed(self) -> None:
        """Server mode is the source of truth. When the user leaves external mode
        (e.g. switches the preset to llama.cpp), a leftover server-alias profile
        must not drag the app back to 'existing' — reset it to the default model.
        Then re-sync the Models page UI."""
        if self._current_server_mode() != "existing":
            for task in ("caption", "bbox"):
                combo = self._profile_combos.get(task)
                if combo is None:
                    continue
                prof = self._profile_for_label(task, combo.currentText())
                if prof.kind == "server":
                    default = profiles_for_task(task)[0]
                    i = combo.findText(default.label)
                    if i >= 0:
                        combo.setCurrentIndex(i)
        self._apply_models_mode()

    def _current_server_mode(self) -> str:
        if "server_start_mode" in self.widgets:
            return self.widgets["server_start_mode"][1].currentText()
        return "local"

    def _apply_models_mode(self) -> None:
        """Existing-server mode shows the live picker; local/custom show the download UI."""
        existing = self._current_server_mode() == "existing"
        for task in ("caption", "bbox"):
            if task in self._picker_host:
                self._picker_host[task].setVisible(existing)
            if task in self._profile_host:
                self._profile_host[task].setVisible(not existing)
            self._update_profile_visibility(task)
        # picker visibility changed; re-assert the bbox lock dimming/enable state
        if hasattr(self, "bbox_same_as_caption"):
            self._set_bbox_fields_enabled(not self.bbox_same_as_caption)

    def _refresh_server_models(self, task: str) -> None:
        combo = self._server_picker.get(task)
        if combo is None:
            return
        hint = self._picker_hint.get(task)
        base = self.widgets["base_url"][1].text().strip() if "base_url" in self.widgets else ""
        key = self.widgets["api_key"][1].text().strip() if "api_key" in self.widgets else ""
        combo.blockSignals(True)
        combo.clear()
        if not base:
            combo.addItem("Set a server URL in Connection/Server")
            combo.setEnabled(False)
            combo.blockSignals(False)
            if hint:
                hint.setText("No server URL set yet.")
            return
        try:
            ids = sorted(server_model_ids(base, key, timeout=4.0))
        except Exception as exc:
            combo.addItem("Couldn't reach the server")
            combo.setEnabled(False)
            combo.blockSignals(False)
            if hint:
                hint.setText(f"{type(exc).__name__}: {exc} — type the name into the field below.")
            return
        combo.setEnabled(True)
        if not ids:
            combo.addItem("Server reported no models")
            combo.blockSignals(False)
            if hint:
                hint.setText("The server is up but has no model loaded.")
            return
        combo.addItem("Select a loaded model…")
        for mid in ids:
            combo.addItem(mid, mid)
        cur = self.widgets[f"{task}_model"][1].text().strip()
        i = combo.findText(cur)
        if i >= 0:
            combo.setCurrentIndex(i)
        combo.blockSignals(False)
        if hint:
            n = len(ids)
            hint.setText(f"{n} model{'s' if n != 1 else ''} reported. Picking one fills the field below.")

    def _apply_server_model(self, task: str, idx: int) -> None:
        combo = self._server_picker.get(task)
        if combo is None:
            return
        mid = combo.itemData(idx)
        if not mid:
            return  # a placeholder row, not a real model
        self.widgets[f"{task}_model"][1].setText(str(mid))
        if task == "caption" and getattr(self, "bbox_same_as_caption", False):
            self._mirror_caption_to_bbox()

    def _add_models_field(self, form, key, label, kind, extra) -> None:
        """Build a generic settings field (registered in self.widgets so _save
        picks it up) and add it to a form on the Models page with its help text."""
        widget = self._make_field(key, kind, extra, None)
        lbl = QLabel(label)
        help_text = self.FIELD_HELP.get(key)
        if help_text:
            lbl.setToolTip(help_text)
            widget.setToolTip(help_text)
        form.addRow(lbl, widget)

    def _short_dir(self, path: Path) -> str:
        text = str(path)
        home = str(Path.home())
        if text.startswith(home):
            text = "~" + text[len(home):]
        return text

    def _tail_dir(self, path: Path) -> str:
        """The last couple of folders, or the whole path when it's already short.

        Model paths share a long common prefix (~/.cache/huggingface/hub/models--…),
        so showing it in full pushes the part that actually differs out of view. The
        complete path stays on the tooltip.
        """
        short = self._short_dir(path)
        parts = [p for p in Path(short).parts if p not in ("/", "\\")]
        if len(short) <= 40 or len(parts) <= 3:
            return short
        # The HF cache buries the model name behind snapshots/<hash>, so taking the
        # last two segments would show the hash and drop the only useful part.
        noise = {"snapshots", "blobs", "refs"}
        meaningful = [p for p in parts
                      if p.lower() not in noise
                      and not re.fullmatch(r"[0-9a-f]{8,}", p.lower())]
        tail = meaningful[-2:] if len(meaningful) >= 2 else parts[-2:]
        return "\u2026/" + "/".join(tail)

    def _settings_with_current_dirs(self):
        """Snapshot of settings reflecting the folder fields as currently typed
        (so Detect uses edits the user hasn't saved yet)."""
        md = self.settings.models_dir
        ex = getattr(self.settings, "extra_model_dirs", "")
        if "models_dir" in self.widgets:
            md = self.widgets["models_dir"][1].text().strip() or md
        if "extra_model_dirs" in self.widgets:
            ex = self.widgets["extra_model_dirs"][1].toPlainText()
        return replace(self.settings, models_dir=md, extra_model_dirs=ex)

    def _use_local_gguf(self, task: str, model_path, dlg) -> None:
        """Apply a discovered local GGUF (chosen in the model picker): switch to the
        custom-local profile, fill the local path, auto-pair an mmproj if one sits
        alongside it, and mirror to bbox when locked."""
        pcombo = self._profile_combos.get(task)
        if pcombo is not None:
            i = pcombo.findText(CUSTOM_LOCAL_PROFILE.label)
            if i >= 0:
                pcombo.setCurrentIndex(i)
        if f"{task}_local_model_path" in self.widgets:
            self.widgets[f"{task}_local_model_path"][1].setText(str(model_path))
        mm = guess_mmproj_for(Path(model_path), getattr(self, "_discovered_mmprojs", []))
        if mm is not None and f"{task}_local_mmproj_path" in self.widgets:
            self.widgets[f"{task}_local_mmproj_path"][1].setText(str(mm))
        self._update_profile_visibility(task)
        if task == "caption" and getattr(self, "bbox_same_as_caption", False):
            self._mirror_caption_to_bbox()
        self._refresh_model_label(task)
        dlg.accept()

    def _refresh_model_label(self, task: str) -> None:
        lbl = self._model_sel_label.get(task)
        combo = self._profile_combos.get(task)
        if lbl is None or combo is None:
            return
        name = combo.currentText()
        prof = self._profile_for_label(task, name)
        # Custom-local: show the chosen file's name rather than the generic label.
        if prof.kind == "custom_local":
            w = self.widgets.get(f"{task}_local_model_path")
            chosen = w[1].text().strip() if w else ""
            if chosen:
                lbl.setText(f"<b>Local: {Path(chosen).name}</b>")
                return
        short = name.split(":", 1)[1].strip() if name.lower().startswith("download:") else name
        badge = ""
        if prof.kind == "hf" and prof.vram_gb > 0:
            vram = self._detected_vram()
            if vram:
                fit = vram_fit(prof.vram_gb, vram)
                colors = {"fits": "#3ddc84", "tight": "#E0A33B", "too_big": "#ff5a52", "unknown": "#9aa4b6"}
                texts = {"fits": "Fits", "tight": "Tight", "too_big": "Too big", "unknown": ""}
                if texts[fit]:
                    badge = f'&nbsp;&nbsp;<span style="color:{colors[fit]}">[{texts[fit]}]</span>'
        # Audio capability sits next to the fit badge: it's the difference between
        # a video caption that can quote dialogue and one that can only watch lips.
        if prof.supports_audio:
            badge += ('&nbsp;&nbsp;<span style="color:#45B964">'
                      '[\U0001F50A Audio]</span>')
        lbl.setText(f"<b>{short}</b>{badge}")
        lbl.setToolTip("Hears the clip's audio \u2014 video captions can include "
                       "dialogue and sound." if prof.supports_audio else "")

    def _open_model_picker(self, task: str) -> None:
        """Model picker. In local (app-managed llama.cpp) mode it lists the
        recommended/download profiles with VRAM fit badges plus the GGUFs already
        on disk, and choosing one configures the app. In external-server mode the
        app can't fetch or load models for the server, so it shows the models
        already on disk (top) and the recommended models with Hugging Face links
        (below), with a note to download/configure those in the server."""
        vram = self._detected_vram()
        rec = recommend_profile_for_vram(task, vram)
        rec_id = rec.id if rec else None
        local_mode = self._current_server_mode() != "existing"

        badge_colors = {"fits": "#3ddc84", "tight": "#E0A33B", "too_big": "#ff5a52", "unknown": "#9aa4b6"}
        badge_text = {"fits": "Fits", "tight": "Tight", "too_big": "Too big", "unknown": "\u2014"}
        rank = {"fits": 0, "tight": 1, "too_big": 2, "unknown": 3}
        # Wider than it was: model names run long, and at 660 they wrapped to two or
        # three lines, so every row was a different height and the list looked ragged.
        CONTENT_W = 860

        dlg = QDialog(self)
        dlg.setWindowTitle("Choose a model")
        dlg.resize(CONTENT_W + 60, 620)
        dlg.setMinimumWidth(CONTENT_W + 40)
        lay = QVBoxLayout(dlg)

        if vram:
            gpu_name = self._detected_gpu_label()
            header = f"Detected {gpu_name} \u2014 about {vram:.0f}GB VRAM."
            if rec:
                rname = rec.label.split(":", 1)[1].strip() if rec.label.lower().startswith("download:") else rec.label
                header += f"  Recommended: {rname}."
        else:
            header = "Couldn't read your VRAM \u2014 showing all models without fit estimates."
        head = QLabel(header)
        head.setObjectName("Hint")
        head.setWordWrap(True)
        lay.addWidget(head)

        if not local_mode:
            ext = QLabel(
                "You're using an external server. The app can't download or load models "
                "into it \u2014 download and configure these in your server (e.g. LM Studio "
                "or Ollama), then pick the model from its list or enter its name."
            )
            ext.setWordWrap(True)
            ext.setStyleSheet("color: #E0A33B;")
            lay.addWidget(ext)

        listing = QListWidget()
        listing.setWordWrap(True)
        listing.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        lay.addWidget(listing, 1)

        def row_width() -> int:
            """Rows follow the viewport, never a hard-coded width.

            They used to be pinned at CONTENT_W while the list was narrower, so the
            last 200-odd pixels hung off the right edge and the Use/HF buttons were
            clipped into unreadable stubs — visible only on rows carrying an extra
            chip, which is what made it look like an audio-specific bug.
            """
            vp = listing.viewport().width()
            return max(360, vp - 2) if vp > 10 else CONTENT_W

        def add_row(row):
            width = row_width()
            row.setFixedWidth(width)
            row.ensurePolished()
            h = row.sizeHint().height()
            lay_ = row.layout()
            if lay_ is not None and lay_.hasHeightForWidth():
                h = max(h, lay_.heightForWidth(width))
            h = max(h, 42)            # never shorter than a Use/HF button row
            item = QListWidgetItem(listing)
            item.setSizeHint(QSize(width, h))
            listing.addItem(item)
            listing.setItemWidget(item, row)

        def add_section(text):
            hdr = QLabel(text)
            hdr.setObjectName("SectionLabel")
            hdr.setContentsMargins(8, 10, 8, 2)
            item = QListWidgetItem(listing)
            item.setFlags(Qt.NoItemFlags)
            item.setSizeHint(QSize(row_width(), hdr.sizeHint().height() + 8))
            listing.addItem(item)
            listing.setItemWidget(item, hdr)

        def add_hint(text):
            e = QLabel(text)
            e.setObjectName("Hint")
            e.setWordWrap(True)
            e.setContentsMargins(8, 2, 8, 6)
            e.setFixedWidth(row_width() - 4)
            h = e.heightForWidth(row_width() - 4)
            if h <= 0:
                h = e.sizeHint().height()
            item = QListWidgetItem(listing)
            item.setFlags(Qt.NoItemFlags)
            item.setSizeHint(QSize(row_width(), h + 4))
            listing.addItem(item)
            listing.setItemWidget(item, e)

        def elided(text: str, width: int, tooltip: str | None = None) -> QLabel:
            """One-line label that shortens with an ellipsis instead of wrapping.

            Wrapping was what made the rows jump around: a long name became two or
            three lines and every row ended up a different height. The full text
            stays available on hover.
            """
            label = QLabel()
            label.setWordWrap(False)
            label.setMinimumWidth(0)
            label.setMaximumWidth(width)
            # Ignored horizontally so the label gives up space instead of forcing
            # the buttons beside it below their own text: an elided label's size
            # hint is as wide as the text it holds, which squeezed "Use" and "HF"
            # into unreadable stubs on rows carrying an extra chip.
            label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            metrics = QFontMetrics(label.font())
            label.setText(metrics.elidedText(text, Qt.ElideRight, width - 6))
            if metrics.horizontalAdvance(text) > width - 6 or tooltip:
                label.setToolTip(tooltip or text)
            return label

        def make_profile_row(p, *, allow_use, show_link=False, downloaded=False):
            name = p.label.split(":", 1)[1].strip() if p.label.lower().startswith("download:") else p.label
            # Two columns: everything descriptive on the left, the action buttons in
            # their own fixed column on the right. Previously the buttons shared a
            # layout with the title and chips while a full-width note sat beneath —
            # so a long note widened the row past the viewport and squeezed the
            # buttons. In a separate column nothing on the left can reach them.
            row = QWidget()
            shell = QHBoxLayout(row)
            shell.setContentsMargins(8, 6, 8, 6)
            shell.setSpacing(8)
            left = QWidget()
            outer = QVBoxLayout(left)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(2)
            left.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
            shell.addWidget(left, 1)
            actions = QHBoxLayout()
            actions.setContentsMargins(0, 0, 0, 0)
            actions.setSpacing(6)
            shell.addLayout(actions, 0)
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            top.setSpacing(8)
            star = "\u2605 " if p.id == rec_id else ""
            title_width = 430 - (96 if p.supports_audio else 0)
            title = elided(f"{star}{name}", title_width, tooltip=p.note or None)
            title.setStyleSheet("font-weight:600;")
            top.addWidget(title, 1)
            if downloaded:
                have = QLabel("On disk")
                have.setToolTip("Already in one of your model folders \u2014 selecting "
                                "this won't download anything.")
                have.setStyleSheet(
                    "color:#3ddc84;border:1px solid #3ddc84;border-radius:6px;"
                    "padding:1px 8px;font-weight:600;")
                top.addWidget(have, 0)
            if p.supports_audio:
                # Audio capability changes what a video caption can contain, so it
                # belongs beside the size/fit chips rather than buried in the note.
                audio_chip = QLabel("\U0001F50A Audio")
                audio_chip.setToolTip(
                    "Hears the clip's audio, so captions can include spoken "
                    "dialogue, music and ambient sound.")
                audio_chip.setStyleSheet(
                    "color:#45B964;border:1px solid #45B964;border-radius:6px;"
                    "padding:1px 8px;font-weight:600;")
                top.addWidget(audio_chip, 0)
            if p.kind == "hf" and p.vram_gb > 0:
                tier = model_size_tier(p.vram_gb)
                chip = QLabel(f"{tier} \u00b7 ~{p.vram_gb:.0f}GB")
                chip.setStyleSheet("color:#9aa4b6;border:1px solid #2a2f3a;border-radius:6px;padding:1px 6px;")
                top.addWidget(chip, 0)
                if vram:
                    fit = vram_fit(p.vram_gb, vram)
                    badge = QLabel(badge_text[fit])
                    badge.setStyleSheet(
                        f"color:{badge_colors[fit]};border:1px solid {badge_colors[fit]};"
                        "border-radius:6px;padding:1px 8px;font-weight:600;"
                    )
                    top.addWidget(badge, 0)
            if allow_use:
                use_btn = QPushButton("Use")
                use_btn.setMinimumWidth(58)
                use_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                use_btn.setCursor(Qt.PointingHandCursor)
                use_btn.clicked.connect(lambda _c, prof=p: self._pick_model(task, prof, dlg))
                actions.addWidget(use_btn, 0)
            if p.kind == "hf" and p.hf_repo and not show_link:
                hf_btn = QPushButton("HF")
                hf_btn.setMinimumWidth(46)
                hf_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
                hf_btn.setToolTip(f"https://huggingface.co/{p.hf_repo}")
                hf_btn.clicked.connect(lambda _c, repo=p.hf_repo: webbrowser.open(f"https://huggingface.co/{repo}"))
                actions.addWidget(hf_btn, 0)
            outer.addLayout(top)
            if show_link and p.kind == "hf" and p.hf_repo:
                link = QLabel(f'<a href="https://huggingface.co/{p.hf_repo}" style="color:#6cb6ff;">huggingface.co/{p.hf_repo}</a>')
                link.setOpenExternalLinks(True)
                link.setObjectName("Hint")
                link.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
                outer.addWidget(link)
            if p.note:
                # One line, elided: a wrapping note made rows 42, 54 or 70px tall
                # depending on how much its author wrote. Full text on hover.
                note = elided(p.note, CONTENT_W - 40, tooltip=p.note)
                note.setObjectName("Hint")
                note.setStyleSheet("color:#A78BFA;")
                outer.addWidget(note)
            return row

        def make_detected_row(path):
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(8, 6, 8, 6)
            rl.setSpacing(8)
            col = QVBoxLayout()
            col.setContentsMargins(0, 0, 0, 0)
            col.setSpacing(0)
            fname = elided(path.name, CONTENT_W - 260)
            fname.setStyleSheet("font-weight:600;")
            # Just the last folder or two: the leading path is nearly always the
            # same across a folder's models and crowds out the part that differs.
            where = elided(self._tail_dir(path.parent), CONTENT_W - 260,
                           tooltip=str(path.parent))
            where.setObjectName("Hint")
            col.addWidget(fname)
            col.addWidget(where)
            rl.addLayout(col, 1)
            # estimated VRAM from file size (+ paired projector), shown like the
            # recommended rows but flagged with "~" since it's a size approximation.
            est = estimate_gguf_vram_gb(path, guess_mmproj_for(path, self._discovered_mmprojs))
            if est > 0:
                tier = model_size_tier(est)
                chip = QLabel(f"{tier} \u00b7 ~{est:.0f}GB")
                chip.setToolTip("Estimated from file size (weights + projector + headroom); "
                                "actual VRAM also depends on context length.")
                chip.setStyleSheet("color:#9aa4b6;border:1px solid #2a2f3a;border-radius:6px;padding:1px 6px;")
                rl.addWidget(chip, 0)
                if vram:
                    fit = vram_fit(est, vram)
                    badge = QLabel("~" + badge_text[fit])
                    badge.setToolTip("Estimated fit on your card.")
                    badge.setStyleSheet(
                        f"color:{badge_colors[fit]};border:1px solid {badge_colors[fit]};"
                        "border-radius:6px;padding:1px 8px;font-weight:600;"
                    )
                    rl.addWidget(badge, 0)
            use = QPushButton("Use")
            use.setCursor(Qt.PointingHandCursor)
            if local_mode:
                use.clicked.connect(lambda _c, p=path: self._use_local_gguf(task, p, dlg))
            else:
                use.clicked.connect(lambda _c, p=path: self._use_detected_external(task, p, dlg))
            rl.addWidget(use, 0)
            return row

        try:
            found, mmprojs = discover_local_gguf_models(self._settings_with_current_dirs())
        except Exception:
            found, mmprojs = [], []
        self._discovered_mmprojs = mmprojs

        hf_profiles = [p for p in profiles_for_task(task) if p.kind == "hf"]
        hf_profiles.sort(key=lambda p: (0 if p.id == rec_id else 1,
                                        rank[vram_fit(p.vram_gb, vram)], -p.vram_gb))

        if local_mode:
            # 1) Models already on disk — the usual pick when running local llama.cpp
            add_section("Downloaded in your folders")
            if not found:
                # Name the folders actually searched. "Nothing found" is unactionable
                # when a download went somewhere the scan doesn't look — usually
                # because Model download location is set to the HF cache, or the
                # models folder was edited but not saved.
                roots = model_search_roots(self._settings_with_current_dirs())
                listed = "\n".join(f"\u2022 {self._short_dir(r)}"
                                    for r in roots[:6] if r.exists())
                add_hint("No GGUF files found. Searched:\n"
                         + (listed or "(none of the configured folders exist)")
                         + "\n\nDownloads go to the folder set by 'Model download "
                           "location' on this page \u2014 if that's the Hugging Face "
                           "cache, files land there rather than your models folder.")
            else:
                for path in found:
                    add_row(make_detected_row(path))
            # No "Custom & server options" here. The server aliases only mean
            # something against an external server — in local mode the app launches
            # its own, and selecting one used to be silently reset when settings were
            # applied. Custom local paths are already covered by the detected list
            # and the Browse… field on the Models page, and the custom-HF form is a
            # footer button rather than a row pretending to be a model.
            # A recommended model you've already fetched shouldn't keep sitting under
            # "download" — that's what makes it look like the download went nowhere.
            settings_now = self._settings_with_current_dirs()
            on_disk, to_get = [], []
            for p in hf_profiles:
                have = False
                try:
                    first = _split_filenames(p.model_filename)
                    have = bool(first) and locate_existing_model_file(
                        settings_now, p.hf_repo, first[0]) is not None
                except Exception:
                    have = False
                (on_disk if have else to_get).append(p)
            if on_disk:
                add_section("Already downloaded")
                for p in on_disk:
                    add_row(make_profile_row(p, allow_use=True, downloaded=True))
            add_section("Recommended to download")
            for p in to_get:
                add_row(make_profile_row(p, allow_use=True))
        else:
            add_section("Detected models")
            if not found:
                add_hint("No downloaded GGUF files found. Add your server's model folders on "
                         "the LLM Models page (Browse\u2026 / Detect model folders).")
            else:
                for path in found:
                    add_row(make_detected_row(path))
            # Server aliases belong here, not in local mode: they name a model the
            # external server already has loaded.
            aliases = [p for p in profiles_for_task(task) if p.kind == "server"]
            if aliases:
                add_section("Name your server's model")
                add_hint("Use one of these if your server already has the model "
                         "loaded under that name.")
                for p in aliases:
                    add_row(make_profile_row(p, allow_use=True))
            add_section("Recommended models")
            add_hint("Download and configure these in your external server, then select the "
                     "model there or enter its name. Links go to Hugging Face.")
            for p in hf_profiles:
                add_row(make_profile_row(p, allow_use=False, show_link=True))

        def _sync_widths():
            width = row_width()
            for i in range(listing.count()):
                item = listing.item(i)
                widget = listing.itemWidget(item)
                if widget is not None:
                    widget.setFixedWidth(width)
                item.setSizeHint(QSize(width, item.sizeHint().height()))

        class _WidthSync(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Resize:
                    _sync_widths()
                return False

        sync = _WidthSync(dlg)
        listing.viewport().installEventFilter(sync)

        box = QDialogButtonBox(QDialogButtonBox.Close)
        # The custom-HF path is a form, not a model, so it's a button rather than a
        # row sitting among entries that have sizes and fit badges.
        rescan = QPushButton("Rescan folders")
        rescan.setToolTip("Look again for GGUF files — use after a download finishes")
        rescan.clicked.connect(lambda: (dlg.accept(), self._open_model_picker(task)))
        box.addButton(rescan, QDialogButtonBox.ResetRole)
        custom = [p for p in profiles_for_task(task) if p.kind == "custom_hf"]
        if custom:
            hf_btn = QPushButton("Use a Hugging Face repo\u2026")
            hf_btn.setToolTip("Point at any repo and filename on Hugging Face")
            hf_btn.clicked.connect(
                lambda _c, prof=custom[0]: self._pick_model(task, prof, dlg))
            box.addButton(hf_btn, QDialogButtonBox.ActionRole)
        box.rejected.connect(dlg.reject)
        box.accepted.connect(dlg.accept)
        lay.addWidget(box)
        dlg.exec()

    def _external_model_key(self, path: Path) -> str:
        """Best-guess model name string an external server would expose for a GGUF
        on disk. LM Studio uses a publisher/model key from its folder layout
        (~/.lmstudio/models/<publisher>/<model>/<file>.gguf); otherwise fall back
        to the file stem. The user can edit the API model name afterward."""
        parts = path.parts
        for i, seg in enumerate(parts):
            if seg == "models" and i + 2 < len(parts) and any(a == ".lmstudio" for a in parts[:i]):
                return f"{parts[i + 1]}/{parts[i + 2]}"
        return path.stem

    def _use_detected_external(self, task: str, path, dlg) -> None:
        """External-server mode: set the API model name to the chosen on-disk model's
        likely server key. The server still owns loading (e.g. LM Studio JIT)."""
        name = self._external_model_key(Path(path))
        if f"{task}_model" in self.widgets:
            self.widgets[f"{task}_model"][1].setText(name)
        if task == "caption" and getattr(self, "bbox_same_as_caption", False):
            self._mirror_caption_to_bbox()
        self._refresh_model_label(task)
        dlg.accept()

    def _pick_model(self, task: str, profile, dlg) -> None:
        combo = self._profile_combos.get(task)
        if combo is not None:
            i = combo.findText(profile.label)
            if i >= 0:
                combo.setCurrentIndex(i)   # fires _on_profile_changed + _refresh_model_label
        # Choosing a server alias is an explicit "use the external server" action.
        if profile.kind == "server" and "server_start_mode" in self.widgets:
            mode_combo = self.widgets["server_start_mode"][1]
            j = mode_combo.findText("existing")
            if j >= 0:
                mode_combo.setCurrentIndex(j)
        dlg.accept()

    def _detected_gpus(self, refresh: bool = False):
        """All detected GPUs (cross-vendor), cached for the session. Pass refresh=True
        to re-probe (the picker's Re-detect button)."""
        if refresh or not hasattr(self, "_gpus_cache"):
            try:
                self._gpus_cache = detect_gpus(self.settings)
            except Exception:
                self._gpus_cache = []
        return self._gpus_cache

    def _selected_gpu_devices(self) -> set:
        """llama.cpp device tokens currently ticked in the picker (live), or from
        saved settings if the picker isn't built. Empty set = use all detected GPUs."""
        entry = self.widgets.get("llama_devices")
        if entry and entry[0] == "_gpupicker":
            checks = getattr(entry[1], "_device_checks", [])
            return {dev for dev, cb in checks if cb.isChecked()}
        return {x.strip() for x in str(getattr(self.settings, "llama_devices", "")).split(",")
                if x.strip()}

    def _target_gpu(self):
        """The single GPU model recommendations are sized against: the picked one, or
        the largest detected (which is also what the picker defaults to)."""
        gpus = self._detected_gpus()
        if not gpus:
            return None
        sel = self._selected_gpu_devices()
        if sel:
            for g in gpus:
                if g.device in sel:
                    return g
        return max(gpus, key=lambda g: g.vram_total_gb or 0)

    def _detected_vram(self) -> float | None:
        """VRAM (GB) of the selected GPU — drives the model recommendations and fit
        badges. The captioner uses one GPU, so this is a single card's VRAM."""
        g = self._target_gpu()
        return g.vram_total_gb if g else None

    def _detected_gpu_label(self) -> str:
        """Human label for the selected GPU, for the recommendation header."""
        g = self._target_gpu()
        return g.name if (g and g.name) else "your GPU"

    def _on_gpu_selection_changed(self) -> None:
        """A new GPU pick changes both the model recommendations (different VRAM) and
        which llama.cpp build is offered (different backend). Apply it to the dialog's
        settings immediately — via a fresh copy, so Cancel still discards it — so the
        download prompt and status reflect it without the user having to Save first."""
        sel = self._selected_gpu_devices()
        self.settings = replace(self.settings, llama_devices=(next(iter(sel)) if sel else ""))
        for task in ("caption", "bbox"):
            try:
                self._refresh_model_label(task)
            except Exception:
                pass

    def _append_dir_line(self, edit, path: str) -> None:
        """Add a folder as a new line in a dirlist edit, skipping duplicates."""
        path = path.strip()
        if not path:
            return
        existing = [ln.strip() for ln in edit.toPlainText().splitlines() if ln.strip()]
        if path in existing:
            return
        existing.append(path)
        edit.setPlainText("\n".join(existing))

    def _append_model_dir(self, edit) -> None:
        start = str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "Choose a model folder", start)
        if path:
            self._append_dir_line(edit, path)

    def _detect_server_dirs(self, edit) -> None:
        """Add the default model folders for the built-in servers that actually
        exist on this machine (LM Studio, llama.cpp cache, Ollama), de-duplicated
        against what's already listed."""
        found = known_server_model_dirs()
        existing = {ln.strip() for ln in re.split(r"[\r\n;]+", edit.toPlainText()) if ln.strip()}
        added = []
        for d in found:
            s = str(d)
            if s not in existing:
                self._append_dir_line(edit, s)
                existing.add(s)
                added.append(s)
        if added:
            QMessageBox.information(
                self, "Model folders",
                "Added these model folders:\n\n" + "\n".join(added) +
                "\n\nNote: Ollama stores models as blobs (not .gguf), so its folder "
                "usually won't surface loadable files here.",
            )
        elif found:
            QMessageBox.information(self, "Model folders",
                                    "Your servers' default model folders are already listed.")
        else:
            QMessageBox.information(
                self, "Model folders",
                "No default server model folders were found on this machine "
                "(LM Studio, llama.cpp, Ollama). Use Browse\u2026 to add one manually.",
            )

    def _browse_into(self, edit: QLineEdit, is_dir: bool) -> None:
        start = edit.text().strip() or str(default_models_dir())
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "Choose folder", start)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose file", start)
        if path:
            edit.setText(path)

    def _browse_model_file(self, edit: QLineEdit) -> None:
        start = edit.text().strip() or str(default_models_dir())
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose GGUF file", start, "GGUF files (*.gguf);;All files (*)"
        )
        if path:
            edit.setText(path)

    def _load_custom_presets(self) -> dict:
        raw = self._qsettings.value("server_presets_custom", "")
        out: dict = {}
        if raw:
            try:
                data = json.loads(raw)
                for name, val in data.items():
                    if isinstance(val, (list, tuple)) and len(val) == 3:
                        out[str(name)] = (str(val[0]), str(val[1]), str(val[2]))
            except (ValueError, TypeError):
                pass
        return out

    def _save_custom_presets(self) -> None:
        data = {name: list(val) for name, val in self._custom_presets.items()}
        self._qsettings.setValue("server_presets_custom", json.dumps(data))

    def _all_presets(self) -> dict:
        # Built-ins take precedence; custom names that clash are blocked at save time.
        merged = dict(self.SERVER_PRESETS)
        merged.update(self._custom_presets)
        return merged

    def _populate_preset_combo(self) -> None:
        combo = self._preset_combo
        if combo is None:
            return
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Select a server…")
        combo.addItems(list(self.SERVER_PRESETS.keys()))
        if self._custom_presets:
            combo.insertSeparator(combo.count())
            combo.addItems(list(self._custom_presets.keys()))
        # Remember the last server the user picked; otherwise reflect whichever
        # preset matches the saved settings, so the dropdown isn't always blank.
        target = QSettings("FantasticCaptioningKit", "QtApp").value("last_server_preset")
        if not (isinstance(target, str) and target in self._all_presets()):
            target = self._preset_matching_settings()
        idx = combo.findText(target) if target else -1
        combo.setCurrentIndex(idx if idx > 0 else 0)
        combo.blockSignals(False)

    def _preset_matching_settings(self) -> str | None:
        for name, preset in self._all_presets().items():
            base_url, _key, start_mode = preset
            if base_url == self.settings.base_url and start_mode == self.settings.server_start_mode:
                return name
        return None

    def _apply_preset(self, name: str) -> None:
        preset = self._all_presets().get(name)
        if not preset:
            return  # the "Select a server…" placeholder or a separator
        QSettings("FantasticCaptioningKit", "QtApp").setValue("last_server_preset", name)
        base_url, api_key, start_mode = preset
        if "base_url" in self.widgets:
            self.widgets["base_url"][1].setText(base_url)
        if "api_key" in self.widgets:
            self.widgets["api_key"][1].setText(api_key)
        if "server_start_mode" in self.widgets:
            combo = self.widgets["server_start_mode"][1]
            idx = combo.findText(start_mode)
            if idx >= 0:
                combo.setCurrentIndex(idx)
        self._refresh_server_panel()

    def _manage_presets(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Manage server presets")
        dlg.resize(420, 320)
        lay = QVBoxLayout(dlg)
        lay.addWidget(QLabel("Custom presets (built-in presets can't be edited):"))
        listing = QListWidget()
        lay.addWidget(listing, 1)

        def refresh():
            listing.clear()
            for name, (url, _key, mode) in self._custom_presets.items():
                listing.addItem(f"{name}  —  {url}  ({mode})")
            if not self._custom_presets:
                item = QListWidgetItem("No custom presets yet.")
                item.setFlags(Qt.NoItemFlags)
                listing.addItem(item)

        def selected_name() -> str | None:
            row = listing.currentRow()
            names = list(self._custom_presets.keys())
            if 0 <= row < len(names):
                return names[row]
            return None

        def save_current():
            name, ok = QInputDialog.getText(dlg, "Save preset", "Preset name:")
            name = (name or "").strip()
            if not ok or not name:
                return
            if name in self.SERVER_PRESETS:
                QMessageBox.warning(dlg, "Name in use",
                                    "That name belongs to a built-in preset. Pick another.")
                return
            if name in self._custom_presets:
                if QMessageBox.question(dlg, "Overwrite preset?",
                                        f"A custom preset named “{name}” already exists. "
                                        "Overwrite it?") != QMessageBox.Yes:
                    return
            self._custom_presets[name] = (
                self.widgets["base_url"][1].text().strip(),
                self.widgets["api_key"][1].text().strip(),
                self.widgets["server_start_mode"][1].currentText(),
            )
            self._save_custom_presets()
            self._populate_preset_combo()
            refresh()

        def delete_selected():
            name = selected_name()
            if not name:
                return
            if QMessageBox.question(dlg, "Delete preset?",
                                    f"Delete the custom preset “{name}”?") != QMessageBox.Yes:
                return
            self._custom_presets.pop(name, None)
            self._save_custom_presets()
            self._populate_preset_combo()
            refresh()

        btns = QHBoxLayout()
        save_btn = QPushButton("Save current settings as preset…")
        save_btn.clicked.connect(save_current)
        del_btn = QPushButton("Delete selected")
        del_btn.clicked.connect(delete_selected)
        btns.addWidget(save_btn)
        btns.addWidget(del_btn)
        btns.addStretch(1)
        lay.addLayout(btns)

        box = QDialogButtonBox(QDialogButtonBox.Close)
        box.rejected.connect(dlg.reject)
        box.accepted.connect(dlg.accept)
        lay.addWidget(box)

        refresh()
        dlg.exec()

    def _cached_latest_build(self):
        """Latest build number from the background update check, if any (Stage 4b
        populates this). None until then — age-based 'recommended' still works."""
        try:
            val = QSettings("FantasticCaptioningKit", "QtApp").value("llama_latest_build")
            return int(val) if val is not None else None
        except (TypeError, ValueError):
            return None

    def _refresh_llama_status(self) -> None:
        label = getattr(self, "_llama_status_label", None)
        button = getattr(self, "_llama_action_btn", None)
        if label is None or button is None:
            return
        record = read_installed_llama()
        state = update_state(record, self._cached_latest_build())
        kind = state["state"]
        if kind == "none":
            label.setText("Not installed — fetch a prebuilt build for your system.")
            label.setStyleSheet("color: #9aa3ad;")
            button.setText("Get llama.cpp")
            return
        build = f"b{record.build}" if record.build else "?"
        age = state["age_days"]
        age_str = f", {age}d old" if age is not None else ""
        base = f"Installed: llama.cpp {build} ({record.backend}{age_str})"
        if kind == "recommended":
            label.setText(base + " · update recommended")
            label.setStyleSheet("color: #E0A33B;")
        elif kind == "available":
            label.setText(base + " · newer build available")
            label.setStyleSheet("color: #9aa3ad;")
        elif self._cached_latest_build() is not None:
            label.setText(base + " · up to date")
            label.setStyleSheet("color: #9aa3ad;")
        else:
            label.setText(base)
            label.setStyleSheet("color: #9aa3ad;")
        button.setText("Update")

    def _refresh_llama_path_placeholder(self) -> None:
        """Show the resolved binary as grey placeholder text so the path is
        visible after a Get llama.cpp install, while the override stays blank
        (and thus keeps auto-tracking future updates)."""
        widget = self.widgets.get("llama_server_path")
        if widget is None:
            return
        detected = find_llama_server()
        if detected is not None:
            widget[1].setPlaceholderText(f"Auto-detected: {detected}")
        else:
            widget[1].setPlaceholderText("Auto-detect (managed install or PATH)")

    def _current_start_mode(self) -> str:
        widget = self.widgets.get("server_start_mode")
        if widget is not None:
            try:
                return widget[1].currentText()
            except Exception:
                pass
        return self.settings.server_start_mode

    def _current_external_label(self) -> str:
        base = ""
        widget = self.widgets.get("base_url")
        if widget is not None:
            base = widget[1].text().strip()
        for name, preset in self._all_presets().items():
            if preset[0] == base:
                return name
        return base or "an external server"

    def _refresh_server_panel(self) -> None:
        label = getattr(self, "_srv_panel_label", None)
        button = getattr(self, "_srv_panel_btn", None)
        if label is None or button is None:
            return
        main = self.parent()
        running = bool(getattr(main, "_server_is_running", lambda: False)()) if main else False
        nomodel_btn = getattr(self, "_srv_panel_nomodel_btn", None)
        mode = self._current_start_mode()
        if mode != "local":
            # An external/managed-elsewhere server: nothing for us to start or stop.
            label.setText(f"Running external server \u2014 set to {self._current_external_label()}.")
            label.setStyleSheet("color: #9aa3ad;")
            button.setVisible(False)
            if nomodel_btn is not None:
                nomodel_btn.setVisible(False)
            return
        button.setVisible(True)
        binary = find_llama_server()
        if nomodel_btn is not None:
            nomodel_btn.setVisible(
                bool(not running and binary is not None and llama_server_supports_router(binary))
            )
        button.setEnabled(True)
        if running:
            label.setText("Local llama-server is running.")
            label.setStyleSheet("color: #3ddc84;")
            button.setText("Stop")
        elif binary is None:
            label.setText("llama.cpp isn't installed yet \u2014 use \u201cGet llama.cpp\u201d below.")
            label.setStyleSheet("color: #9aa3ad;")
            button.setText("Start")
            button.setEnabled(False)
        elif not has_model_config(self.settings, "caption"):
            label.setText("No model configured yet \u2014 pick one to start the server.")
            label.setStyleSheet("color: #9aa3ad;")
            button.setText("Choose model")
        else:
            label.setText("Local llama-server is stopped.")
            label.setStyleSheet("color: #9aa3ad;")
            button.setText("Start")

    def _start_nomodel_from_prefs(self) -> None:
        main = self.parent()
        if main is None:
            return
        main._launch_local_server(model_less=True)
        QTimer.singleShot(500, self._refresh_server_panel)

    def _toggle_local_server_from_prefs(self) -> None:
        main = self.parent()
        if main is None:
            return
        if getattr(main, "_server_is_running", lambda: False)():
            main._stop_local_server()
        elif find_llama_server() is None:
            return  # button is disabled in this state anyway
        elif not has_model_config(main.settings, "caption"):
            items = self.nav.findItems("LLM Models", Qt.MatchExactly)
            if items:
                self.nav.setCurrentRow(self.nav.row(items[0]))
            return
        else:
            main._launch_local_server()   # binary present — launch directly
        QTimer.singleShot(500, self._refresh_server_panel)

    def _live_settings(self):
        """Settings with the not-yet-saved GPU pick and backend override applied, so
        the 'Get llama.cpp' build choice matches what's selected in the dialog right
        now (e.g. picking the Vulkan iGPU on an NVIDIA+iGPU laptop fetches the Vulkan
        build, not CUDA)."""
        kw = {}
        sel = self._selected_gpu_devices()
        if sel:
            kw["llama_devices"] = next(iter(sel))
        hint = self.widgets.get("llama_backend_hint")
        if hint and hint[0] == "choice":
            kw["llama_backend_hint"] = hint[1].currentText()
        return replace(self.settings, **kw) if kw else self.settings

    def _acquire_llama(self) -> None:
        button = self._llama_action_btn
        button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            plan = plan_llama_acquisition(self._live_settings())
        finally:
            QApplication.restoreOverrideCursor()
        if plan is None:
            button.setEnabled(True)
            QMessageBox.information(
                self, "llama.cpp",
                "Couldn't find a prebuilt build for your system (or the release "
                "service is unreachable). You can set a llama-server path manually "
                "in the field below, or build from source.",
            )
            return
        proceed = QMessageBox.question(
            self, "Download llama.cpp",
            f"Download the {plan.description}?\n\n"
            f"Source: {plan.repo}\n"
            f"The download is SHA-256 verified before it's installed.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if proceed != QMessageBox.Yes:
            button.setEnabled(True)
            return
        self._llama_progress.setRange(0, 0)   # busy until we get a percentage
        self._llama_progress.setFormat("Starting\u2026")
        self._llama_progress.setVisible(True)
        self._llama_thread = LlamaInstallThread(plan)
        self._llama_thread.progress.connect(self._on_llama_progress)
        self._llama_thread.done.connect(self._on_llama_installed)
        self._llama_thread.error.connect(self._on_llama_install_error)
        self._llama_thread.start()

    def _on_llama_progress(self, text: str) -> None:
        self._llama_status_label.setText(text)
        match = re.search(r"(\d+)%", text)
        if match:
            self._llama_progress.setRange(0, 100)
            self._llama_progress.setValue(int(match.group(1)))
            self._llama_progress.setFormat("%p%")
        else:
            self._llama_progress.setRange(0, 0)   # indeterminate for verify/extract
            self._llama_progress.setFormat(text)

    def _on_llama_installed(self, record) -> None:
        self._llama_progress.setVisible(False)
        self._llama_action_btn.setEnabled(True)
        self._refresh_llama_status()
        self._refresh_llama_path_placeholder()
        if getattr(self, "_srv_panel_btn", None) is not None:
            self._refresh_server_panel()
        QMessageBox.information(
            self, "llama.cpp",
            f"Installed llama.cpp b{record.build} ({record.backend}).",
        )

    def _on_llama_install_error(self, message: str) -> None:
        self._llama_progress.setVisible(False)
        self._llama_action_btn.setEnabled(True)
        self._refresh_llama_status()
        if has_llama_backup():
            roll = QMessageBox.question(
                self, "Install failed",
                f"The llama.cpp install failed:\n\n{message}\n\n"
                "Roll back to the previously installed build?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if roll == QMessageBox.Yes and rollback_llama():
                self._refresh_llama_status()
                QMessageBox.information(self, "Rolled back", "Restored the previous llama.cpp build.")
                return
        QMessageBox.warning(self, "Install failed", message)

    def _test_server(self) -> None:
        base = self.widgets["base_url"][1].text().strip()
        key = self.widgets["api_key"][1].text().strip()
        if not base:
            QMessageBox.warning(self, "No server URL", "Enter a server URL first.")
            return
        try:
            model_ids = server_model_ids(base, key, timeout=5.0)
        except Exception as exc:
            QMessageBox.warning(
                self, "Test failed",
                "Couldn't reach the server.\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                "Check that the server is running, the URL and port are correct, "
                "and the path ends in /v1.",
            )
            return
        count = len(model_ids)
        if count:
            QMessageBox.information(
                self, "Test passed",
                "Test passed — the server is responding properly.\n\n"
                f"Reported {count} model{'s' if count != 1 else ''} at /models.",
            )
        else:
            QMessageBox.warning(
                self, "No models loaded",
                "The server responded, but /models returned no models.\n\n"
                "Load a model (a vision model is required for captioning), then test again.",
            )

    def _open_profiles_file(self) -> None:
        path = default_profiles_path()
        try:
            if not path.exists():
                path.write_text(json.dumps(profile_seed_data(), indent=2), encoding="utf-8")
            if hasattr(os, "startfile"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except OSError as exc:
            QMessageBox.critical(self, "Could not open profiles file", str(exc))

    def _collect(self) -> None:
        # Prompts and frame rules live outside the settings dataclass, so they
        # commit here rather than through _make_field.
        if hasattr(self, "_pe_pending"):
            self._pe_commit()
        if hasattr(self, "_tg_working"):
            self._tg_commit()
        if hasattr(self, "_mt_working"):
            self._mt_commit()
        kwargs = {}
        for key, (kind, w) in self.widgets.items():
            if kind == "text":
                kwargs[key] = w.text().strip()
            elif kind == "bool":
                kwargs[key] = w.isChecked()
            elif kind == "int":
                kwargs[key] = w.value()
            elif kind == "float":
                kwargs[key] = w.value()
            elif kind == "multiline":
                kwargs[key] = w.toPlainText()
            elif kind == "choice":
                kwargs[key] = w.currentText()
            elif kind == "_gpupicker":
                checks = getattr(w, "_device_checks", [])
                if checks:
                    kwargs[key] = ",".join(dev for dev, cb in checks if cb.isChecked())
                else:
                    # single GPU / detection failed: no checkboxes, keep saved value
                    kwargs[key] = getattr(self.settings, key)
            elif kind == "font":
                kwargs[key] = "" if w.currentText() == "(auto)" else w.currentText().strip()
            elif kind == "color":
                kwargs[key] = w.text().strip() or getattr(self.settings, key)
        # profile ids come from the Models-page combos, not the generic widgets
        kwargs["caption_profile_id"] = profile_id_from_label(
            "caption", self._profile_combos["caption"].currentText()
        )
        kwargs["bbox_profile_id"] = profile_id_from_label(
            "bbox", self._profile_combos["bbox"].currentText()
        )
        # When linked, the bbox model fields are authoritatively mirrored from caption.
        if self.bbox_same_as_caption:
            kwargs["bbox_profile_id"] = kwargs["caption_profile_id"]
            for suffix in ("model", "hf_repo", "model_filename", "mmproj_filename",
                           "local_model_path", "local_mmproj_path"):
                kwargs[f"bbox_{suffix}"] = kwargs[f"caption_{suffix}"]
        self.result = replace(self.settings, **kwargs)
        if hasattr(self, "_tags_list"):
            seen, tags = set(), []
            for i in range(self._tags_list.count()):
                t = self._tags_list.item(i).text().strip()
                if t and t not in seen:
                    seen.add(t); tags.append(t)
            self.tags_result = tags

    def _save(self) -> None:
        self._collect()
        self.accept()

    def _apply(self) -> None:
        """Commit current settings to the running app without closing, so the user
        can set up a model/server, see it take effect, and keep editing."""
        self._collect()
        parent = self.parent()
        if parent is not None and hasattr(parent, "_apply_preferences_result"):
            parent._apply_preferences_result(self)
        btn = self._apply_btn
        if btn is not None:
            btn.setText("Applied \u2713")
            btn.setEnabled(False)
            QTimer.singleShot(1100, lambda: (btn.setText("Apply"), btn.setEnabled(True)))


class FilmstripDelegate(QStyledItemDelegate):
    """Custom-paints filmstrip cells so unsaved items show red, shadowed text.

    A stylesheet on the list makes Qt ignore per-item foreground brushes, and QSS
    has no text-shadow, so the only reliable place to do both is a delegate.
    """

    def __init__(self, window: "MainWindow") -> None:
        super().__init__(window)
        self.window = window

    def paint(self, painter, option, index) -> None:
        t = self.window.theme
        rect = option.rect
        unsaved = bool(index.data(UNSAVED_ROLE))
        selected = bool(option.state & QStyle.State_Selected)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setRenderHint(QPainter.TextAntialiasing, True)

        if bool(index.data(SEPARATOR_ROLE)):
            # The divider: everything past it is out of the dataset.
            painter.setPen(QPen(QColor(t.border), 1, Qt.DashLine))
            x = rect.center().x()
            painter.drawLine(QPointF(x, rect.top() + 6), QPointF(x, rect.bottom() - 6))
            painter.setPen(QColor(t.text_secondary))
            font = painter.font()
            font.setPointSizeF(max(7.0, font.pointSizeF() - 2))
            painter.setFont(font)
            painter.drawText(rect, int(Qt.AlignCenter), "bypassed")
            painter.restore()
            return
        # Bypassed files are dimmed: still visible and still individually
        # captionable, but visibly not part of the dataset.
        if bool(index.data(BYPASS_ROLE)):
            painter.setOpacity(0.42)

        # icon, centered near the top of the cell
        isz = option.decorationSize
        icon = index.data(Qt.DecorationRole)
        top = rect.y() + 8
        icon_rect = None
        if isinstance(icon, QIcon):
            pm = icon.pixmap(isz)
            px = rect.x() + (rect.width() - pm.width()) // 2
            painter.drawPixmap(px, top, pm)
            icon_rect = QRect(px, top, pm.width(), pm.height())

        # selected: a 2px accent border hugging the thumbnail (mockup look)
        if selected and icon_rect is not None:
            pen = QPen(QColor(t.accent))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(QRectF(icon_rect).adjusted(-1, -1, 1, 1), 6, 6)

        # video duration badge: dark pill at the thumbnail's bottom-right, so clips
        # are recognisable at a glance without a play overlay cluttering the strip.
        duration = index.data(DURATION_ROLE)
        if duration and icon_rect is not None:
            bfont = QFont(option.font)
            bfont.setPointSizeF(max(6.5, option.font.pointSizeF() - 1.5))
            bfm = QFontMetrics(bfont)
            bw = bfm.horizontalAdvance(str(duration)) + 8
            bh = bfm.height() + 2
            brect = QRectF(icon_rect.right() - bw - 2, icon_rect.bottom() - bh - 2, bw, bh)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(0, 0, 0, 190))
            painter.drawRoundedRect(brect, 4, 4)
            painter.setFont(bfont)
            painter.setPen(QColor(235, 238, 241))
            painter.drawText(brect, Qt.AlignCenter, str(duration))
            painter.setFont(option.font)

        # filename below the icon, single elided line with a drop shadow.
        # selected (accent) wins over unsaved (amber); otherwise muted secondary.
        text = index.data(Qt.DisplayRole) or ""
        font = option.font
        painter.setFont(font)
        fm = QFontMetrics(font)
        text_rect = QRect(
            rect.x() + 2,
            top + isz.height() + 3,
            rect.width() - 4,
            fm.height() + 4,
        )
        elided = fm.elidedText(text, Qt.ElideRight, text_rect.width())
        flags = int(Qt.AlignHCenter | Qt.AlignTop)
        if selected:
            color = QColor(t.accent)
        elif unsaved:
            color = QColor(t.warning)
        else:
            color = QColor(t.text_secondary)
        painter.setPen(QColor(0, 0, 0, 200))
        painter.drawText(text_rect.translated(1, 1), flags, elided)
        painter.setPen(color)
        painter.drawText(text_rect, flags, elided)

        # unsaved-changes dot: amber circle at the thumbnail's top-right corner,
        # ringed in the filmstrip background so it punches off the image and the
        # selected border alike. Scale + fade is driven per-item by an animation.
        if icon_rect is not None:
            key = index.data(Qt.UserRole)
            progress = self.window._dirty_dot.get(key, 1.0 if unsaved else 0.0)
            if progress > 0.001:
                r = 5.0 * progress
                ring = 2.0
                cx = min(float(icon_rect.right()) + 1.0, float(rect.right()) - (r + ring))
                cy = max(float(icon_rect.top()) - 1.0, float(rect.top()) + (r + ring))
                painter.setOpacity(progress)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(t.surface_1))
                painter.drawEllipse(QPointF(cx, cy), r + ring, r + ring)
                painter.setBrush(QColor(t.warning))
                painter.drawEllipse(QPointF(cx, cy), r, r)
                painter.setOpacity(1.0)

        # guidance-changed dot: violet circle at the TOP-LEFT corner, ringed the
        # same way so it never collides with the amber unsaved dot opposite it.
        if icon_rect is not None and bool(index.data(STALE_ROLE)):
            r = 5.0
            ring = 2.0
            cx = max(float(icon_rect.left()) - 1.0, float(rect.left()) + (r + ring))
            cy = max(float(icon_rect.top()) - 1.0, float(rect.top()) + (r + ring))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(t.surface_1))
            painter.drawEllipse(QPointF(cx, cy), r + ring, r + ring)
            painter.setBrush(QColor(STALE_COLOR))
            painter.drawEllipse(QPointF(cx, cy), r, r)
        # omit marker: violet slashed dot on the LEFT edge, just below the stale dot,
        # so it reads as a distinct shape in the same guidance colour family — "this
        # image's source .txt is omitted (image-only) even though convert mode is on".
        if icon_rect is not None and bool(index.data(OMIT_ROLE)):
            r = 5.0
            ring = 2.0
            cx = max(float(icon_rect.left()) - 1.0, float(rect.left()) + (r + ring))
            cy = max(float(icon_rect.top()) - 1.0, float(rect.top()) + (r + ring)) + 2 * (r + ring) + 2.0
            cy = min(cy, float(rect.bottom()) - (r + ring))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(t.surface_1))
            painter.drawEllipse(QPointF(cx, cy), r + ring, r + ring)
            painter.setBrush(QColor(OMIT_COLOR))
            painter.drawEllipse(QPointF(cx, cy), r, r)
            # diagonal slash, ringed-coloured so it reads as "struck out"
            d = r * 0.7
            painter.setPen(QPen(QColor(t.surface_1), 1.6, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(cx - d, cy + d), QPointF(cx + d, cy - d))
        # unapplied-edit marker: amber scissors-ish bracket at the BOTTOM-LEFT of
        # the poster. Distinct from the caption "unsaved" dot because it's a
        # different kind of unsaved: pixels, not text.
        if icon_rect is not None and bool(index.data(VIDEO_EDIT_ROLE)):
            r = 5.0
            cx = max(float(icon_rect.left()) - 1.0, float(rect.left()) + (r + 2))
            cy = min(float(icon_rect.bottom()) + 1.0, float(rect.bottom()) - (r + 2))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(t.surface_1))
            painter.drawEllipse(QPointF(cx, cy), r + 2, r + 2)
            painter.setPen(QPen(QColor(t.warning), 2.0))
            painter.setBrush(Qt.NoBrush)
            painter.drawArc(QRectF(cx - r, cy - r, r * 2, r * 2), 0, 270 * 16)
        # spec marker: amber triangle on the RIGHT edge — "this clip won't train
        # as-is on the selected model". A triangle rather than a dot so it reads as
        # a warning distinct from the round status dots.
        if icon_rect is not None and bool(index.data(SPEC_ROLE)):
            r = 5.5
            cx = min(float(icon_rect.right()) + 1.0, float(rect.right()) - (r + 2))
            cy = float(icon_rect.center().y())
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(t.surface_1))
            painter.drawEllipse(QPointF(cx, cy), r + 2, r + 2)
            tri = QPolygonF([QPointF(cx, cy - r), QPointF(cx + r, cy + r * 0.8),
                             QPointF(cx - r, cy + r * 0.8)])
            painter.setBrush(QColor(SPEC_COLOR))
            painter.drawPolygon(tri)
            painter.setPen(QPen(QColor(t.surface_1), 1.4, Qt.SolidLine, Qt.RoundCap))
            painter.drawLine(QPointF(cx, cy - r * 0.25), QPointF(cx, cy + r * 0.3))
        if icon_rect is not None and bool(index.data(REVIEW_ROLE)):
            r = 5.0
            ring = 2.0
            cx = max(float(icon_rect.left()) - 1.0, float(rect.left()) + (r + ring))
            cy = min(float(icon_rect.bottom()) + 1.0, float(rect.bottom()) - (r + ring))
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(t.surface_1))
            painter.drawEllipse(QPointF(cx, cy), r + ring, r + ring)
            painter.setBrush(QColor(REVIEW_COLOR))
            painter.drawEllipse(QPointF(cx, cy), r, r)
        # user review flag: a small red flag at the BOTTOM-RIGHT corner.
        if icon_rect is not None and bool(index.data(FLAG_ROLE)):
            fx = min(float(icon_rect.right()) - 3.0, float(rect.right()) - 11.0)
            bottom = min(float(icon_rect.bottom()) - 1.0, float(rect.bottom()) - 2.0)
            top = bottom - 13.0
            pennant = QPainterPath()
            pennant.moveTo(fx, top)
            pennant.lineTo(fx + 9.0, top + 3.0)
            pennant.lineTo(fx, top + 6.0)
            pennant.closeSubpath()
            # light halo so the flag stays visible on any thumbnail
            painter.setBrush(Qt.NoBrush)
            painter.setPen(QPen(QColor("#FFFFFF"), 3.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(QPointF(fx, top), QPointF(fx, bottom))
            painter.drawPath(pennant)
            # red flag on top
            painter.setPen(QPen(QColor(FLAG_COLOR), 1.8, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(QPointF(fx, top), QPointF(fx, bottom))
            painter.setBrush(QColor(FLAG_COLOR))
            painter.drawPath(pennant)
        painter.restore()

    def sizeHint(self, option, index) -> QSize:
        isz = option.decorationSize
        fm = QFontMetrics(option.font)
        return QSize(max(isz.width() + 20, 64), isz.height() + fm.height() + 20)


class FilmstripPreview(QWidget):
    """Designed hover-preview popup: a rounded card holding a 196x147 image plus
    a mono filename and 'i / N' index, with a diamond pointer beneath it. Floats
    above the hovered thumbnail and shows instantly on hover (no dwell/fade).
    Dark theme only.
    """

    _MONO = "'IBM Plex Mono', 'DejaVu Sans Mono', 'Consolas', monospace"
    _BORDER = "#0f848a"   # teal — popup card border + diamond pointer edges

    def __init__(self, theme: "Theme", parent=None) -> None:
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._t = theme
        self._margin = 18                     # room for the drop shadow + arrow
        self._arrow_x = self._margin + PREVIEW_W // 2

        self.card = QWidget(self)
        self.card.setObjectName("PreviewCard")
        self.card.setStyleSheet(
            f"#PreviewCard {{ background: {theme.surface_2};"
            f" border: 1px solid {self._BORDER}; border-radius: 8px; }}"
        )
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(32)
        shadow.setXOffset(0)
        shadow.setYOffset(12)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.card.setGraphicsEffect(shadow)

        cl = QVBoxLayout(self.card)
        cl.setContentsMargins(PREVIEW_PAD, PREVIEW_PAD, PREVIEW_PAD, PREVIEW_PAD)
        cl.setSpacing(0)

        # A vertical stack of marker banners, rebuilt per image (unsaved, guidance
        # changed, omitted, problems, flagged). Empty -> zero height, image sits at top.
        self._banner_box = QWidget(self.card)
        self._banner_lay = QVBoxLayout(self._banner_box)
        self._banner_lay.setContentsMargins(0, 0, 0, 0)
        self._banner_lay.setSpacing(4)
        self._banner_labels: list[QLabel] = []
        cl.addWidget(self._banner_box)

        self.image = QLabel(self.card)
        self.image.setFixedSize(PREVIEW_IMG_W, PREVIEW_IMG_H)
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setStyleSheet(f"background: {theme.surface_0}; border-radius: 5px;")
        cl.addWidget(self.image)

        meta = QHBoxLayout()
        meta.setContentsMargins(4, 7, 4, 2)
        meta.setSpacing(8)
        self.name = QLabel(self.card)
        self.name.setStyleSheet(
            f"font-family: {self._MONO}; font-size: 11px; color: {theme.text_primary};"
        )
        self.idx = QLabel(self.card)
        self.idx.setStyleSheet(
            f"font-family: {self._MONO}; font-size: 10px; color: {theme.text_muted};"
        )
        meta.addWidget(self.name, 1)
        meta.addWidget(self.idx, 0)
        cl.addLayout(meta)

        self.card.setFixedWidth(PREVIEW_W)
        self.card.move(self._margin, self._margin)
        self._resize_to_card()

    def _resize_to_card(self) -> None:
        """Size the window to the card's current height (which changes when the
        unsaved banner is shown or hidden) plus shadow margin and the arrow."""
        self.card.adjustSize()
        ch = self.card.height()
        self.resize(PREVIEW_W + 2 * self._margin, ch + 2 * self._margin + PREVIEW_ARROW)

    # ---- painting -------------------------------------------------------
    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        half = PREVIEW_ARROW / 2
        cx = float(self._arrow_x)
        cy = float(self._margin + self.card.height())   # at the card's bottom edge
        top = QPointF(cx, cy - half)
        right = QPointF(cx + half, cy)
        bottom = QPointF(cx, cy + half)
        left = QPointF(cx - half, cy)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self._t.surface_2))
        p.drawPolygon(QPolygonF([top, right, bottom, left]))
        pen = QPen(QColor(self._BORDER))
        pen.setWidth(1)
        p.setPen(pen)
        p.drawLine(right, bottom)     # the two lower edges read as the tail outline
        p.drawLine(bottom, left)

    # ---- content + show/hide -------------------------------------------
    def set_content(self, pixmap: QPixmap, name: str, index_text: str,
                    banners: tuple = ()) -> None:
        """banners: an iterable of (text, bg_color, fg_color, tooltip) specs, painted
        as a vertical stack above the image in the order given."""
        fm = QFontMetrics(self.name.font())
        self.name.setText(fm.elidedText(name, Qt.ElideRight, PREVIEW_IMG_W - 60))
        self.idx.setText(index_text)
        if not pixmap.isNull():
            self.image.setPixmap(pixmap)
        # rebuild the banner stack
        for lbl in self._banner_labels:
            self._banner_lay.removeWidget(lbl)
            lbl.deleteLater()
        self._banner_labels = []
        for spec in banners:
            text, bg, fg = spec[0], spec[1], spec[2]
            tip = spec[3] if len(spec) > 3 else ""
            lbl = QLabel(text, self._banner_box)
            lbl.setAlignment(Qt.AlignCenter)
            lbl.setStyleSheet(
                f"background: {bg}; color: {fg};"
                f" font-family: {self._MONO}; font-size: 10px; font-weight: 600;"
                f" border-radius: 4px; padding: 3px 6px;"
            )
            if tip:
                lbl.setToolTip(tip)
            self._banner_lay.addWidget(lbl)
            self._banner_labels.append(lbl)
        self._banner_lay.setContentsMargins(0, 0, 0, 5 if self._banner_labels else 0)
        self._resize_to_card()

    def show_at(self, final_pos: QPoint, arrow_x: int) -> None:
        self._arrow_x = arrow_x
        self.move(final_pos)
        self.setWindowOpacity(1.0)
        self.show()
        self.update()


class GuidanceDiffPopup(QWidget):
    """A hover card showing the full 'guidance changed' diff (added lines in the stale
    violet, removed lines struck through and muted). Used when the sidebar section is
    too short to show the diff inline — hovering the section reveals it here. Mirrors
    the filmstrip preview's card styling. Dark theme only."""

    _BORDER = STALE_COLOR  # violet — guidance-changed family

    def __init__(self, theme: "Theme", parent=None) -> None:
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._margin = 18  # room for the drop shadow
        self.card = QWidget(self)
        self.card.setObjectName("DiffCard")
        self.card.setStyleSheet(
            f"#DiffCard {{ background: {theme.surface_2};"
            f" border: 1px solid {self._BORDER}; border-radius: 8px; }}"
        )
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(32)
        shadow.setXOffset(0)
        shadow.setYOffset(12)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.card.setGraphicsEffect(shadow)
        cl = QVBoxLayout(self.card)
        cl.setContentsMargins(12, 10, 12, 12)
        cl.setSpacing(6)
        head = QLabel("Guidance changed since last caption")
        head.setWordWrap(True)
        head.setStyleSheet(f"color: {STALE_COLOR}; font-weight: 600; font-size: 11px;")
        self._diff = QLabel()
        self._diff.setObjectName("Hint")
        self._diff.setWordWrap(True)
        self._diff.setTextFormat(Qt.RichText)
        cl.addWidget(head)
        cl.addWidget(self._diff)
        self.card.setFixedWidth(320)
        v = QVBoxLayout(self)
        v.setContentsMargins(self._margin, self._margin, self._margin, self._margin)
        v.addWidget(self.card)

    def show_diff(self, diff_html: str, target_global: QPoint, screen=None) -> None:
        """Show the diff with the card's top-left anchored near target_global (the
        section's top-right corner), clamped onto the given screen rect."""
        self._diff.setText(diff_html)
        self.adjustSize()
        gap = 8
        x = target_global.x() + gap - self._margin
        y = target_global.y() - self._margin
        if screen is not None:
            x = min(x, screen.right() - self.width())
            x = max(x, screen.left())
            y = min(y, screen.bottom() - self.height())
            y = max(y, screen.top())
        self.move(int(x), int(y))
        self.show()


class TagListPopup(QWidget):
    """A hover card listing all 'tags used' for the current image, one per line, shown
    when there are too many (or too long) to fit inline without crowding the sidebar.
    Mirrors the diff pop-out styling. Dark theme only."""

    _BORDER = STALE_COLOR  # purple — tag family

    def __init__(self, theme: "Theme", parent=None) -> None:
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._theme = theme
        self._margin = 18
        self.card = QWidget(self)
        self.card.setObjectName("TagCard")
        # The main-window stylesheet doesn't cascade into this separate top-level
        # window, so the #UsedPill rule is restated here to keep the chips styled.
        self.card.setStyleSheet(
            f"#TagCard {{ background: {theme.surface_2};"
            f" border: 1px solid {self._BORDER}; border-radius: 8px; }}"
            f" #UsedPill {{ background: {theme.accent_subtle};"
            f" border: 1px solid {theme.accent_subtle_border}; border-radius: 12px;"
            f" color: {theme.accent_on_subtle}; padding: 3px 10px; }}"
        )
        shadow = QGraphicsDropShadowEffect(self.card)
        shadow.setBlurRadius(32)
        shadow.setXOffset(0)
        shadow.setYOffset(12)
        shadow.setColor(QColor(0, 0, 0, 140))
        self.card.setGraphicsEffect(shadow)
        cl = QVBoxLayout(self.card)
        cl.setContentsMargins(12, 10, 12, 12)
        cl.setSpacing(6)
        head = QLabel("Tags used")
        head.setStyleSheet(f"color: {STALE_COLOR}; font-weight: 600; font-size: 11px;")
        cl.addWidget(head)
        self._list = QWidget()
        self._list_lay = QVBoxLayout(self._list)
        self._list_lay.setContentsMargins(0, 0, 0, 0)
        self._list_lay.setSpacing(5)
        cl.addWidget(self._list)
        self.card.setFixedWidth(300)
        v = QVBoxLayout(self)
        v.setContentsMargins(self._margin, self._margin, self._margin, self._margin)
        v.addWidget(self.card)

    def show_tags(self, make_pill, tags, target_global: QPoint, screen=None) -> None:
        while self._list_lay.count():
            item = self._list_lay.takeAt(0)
            wdg = item.widget()
            if wdg is not None:
                wdg.setParent(None)
                wdg.deleteLater()
        for tag in tags:
            pill = make_pill(tag)
            pill.setWordWrap(True)          # let a long re-used phrase wrap inside its chip
            pill.setMaximumWidth(264)
            self._list_lay.addWidget(pill, 0, Qt.AlignLeft)  # one per line, hugging content
        self.adjustSize()
        gap = 8
        x = target_global.x() + gap - self._margin
        y = target_global.y() - self._margin
        if screen is not None:
            x = min(x, screen.right() - self.width())
            x = max(x, screen.left())
            y = min(y, screen.bottom() - self.height())
            y = max(y, screen.top())
        self.move(int(x), int(y))
        self.show()


class _FfmpegInstallThread(QThread):
    """Managed ffmpeg download off the UI thread, with status text for the dialog."""
    status = Signal(str)
    finished_with = Signal(bool, str)

    def __init__(self) -> None:
        super().__init__()
        self.ok = False
        self.message = ""

    def run(self) -> None:
        try:
            install_ffmpeg(progress=self.status.emit)
        except Exception as exc:
            self.ok, self.message = False, str(exc)
        else:
            self.ok, self.message = True, ""
        self.finished_with.emit(self.ok, self.message)


class _VideoEditThread(QThread):
    """Runs the ffmpeg re-encode off the UI thread, writing to a temp file and
    swapping it in only on success so a failure can't leave a truncated clip."""
    finished_with = Signal(bool, str)

    def __init__(self, path: Path, plan) -> None:
        super().__init__()
        self.path = Path(path)
        self.plan = plan
        # Read directly after wait() rather than via the signal: a queued signal is
        # only delivered inside processEvents(), so a thread that finishes before
        # the caller's wait loop starts would never report its result at all.
        self.ok = False
        self.message = ""

    def _finish(self, ok: bool, message: str) -> None:
        self.ok, self.message = ok, message
        self.finished_with.emit(ok, message)

    def run(self) -> None:
        tmp = self.path.with_name(f".{self.path.stem}.edit{self.path.suffix}")
        try:
            ok, message = run_video_edit(self.path, tmp, self.plan)
            if not ok:
                tmp.unlink(missing_ok=True)
                self._finish(False, message)
                return
            os.replace(tmp, self.path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            self._finish(False, str(exc))
            return
        self._finish(True, "")


class TrimBar(QWidget):
    """Scrub bar with draggable in/out handles.

    A QSlider can only carry one value, and trimming needs three (playhead, in,
    out) on the same timeline, so this paints its own track: the kept span is
    highlighted, the discarded ends are dimmed, and the handles are grabbable.
    """
    positionRequested = Signal(int)     # ms
    trimChanged = Signal(int, int)      # in_ms, out_ms
    muteChanged = Signal(int, int)      # mute in_ms, out_ms
    HANDLE = 9
    GRIP = 7          # half-width of the playhead square's grab zone

    def __init__(self, theme: "Theme", parent=None) -> None:
        super().__init__(parent)
        self.theme = theme
        self.setMinimumHeight(34)
        self._peaks: list[float] = []
        # Optional (in_ms, out_ms, which) -> (in_ms, out_ms) hook that pulls a drag
        # onto a legal length. Set by the stage from the selected model target.
        self._snap = None
        # "playhead" moves the cursor, "select" slides the whole selection. One or
        # the other, chosen by the toolbar, rather than inferred from where you
        # happened to click — which left the playhead unreachable whenever the
        # selection covered the timeline.
        self._tool = "playhead"
        self.setMouseTracking(True)
        self._duration = 0
        self._position = 0
        self._in = 0
        self._out = 0
        self._drag: str | None = None      # 'in' | 'out' | 'window' | 'playhead' | 'seek'
        self._grab_ms = 0
        self._grab_in = 0
        # Optional second range, drawn as a red band under the trim track: the span
        # whose audio will be silenced. Kept separate from the trim because it
        # answers a different question — the trim decides which frames ship, the
        # mute decides which of them are silent.
        self._mute_visible = False
        self._mute_in = 0
        self._mute_out = 0
        self._peaks: list[float] = []

    # ---- state ----

    def set_duration(self, ms: int) -> None:
        self._duration = max(0, int(ms))
        if self._out <= 0 or self._out > self._duration:
            self._in, self._out = 0, self._duration
            self.trimChanged.emit(self._in, self._out)
        self.update()

    def set_position(self, ms: int) -> None:
        self._position = max(0, min(self._duration, int(ms)))
        self.update()

    def set_trim(self, in_ms: int, out_ms: int) -> None:
        in_ms = max(0, min(self._duration, int(in_ms)))
        out_ms = max(in_ms, min(self._duration, int(out_ms)))
        if (in_ms, out_ms) != (self._in, self._out):
            self._in, self._out = in_ms, out_ms
            self.trimChanged.emit(self._in, self._out)
        self.update()

    def trim(self) -> tuple[int, int]:
        return self._in, self._out

    def position(self) -> int:
        return self._position

    # ---- mute range ----

    def set_tool(self, tool: str) -> None:
        self._tool = "select" if tool == "select" else "playhead"
        self.update()

    def tool(self) -> str:
        return self._tool

    def set_snap(self, fn) -> None:
        """Install (or clear, with None) the length-snapping hook."""
        self._snap = fn

    def _snapped(self, in_ms: int, out_ms: int, which: str) -> tuple[int, int]:
        if self._snap is None:
            return in_ms, out_ms
        try:
            return self._snap(in_ms, out_ms, which)
        except Exception:
            return in_ms, out_ms

    def set_peaks(self, peaks: list[float]) -> None:
        """Waveform for the track background. Placing a mute on a word boundary by
        eye is guesswork without it."""
        had = bool(self._peaks)
        self._peaks = list(peaks or [])
        if bool(self._peaks) != had:
            self._resize_for_content()
        self.update()

    def _resize_for_content(self) -> None:
        # One track either way now — turning mute on no longer changes the height.
        base = 28 if self._peaks else 8
        self.setMinimumHeight(base + 20)
        self.updateGeometry()

    def set_mute_visible(self, on: bool) -> None:
        self._mute_visible = bool(on)
        if on and self._mute_out <= self._mute_in:
            # Default to the middle third of the current selection: a visible band
            # you can drag, rather than a zero-width one you have to find.
            span = self._out - self._in
            self._mute_in = self._in + span // 3
            self._mute_out = self._in + 2 * span // 3
        self._resize_for_content()
        self.update()

    def mute_visible(self) -> bool:
        return self._mute_visible

    def set_mute_range(self, in_ms: int, out_ms: int) -> None:
        in_ms = max(0, min(self._duration, int(in_ms)))
        out_ms = max(in_ms, min(self._duration, int(out_ms)))
        self._mute_in, self._mute_out = in_ms, out_ms
        self.muteChanged.emit(in_ms, out_ms)
        self.update()

    def mute_range(self) -> tuple[int, int]:
        return self._mute_in, self._mute_out

    def _mute_band(self) -> QRectF:
        """The muted span, drawn on the same track as everything else.

        It used to be a separate strip underneath, which had to be tall enough to
        grab and kept fighting the waveform for height. Red brackets on the one
        timeline are simpler and always the full height of the track.
        """
        track = self._track_rect()
        return QRectF(track.left(), track.top(), track.width(), track.height())

    def duration(self) -> int:
        return self._duration

    # ---- geometry ----

    def _track_rect(self) -> QRectF:
        # The bar grows a real track when there's a waveform to show: an 8px strip
        # renders peaks 4px tall, which is present but useless for placing a cut on
        # a word boundary.
        height = 28.0 if self._peaks else 8.0
        top = max(4.0, (self.height() - height) / 2)
        return QRectF(self.HANDLE, top,
                      max(1, self.width() - 2 * self.HANDLE), height)

    def _x_for(self, ms: int) -> float:
        track = self._track_rect()
        if self._duration <= 0:
            return track.left()
        return track.left() + track.width() * (ms / self._duration)

    def _ms_for(self, x: float) -> int:
        track = self._track_rect()
        if track.width() <= 0 or self._duration <= 0:
            return 0
        frac = (x - track.left()) / track.width()
        return int(max(0.0, min(1.0, frac)) * self._duration)

    # ---- interaction ----

    def mousePressEvent(self, event) -> None:
        x = event.position().x()
        y = event.position().y()
        if self._mute_visible:
            # Both ranges share one track now, so the mute brackets are tested
            # before the trim handles while mute mode is on — that's the range
            # you're there to adjust.
            if abs(x - self._x_for(self._mute_in)) <= self.HANDLE:
                self._drag = "mute_in"
                self.update()
                return
            if abs(x - self._x_for(self._mute_out)) <= self.HANDLE:
                self._drag = "mute_out"
                self.update()
                return
            if self._x_for(self._mute_in) < x < self._x_for(self._mute_out):
                self._drag = "mute_window"
                self._grab_ms = self._ms_for(x)
                self._grab_in = self._mute_in
                self.update()
                return
        # The playhead grip is drawn on top, so it's tested first — otherwise the
        # in-handle sitting at the same position (as it does on every fresh clip,
        # both at 0) swallows every attempt to scrub. Its target is deliberately
        # tight: grab the square to move the playhead, the wider bar to move a
        # bracket.
        if abs(x - self._x_for(self._position)) <= self.GRIP:
            self._drag = "playhead"
            self.positionRequested.emit(self._ms_for(x))
            self.setCursor(Qt.SizeHorCursor)
            return
        if self._tool == "playhead" and not self._mute_visible:
            # Playhead tool: a click anywhere that isn't a bracket moves the cursor,
            # including inside the selection.
            near_bracket = (abs(x - self._x_for(self._in)) <= self.HANDLE + 2
                            or abs(x - self._x_for(self._out)) <= self.HANDLE + 2)
            if not near_bracket:
                self._drag = "playhead"
                self.positionRequested.emit(self._ms_for(x))
                self.setCursor(Qt.SizeHorCursor)
                return
        near_in = abs(x - self._x_for(self._in)) <= self.HANDLE + 2
        near_out = abs(x - self._x_for(self._out)) <= self.HANDLE + 2
        if near_in and near_out:
            # overlapping handles: pick whichever side the click leans towards
            self._drag = "in" if x <= self._x_for(self._in) else "out"
        elif near_in:
            self._drag = "in"
        elif near_out:
            self._drag = "out"
        elif (self._tool == "select"
              and self._x_for(self._in) < x < self._x_for(self._out)):
            # Grab inside the selection to slide the whole window. This is what
            # makes "Fit to target" usable: the fitted length is legal, and moving
            # it preserves that length, so the user picks *which* seconds to keep
            # rather than being stuck with the start of the clip.
            self._drag = "window"
            self._grab_ms = self._ms_for(x)
            self._grab_in = self._in
        else:
            self._drag = "seek"
            self.positionRequested.emit(self._ms_for(x))
        self.update()

    def mouseMoveEvent(self, event) -> None:
        x = event.position().x()
        if self._drag == "in":
            self.set_trim(*self._snapped(
                min(self._ms_for(x), self._out), self._out, "in"))
        elif self._drag == "out":
            self.set_trim(*self._snapped(
                self._in, max(self._ms_for(x), self._in), "out"))
        elif self._drag == "window":
            span = self._out - self._in
            new_in = self._grab_in + (self._ms_for(x) - self._grab_ms)
            new_in = max(0, min(new_in, self._duration - span))
            self.set_trim(new_in, new_in + span)
            self.positionRequested.emit(new_in)
        elif self._drag == "mute_in":
            self.set_mute_range(min(self._ms_for(x), self._mute_out), self._mute_out)
        elif self._drag == "mute_out":
            self.set_mute_range(self._mute_in, max(self._ms_for(x), self._mute_in))
        elif self._drag == "mute_window":
            span = self._mute_out - self._mute_in
            new_in = self._grab_in + (self._ms_for(x) - self._grab_ms)
            new_in = max(0, min(new_in, self._duration - span))
            self.set_mute_range(new_in, new_in + span)
        elif self._drag in ("seek", "playhead"):
            self.positionRequested.emit(self._ms_for(x))
        else:
            on_playhead = abs(x - self._x_for(self._position)) <= self.GRIP
            near_handle = (not on_playhead
                           and (abs(x - self._x_for(self._in)) <= self.HANDLE + 2
                                or abs(x - self._x_for(self._out)) <= self.HANDLE + 2))
            inside = self._x_for(self._in) < x < self._x_for(self._out)
            if near_handle or on_playhead:
                self.setCursor(Qt.SizeHorCursor)
            elif self._tool == "select" and inside:
                self.setCursor(Qt.OpenHandCursor)
            else:
                self.setCursor(Qt.PointingHandCursor)

    def mouseReleaseEvent(self, event) -> None:
        # Sliding the selection used to yank the playhead to the new start, so a
        # click inside the selection looked like the cursor jumping to frame one.
        # Where you were watching is independent of where the selection sits.
        if self._drag in ("in", "out"):
            # park the playhead on the handle just moved, so you see the edit frame
            self.positionRequested.emit(self._in if self._drag == "in" else self._out)
        self._drag = None

    # ---- painting ----

    def paintEvent(self, event) -> None:
        t = self.theme
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        track = self._track_rect()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(t.surface_3))
        painter.drawRoundedRect(track, 4, 4)
        if self._duration > 0:
            keep = QRectF(self._x_for(self._in), track.top(),
                          max(1.0, self._x_for(self._out) - self._x_for(self._in)),
                          track.height())
            painter.setBrush(QColor(t.accent))
            painter.drawRoundedRect(keep, 4, 4)
        if self._peaks:
            # Painted AFTER the selection fill: drawing it first meant the accent
            # rect covered the whole waveform wherever the clip was selected,
            # which is normally all of it.
            painter.save()
            path = QPainterPath()
            path.addRoundedRect(track, 4, 4)
            painter.setClipPath(path)
            mid = track.center().y()
            half = track.height() / 2
            step = track.width() / len(self._peaks)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(t.surface_0))
            for i, peak in enumerate(self._peaks):
                h = max(0.6, peak * half)
                painter.drawRect(QRectF(track.left() + i * step, mid - h,
                                        max(1.0, step - 0.5), h * 2))
            painter.restore()
        if self._duration > 0:
            # grip marks: signal that the selection itself can be dragged
            if keep.width() > 26:
                cx, cy = keep.center().x(), keep.center().y()
                painter.setPen(QPen(QColor(t.surface_0), 1))
                for dx in (-4, 0, 4):
                    painter.drawLine(QPointF(cx + dx, cy - 2), QPointF(cx + dx, cy + 2))
            if self._mute_visible:
                # A red wash over the span plus a bracket at each end, on the same
                # track as the waveform — so you can see which sound you're cutting.
                mx1, mx2 = self._x_for(self._mute_in), self._x_for(self._mute_out)
                wash = QColor(MUTE_COLOR)
                wash.setAlpha(90)
                painter.setPen(Qt.NoPen)
                painter.setBrush(wash)
                painter.drawRect(QRectF(mx1, track.top(),
                                        max(1.0, mx2 - mx1), track.height()))
                pen = QPen(QColor(MUTE_COLOR), 2.5, Qt.SolidLine, Qt.SquareCap)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                top_y, bot_y = track.top() - 3, track.bottom() + 3
                arm = 5.0
                # "[" at the start, "]" at the end, so the direction is readable.
                for hx, direction in ((mx1, 1), (mx2, -1)):
                    painter.drawLine(QPointF(hx, top_y), QPointF(hx, bot_y))
                    painter.drawLine(QPointF(hx, top_y),
                                     QPointF(hx + arm * direction, top_y))
                    painter.drawLine(QPointF(hx, bot_y),
                                     QPointF(hx + arm * direction, bot_y))
            # handles
            painter.setPen(QPen(QColor(t.surface_0), 1))
            painter.setBrush(QColor(t.text_primary))
            for ms in (self._in, self._out):
                hx = self._x_for(ms)
                painter.drawRoundedRect(
                    QRectF(hx - 3, track.top() - 6, 6, track.height() + 12), 2, 2)

        # Playhead LAST so nothing paints over it: it's the thing you grab, and the
        # mute wash, handles and waveform all share this strip. Outside the waveform
        # branch too — moving the waveform above the selection fill had accidentally
        # swallowed it, so clips with no audio drew no playhead at all.
        px = self._x_for(self._position)
        painter.setPen(QPen(QColor(PLAYHEAD_COLOR), 2))
        painter.drawLine(QPointF(px, track.top() - 8), QPointF(px, track.bottom() + 8))
        grip = QRectF(px - 6, track.center().y() - 6, 12, 12)
        painter.setPen(QPen(QColor(t.surface_0), 1))
        painter.setBrush(QColor(PLAYHEAD_COLOR))
        painter.drawRoundedRect(grip, 2, 2)
        painter.end()


class VideoStage(QWidget):
    """Inline video player for the centre stage — the video equivalent of the image
    canvas, not a popup. Transport sits directly under the picture so playback,
    scrubbing and (from the next increment) trim handles are all in the main window.

    Qt 6.5+ ships an FFmpeg-based multimedia backend with PySide6, so this needs no
    system codecs. If QtMultimedia is missing we fall back to the poster frame and
    say so, rather than leaving a dead panel.
    """

    def __init__(self, controller: "MainWindow") -> None:
        super().__init__(controller)
        self.controller = controller
        self.setObjectName("Stage")
        t = controller.theme
        self._player = None
        self._path: Path | None = None
        self._duration_ms = 0
        self._info: "VideoInfo | None" = None
        self._crop_item: CropRectItem | None = None
        self._rotation = 0
        # Keyed by path so switching clips keeps each one's in-progress edit.
        self._pending_edits: dict[str, dict] = {}
        self._audio = None
        self._base_volume = 1.0
        self._peaks_cache: dict[str, list] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._surface = None
        self._fallback = QLabel("")
        self._fallback.setAlignment(Qt.AlignCenter)
        self._fallback.setVisible(False)
        try:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            from PySide6.QtMultimediaWidgets import QGraphicsVideoItem
        except ImportError:
            self._available = False
        else:
            self._available = True
            # The video renders into a QGraphicsScene rather than a plain
            # QVideoWidget so the existing CropRectItem can be laid over it — video
            # crop then behaves exactly like image crop instead of being a second,
            # parallel implementation.
            self._scene = QGraphicsScene(self)
            self._video_item = QGraphicsVideoItem()
            self._scene.addItem(self._video_item)
            self._surface = QGraphicsView(self._scene)
            self._surface.setObjectName("Stage")
            self._surface.setFrameShape(QFrame.NoFrame)
            self._surface.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
            self._surface.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._surface.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            self._video_item.nativeSizeChanged.connect(self._on_native_size)
            self._player = QMediaPlayer(self)
            self._audio = QAudioOutput(self)
            self._player.setAudioOutput(self._audio)
            self._player.setVideoOutput(self._video_item)
            self._player.positionChanged.connect(self._on_position)
            self._player.durationChanged.connect(self._on_duration)
            self._player.playbackStateChanged.connect(self._on_state)
            self._player.errorOccurred.connect(self._on_error)
            # The plugin (and its bundled libavutil) is resident now, so its noisy
            # default logging can be turned down.
            quiet_ffmpeg_logs()
        if self._surface is not None:
            lay.addWidget(self._surface, 1)
        lay.addWidget(self._fallback, 1)

        bar = QFrame()
        bar.setObjectName("VideoBar")
        bar.setFixedHeight(46)
        bar.setStyleSheet(
            f"#VideoBar {{ background: {t.surface_1}; border-top: 1px solid {t.border}; }}")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(8)

        self._play_btn = QToolButton()
        self._play_btn.setObjectName("NavBtn")
        self._play_btn.setIcon(lucide_icon("play", t.text_primary, 18))
        self._play_btn.setToolTip("Play / pause (Space)")
        self._play_btn.clicked.connect(self.toggle_play)
        row.addWidget(self._play_btn)

        self._back_btn = QToolButton()
        self._back_btn.setObjectName("NavBtn")
        self._back_btn.setIcon(lucide_icon("chevron-left", t.text_secondary, 16))
        self._back_btn.setToolTip("Back one frame")
        self._back_btn.clicked.connect(lambda: self.step_frames(-1))
        row.addWidget(self._back_btn)

        self._fwd_btn = QToolButton()
        self._fwd_btn.setObjectName("NavBtn")
        self._fwd_btn.setIcon(lucide_icon("chevron-right", t.text_secondary, 16))
        self._fwd_btn.setToolTip("Forward one frame")
        self._fwd_btn.clicked.connect(lambda: self.step_frames(1))
        row.addWidget(self._fwd_btn)

        self._pos_label = QLabel("0:00")
        self._pos_label.setObjectName("NavCount")
        self._pos_label.setFixedWidth(52)
        self._pos_label.setAlignment(Qt.AlignCenter)
        row.addWidget(self._pos_label)

        # Frame number, not just a timestamp: everything about conforming a clip is
        # counted in frames, so "f0073" is the number you can act on.
        self._frame_label = QLabel("")
        self._frame_label.setObjectName("Hint")
        # Fixed, not minimum: this label grows as the number does ("f9 / 141" ->
        # "f141 / 141"), and a growing label re-lays out the row, which shifted the
        # trim bar sideways while you scrubbed.
        self._frame_label.setFixedWidth(96)
        self._frame_label.setAlignment(Qt.AlignCenter)
        self._frame_label.setToolTip(
            "Frame under the playhead, counted from the start of the clip")
        row.addWidget(self._frame_label)

        self._slider = TrimBar(t)
        self._slider.positionRequested.connect(self._seek)
        self._slider.trimChanged.connect(self._on_trim_changed)
        row.addWidget(self._slider, 1)

        self._len_label = QLabel("0:00")
        self._len_label.setObjectName("NavCount")
        self._len_label.setMinimumWidth(46)
        self._len_label.setAlignment(Qt.AlignCenter)
        row.addWidget(self._len_label)

        self._mute_btn = QToolButton()
        self._mute_btn.setObjectName("NavBtn")
        self._mute_btn.setIcon(lucide_icon("volume-2", t.text_secondary, 16))
        self._mute_btn.setToolTip("Mute / unmute")
        self._mute_btn.clicked.connect(self.toggle_mute)
        row.addWidget(self._mute_btn)

        self._meta_label = QLabel("")
        self._meta_label.setObjectName("Hint")
        row.addWidget(self._meta_label)

        lay.addWidget(bar)
        self._bar = bar
        lay.addWidget(self._build_edit_bar(t))

    def _build_edit_bar(self, theme: "Theme") -> QWidget:
        """Trim/conform controls, inline under the transport. Everything here acts on
        the selection shown in the trim bar above it."""
        bar = QFrame()
        bar.setObjectName("EditBar")
        bar.setStyleSheet(
            f"#EditBar {{ background: {theme.surface_1}; "
            f"border-top: 1px solid {theme.border}; }}")
        rows = QVBoxLayout(bar)
        rows.setContentsMargins(12, 6, 12, 8)
        rows.setSpacing(4)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        rows.addLayout(row)

        # Tool selector: one or the other, so what a click does is a decision you
        # made rather than a guess from where you clicked.
        self._tool_playhead = QToolButton()
        self._tool_playhead.setObjectName("NavBtn")
        self._tool_playhead.setCheckable(True)
        self._tool_playhead.setChecked(True)
        self._tool_playhead.setIcon(
            lucide_icon("mouse-pointer", theme.text_secondary, 15))
        self._tool_playhead.setToolTip("Move the playhead (V)")
        self._tool_playhead.clicked.connect(lambda: self.set_tool("playhead"))
        row.addWidget(self._tool_playhead)

        self._tool_select = QToolButton()
        self._tool_select.setObjectName("NavBtn")
        self._tool_select.setCheckable(True)
        self._tool_select.setIcon(lucide_icon("hand", theme.text_secondary, 15))
        self._tool_select.setToolTip("Slide the selection, keeping its length (H)")
        self._tool_select.clicked.connect(lambda: self.set_tool("select"))
        row.addWidget(self._tool_select)

        sep = QFrame()
        sep.setFrameShape(QFrame.VLine)
        sep.setStyleSheet(f"color: {theme.border};")
        row.addWidget(sep)

        self._in_btn = QPushButton("Set in")
        self._in_btn.setToolTip("Start the clip at the current frame ([)")
        self._in_btn.clicked.connect(self.set_in_at_playhead)
        row.addWidget(self._in_btn)
        self._out_btn = QPushButton("Set out")
        self._out_btn.setToolTip("End the clip at the current frame (])")
        self._out_btn.clicked.connect(self.set_out_at_playhead)
        row.addWidget(self._out_btn)
        self._reset_btn = QPushButton("Reset")
        self._reset_btn.setToolTip("Select the whole clip again")
        self._reset_btn.clicked.connect(self.reset_trim)
        row.addWidget(self._reset_btn)

        # These stretched to fill; they only ever hold two short words, and the
        # space is better spent on the frame boxes.
        for btn in (self._in_btn, self._out_btn, self._reset_btn):
            btn.setMaximumWidth(78)

        # Editable in/out frames. Frames, not seconds: the grid rules are counted
        # in them, so typing 73 is exact where 3.04s is a guess.
        #
        # Each label sits in a tight box with its field, otherwise the row's spare
        # space lands between them and "In" ends up stranded from the number it
        # names.
        row.addSpacing(10)
        for text, attr, tip in (
            ("In", "_in_frame", "First frame kept. Type a number to set it exactly."),
            ("Out", "_out_frame", "Last frame kept. Type a number to set it exactly."),
        ):
            pair = QHBoxLayout()
            pair.setContentsMargins(0, 0, 0, 0)
            pair.setSpacing(4)
            label = QLabel(text)
            label.setStyleSheet(f"color: {theme.text_secondary};")
            pair.addWidget(label)
            box = QSpinBox()
            box.setRange(1, 9_999_999)
            box.setFixedWidth(84)
            box.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            box.setToolTip(tip)
            pair.addWidget(box)
            setattr(self, attr, box)
            holder = QWidget()
            holder.setLayout(pair)
            holder.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
            row.addWidget(holder)
        self._in_frame.editingFinished.connect(lambda: self._frame_box_edited("in"))
        self._out_frame.editingFinished.connect(lambda: self._frame_box_edited("out"))

        row.addSpacing(10)
        row.addWidget(QLabel("Target:"))
        self._target_combo = QComboBox()
        self._target_combo.addItem("None (keep source)", "")
        for key, target in self.controller.model_targets.items():
            self._target_combo.addItem(target.label, key)
        self._target_combo.setToolTip(
            "Conform the clip to a video model's frame rate and frame-count grid")
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)
        row.addWidget(self._target_combo)

        # Second row. All of this in one row demanded 1316px, which forced the whole
        # window to a 1880px minimum — wider than a 1600x900 laptop screen.
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)
        rows.addLayout(row)

        self._rot_ccw = QToolButton()
        self._rot_ccw.setObjectName("NavBtn")
        self._rot_ccw.setIcon(lucide_icon("rotate-ccw", theme.text_secondary, 15))
        self._rot_ccw.setToolTip("Rotate 90\u00b0 anticlockwise")
        self._rot_ccw.clicked.connect(lambda: self.rotate_by(-90))
        row.addWidget(self._rot_ccw)
        self._rot_cw = QToolButton()
        self._rot_cw.setObjectName("NavBtn")
        self._rot_cw.setIcon(lucide_icon("rotate-cw", theme.text_secondary, 15))
        self._rot_cw.setToolTip("Rotate 90\u00b0 clockwise")
        self._rot_cw.clicked.connect(lambda: self.rotate_by(90))
        row.addWidget(self._rot_cw)

        self._crop_btn = QPushButton("Crop")
        self._crop_btn.setCheckable(True)
        self._crop_btn.setToolTip("Draw a crop region over the video")
        self._crop_btn.toggled.connect(self.set_crop_enabled)
        row.addWidget(self._crop_btn)
        self._aspect_combo = QComboBox()
        for label, _ratio in ASPECT_PRESETS:
            self._aspect_combo.addItem(label)
        self._aspect_combo.setToolTip("Constrain the crop to an aspect ratio")
        self._aspect_combo.setVisible(False)
        self._aspect_combo.currentIndexChanged.connect(self._on_aspect_changed)
        row.addWidget(self._aspect_combo)

        self._fit_btn = QPushButton("Fit to target")
        self._fit_btn.setToolTip("Snap the selection to the target's legal length, "
                                 "then drag it to pick which part to keep")
        self._fit_btn.clicked.connect(self.fit_to_target)
        row.addWidget(self._fit_btn)

        self._nudge_back = QToolButton()
        self._nudge_back.setObjectName("NavBtn")
        self._nudge_back.setIcon(lucide_icon("chevron-left", theme.text_secondary, 15))
        self._nudge_back.setToolTip("Move the selection earlier (keeps its length)")
        self._nudge_back.clicked.connect(lambda: self.nudge_window(-1))
        row.addWidget(self._nudge_back)
        self._nudge_fwd = QToolButton()
        self._nudge_fwd.setObjectName("NavBtn")
        self._nudge_fwd.setIcon(lucide_icon("chevron-right", theme.text_secondary, 15))
        self._nudge_fwd.setToolTip("Move the selection later (keeps its length)")
        self._nudge_fwd.clicked.connect(lambda: self.nudge_window(1))
        row.addWidget(self._nudge_fwd)

        self._trim_label = QLabel("")
        self._trim_label.setObjectName("Hint")
        row.addWidget(self._trim_label, 1)

        # Whether this clip's sound will reach the model, visible before you run
        # rather than inferable from the caption afterwards.
        self._audio_badge = QLabel("")
        self._audio_badge.setObjectName("Hint")
        row.addWidget(self._audio_badge)

        self._save_frame_btn = QPushButton("Save frame\u2026")
        self._save_frame_btn.setToolTip(
            "Write the frame under the playhead to an image file")
        self._save_frame_btn.clicked.connect(self.save_current_frame)
        row.addWidget(self._save_frame_btn)

        self._save_audio_btn = QPushButton("Save audio\u2026")
        self._save_audio_btn.setToolTip(
            "Write this clip's audio to a file. Uses the trimmed span when one is "
            "set, at full quality.")
        self._save_audio_btn.clicked.connect(self.save_clip_audio)
        row.addWidget(self._save_audio_btn)

        self._snap_btn = QPushButton("Snap")
        self._snap_btn.setCheckable(True)
        # Default on: a trim that isn't on the grid gets silently truncated or
        # padded by the trainer, so the useful default is the one that can't.
        self._snap_btn.setChecked(
            self.controller.qsettings.value("trim_snap", True, bool))
        self._snap_btn.setToolTip(
            "Pull the trim onto a frame count the selected model accepts "
            "instead of moving frame by frame")
        self._snap_btn.toggled.connect(self._on_snap_toggled)
        row.addWidget(self._snap_btn)

        self._mute_section_btn = QPushButton("Mute section")
        self._mute_section_btn.setCheckable(True)
        self._mute_section_btn.setToolTip(
            "Silence part of the clip's audio \u2014 useful when a trim cuts a word "
            "in half. The picture is untouched.")
        self._mute_section_btn.toggled.connect(self._on_mute_section_toggled)
        row.addWidget(self._mute_section_btn)

        # One button, no preview render: playback is already silenced live inside
        # the brackets, so you audition by pressing play. Rendering a file just to
        # listen left a stray clip in the dataset folder if anything went wrong.
        self._mute_apply_btn = QPushButton("Apply mute")
        self._mute_apply_btn.setObjectName("Primary")
        self._mute_apply_btn.setToolTip(
            "Write the silence to the clip. Press play first \u2014 the bracketed "
            "span is already muted during playback.")
        self._mute_apply_btn.clicked.connect(self.commit_mute)
        self._mute_apply_btn.setVisible(False)
        row.addWidget(self._mute_apply_btn)

        self._pending_banner = QLabel("")
        self._pending_banner.setObjectName("PendingEdits")
        self._pending_banner.setWordWrap(True)
        self._pending_banner.setVisible(False)
        self._pending_banner.setStyleSheet(
            f"#PendingEdits {{ background: {theme.warning}; "
            f"color: {theme.surface_0}; border-radius: 4px; "
            f"padding: 3px 8px; font-weight: 600; }}")

        self._apply_btn = QPushButton("Apply edit\u2026")
        self._apply_btn.setToolTip("Re-encode this clip with the changes above")
        self._apply_btn.clicked.connect(self.apply_edit)
        row.addWidget(self._apply_btn)

        wrap = QWidget()
        col = QVBoxLayout(wrap)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(0)
        col.addWidget(bar)
        col.addWidget(self._pending_banner)
        return wrap

    # ---- pending edits ----

    def _edit_state(self) -> dict:
        in_ms, out_ms = self._slider.trim()
        return {
            "trim": (in_ms, out_ms),
            "duration": self._slider.duration(),
            "crop": self.crop_rect(),
            "rotation": self._rotation,
            "target": self._target_combo.currentData() or "",
        }

    def _is_default_state(self, state: dict) -> bool:
        """No edit means: whole clip, no crop, no rotation. A chosen target alone
        isn't an edit — it only describes what Apply would do."""
        return (state["crop"] is None
                and not state["rotation"]
                and state["trim"] == (0, state["duration"]))

    def has_pending_edits(self, path: Path | None = None) -> bool:
        if path is None:
            return bool(self._path and str(self._path) in self._pending_edits)
        return str(path) in self._pending_edits

    def _mark_pending(self) -> None:
        """Remember the in-progress edit against this clip.

        Stored rather than merely warned about: losing a carefully placed trim
        because you clicked the next thumbnail is the failure worth designing out,
        and a modal on every navigation would be its own annoyance.
        """
        if self._path is None or self._info is None:
            return
        key = str(self._path)
        state = self._edit_state()
        if self._is_default_state(state):
            self._pending_edits.pop(key, None)
        else:
            self._pending_edits[key] = state
        self._refresh_pending_banner()
        self.controller._refresh_video_edit_marker(self._path)

    def _restore_pending(self) -> None:
        """Put a remembered edit back when you return to a clip."""
        if self._path is None:
            return
        state = self._pending_edits.get(str(self._path))
        if not state:
            self._rotation = 0
            self._refresh_pending_banner()
            return
        self._rotation = int(state.get("rotation") or 0)
        target_key = state.get("target") or ""
        idx = self._target_combo.findData(target_key)
        if idx >= 0:
            self._target_combo.blockSignals(True)
            self._target_combo.setCurrentIndex(idx)
            self._target_combo.blockSignals(False)
        trim = state.get("trim")
        if trim:
            self._slider.set_trim(*trim)
        crop = state.get("crop")
        if crop:
            self._crop_btn.blockSignals(True)
            self._crop_btn.setChecked(True)
            self._crop_btn.blockSignals(False)
            self.set_crop_enabled(True)
            if self._crop_item is not None:
                x, y, cw, ch = crop
                self._crop_item.setRect(QRectF(x, y, cw, ch))
        self._apply_preview_rotation()
        self._refresh_pending_banner()

    def clear_pending(self, path: Path | None = None) -> None:
        self._pending_edits.pop(str(path or self._path), None)
        self._refresh_pending_banner()

    def _refresh_pending_banner(self) -> None:
        banner = getattr(self, "_pending_banner", None)
        if banner is None:
            return
        pending = self.has_pending_edits()
        banner.setVisible(pending)
        self._apply_btn.setProperty("Primary", pending)
        if pending:
            state = self._pending_edits.get(str(self._path), {})
            bits = []
            if state.get("trim") != (0, state.get("duration")):
                span = (state["trim"][1] - state["trim"][0]) / 1000.0
                bits.append(f"trimmed to {span:.2f}s")
            if state.get("crop"):
                bits.append("cropped")
            if state.get("rotation"):
                bits.append(f"rotated {state['rotation']}\u00b0")
            banner.setText("Unapplied edits (" + ", ".join(bits)
                           + ") \u2014 press Apply edit to write them to the file.")

    # ---- rotation ----

    def rotate_by(self, degrees: int) -> None:
        """Turn the preview and remember it for the next Apply.

        The scene is rotated rather than the file, so the picture you crop and trim
        against is the picture you'll get — rotating on export alone would have you
        drawing a crop rect on the wrong axes.
        """
        if self._info is None:
            return
        self._rotation = (self._rotation + degrees) % 360
        self._apply_preview_rotation()
        self._mark_pending()
        self._refresh_trim_label()
        self.controller._set_status(
            f"Rotated to {self._rotation}\u00b0 \u2014 press Apply edit to write it."
            if self._rotation else "Rotation cleared.")

    def _apply_preview_rotation(self) -> None:
        if not self._available or self._video_item is None:
            return
        size = self._video_item.size()
        if size.isEmpty():
            return
        item = self._video_item
        item.setTransformOriginPoint(size.width() / 2, size.height() / 2)
        item.setRotation(self._rotation)
        # The scene rect must follow the rotated bounds or the view crops the turn.
        if self._rotation % 180:
            rect = QRectF(0, 0, size.height(), size.width())
            item.setPos((size.height() - size.width()) / 2,
                        (size.width() - size.height()) / 2)
        else:
            rect = QRectF(0, 0, size.width(), size.height())
            item.setPos(0, 0)
        self._scene.setSceneRect(rect)
        self._surface.fitInView(rect, Qt.KeepAspectRatio)
        if self._crop_item is not None:
            self._crop_item.set_bounds(rect)

    # ---- crop ----

    def _on_native_size(self, size) -> None:
        """The real pixel size of the video only arrives once decoding starts; that's
        when the scene can be sized and the crop rect given correct bounds."""
        if size.isEmpty():
            return
        self._video_item.setSize(size)
        self._scene.setSceneRect(QRectF(0, 0, size.width(), size.height()))
        self._surface.fitInView(self._video_item, Qt.KeepAspectRatio)
        if self._crop_item is not None:
            self._crop_item.set_bounds(QRectF(0, 0, size.width(), size.height()))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._available and not self._video_item.size().isEmpty():
            self._surface.fitInView(self._video_item, Qt.KeepAspectRatio)

    def set_crop_enabled(self, on: bool) -> None:
        self._aspect_combo.setVisible(on)
        if not self._available:
            return
        if on:
            size = self._video_item.size()
            if size.isEmpty():
                self._crop_btn.setChecked(False)
                self.controller._set_status(
                    "Crop needs the video's size \u2014 give it a moment to load, "
                    "then try again.")
                return
            bounds = QRectF(0, 0, size.width(), size.height())
            # Starts at the full frame, like the image crop dialog — and note
            # set_aspect() re-fits the rect, so any initial inset would be discarded
            # the moment an aspect preset is applied.
            self._crop_item = CropRectItem(bounds, self._on_crop_changed)
            self._scene.addItem(self._crop_item)
            self._on_aspect_changed(self._aspect_combo.currentIndex())
        elif self._crop_item is not None:
            self._scene.removeItem(self._crop_item)
            self._crop_item = None
        self._refresh_trim_label()
        self._mark_pending()

    def _on_aspect_changed(self, idx: int) -> None:
        if self._crop_item is not None and 0 <= idx < len(ASPECT_PRESETS):
            self._crop_item.set_aspect(ASPECT_PRESETS[idx][1])
            self._mark_pending()

    def _on_crop_changed(self) -> None:
        self._refresh_trim_label()
        self._mark_pending()

    def crop_rect(self) -> tuple[int, int, int, int] | None:
        """(x, y, w, h) in source pixels, or None when cropping is off."""
        if self._crop_item is None:
            return None
        r = self._crop_item.rect()
        return (max(0, int(round(r.x()))), max(0, int(round(r.y()))),
                max(2, int(round(r.width()))), max(2, int(round(r.height()))))

    # ---- trim state ----

    def current_target(self):
        key = self._target_combo.currentData() if hasattr(self, "_target_combo") else ""
        return self.controller.model_targets.get(key) if key else None

    def _on_target_changed(self, _idx: int) -> None:
        self._refresh_snap()
        self._refresh_trim_label()
        self._mark_pending()

    def _on_trim_changed(self, in_ms: int, out_ms: int) -> None:
        self._refresh_frame_boxes()
        self._refresh_trim_label()
        self._mark_pending()
        if self._is_playing() and not (in_ms <= self._slider.position() < out_ms):
            self._player.setPosition(in_ms)

    def set_in_at_playhead(self) -> None:
        _in, out = self._slider.trim()
        self._slider.set_trim(self._slider.position(), out)

    def set_out_at_playhead(self) -> None:
        in_ms, _out = self._slider.trim()
        self._slider.set_trim(in_ms, self._slider.position())

    def reset_trim(self) -> None:
        self._slider.set_trim(0, self._slider.duration())

    def fit_to_target(self) -> None:
        """Shorten the selection to the nearest legal length for the target.

        The fitted window keeps its centre where the current selection was, rather
        than snapping to the start of the clip — and because dragging the middle of
        the trim bar moves the window without changing its length, the user can then
        slide this legal window to whichever seconds they actually want.
        """
        target = self.current_target()
        if target is None or self._info is None:
            return
        in_ms, out_ms = self._slider.trim()
        duration = self._slider.duration()
        span_s = max(0.0, (out_ms - in_ms) / 1000.0)
        frames = target.snap_frames(
            min(int(round(span_s * target.fps)), target.max_frames()), "down")
        frames = max(frames, target.smallest_legal_frames())
        span_ms = min(int(math.ceil(frames / target.fps * 1000)), duration)
        centre = (in_ms + out_ms) // 2
        new_in = max(0, min(centre - span_ms // 2, duration - span_ms))
        self._slider.set_trim(new_in, new_in + span_ms)
        self._refresh_trim_label()
        if frames < target.min_frames():
            # The clip can't reach the model's minimum however it's cut, so trimming
            # further only throws away material that was already too scarce. Say so
            # plainly instead of implying the fit succeeded.
            self.controller._set_status(
                f"{self._path.name if self._path else 'This clip'} is only "
                f"{duration / 1000:.1f}s \u2014 {target.label} wants at least "
                f"{target.seconds_for_frames(target.min_frames()):.1f}s. Snapped to "
                f"{frames} frames, but it's below the model's minimum.")
            return
        self.controller._set_status(
            f"Fitted to {frames} frames for {target.label} \u2014 drag the highlighted "
            "region to choose which part of the clip to keep.")

    def nudge_window(self, direction: int, seconds: float = 0.5) -> None:
        """Slide the selection without changing its length, so a fitted (legal)
        window stays legal."""
        in_ms, out_ms = self._slider.trim()
        span = out_ms - in_ms
        step = int(direction * seconds * 1000)
        new_in = max(0, min(in_ms + step, self._slider.duration() - span))
        self._slider.set_trim(new_in, new_in + span)
        self._slider.set_position(new_in)
        self._seek(new_in)

    def save_current_frame(self) -> None:
        """Write the frame under the playhead wherever the user chooses."""
        if self._path is None:
            return
        ms = self._slider.position()
        # Frame number rather than a timestamp: it's what a grid-aligned dataset is
        # counted in, and it avoids '0:03.4' style characters in filenames.
        info = self._info
        frame_no = int(round(ms / 1000.0 * info.fps)) if info else 0
        suggested = f"{self._path.stem}_f{frame_no:05d}.png"
        target, _ = QFileDialog.getSaveFileName(
            self.controller, "Save frame",
            str(self._path.parent / suggested),
            "PNG image (*.png);;JPEG image (*.jpg)")
        if not target:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok, message = extract_single_frame(self._path, target, ms / 1000.0)
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            QMessageBox.critical(self.controller, "Could not save the frame", message)
            return
        saved = Path(target)
        # Landing it in the dataset folder is a common intent, so re-list when it
        # does rather than leaving a file the filmstrip doesn't know about.
        if self.controller.store is not None and saved.parent == self.controller.store.folder:
            self.controller.images = self.controller.store.images()
            self.controller._rebuild_filmstrip()
            row = self.controller._row_for_path(saved)
            if row is not None:
                self.controller.filmstrip.setCurrentRow(row)
        self.controller._set_status(f"Saved frame {frame_no} to {saved.name}.")

    def save_clip_audio(self) -> None:
        """Write the clip's audio out, honouring the trim."""
        if self._path is None:
            return
        if not has_audio_stream(self._path):
            self.controller._set_status(
                f"{self._path.name} has no audio track \u2014 nothing to save.")
            return
        in_ms, out_ms = self._slider.trim()
        trimmed = (in_ms, out_ms) != (0, self._slider.duration())
        suffix = "_trimmed" if trimmed else ""
        target, _ = QFileDialog.getSaveFileName(
            self.controller, "Save audio",
            str(self._path.parent / f"{self._path.stem}{suffix}.wav"),
            "WAV (*.wav);;FLAC (*.flac);;MP3 (*.mp3);;M4A (*.m4a);;Opus (*.opus)")
        if not target:
            return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok, message = export_audio(
                self._path, target,
                start_s=in_ms / 1000.0 if trimmed else 0.0,
                end_s=out_ms / 1000.0 if trimmed else None)
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            QMessageBox.critical(self.controller, "Could not save the audio", message)
            return
        span = f" ({(out_ms - in_ms) / 1000.0:.2f}s)" if trimmed else ""
        self.controller._set_status(f"Saved audio{span} to {Path(target).name}.")

    # ---- tools and frame entry ----

    def set_tool(self, tool: str) -> None:
        """Playhead or hand — never both, so a click has one meaning."""
        tool = "select" if tool == "select" else "playhead"
        self._slider.set_tool(tool)
        self._tool_playhead.setChecked(tool == "playhead")
        self._tool_select.setChecked(tool == "select")
        self.controller._set_status(
            "Playhead tool: click the timeline to move the cursor."
            if tool == "playhead" else
            "Hand tool: drag inside the selection to slide it, keeping its length.")

    # Frame numbers shown to the user are 1-based: frame 1 is the first frame of
    # the clip, and the out box shows the last frame KEPT rather than the exclusive
    # end. "Frame 0" reads as nothing, and an out of 141 on a 141-frame clip looked
    # like it was keeping one frame too many.
    def _frames_for(self, ms: int) -> int:
        """Zero-based frame index at a timestamp — the internal form."""
        info = self._info
        return int(round(ms / 1000.0 * info.fps)) if info and info.fps else 0

    def _ms_for_frame(self, frame: int) -> int:
        info = self._info
        return int(round(frame / info.fps * 1000)) if info and info.fps else 0

    def _total_frames(self) -> int:
        info = self._info
        if info is None or not info.fps:
            return 0
        return info.frame_count or int(round(info.duration_s * info.fps))

    def _refresh_frame_boxes(self) -> None:
        if self._info is None:
            return
        in_ms, out_ms = self._slider.trim()
        total = self._total_frames()
        first = self._frames_for(in_ms) + 1                 # 1-based for display
        last = max(first, self._frames_for(out_ms))         # inclusive last kept
        if total:
            last = min(last, total)
        for box, value in ((self._in_frame, first), (self._out_frame, last)):
            box.blockSignals(True)
            if total:
                box.setRange(1, total)
            box.setValue(value)
            box.blockSignals(False)
        # No "/ total" suffix: the frame counter beside the playhead already shows
        # it, and here it just crowded the number out of a fixed-width box.

    def _frame_box_edited(self, which: str) -> None:
        """Typing a frame moves that edge, and snap still applies.

        Snapping a typed number could look like the box ignoring you, so when it
        moves the value the status line says so rather than leaving you to spot it.
        """
        if self._info is None:
            return
        in_ms, out_ms = self._slider.trim()
        # Back to zero-based: box 1 means index 0, and an inclusive out of N means
        # the selection ends at the start of index N.
        wanted = (self._ms_for_frame(self._in_frame.value() - 1) if which == "in"
                  else self._ms_for_frame(self._out_frame.value()))
        wanted = max(0, min(wanted, self._slider.duration()))
        if which == "in":
            new_in, new_out = min(wanted, out_ms), out_ms
        else:
            new_in, new_out = in_ms, max(wanted, in_ms)
        asked_frames = self._frames_for(new_out - new_in)
        if self._snap_btn.isChecked() and self.current_target() is not None:
            new_in, new_out = self._snap_trim(new_in, new_out, which)
        self._slider.set_trim(new_in, new_out)
        self._refresh_frame_boxes()
        got_frames = self._frames_for(new_out - new_in)
        if got_frames != asked_frames:
            self.controller._set_status(
                f"Snapped to {got_frames} frames \u2014 the nearest length "
                f"{self.current_target().label} accepts.")

    # ---- snapping to the model's frame grid ----

    def _on_snap_toggled(self, on: bool) -> None:
        self.controller.qsettings.setValue("trim_snap", bool(on))
        self._refresh_snap()
        self._refresh_trim_label()
        self.controller._set_status(
            "Trim snaps to the model's legal frame counts."
            if on else "Trim moves frame by frame.")

    def _refresh_snap(self) -> None:
        """Arm or disarm the snap hook for the current target."""
        target = self.current_target()
        if target is None or not self._snap_btn.isChecked():
            self._slider.set_snap(None)
            self._snap_btn.setEnabled(target is not None)
            return
        self._snap_btn.setEnabled(True)
        self._slider.set_snap(self._snap_trim)

    def _snap_trim(self, in_ms: int, out_ms: int, which: str) -> tuple[int, int]:
        """Pull a dragged edge so the selection lands on a legal frame count.

        The grid is a length rule (frames % modulus == remainder), so the edge
        being dragged moves and the opposite one stays put — snapping both would
        fight the user's intent about where the clip starts.
        """
        target = self.current_target()
        if target is None:
            return in_ms, out_ms
        duration = self._slider.duration()
        # The floor is the model's MINIMUM usable length, not the smallest number
        # that happens to satisfy the grid. For H3 those are 73 frames and 5: the
        # rungs below the minimum (5, 22, 39, 56) are all grid-legal and all
        # useless, so offering one looks like a valid choice and isn't.
        floor = target.min_frames()
        ceiling = target.max_frames()
        # A clip too short to hold one legal length has nothing to snap to. Leave
        # the selection alone and let the amber "under the model's minimum" warning
        # explain, rather than snapping to a length the model can't train on.
        if int(round(duration / 1000.0 * target.fps)) < floor:
            return in_ms, out_ms
        frames = max(1, int(round((out_ms - in_ms) / 1000.0 * target.fps)))
        legal = max(floor, min(target.snap_frames(frames, "nearest"), ceiling))
        span = int(round(target.seconds_for_frames(legal) * 1000))
        if which == "out":
            new_out = in_ms + span
            if new_out > duration:
                # Past the end: step down a rung rather than clamping to an
                # illegal length.
                lower = target.snap_frames(
                    max(1, int((duration - in_ms) / 1000.0 * target.fps)), "down")
                lower = max(floor, min(lower, ceiling))
                new_out = in_ms + int(round(target.seconds_for_frames(lower) * 1000))
            return in_ms, min(duration, new_out)
        new_in = out_ms - span
        if new_in < 0:
            lower = target.snap_frames(
                max(1, int(out_ms / 1000.0 * target.fps)), "down")
            lower = max(floor, min(lower, ceiling))
            new_in = out_ms - int(round(target.seconds_for_frames(lower) * 1000))
        return max(0, new_in), out_ms

    # ---- mute section ----

    def _on_mute_section_toggled(self, on: bool) -> None:
        # The label names the action, not the state: a checked button reading
        # "Mute section" looks like it's telling you what mode you're in when it's
        # actually the thing you press to leave.
        self._mute_section_btn.setText("Discard mute" if on else "Mute section")
        self._mute_section_btn.setToolTip(
            "Drop the mute brackets without changing the clip" if on
            else "Silence part of the clip's audio \u2014 useful when a trim cuts a "
                 "word in half. The picture is untouched.")
        self.set_mute_mode(on)

    def set_mute_mode(self, on: bool) -> None:
        """Show the red mute band and its controls. Leaving the mode always drops
        any un-kept preview, so the file on disk is never left as a temp render."""
        if on and self._path is not None and not has_audio_stream(self._path):
            self._mute_section_btn.setChecked(False)
            self.controller._set_status(
                "This clip has no audio track \u2014 nothing to mute.")
            return
        self._slider.set_mute_visible(on)
        if not on and self._audio is not None:
            self._audio.setVolume(self._base_volume)
        self._mute_apply_btn.setVisible(on)
        self._update_mute_buttons()
        self._refresh_trim_label()

    def _update_mute_buttons(self) -> None:
        lo, hi = self._slider.mute_range()
        self._mute_apply_btn.setEnabled(hi - lo >= 30)

    def commit_mute(self) -> None:
        """Render the mute and swap it in, keeping the untouched original.

        The render happens here and nowhere else, into the folder's own scratch
        space, so there's never a playable stray sitting in the dataset.
        """
        if self._path is None:
            return
        in_ms, out_ms = self._slider.mute_range()
        secs = (out_ms - in_ms) / 1000.0
        if secs < 0.03:
            self.controller._set_status("Drag the red brackets to choose what to mute.")
            return
        if QMessageBox.question(
            self.controller, "Keep the mute",
            f"Silence {secs:.2f}s of audio in {self._path.name}?\n\n"
            "The picture is unchanged, and the untouched original is kept in "
            ".original/.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return
        store = self.controller.store
        if store is None:
            return
        tmp = store.work_dir() / f"mute{self._path.suffix}"
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            ok, message = apply_mute_span(self._path, tmp,
                                          in_ms / 1000.0, out_ms / 1000.0)
        finally:
            QApplication.restoreOverrideCursor()
        if not ok:
            tmp.unlink(missing_ok=True)
            QMessageBox.critical(self.controller, "Could not mute", message)
            return
        try:
            store.backup_original(self._path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            QMessageBox.critical(self.controller, "Could not back up", str(exc))
            return
        # Release the file before replacing it — Windows will refuse otherwise.
        self.release()
        try:
            os.replace(tmp, self._path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            QMessageBox.critical(self.controller, "Could not save", str(exc))
            return
        self._mute_section_btn.setChecked(False)
        self._update_mute_buttons()
        self.controller._purge_poster_cache(self._path)
        self._load_into_player(self._path)
        self.refresh_audio_badge()
        self.controller._set_status(f"Muted {secs:.2f}s in {self._path.name}.")

    def refresh_audio_badge(self) -> None:
        badge = getattr(self, "_audio_badge", None)
        if badge is None:
            return
        if self._info is None:
            badge.setText("")
            return
        theme = self.controller.theme
        ok, why = self.controller.audio_status()
        has_track = False
        if self._path is not None:
            try:
                has_track = has_audio_stream(self._path)
            except Exception:
                has_track = False
        if not has_track:
            badge.setText("\U0001F507 silent clip")
            badge.setToolTip("This clip has no audio track, so there's no sound to "
                             "caption.")
            badge.setStyleSheet(f"color: {theme.text_secondary};")
        elif ok:
            badge.setText("\U0001F50A audio on")
            badge.setToolTip("The clip's audio is sent with the frames, so dialogue "
                             "and sound can be described.")
            badge.setStyleSheet(f"color: {theme.success};")
        else:
            badge.setText("\U0001F507 audio off")
            badge.setToolTip(why)
            badge.setStyleSheet(f"color: {theme.warning};")

    def _refresh_trim_label(self) -> None:
        if self._info is None:
            self._trim_label.setText("")
            return
        in_ms, out_ms = self._slider.trim()
        span_s = max(0.0, (out_ms - in_ms) / 1000.0)
        target = self.current_target()
        if target is None:
            frames = int(round(span_s * (self._info.fps or 0)))
            self._trim_label.setText(f"{span_s:.2f}s \u00b7 ~{frames} frames")
            self._trim_label.setStyleSheet(f"color: {self.controller.theme.text_secondary};")
            return
        frames = int(round(span_s * target.fps))
        legal = target.is_legal_frames(frames)
        capped = frames > target.max_frames()
        short = frames < target.min_frames()
        if legal and not capped and not short:
            text = f"{frames} frames \u00b7 {span_s:.2f}s @ {target.fps:g}fps \u2713"
            colour = self.controller.theme.success
        else:
            nearest = target.snap_frames(min(frames, target.max_frames()), "down")
            reason = ("over the model's maximum" if capped
                      else "under the model's minimum" if short
                      else f"not on the {target.frame_modulus}n"
                           f"+{target.frame_remainder} grid")
            text = f"{frames} frames \u2014 {reason}; nearest {nearest}"
            colour = self.controller.theme.warning
        self._trim_label.setText(text)
        self._trim_label.setStyleSheet(f"color: {colour};")

    # ---- public API ----

    def load(self, path: Path, info: "VideoInfo | None") -> None:
        self._path = Path(path)
        self._info = info
        self._fps = info.fps if info and info.fps else 25.0
        # A new clip starts fully selected; duration arrives from the player.
        self._slider.set_duration(int((info.duration_s * 1000) if info else 0))
        self._slider.set_trim(0, self._slider.duration())
        if self._crop_btn.isChecked():
            self._crop_btn.setChecked(False)     # also tears down the rect
        self._rotation = 0
        self._set_position_labels(0)
        self._refresh_frame_boxes()
        self._refresh_snap()
        self._load_peaks()
        self._restore_pending()
        self._refresh_trim_label()
        self.refresh_audio_badge()
        for w in (self._in_btn, self._out_btn, self._reset_btn, self._fit_btn,
                  self._apply_btn, self._target_combo, self._nudge_back,
                  self._nudge_fwd, self._crop_btn, self._mute_section_btn,
                  self._save_frame_btn, self._save_audio_btn, self._snap_btn,
                  self._rot_ccw, self._rot_cw):
            w.setEnabled(info is not None)
        self._meta_label.setText(
            f"{info.width}\u00d7{info.height} \u00b7 {info.fps:g} fps \u00b7 {info.codec}"
            if info else "")
        if not self._available or self._player is None:
            poster = self.controller._video_poster(self._path)
            self._fallback.setVisible(True)
            if poster:
                self._fallback.setPixmap(QPixmap(str(poster)).scaled(
                    720, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self._fallback.setText(
                    "Video playback needs the QtMultimedia component of PySide6.")
            for w in (self._play_btn, self._back_btn, self._fwd_btn, self._slider,
                      self._mute_btn):
                w.setEnabled(False)
            return
        self._load_into_player(self._path)

    def _load_peaks(self) -> None:
        """Waveform for the current clip, cached per path so scrubbing back and
        forth doesn't re-decode."""
        self._slider.set_peaks([])
        if self._path is None or not self._available:
            return
        key = str(self._path)
        if key not in self._peaks_cache:
            try:
                self._peaks_cache[key] = audio_peaks(self._path, buckets=320)
            except Exception:
                self._peaks_cache[key] = []
        self._slider.set_peaks(self._peaks_cache[key])

    def _load_into_player(self, path: Path | None) -> None:
        """Point the player at a file without disturbing trim/mute state — used for
        the initial load and for swapping between a clip and its muted preview."""
        if self._player is None or path is None:
            return
        self._player.setSource(QUrl.fromLocalFile(str(path)))
        self._player.pause()   # show first frame without autoplaying

    def stop(self) -> None:
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())
        self._path = None

    def toggle_play(self) -> None:
        if self._player is None:
            return
        from PySide6.QtMultimedia import QMediaPlayer
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
            return
        # Start inside the selection, so pressing play always previews the trim
        # rather than whatever happens to be under the playhead.
        in_ms, out_ms = self._slider.trim()
        if out_ms > in_ms and not (in_ms <= self._player.position() < out_ms):
            self._player.setPosition(in_ms)
            self._slider.set_position(in_ms)
        self._player.play()

    def toggle_mute(self) -> None:
        if self._player is None:
            return
        muted = not self._audio.isMuted()
        self._audio.setMuted(muted)
        self._mute_btn.setIcon(lucide_icon(
            "volume-x" if muted else "volume-2", self.controller.theme.text_secondary, 16))

    def toggle_playback(self) -> None:
        if self._player is None:
            return
        if self._is_playing():
            self._player.pause()
        else:
            self._player.play()

    def seek_to_trim(self, edge: str) -> None:
        """Jump the playhead to a trim bracket — the two frames you actually check."""
        in_ms, out_ms = self._slider.trim()
        target = in_ms if edge == "in" else max(in_ms, out_ms - 1)
        self._slider.set_position(target)
        self._seek(target)
        self._set_position_labels(target)

    def step_frames(self, frames: int) -> None:
        """Nudge by whole frames — the precision trim work needs this."""
        if self._player is None:
            return
        self._player.pause()
        step = int(round(1000.0 / max(1.0, getattr(self, "_fps", 25.0)))) * frames
        self._player.setPosition(
            max(0, min(self._duration_ms, self._player.position() + step)))

    # ---- internals ----

    def _set_position_labels(self, ms: int) -> None:
        """Time and frame number for the playhead.

        Frames matter more than the clock here: every rule about conforming a clip
        is expressed in frames, so f0073 is the number you can act on.
        """
        self._pos_label.setText(self._fmt(ms))
        info = self._info
        if info is None or not info.fps:
            self._frame_label.setText("")
            return
        frame = int(round(ms / 1000.0 * info.fps)) + 1        # 1-based, as above
        total = info.frame_count or int(round(info.duration_s * info.fps))
        if total:
            frame = min(frame, total)
        self._frame_label.setText(f"f{frame}" + (f" / {total}" if total else ""))

    @staticmethod
    def _fmt(ms: int) -> str:
        total = max(0, ms // 1000)
        h, rem = divmod(total, 3600)
        m, sec = divmod(rem, 60)
        return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"

    def _seek(self, ms: int) -> None:
        if self._player is not None:
            self._player.setPosition(ms)

    def _on_slider_pressed(self) -> None:
        if self._player is not None:
            self._player.setPosition(self._slider.value())

    def _remember_volume(self) -> None:
        """Capture the user's volume so live-mute restores it rather than a
        hard-coded full."""
        if self._audio is not None and self._audio.volume() > 0.01:
            self._base_volume = self._audio.volume()

    def _apply_live_mute(self, ms: int) -> None:
        """Silence the player while the playhead is inside the mute band.

        Renders nothing: you hear the cut as you scrub, and only pay for an encode
        when you commit. Rendering first meant waiting on a full re-encode just to
        find out the band was 200ms off.
        """
        if self._audio is None or not self._slider.mute_visible():
            return
        lo, hi = self._slider.mute_range()
        inside = lo <= ms < hi and hi > lo
        want = 0.0 if inside else self._base_volume
        if abs(self._audio.volume() - want) > 0.01:
            self._audio.setVolume(want)

    def _on_position(self, ms: int) -> None:
        self._apply_live_mute(ms)
        # Confine playback to the selection: what plays is what the trim will keep.
        # Looping (rather than stopping) lets you watch the cut repeatedly while
        # nudging the handles.
        in_ms, out_ms = self._slider.trim()
        if self._is_playing() and out_ms > in_ms and ms >= out_ms - 40:
            self._player.setPosition(in_ms)
            self._slider.set_position(in_ms)
            self._set_position_labels(in_ms)
            return
        self._slider.set_position(ms)
        self._set_position_labels(ms)

    def _is_playing(self) -> bool:
        if self._player is None:
            return False
        from PySide6.QtMultimedia import QMediaPlayer
        return self._player.playbackState() == QMediaPlayer.PlayingState

    def _on_duration(self, ms: int) -> None:
        if ms <= 0:
            return
        self._duration_ms = ms
        self._slider.set_duration(ms)
        self._len_label.setText(self._fmt(ms))
        self._refresh_trim_label()

    def apply_edit(self) -> None:
        """Hand the current selection + target to the controller to re-encode."""
        if self._path is None or self._info is None:
            return
        in_ms, out_ms = self._slider.trim()
        self.controller.apply_video_edit(
            self._path, self._info, in_ms / 1000.0, out_ms / 1000.0,
            self.current_target(), self.crop_rect(), self._rotation)

    def release(self) -> None:
        """Drop the file handle so the clip can be overwritten on disk."""
        if self._player is not None:
            self._player.stop()
            self._player.setSource(QUrl())

    def _on_error(self, _error, message: str) -> None:
        """Real playback failures belong in the status bar, not buried in stderr."""
        name = self._path.name if self._path else "video"
        self.controller._set_status(
            f"Could not play {name}: {message}" if message
            else f"Could not play {name}.")

    def _on_state(self, state) -> None:
        from PySide6.QtMultimedia import QMediaPlayer
        playing = state == QMediaPlayer.PlayingState
        self._play_btn.setIcon(lucide_icon(
            "pause" if playing else "play", self.controller.theme.text_primary, 18))


# Shared by the image crop dialog and the video crop overlay so the two can't
# drift apart.
ASPECT_PRESETS: tuple[tuple[str, float | None], ...] = (
    ("Freeform", None),
    ("1:1", 1.0),
    ("3:2", 3 / 2), ("2:3", 2 / 3),
    ("4:3", 4 / 3), ("3:4", 3 / 4),
    ("16:9", 16 / 9), ("9:16", 9 / 16),
    ("21:9", 21 / 9), ("9:21", 9 / 21),
)


class CropRectItem(QGraphicsRectItem):
    """Interactive crop rectangle: drag inside to move, drag a corner to resize,
    optionally constrained to a fixed aspect ratio. Lives in image-pixel scene
    coordinates and is always clamped to the image bounds. Calls on_change() after
    every geometry change so the dialog can sync its mask, labels and spins."""

    def __init__(self, bounds: QRectF, on_change) -> None:
        super().__init__(bounds)
        self._bounds = QRectF(bounds)
        self._on_change = on_change
        self._aspect: float | None = None      # w/h, None = freeform
        self._mode: str | None = None          # None | "move" | corner name
        self._press_pos = QPointF()
        self._press_rect = QRectF()
        self.setAcceptHoverEvents(True)
        self.setZValue(10)
        pen = QPen(QColor("#2FC6B3"), 0)
        pen.setCosmetic(True)
        pen.setWidth(2)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.NoBrush))

    # -- geometry helpers -------------------------------------------------
    def _handle_px(self) -> float:
        """Corner hit radius in scene units, scaled so it's grabbable at any zoom."""
        view = self.scene().views()[0] if self.scene() and self.scene().views() else None
        if view is None:
            return 12.0
        return 12.0 / max(view.transform().m11(), 1e-6)

    def _corners(self) -> dict[str, QPointF]:
        r = self.rect()
        return {"tl": r.topLeft(), "tr": r.topRight(),
                "bl": r.bottomLeft(), "br": r.bottomRight()}

    def _corner_at(self, pos: QPointF) -> str | None:
        h = self._handle_px()
        for name, pt in self._corners().items():
            if abs(pos.x() - pt.x()) <= h and abs(pos.y() - pt.y()) <= h:
                return name
        return None

    def set_bounds(self, bounds: QRectF) -> None:
        """Re-clamp to a new frame size — a video's true size only arrives once
        decoding starts, so the rect may be created before it's known."""
        self._bounds = QRectF(bounds)
        r = self.rect().intersected(self._bounds)
        if r.isEmpty():
            r = QRectF(self._bounds)
        self.setRect(r)
        self._on_change()

    def set_aspect(self, aspect: float | None) -> None:
        """Constrain to w/h (None = freeform) and refit the rect: the largest rect of
        that aspect that fits the image, centered — predictable on every switch."""
        self._aspect = aspect
        b = self._bounds
        if aspect is None:
            self.setRect(QRectF(b))
        else:
            w, h = b.width(), b.height()
            if w / h > aspect:
                w = h * aspect
            else:
                h = w / aspect
            x = b.center().x() - w / 2
            y = b.center().y() - h / 2
            self.setRect(QRectF(x, y, w, h))
        self._changed()

    def _changed(self) -> None:
        self.update()
        if self._on_change:
            self._on_change()

    # -- painting ---------------------------------------------------------
    def paint(self, painter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        # corner handles: small filled squares, cosmetic size
        h = self._handle_px()
        painter.setPen(QPen(QColor("#0F1115"), 0))
        painter.setBrush(QBrush(QColor("#2FC6B3")))
        s = h * 0.8
        for pt in self._corners().values():
            painter.drawRect(QRectF(pt.x() - s / 2, pt.y() - s / 2, s, s))

    # -- interaction ------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        pos = event.pos()
        corner = self._corner_at(pos)
        if corner:
            self._mode = corner
        elif self.rect().contains(pos):
            self._mode = "move"
        else:
            self._mode = None
        self._press_pos = pos
        self._press_rect = QRectF(self.rect())
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._mode is None:
            return
        pos = event.pos()
        b = self._bounds
        if self._mode == "move":
            delta = pos - self._press_pos
            r = QRectF(self._press_rect)
            r.translate(delta)
            # clamp inside the image
            dx = max(b.left() - r.left(), 0) or min(b.right() - r.right(), 0)
            dy = max(b.top() - r.top(), 0) or min(b.bottom() - r.bottom(), 0)
            r.translate(dx, dy)
            self.setRect(r)
            self._changed()
            return
        # corner resize: anchor the opposite corner, follow the cursor
        anchor = {"tl": self._press_rect.bottomRight(), "tr": self._press_rect.bottomLeft(),
                  "bl": self._press_rect.topRight(), "br": self._press_rect.topLeft()}[self._mode]
        px = min(max(pos.x(), b.left()), b.right())
        py = min(max(pos.y(), b.top()), b.bottom())
        w = abs(px - anchor.x())
        h = abs(py - anchor.y())
        sx = 1 if px >= anchor.x() else -1
        sy = 1 if py >= anchor.y() else -1
        if self._aspect is not None and w > 0 and h > 0:
            # fit the aspect inside the dragged span, then clamp to the image
            if w / h > self._aspect:
                w = h * self._aspect
            else:
                h = w / self._aspect
            # available room from the anchor in the drag direction
            avail_w = (b.right() - anchor.x()) if sx > 0 else (anchor.x() - b.left())
            avail_h = (b.bottom() - anchor.y()) if sy > 0 else (anchor.y() - b.top())
            scale = min(1.0, (avail_w / w) if w > 0 else 1.0, (avail_h / h) if h > 0 else 1.0)
            w *= scale
            h *= scale
        w = max(w, 8.0)
        h = max(h, 8.0)
        r = QRectF(min(anchor.x(), anchor.x() + sx * w), min(anchor.y(), anchor.y() + sy * h), w, h)
        r = r.intersected(b)
        if r.width() >= 8 and r.height() >= 8:
            self.setRect(r)
            self._changed()

    def mouseReleaseEvent(self, event) -> None:
        self._mode = None
        event.accept()


class CropResizeDialog(QDialog):
    """Crop and/or resize the current image, destructively but safely: the pre-edit
    file is copied to <folder>/.original/<name> first (first backup wins), then the
    edited pixels replace the file in place so the dataset keeps stable filenames.

    Aspect presets cover the common training buckets (1:1, 3:2, 4:3, 16:9, 21:9 and
    their portrait counterparts) plus freeform. Output can optionally be resized;
    the W/H spins stay linked to the crop's aspect so the result is never stretched.
    """

    ASPECTS = ASPECT_PRESETS

    def __init__(self, controller, image_path: Path, theme: "Theme") -> None:
        super().__init__(controller)
        self.controller = controller
        self.image_path = Path(image_path)
        self.theme = theme
        self.setWindowTitle(f"Crop / Resize \u2014 {self.image_path.name}")
        self.resize(860, 620)

        self._rotation = 0
        pm = QPixmap(str(self.image_path))
        if pm.isNull():
            raise ValueError(f"Could not open {self.image_path}")
        self.src_w, self.src_h = pm.width(), pm.height()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 12, 12, 12)
        lay.setSpacing(10)

        # -- canvas -------------------------------------------------------
        self.scene = QGraphicsScene(self)
        self.pix_item = self.scene.addPixmap(pm)
        self.scene.setSceneRect(QRectF(0, 0, self.src_w, self.src_h))
        bounds = QRectF(0, 0, self.src_w, self.src_h)
        # dim everything outside the crop (even-odd fill: outer rect minus crop rect)
        self.mask_item = QGraphicsPathItem()
        self.mask_item.setBrush(QBrush(QColor(0, 0, 0, 140)))
        self.mask_item.setPen(QPen(Qt.NoPen))
        self.mask_item.setZValue(5)
        self.scene.addItem(self.mask_item)
        self.crop_item = CropRectItem(bounds, self._crop_changed)
        self.scene.addItem(self.crop_item)

        self.view = QGraphicsView(self.scene)
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.view.setBackgroundBrush(QBrush(QColor(self.theme.surface_0)))
        self.view.setFrameShape(QFrame.NoFrame)
        lay.addWidget(self.view, 1)

        # -- controls -----------------------------------------------------
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(QLabel("Aspect:"))
        self.aspect_combo = QComboBox()
        for label, _ratio in self.ASPECTS:
            self.aspect_combo.addItem(label)
        self.aspect_combo.currentIndexChanged.connect(self._aspect_selected)
        row.addWidget(self.aspect_combo)

        self.crop_label = QLabel("")
        self.crop_label.setStyleSheet(f"color: {self.theme.text_secondary};")
        row.addWidget(self.crop_label)

        reset = QPushButton("Reset crop")
        reset.clicked.connect(self._reset_crop)
        row.addWidget(reset)

        rot_ccw = QToolButton()
        rot_ccw.setObjectName("NavBtn")
        rot_ccw.setIcon(lucide_icon("rotate-ccw", self.theme.text_secondary, 15))
        rot_ccw.setToolTip("Rotate 90\u00b0 anticlockwise")
        rot_ccw.clicked.connect(lambda: self._rotate_by(-90))
        row.addWidget(rot_ccw)
        rot_cw = QToolButton()
        rot_cw.setObjectName("NavBtn")
        rot_cw.setIcon(lucide_icon("rotate-cw", self.theme.text_secondary, 15))
        rot_cw.setToolTip("Rotate 90\u00b0 clockwise")
        rot_cw.clicked.connect(lambda: self._rotate_by(90))
        row.addWidget(rot_cw)
        self.rot_label = QLabel("")
        self.rot_label.setStyleSheet(f"color: {self.theme.warning};")
        row.addWidget(self.rot_label)
        row.addStretch(1)

        self.resize_check = QCheckBox("Resize output to")
        self.resize_check.toggled.connect(self._resize_toggled)
        row.addWidget(self.resize_check)
        self.w_spin = QSpinBox()
        self.w_spin.setRange(8, 32768)
        self.h_spin = QSpinBox()
        self.h_spin.setRange(8, 32768)
        for s in (self.w_spin, self.h_spin):
            s.setEnabled(False)
            s.setSuffix(" px")
        self.w_spin.valueChanged.connect(self._w_edited)
        self.h_spin.valueChanged.connect(self._h_edited)
        row.addWidget(self.w_spin)
        row.addWidget(QLabel("\u00d7"))
        row.addWidget(self.h_spin)
        lay.addLayout(row)

        # -- buttons ------------------------------------------------------
        btns = QHBoxLayout()
        self.revert_btn = QPushButton("Revert to original")
        self.revert_btn.setToolTip("Restore the backed-up original from .original/ "
                                   "(the backup is kept).")
        self.revert_btn.clicked.connect(self._revert)
        store = getattr(controller, "store", None)
        self.revert_btn.setVisible(bool(store and store.has_original_backup(self.image_path)))
        btns.addWidget(self.revert_btn)
        btns.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        btns.addWidget(cancel)
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.setDefault(True)
        self.apply_btn.clicked.connect(self._apply)
        btns.addWidget(self.apply_btn)
        lay.addLayout(btns)

        self._sync_guard = False
        self._crop_changed()

    # -- view fitting -----------------------------------------------------
    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    # -- crop/aspect sync -------------------------------------------------
    def _aspect_selected(self, idx: int) -> None:
        self.crop_item.set_aspect(self.ASPECTS[idx][1])

    def _rotate_by(self, degrees: int) -> None:
        """Turn the preview so the crop rect is drawn against the final orientation.

        Rotating only on save would have the user cropping one picture and getting
        another.
        """
        self._rotation = (self._rotation + degrees) % 360
        self.view.resetTransform()
        if self._rotation:
            self.view.rotate(self._rotation)
        self.view.fitInView(self.pix_item, Qt.KeepAspectRatio)
        self.rot_label.setText(f"rotated {self._rotation}\u00b0"
                               if self._rotation else "")
        # A quarter turn swaps the output dimensions, so the resize spinboxes stop
        # describing the result.
        if self._rotation % 180 and self.resize_check.isChecked():
            self.resize_check.setChecked(False)
        self.resize_check.setEnabled(not self._rotation % 180)
        self._crop_changed()

    def _reset_crop(self) -> None:
        # back to the full image; keep the chosen aspect constraint applied
        self.crop_item.set_aspect(self.ASPECTS[self.aspect_combo.currentIndex()][1])

    def crop_box(self) -> tuple[int, int, int, int]:
        """Integer (left, top, right, bottom) of the crop in image pixels."""
        r = self.crop_item.rect()
        left = max(0, int(round(r.left())))
        top = max(0, int(round(r.top())))
        right = min(self.src_w, int(round(r.right())))
        bottom = min(self.src_h, int(round(r.bottom())))
        return left, top, max(right, left + 1), max(bottom, top + 1)

    def _crop_changed(self) -> None:
        # mask: outer rect minus crop rect (even-odd)
        path = QPainterPath()
        path.setFillRule(Qt.OddEvenFill)
        path.addRect(QRectF(0, 0, self.src_w, self.src_h))
        path.addRect(self.crop_item.rect())
        self.mask_item.setPath(path)
        l, t, r, b = self.crop_box()
        w, h = r - l, b - t
        self.crop_label.setText(f"{w} \u00d7 {h} px")
        if not self.resize_check.isChecked():
            self._sync_guard = True
            self.w_spin.setValue(w)
            self.h_spin.setValue(h)
            self._sync_guard = False

    def _resize_toggled(self, on: bool) -> None:
        for s in (self.w_spin, self.h_spin):
            s.setEnabled(on)
        if on:
            l, t, r, b = self.crop_box()
            self._sync_guard = True
            self.w_spin.setValue(r - l)
            self.h_spin.setValue(b - t)
            self._sync_guard = False

    def _crop_aspect(self) -> float:
        l, t, r, b = self.crop_box()
        return (r - l) / max(1, (b - t))

    def _w_edited(self, val: int) -> None:
        if self._sync_guard or not self.resize_check.isChecked():
            return
        self._sync_guard = True
        self.h_spin.setValue(max(8, int(round(val / self._crop_aspect()))))
        self._sync_guard = False

    def _h_edited(self, val: int) -> None:
        if self._sync_guard or not self.resize_check.isChecked():
            return
        self._sync_guard = True
        self.w_spin.setValue(max(8, int(round(val * self._crop_aspect()))))
        self._sync_guard = False

    # -- apply / revert ---------------------------------------------------
    def output_size(self) -> tuple[int, int]:
        l, t, r, b = self.crop_box()
        if self.resize_check.isChecked():
            return self.w_spin.value(), self.h_spin.value()
        return r - l, b - t

    def _save_kwargs(self) -> dict:
        suffix = self.image_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            return {"quality": 95}
        if suffix == ".webp":
            return {"quality": 95}
        return {}

    def _apply(self) -> None:
        from PIL import Image
        l, t, r, b = self.crop_box()
        out_w, out_h = self.output_size()
        full = (l, t, r, b) == (0, 0, self.src_w, self.src_h)
        rotation = self._rotation % 360
        if full and (out_w, out_h) == (self.src_w, self.src_h) and not rotation:
            self.reject()  # nothing to do
            return
        store = getattr(self.controller, "store", None)
        try:
            if store is not None:
                store.backup_original(self.image_path)
            with Image.open(self.image_path) as im:
                im.load()
                if not full:
                    im = im.crop((l, t, r, b))
                if rotation:
                    # expand=True so a quarter turn keeps the whole picture rather
                    # than cropping it to the original frame.
                    im = im.rotate(-rotation, expand=True)
                if (out_w, out_h) != im.size and not rotation:
                    im = im.resize((out_w, out_h), Image.LANCZOS)
                im.save(self.image_path, **self._save_kwargs())
        except Exception as exc:
            QMessageBox.critical(self, "Edit failed", str(exc))
            return
        self.accept()

    def _revert(self) -> None:
        store = getattr(self.controller, "store", None)
        if store is None or not store.restore_original(self.image_path):
            QMessageBox.information(self, "No backup",
                                    "No backed-up original exists for this file.")
            return
        self.accept()


class BatchResizeThread(QThread):
    """Resizes a list of images in place, backing each up to .original/ first.

    Pillow work is slow enough on big folders to block the UI, so it runs off the
    main thread and reports progress per image. Failures are collected rather than
    aborting the run — one unreadable file shouldn't stop a 2,000-image folder.
    """

    item_progress = Signal(int, int, str)      # index (1-based), total, filename
    item_done = Signal(str, int, int)          # path str, new width, new height
    item_error = Signal(str, str)              # path str, message
    batch_finished = Signal(int, int, int, bool)  # changed, skipped, failed, cancelled

    def __init__(self, store, paths, plan):
        super().__init__()
        self.store = store
        self.paths = list(paths)
        self.plan = plan   # BatchResizePlan

    def run(self) -> None:
        from PIL import Image
        changed = skipped = failed = 0
        total = len(self.paths)
        for i, path in enumerate(self.paths, start=1):
            if self.isInterruptionRequested():
                self.batch_finished.emit(changed, skipped, failed, True)
                return
            self.item_progress.emit(i, total, path.name)
            try:
                with Image.open(path) as im:
                    im.load()
                    target = self.plan.target_for(im.size)
                    if target is None:
                        skipped += 1
                        continue
                    box = self.plan.crop_box_for(im.size, target)
                    if self.store is not None:
                        self.store.backup_original(path)
                    out = im.crop(box) if box else im
                    if out.size != target:
                        out = out.resize(target, Image.LANCZOS)
                    out.save(path, **self.plan.save_kwargs(path))
                changed += 1
                self.item_done.emit(str(path), target[0], target[1])
            except Exception as exc:
                failed += 1
                self.item_error.emit(str(path), str(exc))
        self.batch_finished.emit(changed, skipped, failed, False)


class BatchResizePlan:
    """Pure geometry for a batch resize — no I/O, so it's cheap to preview and easy
    to test. `mode` is one of: 'longest', 'shortest', 'exact', 'percent'.

    'exact' centre-crops to the target aspect before scaling so images are never
    stretched; the other modes preserve the source aspect and never crop.
    """

    def __init__(self, mode: str, value: int = 1024, width: int = 1024,
                 height: int = 1024, percent: int = 100, allow_upscale: bool = False):
        self.mode = mode
        self.value = int(value)
        self.width = int(width)
        self.height = int(height)
        self.percent = int(percent)
        self.allow_upscale = bool(allow_upscale)

    def target_for(self, size: tuple[int, int]) -> tuple[int, int] | None:
        """Output size for a source size, or None when the image should be skipped
        (already conforming, or would need an upscale that isn't allowed)."""
        w, h = size
        if w <= 0 or h <= 0:
            return None
        if self.mode == "percent":
            if self.percent == 100:
                return None
            if self.percent > 100 and not self.allow_upscale:
                return None
            tw = max(1, round(w * self.percent / 100))
            th = max(1, round(h * self.percent / 100))
        elif self.mode == "exact":
            tw, th = max(1, self.width), max(1, self.height)
            if not self.allow_upscale and (tw > w or th > h):
                return None
        else:
            cur = max(w, h) if self.mode == "longest" else min(w, h)
            if cur == self.value:
                return None
            if cur < self.value and not self.allow_upscale:
                return None
            scale = self.value / cur
            tw = max(1, round(w * scale))
            th = max(1, round(h * scale))
        if (tw, th) == (w, h):
            return None
        return tw, th

    def crop_box_for(self, size: tuple[int, int],
                     target: tuple[int, int]) -> tuple[int, int, int, int] | None:
        """Centre-crop box needed before scaling. Only 'exact' crops (to match the
        target aspect without stretching); other modes return None."""
        if self.mode != "exact":
            return None
        w, h = size
        tw, th = target
        src_aspect = w / h
        dst_aspect = tw / th
        if abs(src_aspect - dst_aspect) < 1e-6:
            return None
        if src_aspect > dst_aspect:      # too wide: trim the sides
            cw = round(h * dst_aspect)
            x = (w - cw) // 2
            return (x, 0, x + cw, h)
        ch = round(w / dst_aspect)       # too tall: trim top/bottom
        y = (h - ch) // 2
        return (0, y, w, y + ch)

    def save_kwargs(self, path: Path) -> dict:
        suffix = Path(path).suffix.lower()
        if suffix in (".jpg", ".jpeg", ".webp"):
            return {"quality": 95}
        return {}


class BatchResizeDialog(QDialog):
    """Resize every image in the folder (or just the flagged ones) in place, with
    each original preserved in .original/ exactly like the single-image editor.

    Shows a live count of how many images the current settings would actually change
    before anything is written, so a mis-set value is obvious up front.
    """

    MODES = (
        ("Longest side at most", "longest"),
        ("Shortest side at most", "shortest"),
        ("Exact size (centre-crop to fit)", "exact"),
        ("Scale by percentage", "percent"),
    )

    def __init__(self, controller, theme: "Theme") -> None:
        super().__init__(controller)
        self.controller = controller
        self.theme = theme
        self.setWindowTitle("Batch resize")
        self.setMinimumWidth(560)
        self._thread: BatchResizeThread | None = None
        self._sizes: dict[str, tuple[int, int]] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(12)

        intro = QLabel(
            "Resizes images in place. Each original is copied to <b>.original/</b> "
            "first, so this can always be undone."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {theme.text_secondary};")
        lay.addWidget(intro)

        # -- scope --------------------------------------------------------
        scope_row = QHBoxLayout()
        scope_row.addWidget(QLabel("Apply to:"))
        self.scope_combo = QComboBox()
        all_n = len(controller.images)
        flagged = self._flagged_paths()
        self.scope_combo.addItem(f"All files ({all_n})", "all")
        self.scope_combo.addItem(f"Flagged for review ({len(flagged)})", "flagged")
        self.scope_combo.currentIndexChanged.connect(self._refresh_estimate)
        scope_row.addWidget(self.scope_combo)
        scope_row.addStretch(1)
        lay.addLayout(scope_row)

        # -- mode ---------------------------------------------------------
        mode_row = QHBoxLayout()
        self.mode_combo = QComboBox()
        for label, _key in self.MODES:
            self.mode_combo.addItem(label)
        self.mode_combo.currentIndexChanged.connect(self._mode_changed)
        mode_row.addWidget(self.mode_combo)

        self.value_spin = QSpinBox()
        self.value_spin.setRange(8, 32768)
        self.value_spin.setValue(1024)
        self.value_spin.setSuffix(" px")
        self.value_spin.valueChanged.connect(self._refresh_estimate)
        mode_row.addWidget(self.value_spin)

        self.w_spin = QSpinBox(); self.w_spin.setRange(8, 32768); self.w_spin.setValue(1024)
        self.h_spin = QSpinBox(); self.h_spin.setRange(8, 32768); self.h_spin.setValue(1024)
        for s in (self.w_spin, self.h_spin):
            s.setSuffix(" px")
            s.valueChanged.connect(self._refresh_estimate)
            s.setVisible(False)
        self.x_label = QLabel("\u00d7")
        self.x_label.setVisible(False)
        mode_row.addWidget(self.w_spin)
        mode_row.addWidget(self.x_label)
        mode_row.addWidget(self.h_spin)

        self.pct_spin = QSpinBox()
        self.pct_spin.setRange(1, 400)
        self.pct_spin.setValue(50)
        self.pct_spin.setSuffix(" %")
        self.pct_spin.setVisible(False)
        self.pct_spin.valueChanged.connect(self._refresh_estimate)
        mode_row.addWidget(self.pct_spin)
        mode_row.addStretch(1)
        lay.addLayout(mode_row)

        self.upscale_check = QCheckBox("Allow upscaling images that are already smaller")
        self.upscale_check.toggled.connect(self._refresh_estimate)
        lay.addWidget(self.upscale_check)

        self.estimate = QLabel("")
        self.estimate.setWordWrap(True)
        lay.addWidget(self.estimate)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)

        btns = QHBoxLayout()
        btns.addStretch(1)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self._cancel)
        btns.addWidget(self.cancel_btn)
        self.run_btn = QPushButton("Resize")
        self.run_btn.setDefault(True)
        self.run_btn.clicked.connect(self._run)
        btns.addWidget(self.run_btn)
        lay.addLayout(btns)

        self._refresh_estimate()

    # -- scope / plan -----------------------------------------------------
    def _flagged_paths(self) -> list[Path]:
        proj = getattr(self.controller, "project", None)
        if proj is None:
            return []
        return [p for p in self.controller.images if proj.is_review_marked(p.name)]

    def target_paths(self) -> list[Path]:
        # Pillow can't open videos; batch resize is an image operation. Video
        # resizing arrives with the ffmpeg edit dialog.
        if self.scope_combo.currentData() == "flagged":
            paths = self._flagged_paths()
        else:
            paths = list(self.controller.images)
        return [p for p in paths if not is_video(p)]

    def plan(self) -> BatchResizePlan:
        mode = self.MODES[self.mode_combo.currentIndex()][1]
        return BatchResizePlan(
            mode=mode, value=self.value_spin.value(),
            width=self.w_spin.value(), height=self.h_spin.value(),
            percent=self.pct_spin.value(),
            allow_upscale=self.upscale_check.isChecked(),
        )

    def _mode_changed(self, idx: int) -> None:
        mode = self.MODES[idx][1]
        self.value_spin.setVisible(mode in ("longest", "shortest"))
        for wdg in (self.w_spin, self.h_spin, self.x_label):
            wdg.setVisible(mode == "exact")
        self.pct_spin.setVisible(mode == "percent")
        self._refresh_estimate()

    def _image_size(self, path: Path) -> tuple[int, int] | None:
        """Cached (w, h) read from the header only — no full decode, so the estimate
        stays responsive on large folders."""
        key = str(path)
        if key not in self._sizes:
            try:
                from PIL import Image
                with Image.open(path) as im:
                    self._sizes[key] = im.size
            except Exception:
                self._sizes[key] = None
        return self._sizes[key]

    def _refresh_estimate(self) -> None:
        paths = self.target_paths()
        plan = self.plan()
        changed = unreadable = 0
        for p in paths:
            size = self._image_size(p)
            if size is None:
                unreadable += 1
            elif plan.target_for(size) is not None:
                changed += 1
        total = len(paths)
        skipped = total - changed - unreadable
        parts = [f"<b>{changed}</b> of {total} image(s) will be resized"]
        if skipped:
            parts.append(f"{skipped} already match and will be skipped")
        if unreadable:
            parts.append(f"{unreadable} could not be read")
        self.estimate.setText(" \u00b7 ".join(parts) + ".")
        self.run_btn.setEnabled(changed > 0)

    # -- run --------------------------------------------------------------
    def _run(self) -> None:
        paths = self.target_paths()
        plan = self.plan()
        work = [p for p in paths
                if (sz := self._image_size(p)) is not None and plan.target_for(sz) is not None]
        if not work:
            return
        resp = QMessageBox.question(
            self, "Resize images",
            f"Resize {len(work)} image(s) in place?\n\n"
            "Each original is copied to .original/ first, so this can be reverted.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if resp != QMessageBox.Yes:
            return
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, len(work))
        self.progress.setValue(0)
        self._errors: list[str] = []
        self._thread = BatchResizeThread(self.controller.store, work, plan)
        self._thread.item_progress.connect(self._on_progress)
        self._thread.item_done.connect(self._on_item_done)
        self._thread.item_error.connect(lambda p, m: self._errors.append(f"{Path(p).name}: {m}"))
        self._thread.batch_finished.connect(self._on_finished)
        self._thread.start()

    def _on_progress(self, i: int, total: int, name: str) -> None:
        self.progress.setValue(i)
        self.estimate.setText(f"Resizing {i} / {total} \u2014 {name}")

    def _on_item_done(self, path_str: str, w: int, h: int) -> None:
        self._sizes[path_str] = (w, h)
        self.controller._refresh_edited_image_cached(Path(path_str))

    def _on_finished(self, changed: int, skipped: int, failed: int, cancelled: bool) -> None:
        self._thread = None
        self.progress.setVisible(False)
        self.controller._refresh_current_after_batch()
        verb = "Cancelled after" if cancelled else "Resized"
        msg = f"{verb} {changed} image(s)."
        if skipped:
            msg += f" {skipped} skipped."
        if failed:
            msg += f" {failed} failed."
        if self._errors:
            msg += "\n\n" + "\n".join(self._errors[:10])
            if len(self._errors) > 10:
                msg += f"\n… and {len(self._errors) - 10} more."
        QMessageBox.information(self, "Batch resize", msg)
        self.accept()

    def _cancel(self) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.requestInterruption()
            return
        self.reject()

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            self._thread.requestInterruption()
            self._thread.wait(5000)
        super().closeEvent(event)


class ToggleSwitch(QAbstractButton):
    """A checkable on/off switch styled to the token spec (track 34×19, knob 15).

    Drop-in for QCheckBox state-wise: checkable, emits toggled. The knob glides
    between states with an eased animation and the track/knob colors crossfade.
    Programmatic state changes made while signals are blocked (our load/sync paths)
    snap instantly, so navigating images doesn't fire a flurry of animations.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self._tw, self._th, self._knob = 34, 19, 15
        self.setFixedSize(self._tw, self._th)
        self._progress = 1.0 if self.isChecked() else 0.0
        self._anim = QPropertyAnimation(self, b"knobProgress", self)
        self._anim.setDuration(MOTION_FAST)
        self._anim.setEasingCurve(QEasingCurve.OutCubic)
        self.toggled.connect(self._glide)

    def sizeHint(self) -> QSize:
        return QSize(self._tw, self._th)

    def _get_progress(self) -> float:
        return self._progress

    def _set_progress(self, value: float) -> None:
        self._progress = float(value)
        self.update()

    knobProgress = Property(float, _get_progress, _set_progress)

    def _target(self) -> float:
        return 1.0 if self.isChecked() else 0.0

    def _glide(self, *_args) -> None:
        self._anim.stop()
        self._anim.setStartValue(self._progress)
        self._anim.setEndValue(self._target())
        self._anim.start()

    def _snap(self) -> None:
        self._anim.stop()
        self._set_progress(self._target())

    def setChecked(self, checked: bool) -> None:
        # Python-level (programmatic) sets land here; user clicks toggle in C++ and
        # animate via the toggled -> _glide connection. When our sync code sets state
        # with signals blocked, snap instead of animating.
        super().setChecked(checked)
        if self.signalsBlocked():
            self._snap()

    @staticmethod
    def _mix(a: QColor, b: QColor, t: float) -> QColor:
        return QColor(
            round(a.red() + (b.red() - a.red()) * t),
            round(a.green() + (b.green() - a.green()) * t),
            round(a.blue() + (b.blue() - a.blue()) * t),
        )

    def paintEvent(self, _event) -> None:
        t = max(0.0, min(1.0, self._progress))
        if not self.isEnabled():
            track, knob = QColor("#1F2A3A"), QColor("#5A6675")
        else:
            track = self._mix(QColor("#373D46"), QColor("#4C8DFF"), t)
            knob = self._mix(QColor("#8A929B"), QColor("#FFFFFF"), t)
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setPen(Qt.NoPen)
        p.setBrush(track)
        p.drawRoundedRect(self.rect(), self._th / 2, self._th / 2)
        inset = 2
        x = inset + t * (self._tw - self._knob - 2 * inset)
        y = (self._th - self._knob) // 2
        p.setBrush(knob)
        p.drawEllipse(QRectF(x, y, self._knob, self._knob))
        p.end()


class VerticalTab(QFrame):
    """A thin clickable edge tab with bottom-to-top text (e.g. 'RAW JSON')."""

    clicked = Signal()

    def __init__(self, text: str, parent=None) -> None:
        super().__init__(parent)
        self._text = text
        self._on = False
        self.setObjectName("JsonTab")
        self.setFixedWidth(26)
        self.setCursor(Qt.PointingHandCursor)

    def set_on(self, on: bool) -> None:
        self._on = on
        self.update()

    def mousePressEvent(self, event) -> None:
        self.clicked.emit()
        event.accept()

    def sizeHint(self) -> QSize:
        fm = self.fontMetrics()
        return QSize(26, fm.horizontalAdvance(self._text) + 28)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)  # QSS background / border
        win = self.window()
        theme = getattr(win, "theme", None)
        color = (theme.accent if (self._on and theme) else
                 (theme.text_secondary if theme else "#A6ADB6"))
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.translate(0, self.height())
        p.rotate(-90)
        p.setPen(QColor(color))
        p.drawText(QRect(0, 0, self.height(), self.width()), int(Qt.AlignCenter), self._text)
        p.end()


class FlowLayout(QLayout):
    """A layout that wraps its widgets onto new rows as width runs out (for pills)."""

    def __init__(self, parent=None, margin: int = 0, spacing: int = 6) -> None:
        super().__init__(parent)
        if parent is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items: list = []

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index: int):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._do_layout(QRect(0, 0, width, 0), True)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(margins.left() + margins.right(), margins.top() + margins.bottom())
        return size

    def _do_layout(self, rect, test_only: bool) -> int:
        x = rect.x()
        y = rect.y()
        line_height = 0
        spacing = self.spacing()
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + spacing
            if next_x - spacing > rect.right() and line_height > 0:
                x = rect.x()
                y = y + line_height + spacing
                next_x = x + hint.width() + spacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y()


class FlowWidget(QWidget):
    """Host widget for a FlowLayout that reports its wrapped height to the parent.

    A plain QWidget advertises only a single row's height to its parent layout, so
    when the flow wraps onto extra rows those rows render outside the widget's
    rectangle and overlap whatever sits below it (e.g. the read-only tag note).
    Enabling height-for-width on the size policy makes the parent layout reserve
    room for every wrapped row.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        sp = self.sizePolicy()
        sp.setHeightForWidth(True)
        sp.setVerticalPolicy(QSizePolicy.Minimum)
        self.setSizePolicy(sp)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        lay = self.layout()
        if lay is not None and lay.hasHeightForWidth():
            return lay.heightForWidth(width)
        return super().heightForWidth(width)


TRIGGER_PROP = QTextFormat.UserProperty + 7


def make_trigger_format(color) -> QTextCharFormat:
    fmt = QTextCharFormat()
    fmt.setForeground(QColor(color))
    fmt.setFontWeight(QFont.DemiBold)
    bg = QColor(color)
    bg.setAlpha(52)
    fmt.setBackground(bg)
    fmt.setProperty(TRIGGER_PROP, True)
    return fmt


def _attach_word_end(text: str, pos: int) -> int:
    """If the insertion point sits inside/at the right edge of a word, advance to the
    end of that word so an inserted or dropped trigger lands after the word rather
    than splitting it. A point already at a boundary (space, or start of a word) is
    left as-is."""
    if pos > 0 and pos - 1 < len(text) and (text[pos - 1].isalnum() or text[pos - 1] == "_"):
        n = len(text)
        while pos < n and (text[pos].isalnum() or text[pos] == "_"):
            pos += 1
    return pos


class DraggableTagButton(QPushButton):
    """A palette tag: click to insert at the cursor, or drag it into the editor."""

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self._press = None
        self._full_text = text  # display text may be elided; drag/insert use the full tag

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._press = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._press is not None and (event.buttons() & Qt.LeftButton):
            if (event.position().toPoint() - self._press).manhattanLength() >= QApplication.startDragDistance():
                self._start_drag()
                return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._press = None
        super().mouseReleaseEvent(event)

    def _start_drag(self) -> None:
        self._press = None
        self.setDown(False)  # don't leave the button stuck pressed
        mime = QMimeData()
        mime.setData(TriggerTextEdit.TRIGGER_MIME, self._full_text.encode("utf-8"))
        mime.setText(self._full_text)
        drag = QDrag(self)
        drag.setMimeData(mime)
        pm = self.grab()
        drag.setPixmap(pm)
        drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
        drag.exec(Qt.CopyAction)


class TriggerTextEdit(QPlainTextEdit):
    """Per-file guidance editor where inserted triggers act as draggable chips."""

    TRIGGER_MIME = "application/x-guidance-trigger"

    def __init__(self, text: str = "", parent=None) -> None:
        super().__init__(text, parent)
        self._trigger_color = "#2f6fed"
        self._normal_color = "#e6e6e6"
        self._press_pos = None
        self._drag_run = None
        self._drag_source = None
        self._drop_pos = None
        self._known_triggers: set[str] = set()
        self.setAcceptDrops(True)
        self.viewport().setMouseTracking(True)
        # Small × that appears on the hovered trigger to remove it from the text.
        self._hover_run = None
        self._del_btn = QToolButton(self.viewport())
        self._del_btn.setObjectName("TriggerDel")
        self._del_btn.setText("×")
        self._del_btn.setCursor(Qt.PointingHandCursor)
        self._del_btn.setFixedSize(15, 15)
        self._del_btn.setToolTip("Remove this tag")
        self._del_btn.hide()
        self._del_btn.clicked.connect(self._remove_hovered_trigger)

    def set_known_triggers(self, triggers) -> None:
        self._known_triggers = set(t for t in triggers if t)
        self.rescan()

    @staticmethod
    def _is_word_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    def rescan(self) -> None:
        """Re-apply chip formatting to every known trigger occurrence (whole word)."""
        if getattr(self, "_suppress", False) or getattr(self, "_pending", False):
            return
        text = self.toPlainText()
        self._suppress = True
        doc = self.document()
        block_cur = QTextCursor(doc)
        block_cur.beginEditBlock()
        try:
            whole = QTextCursor(doc)
            whole.select(QTextCursor.Document)
            normal = QTextCharFormat()
            normal.setForeground(QColor(self._normal_color))
            whole.setCharFormat(normal)
            for trig in sorted(self._known_triggers, key=len, reverse=True):
                length = len(trig)
                start = 0
                while True:
                    idx = text.find(trig, start)
                    if idx < 0:
                        break
                    before = text[idx - 1] if idx > 0 else " "
                    after = text[idx + length] if idx + length < len(text) else " "
                    if not self._is_word_char(before) and not self._is_word_char(after):
                        tc = QTextCursor(doc)
                        tc.setPosition(idx)
                        tc.setPosition(idx + length, QTextCursor.KeepAnchor)
                        tc.mergeCharFormat(make_trigger_format(self._trigger_color))
                    start = idx + length
        finally:
            block_cur.endEditBlock()
            self._suppress = False
        self.viewport().update()

    def _char_has_trigger(self, i: int) -> bool:
        if i < 0 or i >= self.document().characterCount() - 1:
            return False
        cur = QTextCursor(self.document())
        cur.setPosition(i)
        cur.setPosition(i + 1, QTextCursor.KeepAnchor)
        sel = cur.selectedText()
        if not sel or sel == "\u2029":
            return False
        return cur.charFormat().hasProperty(TRIGGER_PROP)

    def _trigger_run_at(self, pos: int):
        target = None
        if self._char_has_trigger(pos):
            target = pos
        elif pos > 0 and self._char_has_trigger(pos - 1):
            target = pos - 1
        if target is None:
            return None
        start = target
        while start > 0 and self._char_has_trigger(start - 1):
            start -= 1
        end = target + 1
        total = self.document().characterCount()
        while end < total and self._char_has_trigger(end):
            end += 1
        cur = QTextCursor(self.document())
        cur.setPosition(start)
        cur.setPosition(end, QTextCursor.KeepAnchor)
        return (start, end, cur.selectedText())

    def _chip_pixmap(self, text: str) -> QPixmap:
        fm = self.fontMetrics()
        w = fm.horizontalAdvance(text) + 18
        h = fm.height() + 8
        pm = QPixmap(w, h)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing, True)
        bg = QColor(self._trigger_color)
        bg.setAlpha(70)
        p.setBrush(bg)
        p.setPen(QColor(self._trigger_color))
        p.drawRoundedRect(0, 0, w - 1, h - 1, h / 2, h / 2)
        p.setPen(QColor(self._trigger_color))
        p.drawText(pm.rect(), Qt.AlignCenter, text)
        p.end()
        return pm

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_run = self._trigger_run_at(self.cursorForPosition(event.pos()).position())
            self._press_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if (event.buttons() & Qt.LeftButton) and self._drag_run is not None and self._press_pos is not None:
            if (event.pos() - self._press_pos).manhattanLength() >= QApplication.startDragDistance():
                self._begin_drag()
                return
        elif not (event.buttons() & Qt.LeftButton):
            run = self._trigger_run_at(self.cursorForPosition(event.pos()).position())
            self.viewport().setCursor(Qt.OpenHandCursor if run is not None else Qt.IBeamCursor)
            self._update_del_btn(run, event.pos())
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:
        # Hide the × unless the pointer moved onto the button itself.
        if not self._del_btn.underMouse():
            self._hide_del_btn()
        super().leaveEvent(event)

    def _update_del_btn(self, run, pos=None) -> None:
        if run is None:
            # Don't drop the button while the pointer is bridging the gap toward it.
            if pos is not None and self._del_btn.isVisible():
                if self._del_btn.geometry().adjusted(-10, -10, 10, 10).contains(pos):
                    return
            self._hide_del_btn()
            return
        start, end, _text = run
        self._hover_run = run
        doc = self.document()
        last = doc.characterCount() - 1
        cs = QTextCursor(doc); cs.setPosition(min(start, last))
        ce = QTextCursor(doc); ce.setPosition(min(end, last))
        r_start = self.cursorRect(cs)
        r_end = self.cursorRect(ce)
        bw, bh = self._del_btn.width(), self._del_btn.height()
        # Sit on the chip's top-right corner, overlapping it slightly so there's no
        # dead gap to cross on the way to the button.
        x = r_end.left() - bw + 4
        y = r_start.top() - 4
        x = max(0, min(x, self.viewport().width() - bw - 1))
        y = max(0, y)
        self._del_btn.move(x, y)
        self._del_btn.raise_()
        self._del_btn.show()

    def _hide_del_btn(self) -> None:
        self._hover_run = None
        self._del_btn.hide()

    def _remove_hovered_trigger(self) -> None:
        if self._hover_run is None:
            return
        start, end, _text = self._hover_run
        doc = self.document()
        total = doc.characterCount()
        s, e = start, end
        # Swallow one adjacent space (prefer the trailing one) to avoid leftover gaps.
        probe = QTextCursor(doc)
        if e < total - 1:
            probe.setPosition(e)
            probe.setPosition(e + 1, QTextCursor.KeepAnchor)
            if probe.selectedText() == " ":
                e += 1
        if e == end and s > 0:
            probe.setPosition(s - 1)
            probe.setPosition(s, QTextCursor.KeepAnchor)
            if probe.selectedText() == " ":
                s -= 1
        cur = QTextCursor(doc)
        cur.setPosition(s)
        cur.setPosition(e, QTextCursor.KeepAnchor)
        cur.removeSelectedText()
        self._hide_del_btn()
        self.rescan()

    def mouseReleaseEvent(self, event) -> None:
        self._drag_run = None
        self._press_pos = None
        super().mouseReleaseEvent(event)

    def _begin_drag(self) -> None:
        if self._drag_run is None:
            return
        start, end, text = self._drag_run
        self._drag_source = (start, end)
        mime = QMimeData()
        mime.setData(self.TRIGGER_MIME, text.encode("utf-8"))
        mime.setText(text)
        drag = QDrag(self)
        drag.setMimeData(mime)
        pm = self._chip_pixmap(text)
        drag.setPixmap(pm)
        drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
        drag.exec(Qt.MoveAction)
        self._drag_run = None
        self._press_pos = None
        self._drag_source = None
        self._drop_pos = None
        self.viewport().update()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasFormat(self.TRIGGER_MIME):
            self._drop_pos = self.cursorForPosition(event.position().toPoint()).position()
            self.viewport().update()
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if event.mimeData().hasFormat(self.TRIGGER_MIME):
            self._drop_pos = self.cursorForPosition(event.position().toPoint()).position()
            self.viewport().update()
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dragLeaveEvent(self, event) -> None:
        self._drop_pos = None
        self.viewport().update()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        md = event.mimeData()
        if not md.hasFormat(self.TRIGGER_MIME):
            super().dropEvent(event)
            return
        text = bytes(md.data(self.TRIGGER_MIME)).decode("utf-8")
        drop_pos = self.cursorForPosition(event.position().toPoint()).position()
        # Internal drag reorders (remove + reinsert); an external palette chip just inserts.
        source = self._drag_source if event.source() is self else None
        self._move_trigger(source, drop_pos, text)
        self._drop_pos = None
        self.viewport().update()
        event.acceptProposedAction()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._drop_pos is None:
            return
        cur = QTextCursor(self.document())
        cur.setPosition(min(self._drop_pos, self.document().characterCount() - 1))
        rect = self.cursorRect(cur)
        painter = QPainter(self.viewport())
        pen = QPen(QColor(self._trigger_color))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.drawLine(rect.left(), rect.top(), rect.left(), rect.bottom())
        painter.end()

    def _move_trigger(self, source, drop_pos: int, text: str) -> None:
        self._suppress = True
        cur = QTextCursor(self.document())
        cur.beginEditBlock()
        try:
            if source is not None:
                s, e = source
                if e < self.document().characterCount() - 1:
                    probe = QTextCursor(self.document())
                    probe.setPosition(e)
                    probe.setPosition(e + 1, QTextCursor.KeepAnchor)
                    if probe.selectedText() == " ":
                        e += 1
                cur.setPosition(s)
                cur.setPosition(e, QTextCursor.KeepAnchor)
                cur.removeSelectedText()
                removed = e - s
                if drop_pos >= e:
                    drop_pos -= removed
                elif drop_pos > s:
                    drop_pos = s
            # Snap to the end of the word under the drop point so a mid-word drop
            # doesn't split the word, then space-separate the trigger.
            full = self.toPlainText()
            drop_pos = _attach_word_end(full, drop_pos)
            cur.setPosition(max(0, min(drop_pos, self.document().characterCount() - 1)))
            normal = QTextCharFormat()
            normal.setForeground(QColor(self._normal_color))
            full = self.toPlainText()
            ip = cur.position()
            if ip > 0 and ip - 1 < len(full) and not full[ip - 1].isspace():
                cur.insertText(" ", normal)
            cur.insertText(text, make_trigger_format(self._trigger_color))
            after = self.toPlainText()
            np = cur.position()
            if np >= len(after) or not after[np].isspace():
                cur.insertText(" ", normal)
        finally:
            cur.endEditBlock()
            self._suppress = False


class ElementRow(QWidget):
    """A rich element-list row: reorder, color dot, type pill, label, delete.

    Plain labels don't consume mouse presses, so clicking the dot/pill/label
    bubbles up here and selects the row; the buttons consume their own clicks.
    """

    clicked = Signal(int)

    def __init__(self, index: int, parent=None) -> None:
        super().__init__(parent)
        self._index = index
        self.setObjectName("ElementRow")
        self.setAttribute(Qt.WA_StyledBackground, True)

    def mousePressEvent(self, event) -> None:
        self.clicked.emit(self._index)
        super().mousePressEvent(event)


class GuidanceDialog(QDialog):
    """Dialog whose close (X / Esc / Close button) is routed through a gate that
    can apply, discard, or veto (keep open)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._gate = None

    def set_close_gate(self, gate) -> None:
        self._gate = gate

    def closeEvent(self, event) -> None:
        if self._gate is None or self._gate():
            event.accept()
        else:
            event.ignore()

    def reject(self) -> None:
        if self._gate is None or self._gate():
            super().reject()


class SourcePopout(QDialog):
    """Modeless source-caption inspector. Plain Left/Right arrows navigate to the
    previous/next image (mirroring the main window). An event filter catches the
    arrows even when the read-only text field has focus, while modified arrows
    (Shift/Ctrl) still pass through so text selection and copy keep working."""

    def __init__(self, parent, on_prev, on_next) -> None:
        super().__init__(parent)
        self._on_prev = on_prev
        self._on_next = on_next

    def _nav_key(self, event) -> bool:
        if event.modifiers() == Qt.NoModifier:
            if event.key() in (Qt.Key_Left, Qt.Key_A):
                self._on_prev()
                return True
            if event.key() in (Qt.Key_Right, Qt.Key_D):
                self._on_next()
                return True
        return False

    def keyPressEvent(self, event) -> None:
        if self._nav_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.KeyPress and self._nav_key(event):
            return True
        return super().eventFilter(obj, event)


class MainWindow(QMainWindow):
    """The application window.

    Note on keys: arrows and A/D mean "previous/next box" for Ideogram bounding
    boxes and "previous/next frame" on a clip. Which one you get follows whatever
    is on screen, so the same keys do the obvious thing without a mode to learn.
    """

    _SC_THUMB_H = 300  # fixed height of the source-popout thumbnail box (stops layout jitter)

    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.theme = Theme(self.settings)
        self.qsettings = QSettings("FantasticCaptioningKit", "QtApp")
        self._default_tags = self._load_default_tags()
        self.store: CaptionStore | None = None
        self.images: list[Path] = []
        self._has_source_txt = False
        self.current: Path | None = None
        self.current_caption: dict = default_caption()
        # Plain presets keep their caption as a string; structured ones use the dict
        # above. self.preset decides which is authoritative.
        self.current_text: str = ""
        self._video_info: dict[str, VideoInfo] = {}
        # Model specs live in a user-editable JSON beside the app; built-ins are the
        # fallback so a bad or missing file can't stop the app starting.
        self.model_targets = load_targets(app_base_dir())
        self.preset: CaptionPreset = get_preset(None)
        self.project: ProjectConfig = ProjectConfig()
        self.selected_element_index: int | None = None
        self.box_items: list = []
        self._next_color_id = 0
        self._loading = False
        self._dirty = False
        self._guidance_dirty = False
        self._pending: dict[str, dict] = {}
        self._thumb_items: dict[str, QListWidgetItem] = {}
        self._dirty_dot: dict[str, float] = {}        # path -> unsaved-dot progress 0..1
        self._dirty_dot_anims: dict[str, QVariantAnimation] = {}
        self._thumb_base: dict[str, QPixmap] = {}
        self._autosave = False
        self._syncing = False
        # Debounced live refresh of the raw-JSON panel while typing.
        self._json_live_timer = QTimer(self)
        self._json_live_timer.setSingleShot(True)
        self._json_live_timer.setInterval(150)
        self._json_live_timer.timeout.connect(self._live_json_refresh)
        self._user_zoomed = False
        self._ai_thread: AiJobThread | None = None
        self._job_running = False
        self._job_cancelled = False
        self._read_only = False
        self._server_proc = None   # llama-server process we launched (local mode)
        self._server_popover = None
        self._server_reachable = None
        self._server_modelless = False

        self.setWindowTitle(APP_TITLE)
        # Restore the last window size/position (and maximized/screen state);
        # fall back to a sensible default on first run or if the saved blob is bad.
        geo = self.qsettings.value("window_geometry")
        if not (geo is not None and self.restoreGeometry(geo)):
            self.resize(1600, 1000)
        self.apply_appearance(self.settings)

        self._build_toolbar()
        self.setAcceptDrops(True)   # drag media anywhere onto the window
        self._build_body()
        self._restore_autosave_pref()
        self._load_guidance_presets()
        self._folder_tags: list[str] = []
        play_sc = QShortcut(QKeySequence(Qt.Key_Space), self)
        play_sc.setContext(Qt.WidgetWithChildrenShortcut)
        play_sc.activated.connect(self.toggle_video_playback)
        self._build_server_status()
        self._start_server_monitor()
        self._maybe_check_llama_update()
        # The panel is built in its structured form; sync it to the active preset so a
        # fresh launch (plain text by default) doesn't show the JSON-only controls.
        self._apply_preset_ui()
        self._set_status("Open a folder to begin.")

    # ---- layout ----------------------------------------------------------
    def _build_toolbar(self) -> None:
        """Builds the chrome as a left icon rail + a slim top bar (no top toolbar).

        Actions are registered on the window so their shortcuts work regardless of
        where the button lives.
        """
        ic = self.theme.text_secondary

        open_action = QAction(lucide_icon("folder-open", ic), "Open folder", self)
        open_action.setShortcut("Ctrl+O")
        open_action.setToolTip("Open folder (Ctrl+O)")
        open_action.triggered.connect(self.open_folder)

        guidance_action = QAction(lucide_icon("pencil", ic), "Guidance Settings", self)
        guidance_action.setToolTip("Open the full guidance editor")
        guidance_action.triggered.connect(self._open_guidance_expand)

        self.panels_action = QAction(lucide_icon("panel-left-close", ic), "Collapse guidance panel", self)
        self.panels_action.setShortcut("Ctrl+\\")
        self.panels_action.setToolTip("Collapse guidance panel (Ctrl+\\)")
        self.panels_action.triggered.connect(self.toggle_left_panel)

        fit_action = QAction(lucide_icon("maximize", ic), "Fit", self)
        fit_action.setShortcut("Ctrl+0")
        fit_action.setToolTip("Fit image to view (Ctrl+0)")
        fit_action.triggered.connect(self.fit_view)

        save_action = QAction(lucide_icon("save", ic), "Save", self)
        save_action.setShortcut("Ctrl+S")
        save_action.setToolTip("Save current caption (Ctrl+S)")
        save_action.triggered.connect(self.save_current)

        save_all_action = QAction(lucide_icon("save-all", ic), "Save all", self)
        save_all_action.setShortcut("Ctrl+Shift+S")
        save_all_action.setToolTip("Save all captions (Ctrl+Shift+S)")
        save_all_action.triggered.connect(self.save_all)

        # Raw JSON now lives as a button under Background in the Caption tab; this
        # slot became the system-prompt editor, which applies to every preset.
        self.sysprompt_action = QAction(lucide_icon("braces", ic), "System prompt", self)
        self.sysprompt_action.setToolTip("Edit the LLM system prompt for this preset")
        self.sysprompt_action.triggered.connect(self.open_system_prompt)

        self.json_action = QAction("LLM instructions", self)
        self.json_action.setCheckable(True)
        self.json_action.setShortcut("Ctrl+J")
        self.json_action.toggled.connect(self.toggle_json_panel)

        # Image navigation — bracket keys avoid colliding with text-field editing.
        prev_action = QAction("Previous image", self)
        prev_action.setShortcut("Ctrl+[")
        prev_action.triggered.connect(self.prev_image)
        next_action = QAction("Next image", self)
        next_action.setShortcut("Ctrl+]")
        next_action.triggered.connect(self.next_image)

        prefs_action = QAction(lucide_icon("settings", ic), "Preferences…", self)
        prefs_action.setShortcut("Ctrl+,")
        prefs_action.triggered.connect(self.open_preferences)

        # Manual review flag — a metadata mark, so it stays usable in batch read-only mode.
        self.flag_action = QAction(lucide_icon("flag", ic), "Flag for review", self)
        self.flag_action.setCheckable(True)
        self.flag_action.setShortcut("F")
        self.flag_action.setToolTip("Flag this file for manual review (F)")
        self.flag_action.triggered.connect(self._toggle_review_flag)

        add_media_action = QAction(lucide_icon("image-plus", ic), "Add media\u2026", self)
        add_media_action.setToolTip(
            "Copy images or videos into this dataset folder (or just drag them in)")
        add_media_action.setShortcut("Ctrl+Shift+O")
        add_media_action.triggered.connect(self.open_add_media)

        dupe_action = QAction(lucide_icon("copy", ic), "Duplicate / back up\u2026", self)
        dupe_action.setToolTip(
            "Copy this dataset to another folder, choosing what comes along")
        dupe_action.triggered.connect(self.open_duplicate_dataset)

        self.remove_action = QAction(lucide_icon("trash-2", ic), "Remove\u2026", self)
        self.remove_action.setToolTip("Bypass this file, or delete it permanently")
        self.remove_action.triggered.connect(lambda: self.remove_media())

        self.bypass_action = QAction(lucide_icon("eye-off", ic), "Bypass", self)
        self.bypass_action.setToolTip(
            "Move this file out of the dataset into .bypass/ (still captionable)")
        self.bypass_action.setShortcut("Ctrl+B")
        self.bypass_action.triggered.connect(lambda: self.toggle_bypass())
        self.crop_action = QAction(lucide_icon("crop", ic), "Crop / Resize", self)
        self.crop_action.setToolTip(
            "Crop, rotate or resize this image (original kept in .original/)")
        self.crop_action.triggered.connect(self.open_crop_dialog)
        self.crop_action.setEnabled(False)  # until an image is shown

        self.batch_resize_action = QAction(lucide_icon("scaling", ic), "Batch resize…", self)
        self.batch_resize_action.setToolTip(
            "Resize every image in the folder (originals kept in .original/)")
        self.batch_resize_action.triggered.connect(self.open_batch_resize)

        next_flag_action = QAction("Next flagged image", self)
        next_flag_action.setShortcut("Shift+F")
        next_flag_action.setToolTip("Jump to the next image flagged for review (Shift+F)")
        next_flag_action.triggered.connect(self._next_flagged_image)

        about_action = QAction(lucide_icon("info", ic), "About", self)
        about_action.triggered.connect(self.show_about)

        for act in (open_action, add_media_action, guidance_action,
                    self.panels_action, fit_action,
                    save_action, save_all_action, self.json_action,
                    prev_action, next_action, self.flag_action, self.bypass_action,
                    self.remove_action, self.crop_action,
                    self.batch_resize_action, self.sysprompt_action, next_flag_action,
                    prefs_action, about_action):
            self.addAction(act)

        # ---- left icon rail ----
        rail = QFrame()
        rail.setObjectName("Rail")
        rail.setFixedWidth(50)
        rlay = QVBoxLayout(rail)
        rlay.setContentsMargins(7, 10, 7, 10)
        rlay.setSpacing(6)
        for act in (open_action, add_media_action, guidance_action, fit_action,
                    self.panels_action,
                    self.sysprompt_action, self.flag_action, self.bypass_action,
                    self.crop_action, self.remove_action, dupe_action,
                    self.batch_resize_action):
            rlay.addWidget(self._rail_button(act))
        rlay.addStretch(1)
        rlay.addWidget(self._rail_button(prefs_action))
        self.rail = rail

        # ---- slim top bar ----
        top = QFrame()
        top.setObjectName("TopBar")
        top.setFixedHeight(46)
        tlay = QHBoxLayout(top)
        tlay.setContentsMargins(14, 6, 12, 6)
        tlay.setSpacing(8)
        self.title_label = QLabel("")
        self.title_label.setObjectName("TitleLabel")
        self.title_label.setAlignment(Qt.AlignCenter)
        tlay.addStretch(1)
        tlay.addWidget(self.title_label)
        tlay.addStretch(1)
        save_btn = QPushButton("Save")
        save_btn.setToolTip("Save current caption (Ctrl+S)")
        save_btn.clicked.connect(self.save_current)
        save_all_btn = QPushButton("Save all")
        save_all_btn.setToolTip("Save all captions (Ctrl+Shift+S)")
        save_all_btn.clicked.connect(self.save_all)
        overflow = QToolButton()
        overflow.setObjectName("RailButton")
        overflow.setIcon(lucide_icon("ellipsis", ic))
        overflow.setToolTip("More")
        overflow.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(overflow)
        menu.addAction(about_action)
        overflow.setMenu(menu)
        tlay.addWidget(save_btn)
        tlay.addWidget(save_all_btn)
        tlay.addWidget(overflow)
        self.top_bar = top

    def _rail_button(self, action: QAction) -> QToolButton:
        btn = QToolButton()
        btn.setObjectName("RailButton")
        btn.setDefaultAction(action)  # icon, tooltip, checked-state, trigger all follow the action
        btn.setIconSize(QSize(20, 20))
        btn.setFixedSize(36, 36)
        return btn

    def next_image(self) -> None:
        if getattr(self, "_nav_locked", False):
            return
        count = self.filmstrip.count()
        if count == 0:
            return
        self.filmstrip.setCurrentRow(min(self.filmstrip.currentRow() + 1, count - 1))

    def prev_image(self) -> None:
        if getattr(self, "_nav_locked", False):
            return
        if self.filmstrip.count() == 0:
            return
        self.filmstrip.setCurrentRow(max(self.filmstrip.currentRow() - 1, 0))

    def fit_view(self) -> None:
        if self.pixmap_item is not None:
            self._user_zoomed = False
            self.view.resetTransform()
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self._update_zoom_label()

    def show_about(self) -> None:
        QMessageBox.about(
            self,
            f"About {APP_TITLE}",
            f"{APP_TITLE}\n\n"
            "A local tool for preparing image and video caption datasets: crop, trim "
            "images, then edit and generate captions with a local vision model. "
            "Ships with structured Ideogram 4 JSON support.\n\n"
            "Built with PySide6 (Qt for Python), used under the LGPL v3.",
        )

    def _panel(self, title: str) -> QWidget:
        w = QWidget()
        w.setObjectName("Panel")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        label = QLabel(title)
        label.setObjectName("SectionLabel")
        lay.addWidget(label)
        return w

    def _field_label(self, text: str) -> QLabel:
        lab = QLabel(text)
        lab.setObjectName("FieldHead")
        return lab

    def _expand_icon(self) -> QIcon:
        cached = getattr(self, "_expand_icon_cache", None)
        if cached is not None:
            return cached
        self._expand_icon_cache = lucide_icon("maximize-2", self.theme.text_secondary, 14)
        return self._expand_icon_cache

    def _attach_expand(self, field, title: str, single_line: bool = False, with_tags: bool = False) -> QWidget:
        """Wrap a text field with a small expand button that opens a big editor.

        The field reference is unchanged, so all existing commit/sync wiring works.
        When with_tags is set, the pop-out uses the trigger editor + tag palette.
        """
        cont = QWidget()
        h = QHBoxLayout(cont)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(4)
        h.addWidget(field, 1)
        btn = QToolButton()
        btn.setObjectName("ExpandBtn")
        btn.setIcon(self._expand_icon())
        btn.setIconSize(QSize(14, 14))
        btn.setFixedSize(22, 22)
        btn.setToolTip("Expand editor")
        btn.clicked.connect(
            lambda _c, f=field, t=title, sl=single_line, wt=with_tags: self._open_text_expand(f, t, sl, wt)
        )
        align = Qt.AlignTop if isinstance(field, QPlainTextEdit) else Qt.AlignVCenter
        h.addWidget(btn, 0, align)
        return cont

    def _open_text_expand(self, field, title: str, single_line: bool, with_tags: bool = False) -> None:
        if not field.isEnabled():
            return
        read_only = bool(getattr(field, "isReadOnly", lambda: False)())
        current = field.toPlainText() if isinstance(field, QPlainTextEdit) else field.text()
        use_tags = with_tags and self.store is not None and not read_only
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.resize(900, 680 if use_tags else 600)
        v = QVBoxLayout(dlg)
        if use_tags:
            editor = TriggerTextEdit(current)
            editor._trigger_color = self.theme.accent
            editor._normal_color = self.theme.text_primary
            editor._pending = False
            editor._suppress = False
            editor.textChanged.connect(lambda e=editor: self._on_editor_text_changed(e))
            v.addWidget(editor, 1)
            self._build_tag_palette(v, editor)
        else:
            editor = QPlainTextEdit()
            editor.setPlainText(current)
            editor.setReadOnly(read_only)
            v.addWidget(editor, 1)
        if read_only:
            buttons = QDialogButtonBox(QDialogButtonBox.Close)
            buttons.rejected.connect(dlg.reject)
        else:
            buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
            buttons.accepted.connect(dlg.accept)
            buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        editor.setFocus()
        if dlg.exec():  # Save only; read-only Close rejects, so no write-back
            text = editor.toPlainText()
            if single_line:
                text = text.replace("\r", " ").replace("\n", " ")
            if isinstance(field, QPlainTextEdit):
                field.setPlainText(text)
            else:
                field.setText(text)

    # ---- folder-wide tag palette (persists per dataset) -----------------
    def _load_default_tags(self) -> list[str]:
        """Global default tags, shown on every folder. Stored app-wide in QSettings;
        falls back to the built-in seed list when unset."""
        raw = self.qsettings.value("default_tags", None)
        tags: list[str] = []
        if isinstance(raw, str) and raw:
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    tags = [t for t in data if isinstance(t, str) and t.strip()]
            except json.JSONDecodeError:
                tags = []
        elif isinstance(raw, (list, tuple)):
            tags = [str(t) for t in raw if str(t).strip()]
        if not tags and raw is None:
            tags = list(GENERAL_TAGS)  # first run: seed with the built-in defaults
        # de-dupe, preserve order
        seen, out = set(), []
        for t in tags:
            if t not in seen:
                seen.add(t); out.append(t)
        return out

    def _save_default_tags(self, tags: list[str]) -> None:
        self.qsettings.setValue("default_tags", json.dumps(list(tags), ensure_ascii=False))

    def _folder_tags_path(self) -> Path:
        return self.store.project_path().parent / FOLDER_TAGS_FILENAME

    def _load_folder_tags(self) -> None:
        self._folder_tags = []
        if self.store is None:
            return
        try:
            path = self._folder_tags_path()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._folder_tags = [
                        t for t in data.get("tags", []) if isinstance(t, str) and t.strip()
                    ]
        except (OSError, json.JSONDecodeError):
            self._folder_tags = []

    def _save_folder_tags(self) -> None:
        if self.store is None:
            return
        try:
            path = self._folder_tags_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"tags": self._folder_tags}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            self._set_status(f"Could not save tags: {exc}")

    def _add_folder_tag(self, text: str, rebuild) -> None:
        text = text.strip()
        if not text or text in self._folder_tags or text in self._default_tags:
            return
        self._folder_tags.append(text)
        self._save_folder_tags()
        rebuild()

    def _remove_folder_tag(self, text: str, rebuild) -> None:
        if text in self._folder_tags:
            self._folder_tags.remove(text)
            self._save_folder_tags()
            rebuild()

    def _insert_tag(self, editor: QPlainTextEdit, trigger: str) -> None:
        self._commit_pending(editor)  # finalise any red preset first
        cursor = editor.textCursor()
        normal = QTextCharFormat()
        normal.setForeground(QColor(self.theme.text_primary))
        text = editor.toPlainText()
        # Land after the word the cursor is on (don't split it) and don't eat a selection.
        pos = cursor.selectionEnd() if cursor.hasSelection() else cursor.position()
        pos = _attach_word_end(text, pos)
        cursor.setPosition(pos)
        if pos > 0 and pos - 1 < len(text) and not text[pos - 1].isspace():
            cursor.insertText(" ", normal)
        cursor.insertText(trigger, make_trigger_format(self.theme.accent))
        after = editor.toPlainText()
        npos = cursor.position()
        if npos >= len(after) or not after[npos].isspace():
            cursor.insertText(" ", normal)
        editor.setCurrentCharFormat(normal)
        editor.setTextCursor(cursor)
        editor.setFocus()

    def _clear_layout(self, layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()

    def _make_tag_pill(self, text: str, removable: bool, image_ed: QPlainTextEdit, rebuild) -> QFrame:
        pill = QFrame()
        pill.setObjectName("CustomPill" if removable else "GrayPill")
        h = QHBoxLayout(pill)
        h.setContentsMargins(2, 0, 2, 0)
        h.setSpacing(0)
        btn = DraggableTagButton(text)
        btn.setObjectName("PillText")
        btn.setCursor(Qt.PointingHandCursor)
        # Cap the display width so a long re-used phrase becomes an elided chip with the
        # full text on hover, instead of stretching the pill (and the whole window) wide.
        # The full text is retained on the button for click/drag insertion.
        _MAX = 360
        _disp = btn.fontMetrics().elidedText(text, Qt.ElideRight, _MAX)
        btn.setText(_disp)
        btn.setMaximumWidth(_MAX + 16)
        if _disp != text:
            btn.setToolTip(text)  # full phrase on hover
        else:
            btn.setToolTip("Click to insert at the cursor, or drag into the text")
        btn.clicked.connect(lambda _c=False, t=text: self._insert_tag(image_ed, t))
        h.addWidget(btn)
        if removable:
            x = QToolButton()
            x.setText("×")
            x.setObjectName("PillX")
            x.setCursor(Qt.PointingHandCursor)
            x.setToolTip("Remove this tag")
            x.clicked.connect(lambda _c=False, t=text: self._remove_folder_tag(t, rebuild))
            h.addWidget(x)
        return pill

    def _build_tag_palette(self, parent_layout, image_ed: QPlainTextEdit) -> None:
        label = QLabel("Tags — click or drag to insert a trigger")
        label.setObjectName("Hint")
        parent_layout.addWidget(label)

        custom_host = FlowWidget()
        custom_flow = FlowLayout(custom_host, 0, 6)
        parent_layout.addWidget(custom_host)

        gray_host = FlowWidget()
        gray_flow = FlowLayout(gray_host, 0, 6)
        for tag in self._default_tags:
            gray_flow.addWidget(self._make_tag_pill(tag, False, image_ed, None))
        parent_layout.addWidget(gray_host)

        add_row = QHBoxLayout()
        tag_input = QLineEdit()
        tag_input.setPlaceholderText("New tag…")
        add_btn = QPushButton("+ Add")
        add_row.addWidget(tag_input, 1)
        add_row.addWidget(add_btn)
        parent_layout.addLayout(add_row)

        def rebuild() -> None:
            self._clear_layout(custom_flow)
            for tag in self._folder_tags:
                custom_flow.addWidget(self._make_tag_pill(tag, True, image_ed, rebuild))
            custom_host.adjustSize()
            if isinstance(image_ed, TriggerTextEdit):
                image_ed.set_known_triggers(set(self._folder_tags) | set(self._default_tags))

        def do_add() -> None:
            self._add_folder_tag(tag_input.text(), rebuild)
            tag_input.clear()

        add_btn.clicked.connect(do_add)
        tag_input.returnPressed.connect(do_add)
        rebuild()

    # ---- guidance presets (managed inside the expand popup) -------------
    def _guidance_presets_path(self) -> Path:
        return default_profiles_path().parent / GUIDANCE_PRESETS_FILENAME

    def _load_guidance_presets(self) -> None:
        """User presets live in the file; built-ins come from code. Legacy files
        (without the v2 marker) are ignored so old placeholders are retired."""
        path = self._guidance_presets_path()
        user = {"folder": [], "image": []}
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict) and data.get("_format") == 2:
                    for scope in ("folder", "image"):
                        user[scope] = [p for p in data.get(scope, []) if isinstance(p, dict)]
        except (OSError, json.JSONDecodeError):
            pass
        self._user_presets = user
        self._save_guidance_presets()  # normalise to current v2 format

    def _save_guidance_presets(self) -> None:
        try:
            path = self._guidance_presets_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "_format": 2,
                "folder": self._user_presets.get("folder", []),
                "image": self._user_presets.get("image", []),
            }
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        except OSError as exc:
            self._set_status(f"Could not save presets: {exc}")

    def _builtin_presets(self, scope: str) -> list[dict]:
        src = FOLDER_GUIDANCE_PRESETS if scope == "folder" else IMAGE_GUIDANCE_PRESETS
        return [{"name": n, "text": t, "builtin": True} for n, t in src]

    def _all_presets(self, scope: str) -> list[dict]:
        user = [
            {"name": p.get("name", ""), "text": p.get("text", ""), "builtin": False}
            for p in self._user_presets.get(scope, [])
        ]
        return self._builtin_presets(scope) + user

    def _reload_preset_combo(self, combo: QComboBox, scope: str) -> None:
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Select preset…")
        for preset in self._all_presets(scope):
            label = preset["name"] + ("" if preset["builtin"] else "  (custom)")
            combo.addItem(label)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _on_preset_selected(self, scope: str, editor: QPlainTextEdit, combo: QComboBox, index: int) -> None:
        if index <= 0:
            return
        presets = self._all_presets(scope)
        if index - 1 >= len(presets):
            return
        text = presets[index - 1]["text"]
        # If the current insert is still uncommitted (red, unedited), swap it out
        # rather than appending. Once committed (✓ or edited), a new pick appends.
        if getattr(editor, "_pending", False):
            self._reject_pending(editor)
        self._insert_pending(editor, combo, text)

    def _insert_pending(self, editor: QPlainTextEdit, combo: QComboBox, text: str) -> None:
        editor._suppress = True
        try:
            prior = editor.toPlainText()
            editor._prior_text = prior
            cursor = editor.textCursor()
            cursor.movePosition(QTextCursor.End)
            start = cursor.position()
            if prior.strip():
                cursor.insertText("\n\n")
            red = QTextCharFormat()
            red.setForeground(QColor(UNSAVED_GLOW))
            cursor.insertText(text, red)
        finally:
            editor._suppress = False
        editor._pending_start = start
        editor._pending = True
        end_cursor = editor.textCursor()
        end_cursor.movePosition(QTextCursor.End)
        editor.setTextCursor(end_cursor)
        default = QTextCharFormat()
        default.setForeground(QColor(self.theme.text_primary))
        editor.setCurrentCharFormat(default)
        editor._accept_btn.setEnabled(True)
        editor._reject_btn.setEnabled(True)
        combo.blockSignals(True)
        combo.setCurrentIndex(0)
        combo.blockSignals(False)

    def _on_editor_text_changed(self, editor: QPlainTextEdit) -> None:
        # a user edit (not our own programmatic change) commits the pending insert
        if getattr(editor, "_suppress", False):
            return
        if getattr(editor, "_pending", False):
            self._commit_pending(editor)
        if isinstance(editor, TriggerTextEdit):
            editor.rescan()

    def _commit_pending(self, editor: QPlainTextEdit) -> None:
        if not getattr(editor, "_pending", False):
            return
        editor._suppress = True
        try:
            cursor = editor.textCursor()
            cursor.setPosition(editor._pending_start)
            cursor.movePosition(QTextCursor.End, QTextCursor.KeepAnchor)
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(self.theme.text_primary))
            cursor.mergeCharFormat(fmt)
        finally:
            editor._suppress = False
        default = QTextCharFormat()
        default.setForeground(QColor(self.theme.text_primary))
        editor.setCurrentCharFormat(default)
        editor._pending = False
        editor._accept_btn.setEnabled(False)
        editor._reject_btn.setEnabled(False)

    def _reject_pending(self, editor: QPlainTextEdit) -> None:
        if not getattr(editor, "_pending", False):
            return
        editor._suppress = True
        try:
            editor.setPlainText(getattr(editor, "_prior_text", ""))
        finally:
            editor._suppress = False
        editor._pending = False
        editor._accept_btn.setEnabled(False)
        editor._reject_btn.setEnabled(False)

    def _save_preset_as(self, scope: str, editor: QPlainTextEdit, combo: QComboBox) -> None:
        if not editor.toPlainText().strip():
            QMessageBox.information(self, "Nothing to save", "The field is empty.")
            return
        name, ok = QInputDialog.getText(self, "Save preset", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        if any(p["name"] == name for p in self._builtin_presets(scope)):
            QMessageBox.information(
                self, "Name in use",
                f"“{name}” is a built-in preset name. Please choose a different name.",
            )
            return
        user = self._user_presets.setdefault(scope, [])
        existing = next((p for p in user if p.get("name") == name), None)
        if existing is not None:
            confirm = QMessageBox.question(
                self, "Overwrite preset",
                f"A preset named “{name}” already exists. Overwrite it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if confirm != QMessageBox.Yes:
                return
            existing["text"] = editor.toPlainText()
        else:
            user.append({"name": name, "text": editor.toPlainText()})
        self._save_guidance_presets()
        self._reload_preset_combo(combo, scope)
        idx = combo.findText(f"{name}  (custom)")
        if idx >= 0:
            combo.blockSignals(True)
            combo.setCurrentIndex(idx)
            combo.blockSignals(False)

    def _delete_preset(self, scope: str, combo: QComboBox) -> None:
        index = combo.currentIndex()
        presets = self._all_presets(scope)
        if index <= 0 or index - 1 >= len(presets):
            QMessageBox.information(self, "No preset selected", "Pick a preset to delete first.")
            return
        target = presets[index - 1]
        if target["builtin"]:
            QMessageBox.information(
                self, "Built-in preset",
                "Built-in presets can't be deleted. You can edit the field and use "
                "“Save as…” to make your own.",
            )
            return
        confirm = QMessageBox.question(
            self, "Delete preset", f"Delete preset “{target['name']}”?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        user_index = index - 1 - len(self._builtin_presets(scope))
        user = self._user_presets.get(scope, [])
        if 0 <= user_index < len(user):
            user.pop(user_index)
            self._save_guidance_presets()
            self._reload_preset_combo(combo, scope)

    def _build_popup_scope(self, parent_layout, scope: str, label: str, initial: str) -> QPlainTextEdit:
        section = QLabel(label)
        section.setObjectName("SectionLabel")
        parent_layout.addWidget(section)

        editor = TriggerTextEdit(initial) if scope == "image" else QPlainTextEdit(initial)
        if scope == "image":
            editor._trigger_color = self.theme.accent
            editor._normal_color = self.theme.text_primary
            editor.setPlaceholderText(
                "Just this file \u2014 name the specific characters or objects you want called out."
            )
        else:
            editor.setPlaceholderText(
                "Applied to every file here \u2014 art style, lighting, composition, "
                "things to always mention or avoid."
            )
        editor._pending = False
        editor._suppress = False
        editor.textChanged.connect(lambda e=editor: self._on_editor_text_changed(e))
        row = QHBoxLayout()
        row.addWidget(QLabel("Preset:"))
        combo = QComboBox()
        self._reload_preset_combo(combo, scope)
        accept_btn = QToolButton()
        accept_btn.setIcon(lucide_icon("check", Theme.success, 16))
        accept_btn.setToolTip("Keep the inserted preset text")
        accept_btn.setEnabled(False)
        reject_btn = QToolButton()
        reject_btn.setIcon(lucide_icon("x", Theme.error, 16))
        reject_btn.setToolTip("Discard the inserted preset text")
        reject_btn.setEnabled(False)
        editor._accept_btn = accept_btn
        editor._reject_btn = reject_btn
        combo.currentIndexChanged.connect(
            lambda i, s=scope, e=editor, c=combo: self._on_preset_selected(s, e, c, i)
        )
        accept_btn.clicked.connect(lambda _c, e=editor: self._commit_pending(e))
        reject_btn.clicked.connect(lambda _c, e=editor: self._reject_pending(e))
        save_btn = QPushButton("Save as…")
        save_btn.setToolTip("Save the current field text as a new custom preset.")
        save_btn.clicked.connect(lambda _c, s=scope, e=editor, c=combo: self._save_preset_as(s, e, c))
        del_btn = QPushButton("Delete")
        del_btn.setToolTip("Delete the selected custom preset.")
        del_btn.clicked.connect(lambda _c, s=scope, c=combo: self._delete_preset(s, c))
        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Clear this field.")
        clear_btn.clicked.connect(lambda _c, e=editor: e.clear())
        row.addWidget(combo, 1)
        row.addWidget(accept_btn)
        row.addWidget(reject_btn)
        row.addWidget(save_btn)
        row.addWidget(del_btn)
        row.addWidget(clear_btn)
        parent_layout.addLayout(row)
        parent_layout.addWidget(editor, 1)
        return editor

    def _open_guidance_expand(self) -> None:
        has_images = self.store is not None and bool(self.images)
        if has_images:
            self.commit_guidance()  # make the project reflect the current main fields
            self._refresh_source_availability()

        dlg = GuidanceDialog(self)
        dlg.setWindowTitle("Custom Caption Guidance")
        outer = QVBoxLayout(dlg)
        body = QHBoxLayout()
        outer.addLayout(body, 1)

        left = QVBoxLayout()
        body.addLayout(left, 3)
        if has_images:
            dlg_convert = ToggleSwitch()
            dlg_convert.setChecked(self._convert_active())
            dlg_convert.setEnabled(getattr(self, "_has_source_txt", False))
            dlg_convert.toggled.connect(self._set_convert_mode)
            conv_row = self._explained_toggle_row(
                "Use existing .txt captions as guidance",
                "Each file's matching .txt sidecar is fed to the captioner to upgrade into "
                "structured JSON. Images without a .txt fall back to image-only captioning. "
                "Folder-wide; applies as soon as you toggle it.",
                dlg_convert,
            )
            self._style_convert_row(conv_row, teal_title=True)
            dlg_convert.toggled.connect(
                lambda _checked, r=conv_row: self._style_convert_row(r, teal_title=True))
            left.addWidget(conv_row)
            conv_div = QFrame()
            conv_div.setObjectName("PanelDivider")
            conv_div.setFrameShape(QFrame.HLine)
            left.addWidget(conv_div)
        folder_initial = self.project.folder_guidance if has_images else self.g_folder.toPlainText()
        folder_ed = self._build_popup_scope(left, "folder", "Folder \u00b7 all files", folder_initial)

        if has_images:
            start_idx = self.images.index(self.current) if self.current in self.images else 0
            # work on a copy of the per-file guidance so edits can be discarded
            work_per_file = dict(self.project.per_file)
            image_initial = work_per_file.get(self.images[start_idx].name, "")
        else:
            start_idx = 0
            work_per_file = {}
            image_initial = ""
        image_ed = self._build_popup_scope(left, "image", "This file", image_initial)
        state = {"idx": start_idx}

        if has_images:
            dlg_omit = ToggleSwitch()
            left.addWidget(self._toggle_row(
                "Use this file's .txt caption", dlg_omit,
                "Off = caption this image from the image alone, even though convert mode is on. "
                "Available when convert mode is on and this file has a matching .txt."))
            self._dlg_omit_toggle = dlg_omit

            def refresh_omit() -> None:
                img = self.images[state["idx"]]
                self._dlg_omit_name = img.name
                on = bool(self._convert_active() and self.store is not None
                          and self.store.has_source_text(img))
                dlg_omit.setEnabled(on)
                dlg_omit.blockSignals(True)
                dlg_omit.setChecked(on and not self.project.is_convert_omitted(img.name))
                dlg_omit.blockSignals(False)

            dlg_omit.toggled.connect(
                lambda checked: self._set_image_omit(self.images[state["idx"]].name, omit=not checked))
            dlg_convert.toggled.connect(lambda *_: refresh_omit())
            refresh_omit()

        original_folder = folder_initial
        original_per_file = {k: v for k, v in work_per_file.items() if v.strip()}
        original_image = image_initial

        def save_image_field() -> None:
            if not has_images:
                return
            self._commit_pending(image_ed)
            img = self.images[state["idx"]]
            text = image_ed.toPlainText()
            if text.strip():
                work_per_file[img.name] = text
            else:
                work_per_file.pop(img.name, None)

        if has_images:
            dlg.resize(1180, 820)
            self._build_tag_palette(left, image_ed)
            right = QVBoxLayout()
            body.addLayout(right, 2)
            nav = QHBoxLayout()
            prev_btn = QToolButton()
            prev_btn.setIcon(lucide_icon("chevron-left", self.theme.text_secondary, 18))
            prev_btn.setToolTip("Previous image")
            next_btn = QToolButton()
            next_btn.setIcon(lucide_icon("chevron-right", self.theme.text_secondary, 18))
            next_btn.setToolTip("Next image")
            name_label = QLabel()
            name_label.setObjectName("Hint")
            name_label.setAlignment(Qt.AlignCenter)
            nav.addWidget(prev_btn)
            nav.addWidget(name_label, 1)
            nav.addWidget(next_btn)
            right.addLayout(nav)
            preview = QLabel()
            preview.setObjectName("Panel")
            preview.setAlignment(Qt.AlignCenter)
            preview.setMinimumWidth(380)
            right.addWidget(preview, 1)

            def refresh_preview() -> None:
                img = self.images[state["idx"]]
                pm = self.preview_pixmap(img)
                if pm.isNull():
                    preview.setText("(no preview available)")
                else:
                    preview.setPixmap(pm.scaled(440, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                name_label.setText(f"{img.name}   ({state['idx'] + 1} / {len(self.images)})")
                prev_btn.setEnabled(state["idx"] > 0)
                next_btn.setEnabled(state["idx"] < len(self.images) - 1)

            def go(delta: int) -> None:
                save_image_field()
                state["idx"] = max(0, min(state["idx"] + delta, len(self.images) - 1))
                image_ed._suppress = True
                image_ed.setPlainText(work_per_file.get(self.images[state["idx"]].name, ""))
                image_ed._suppress = False
                image_ed._pending = False
                image_ed._accept_btn.setEnabled(False)
                image_ed._reject_btn.setEnabled(False)
                image_ed.rescan()
                refresh_preview()
                refresh_omit()

            prev_btn.clicked.connect(lambda: go(-1))
            next_btn.clicked.connect(lambda: go(1))
            refresh_preview()
        else:
            dlg.resize(900, 760)

        def is_dirty() -> bool:
            self._commit_pending(folder_ed)
            save_image_field()
            if folder_ed.toPlainText() != original_folder:
                return True
            if has_images:
                current = {k: v for k, v in work_per_file.items() if v.strip()}
                return current != original_per_file
            return image_ed.toPlainText() != original_image

        def apply() -> None:
            self._commit_pending(folder_ed)
            save_image_field()
            folder_text = folder_ed.toPlainText()
            if has_images:
                self.project.folder_guidance = folder_text
                self.project.per_file = dict(work_per_file)
                self.g_folder.setPlainText(folder_text)
                if self.current is not None:
                    self.load_per_file_guidance(self.current.name)
                for img in self.images:
                    self._refresh_thumb_marker(img)
                self._guidance_dirty = True
                self.persist_guidance_if_dirty()
            else:
                self.g_folder.setPlainText(folder_text)
                image_text = image_ed.toPlainText()
                if image_text.strip():
                    self.g_per_file.setPlainText(image_text)
            nonlocal original_folder, original_per_file, original_image
            original_folder = folder_text
            original_per_file = {k: v for k, v in work_per_file.items() if v.strip()}
            original_image = image_ed.toPlainText()

        def gate() -> bool:
            if not is_dirty():
                return True
            resp = QMessageBox.question(
                dlg, "Apply changes?",
                "You have unapplied guidance changes. Apply them before closing?",
                QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel, QMessageBox.Yes,
            )
            if resp == QMessageBox.Cancel:
                return False
            if resp == QMessageBox.Yes:
                apply()
            return True

        dlg.set_close_gate(gate)

        buttons = QDialogButtonBox(QDialogButtonBox.Apply | QDialogButtonBox.Close)
        buttons.button(QDialogButtonBox.Apply).clicked.connect(apply)
        buttons.button(QDialogButtonBox.Close).clicked.connect(dlg.reject)
        outer.addWidget(buttons)

        folder_ed.setFocus()
        dlg.exec()
        self._dlg_omit_toggle = None
        self._dlg_omit_name = None

    def _build_guidance_panel(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Panel")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(4)

        # widgets first, so preset pickers can reference their fields
        self.g_folder = QPlainTextEdit()
        self.g_folder.setObjectName("GuidanceBoxRO")
        self.g_folder.setReadOnly(True)
        self.g_folder.setFixedHeight(96)
        self.g_folder.setPlaceholderText("No folder guidance — edit in Guidance Settings.")
        self.g_folder.setToolTip(
            "Folder guidance (read-only here). Edit it in Guidance Settings."
        )
        self.g_folder_enabled = ToggleSwitch()
        self.g_folder_enabled.setEnabled(False)
        self._folder_enabled_row = self._toggle_row(
            "Apply folder guidance", self.g_folder_enabled,
            "When off, the folder guidance is ignored for every file. Enabled once you add folder guidance.",
        )
        self.g_per_file_enabled = ToggleSwitch()
        self.g_per_file_enabled.setEnabled(False)
        self._per_file_enabled_row = self._toggle_row(
            "Apply this-file guidance", self.g_per_file_enabled,
            "When off, this file's guidance is kept but not applied. Enabled once you add per-file guidance.",
        )
        self.g_mode = QComboBox()
        self.g_mode.addItems(list(GUIDANCE_MODES))
        self.g_mode.setToolTip(
            "Controls faithfulness vs. creativity for JSON generation:\n"
            "• Inherit — use the global Creative JSON preference\n"
            "• Faithful — describe only what is actually in the image\n"
            "• Creative — allow more imaginative elaboration"
        )
        self.g_mode.setItemData(0, "Use the global Creative JSON preference from Settings.", Qt.ToolTipRole)
        self.g_mode.setItemData(1, "Describe only what is actually visible in the image.", Qt.ToolTipRole)
        self.g_mode.setItemData(2, "Allow the model to elaborate more imaginatively.", Qt.ToolTipRole)
        self.g_per_file = QPlainTextEdit()
        self.g_per_file.setObjectName("GuidanceBoxRO")
        self.g_per_file.setReadOnly(True)
        self.g_per_file.setFixedHeight(120)
        self.g_per_file.setPlaceholderText("No guidance for this file \u2014 edit in Guidance Settings.")
        self.g_per_file.setToolTip(
            "Per-file guidance (read-only here). Edit it in Guidance Settings. Added on top "
            "of the folder guidance for this file only."
        )
        # Convert mode (folder-wide): feed each image's .txt sidecar to the captioner
        # as a source caption to upgrade into structured JSON.
        self.g_convert_enabled = ToggleSwitch()
        self.g_convert_enabled.setEnabled(False)
        self._convert_row = self._explained_toggle_row(
            "Use existing .txt captions as guidance",
            "Upgrade each file's .txt into structured JSON — no .txt means media-only.",
            self.g_convert_enabled,
            "When on, each image's matching .txt sidecar is fed to the captioner as a source "
            "caption to upgrade into structured JSON. Images without a .txt use image-only captioning.",
        )
        self.g_convert_enabled.toggled.connect(self._on_convert_toggled)
        # Read-only preview of the detected .txt for the current image.
        self.g_source_caption = QPlainTextEdit()
        self.g_source_caption.setObjectName("GuidanceBoxRO")
        self.g_source_caption.setReadOnly(True)
        self.g_source_caption.setFixedHeight(72)
        self.g_source_caption.setToolTip("The .txt source caption fed to the captioner for this file (read-only).")

        title = QLabel("Caption Guidance")
        title.setObjectName("SectionLabel")
        title.setToolTip(
            "Extra natural-language instructions injected into the model's prompt "
            "when generating JSON from an image."
        )
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.addWidget(title)
        header.addStretch(1)
        collapse_btn = QToolButton()
        collapse_btn.setObjectName("CollapseChevron")
        collapse_btn.setIcon(lucide_icon("chevrons-left", self.theme.text_secondary, 16))
        collapse_btn.setToolTip("Collapse side panels (Ctrl+\\)")
        collapse_btn.clicked.connect(self.toggle_left_panel)
        header.addWidget(collapse_btn)
        lay.addLayout(header)
        settings_btn = QPushButton("Guidance Settings")
        settings_btn.setToolTip(
            "Open the full editor \u2014 edit folder & per-file guidance, browse files, "
            "and manage presets."
        )
        settings_btn.clicked.connect(self._open_guidance_expand)
        lay.addWidget(settings_btn)
        lay.addWidget(self._convert_row)

        # ---- Global (applies to every file) ----
        lay.addWidget(self._folder_enabled_row)
        mode_label = self._field_label("Mode")
        mode_label.setToolTip("How closely generation should follow the source (applies to the whole folder).")
        lay.addWidget(mode_label)
        lay.addWidget(self.g_mode)
        folder_label = self._field_label("Folder \u00b7 all files")
        folder_label.setToolTip("Guidance applied to every file in this folder.")
        lay.addWidget(folder_label)
        lay.addWidget(self.g_folder)

        divider = QFrame()
        divider.setObjectName("PanelDivider")
        divider.setFrameShape(QFrame.HLine)
        lay.addWidget(divider)

        # ---- This image ----
        lay.addWidget(self._per_file_enabled_row)
        image_label = self._field_label("This file")
        image_label.setToolTip("Guidance applied only to the currently selected file.")
        lay.addWidget(image_label)
        lay.addWidget(self.g_per_file)

        # Source caption sub-section (only visible in convert mode): a status line
        # and a read-only preview of the detected .txt, with an expand handle.
        self._source_caption_box = QWidget()
        sc_lay = QVBoxLayout(self._source_caption_box)
        sc_lay.setContentsMargins(0, 6, 0, 0)
        sc_lay.setSpacing(4)
        sc_head = QHBoxLayout()
        sc_head.setContentsMargins(0, 0, 0, 0)
        sc_label = self._field_label("Source caption")
        sc_label.setToolTip("The .txt fed to the captioner as source material for this file.")
        sc_head.addWidget(sc_label)
        sc_head.addStretch(1)
        self._source_status = QLabel("")
        self._source_status.setObjectName("Hint")
        sc_head.addWidget(self._source_status)
        self.g_source_use = ToggleSwitch()
        self.g_source_use.setToolTip(
            "Use this file's .txt caption. Turn off to caption it from the media alone.")
        self.g_source_use.toggled.connect(self._on_source_use_toggled)
        sc_head.addWidget(self.g_source_use)
        sc_lay.addLayout(sc_head)
        sc_field_row = QWidget()
        sc_field_h = QHBoxLayout(sc_field_row)
        sc_field_h.setContentsMargins(0, 0, 0, 0)
        sc_field_h.setSpacing(4)
        sc_field_h.addWidget(self.g_source_caption, 1)
        sc_expand = QToolButton()
        sc_expand.setObjectName("ExpandBtn")
        sc_expand.setIcon(self._expand_icon())
        sc_expand.setIconSize(QSize(14, 14))
        sc_expand.setFixedSize(22, 22)
        sc_expand.setToolTip("Pop out the source caption — stays open and follows the file you're on")
        sc_expand.clicked.connect(self._open_source_popout)
        sc_field_h.addWidget(sc_expand, 0, Qt.AlignTop)
        sc_lay.addWidget(sc_field_row)
        self._source_caption_box.setVisible(False)
        lay.addWidget(self._source_caption_box)

        # Tags used — read-only reflection of which palette tags appear in THIS
        # image's per-file guidance. Editing happens only in Guidance Settings.
        used_label = self._field_label("Tags used")
        used_label.setToolTip("Trigger tags referenced in this file's guidance.")
        lay.addWidget(used_label)
        # Note sits ABOVE the chips so its position is fixed — the flow host below
        # can grow/shrink rows without ever shoving this line around.
        self._used_tags_hint = QLabel("Read-only · manage in Guidance Settings")
        self._used_tags_hint.setObjectName("Hint")
        self._used_tags_hint.setWordWrap(True)
        lay.addWidget(self._used_tags_hint)
        self._used_tags_host = FlowWidget()
        self._used_tags_flow = FlowLayout(self._used_tags_host, 0, 6)
        lay.addWidget(self._used_tags_host)
        # When the tags get numerous or long enough to crowd the panel, the inline
        # pills collapse to this purple "View tags" pill whose hover reveals the full
        # list in a pop-out (keeps the common, few-tags case flat and quick-reference).
        self._used_tags_collapsed = QLabel("View tags")
        self._used_tags_collapsed.setObjectName("ViewTagsPill")
        self._used_tags_collapsed.setStyleSheet(
            "QLabel#ViewTagsPill { background: rgba(167,139,250,0.16); color: #A78BFA;"
            " border: 1px solid #A78BFA; border-radius: 10px;"
            " padding: 3px 10px; font-size: 11px; font-weight: 600; }"
        )
        self._used_tags_collapsed.setCursor(Qt.PointingHandCursor)
        self._used_tags_collapsed.setVisible(False)
        self._used_tags_collapsed.installEventFilter(self)  # hover -> tag-list pop-out
        _vt_row = QHBoxLayout()
        _vt_row.setContentsMargins(0, 0, 0, 0)
        _vt_row.addWidget(self._used_tags_collapsed)
        _vt_row.addStretch(1)
        lay.addLayout(_vt_row)

        # Guidance-changed section — shown when THIS image's effective guidance has
        # changed since its caption was generated. The full color-coded diff would be
        # variable-length and clip at the bottom of the panel, so it lives in a hover
        # pop-out (GuidanceDiffPopup); only this compact header + hint stay inline.
        self._gchg_box = QWidget()
        gv = QVBoxLayout(self._gchg_box)
        gv.setContentsMargins(0, 10, 0, 0)
        gv.setSpacing(2)
        self._gchg_head = QLabel("Guidance changed since last caption")
        self._gchg_head.setWordWrap(True)
        self._gchg_head.setStyleSheet(
            f"color: {STALE_COLOR}; font-weight: 600; font-size: 11px;"
        )
        self._gchg_hint = QLabel("Hover to see what changed")
        self._gchg_hint.setObjectName("Hint")
        self._gchg_hint.setWordWrap(True)
        gv.addWidget(self._gchg_head)
        gv.addWidget(self._gchg_hint)
        self._gchg_box.setVisible(False)
        self._gchg_box.setCursor(Qt.PointingHandCursor)
        self._gchg_box.installEventFilter(self)  # Enter/Leave -> show/hide the diff pop-out
        lay.addWidget(self._gchg_box)
        lay.addStretch(1)

        # Debounced live staleness refresh as guidance is edited.
        self._stale_timer = QTimer(self)
        self._stale_timer.setSingleShot(True)
        self._stale_timer.setInterval(220)
        self._stale_timer.timeout.connect(self._refresh_stale_state)

        self.g_folder.textChanged.connect(self._mark_guidance_dirty)
        self.g_folder.textChanged.connect(self._sync_folder_toggle)
        self.g_folder.textChanged.connect(self._schedule_stale_refresh)
        self.g_per_file.textChanged.connect(self._mark_guidance_dirty)
        self.g_per_file.textChanged.connect(self._refresh_tags_used)
        self.g_per_file.textChanged.connect(self._sync_per_file_toggle)
        self.g_per_file.textChanged.connect(self._schedule_stale_refresh)
        self.g_folder_enabled.toggled.connect(self._on_folder_enabled_toggled)
        self.g_per_file_enabled.toggled.connect(self._on_per_file_enabled_toggled)
        self.g_folder_enabled.toggled.connect(self._schedule_stale_refresh)
        self.g_per_file_enabled.toggled.connect(self._schedule_stale_refresh)
        self.g_mode.currentTextChanged.connect(self._mark_guidance_dirty)
        self._refresh_tags_used()
        return w

    def _toggle_row(self, text: str, switch: "ToggleSwitch", tooltip: str = "") -> QWidget:
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        lab = QLabel(text)
        lab.setWordWrap(True)  # narrow panel: wrap instead of clipping the label
        lab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        if tooltip:
            lab.setToolTip(tooltip)
            switch.setToolTip(tooltip)
        h.addWidget(lab, 1)
        h.addWidget(switch, 0, Qt.AlignVCenter)
        return row

    def _explained_toggle_row(self, title: str, description: str, switch: "ToggleSwitch",
                              tooltip: str = "") -> QWidget:
        """A toggle row with a title and a muted one-line description beneath it, for
        settings that warrant more than a bare label. Toggle stays right-aligned."""
        row = QWidget()
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(8)
        col = QVBoxLayout()
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(1)
        lab = QLabel(title)
        lab.setWordWrap(True)
        desc = QLabel(description)
        desc.setObjectName("Hint")
        desc.setWordWrap(True)
        col.addWidget(lab)
        col.addWidget(desc)
        h.addLayout(col, 1)
        h.addWidget(switch, 0, Qt.AlignVCenter)
        if tooltip:
            lab.setToolTip(tooltip)
            switch.setToolTip(tooltip)
        row._title_lbl = lab          # exposed so callers can recolour / re-text
        row._desc_lbl = desc
        row._avail_desc = description  # the "feature available" description to restore
        return row

    _CONVERT_NO_TXT_DESC = ("No .txt caption files were found in this folder. Add a .txt "
                            "caption file whose name matches a file here to use this "
                            "feature.")

    def _refresh_source_availability(self) -> None:
        """Recompute whether the folder has any .txt sidecars (folder-level gate)."""
        self._has_source_txt = bool(
            self.store is not None and self.images and self.store.any_source_text(self.images))

    def _convert_active(self) -> bool:
        """Convert mode is only effective when it's on AND the folder actually has
        at least one matching .txt to draw from."""
        return bool(self.project is not None and self.project.convert_txt_to_json
                    and getattr(self, "_has_source_txt", False))

    def _style_convert_row(self, row, *, teal_title: bool) -> None:
        """Colour and text a convert toggle row by availability. Title goes teal in
        the popup; the live description is amber there while convert is on and gray
        when it's off (a quick at-a-glance indicator). Both surfaces swap to a muted
        'no .txt found' note when the folder has no source captions."""
        if row is None:
            return
        avail = getattr(self, "_has_source_txt", False)
        on = bool(self.project is not None and self.project.convert_txt_to_json)
        title = getattr(row, "_title_lbl", None)
        desc = getattr(row, "_desc_lbl", None)
        if title is not None:
            title.setStyleSheet("color:#2FC6B3; font-weight:600;" if teal_title else "")
        if desc is None:
            return
        if avail:
            desc.setText(getattr(row, "_avail_desc", ""))
            if teal_title:
                amber = getattr(self.theme, "warning", "#E0A33B")
                desc.setStyleSheet(f"color:{amber};" if on else "color:#9aa4b6;")
            else:
                desc.setStyleSheet("")
        else:
            desc.setText(self._CONVERT_NO_TXT_DESC)
            desc.setStyleSheet("color:#9aa4b6;")

    def _set_convert_mode(self, checked: bool) -> None:
        """Apply convert mode (folder-wide). Used by both the panel toggle and the
        Guidance Settings dialog toggle, keeping the two in sync."""
        if self.store is None:
            return
        self.project.convert_txt_to_json = bool(checked)
        self._guidance_dirty = True
        self.persist_guidance_if_dirty()
        sw = getattr(self, "g_convert_enabled", None)
        if sw is not None and sw.isChecked() != bool(checked):
            sw.blockSignals(True)
            sw.setChecked(bool(checked))
            sw.blockSignals(False)
        self._refresh_source_caption()
        self._refresh_omit_markers()  # convert on/off flips every image's omit marker

    def _set_image_omit(self, name: str, omit: bool) -> None:
        """Per-file override of convert mode. Used by the sidebar, pop-out, and
        dialog toggles, all kept in sync. The toggles are framed positively ("use
        this image's .txt"), so checked = not omitted."""
        if self.store is None or self.project is None:
            return
        self.project.set_convert_omit(name, omit)
        self._guidance_dirty = True
        self.persist_guidance_if_dirty()
        self._refresh_source_caption()  # restyles strip + pop-out toggles/text/status
        self._sync_dialog_omit_toggle(name)
        path = next((p for p in self.images if p.name == name), None)
        if path is not None:
            self._refresh_thumb_marker(path)

    def _sync_dialog_omit_toggle(self, name: str) -> None:
        tog = getattr(self, "_dlg_omit_toggle", None)
        if tog is None or getattr(self, "_dlg_omit_name", None) != name:
            return
        checked = not self.project.is_convert_omitted(name)
        if tog.isChecked() != checked:
            tog.blockSignals(True)
            tog.setChecked(checked)
            tog.blockSignals(False)

    def _image_is_omit_marked(self, img: Path) -> bool:
        """Whether the filmstrip should show the violet omit marker: convert active,
        the image has a .txt, and the user omitted it."""
        return bool(self._convert_active() and self.store is not None
                    and self.store.has_source_text(img)
                    and self.project.is_convert_omitted(img.name))

    def _refresh_omit_markers(self) -> None:
        """Update the omit marker on every thumbnail (e.g. when convert toggles)."""
        items = getattr(self, "_thumb_items", {})
        for path in self.images:
            item = items.get(str(path))
            if item is not None:
                item.setData(OMIT_ROLE, self._image_is_omit_marked(path))
                item.setToolTip(self._thumb_tooltip(path))
        vp = getattr(self, "filmstrip", None)
        if vp is not None:
            self.filmstrip.viewport().update()

    @staticmethod
    def _tag_used_in(text: str, tag: str) -> bool:
        if not tag:
            return False
        length = len(tag)
        start = 0
        while True:
            idx = text.find(tag, start)
            if idx < 0:
                return False
            before = text[idx - 1] if idx > 0 else " "
            after = text[idx + length] if idx + length < len(text) else " "
            if not (before.isalnum() or before == "_") and not (after.isalnum() or after == "_"):
                return True
            start = idx + length

    def _make_used_pill(self, text: str) -> QLabel:
        pill = QLabel(text)
        pill.setObjectName("UsedPill")
        pill.setFont(QFont(self.settings.mono_font_family or "Monospace"))
        return pill

    # Tags stay inline (flat, quick-reference) until they'd crowd the narrow panel:
    # more than this many, or any single one this long (a re-used phrase, not a name).
    _TAGS_INLINE_MAX = 6
    _TAG_LEN_INLINE_MAX = 22

    def _tags_overflow(self, tags: list[str]) -> bool:
        return (len(tags) > self._TAGS_INLINE_MAX
                or any(len(t) > self._TAG_LEN_INLINE_MAX for t in tags))

    def _refresh_tags_used(self) -> None:
        if not hasattr(self, "_used_tags_flow"):
            return
        text = self.g_per_file.toPlainText()
        known = list(getattr(self, "_folder_tags", [])) + [
            t for t in self._default_tags if t not in getattr(self, "_folder_tags", [])
        ]
        used = [t for t in known if self._tag_used_in(text, t)]
        self._used_tags_used = used
        self._clear_layout(self._used_tags_flow)
        if used and self._tags_overflow(used):
            # collapse to the "View tags" pill; the full list lives in the hover pop-out
            self._used_tags_host.setVisible(False)
            self._used_tags_collapsed.setText(f"View tags ({len(used)})  \u2197")
            self._used_tags_collapsed.setVisible(True)
        else:
            self._used_tags_collapsed.setVisible(False)
            self._hide_tags_popup()
            for tag in used:
                self._used_tags_flow.addWidget(self._make_used_pill(tag))
            self._used_tags_host.setVisible(bool(used))
            # adjustSize() collapsed the host to one pill's width (forcing a single
            # column and a too-short height on re-populate). Invalidate the flow and
            # let the parent re-query heightForWidth at the real panel width instead.
            self._used_tags_flow.invalidate()
            self._used_tags_host.updateGeometry()

    def _mark_guidance_dirty(self, *args) -> None:
        if not self._loading:
            self._guidance_dirty = True

    def _sync_folder_toggle(self, *args) -> None:
        """Folder toggle: interactive only when folder guidance exists; reflects the
        stored enable flag (default on, so newly-added guidance is applied)."""
        has = bool(self.g_folder.toPlainText().strip())
        sw = self.g_folder_enabled
        sw.setEnabled(has)
        proj = getattr(self, "project", None)
        on = has and (proj.folder_guidance_enabled if proj is not None else True)
        sw.blockSignals(True)
        sw.setChecked(on)
        sw.blockSignals(False)

    def _sync_per_file_toggle(self, *args) -> None:
        """Per-file toggle: interactive only when this image has guidance; reflects
        the stored per-file override (default on)."""
        has = bool(self.g_per_file.toPlainText().strip())
        sw = self.g_per_file_enabled
        sw.setEnabled(has)
        proj = getattr(self, "project", None)
        name = self.current.name if self.current is not None else None
        active = proj.per_file_active(name) if (proj is not None and name) else True
        sw.blockSignals(True)
        sw.setChecked(has and active)
        sw.blockSignals(False)

    def _on_folder_enabled_toggled(self, checked: bool) -> None:
        # Fires only on real user interaction (sync uses blockSignals), and the
        # toggle is only enabled when folder text exists — so this is always valid.
        if self._loading:
            return
        self.project.folder_guidance_enabled = checked
        self._guidance_dirty = True

    def _on_per_file_enabled_toggled(self, checked: bool) -> None:
        if self._loading or self.current is None:
            return
        self.project.per_file_enabled[self.current.name] = checked
        self._guidance_dirty = True

    def load_project_into_ui(self) -> None:
        self._loading = True
        try:
            self.g_folder.setPlainText(self.project.folder_guidance)
            self.g_mode.setCurrentText(CREATIVE_TO_MODE.get(self.project.creative_json, "Inherit"))
            self._sync_folder_toggle()
            self._refresh_source_availability()
            avail = getattr(self, "_has_source_txt", False)
            self.g_convert_enabled.setEnabled(self.store is not None and avail)
            self.g_convert_enabled.blockSignals(True)
            self.g_convert_enabled.setChecked(self._convert_active())
            self.g_convert_enabled.blockSignals(False)
            self._style_convert_row(getattr(self, "_convert_row", None), teal_title=False)
        finally:
            self._loading = False
        self._guidance_dirty = False
        self._refresh_source_caption()

    def load_per_file_guidance(self, filename: str) -> None:
        self._loading = True
        try:
            self.g_per_file.setPlainText(self.project.per_file_guidance(filename))
            self._sync_per_file_toggle()
        finally:
            self._loading = False
        self._refresh_guidance_changes()
        self._refresh_source_caption()

    def _on_convert_toggled(self, checked: bool) -> None:
        if self._loading or self.store is None:
            return
        self._set_convert_mode(checked)

    def _on_source_use_toggled(self, checked: bool) -> None:
        if self._loading or self.store is None or self.current is None:
            return
        self._set_image_omit(self.current.name, omit=not checked)

    @staticmethod
    def _elide_middle(text: str, limit: int = 26) -> str:
        if len(text) <= limit:
            return text
        keep = max(1, limit - 1)
        head = keep // 2
        return text[:head] + "\u2026" + text[-(keep - head):]

    def _image_uses_source(self, img: Path) -> bool:
        """True if this image's .txt should be fed to the captioner: convert mode is
        active, the image has a matching .txt, and the user hasn't omitted it."""
        return bool(self._convert_active() and self.store is not None
                    and self.store.has_source_text(img)
                    and not self.project.is_convert_omitted(img.name))

    def _apply_source_strikethrough(self, field, omitted: bool) -> None:
        """Strike through the source-caption text when this image is omitted, so the
        skipped caption reads as struck out."""
        cur = field.textCursor()
        cur.select(QTextCursor.Document)
        fmt = QTextCharFormat()
        fmt.setFontStrikeOut(bool(omitted))
        cur.mergeCharFormat(fmt)
        cur.clearSelection()
        field.setTextCursor(cur)

    def _current_source_caption(self):
        """(found_text, status_label, status_color, placeholder, omitted) for the
        current image. found_text is "" when there is no .txt. Returns None when
        convert mode is off or no folder/image is active."""
        if not self._convert_active():
            return None
        if self.store is None or self.current is None:
            return None
        text = self.store.load_source_text(self.current)
        if text:
            if self.project.is_convert_omitted(self.current.name):
                return text, "omitted \u00b7 image-only", OMIT_COLOR, "", True
            name = self.store.source_text_path(self.current).name
            return text, "\u2713 " + self._elide_middle(name), "#3ddc84", "", False
        warn = getattr(self.theme, "warning", "#E0A33B")
        return ("", "no .txt \u00b7 image-only", warn,
                "No source caption for this file — the captioner will work from the media alone.", False)

    def _refresh_source_caption(self) -> None:
        box = getattr(self, "_source_caption_box", None)
        if box is None:
            return
        convert_on = self._convert_active()
        box.setVisible(convert_on)
        if not convert_on:
            self._close_source_popout()
        # the per-file "use this .txt" toggle (only meaningful with a .txt present)
        tog = getattr(self, "g_source_use", None)
        if tog is not None:
            has_txt = bool(convert_on and self.store is not None and self.current is not None
                           and self.store.has_source_text(self.current))
            tog.setEnabled(has_txt)
            tog.blockSignals(True)
            tog.setChecked(has_txt and not self.project.is_convert_omitted(self.current.name))
            tog.blockSignals(False)
        info = self._current_source_caption()
        if info is None:
            self.g_source_caption.setPlainText("")
            self._source_status.setText("")
            self._source_status.setToolTip("")
            self._apply_source_strikethrough(self.g_source_caption, False)
        else:
            text, status, color, placeholder, omitted = info
            self.g_source_caption.setPlainText(text)
            # Compact indicator next to the title (the full status would crowd out the
            # title in this narrow pane): green check = this .txt is used, purple X =
            # omitted. No glyph when there's no .txt. The full status sits on its tooltip.
            if omitted:
                glyph, gcolor = "\u2717", OMIT_COLOR          # ✗ purple
            elif text:
                glyph, gcolor = "\u2713", "#3ddc84"           # ✓ green
            else:
                glyph, gcolor = "", color
            self._source_status.setText(glyph)
            self._source_status.setStyleSheet(
                f"color:{gcolor}; font-size:13px; font-weight:600;")
            self._source_status.setToolTip(status)
            self.g_source_caption.setPlaceholderText(placeholder)
            self._apply_source_strikethrough(self.g_source_caption, omitted)
        self._update_source_popout()

    def _open_source_popout(self) -> None:
        """A modeless source-caption inspector: it stays open while you browse and
        follows the current image (thumbnail + .txt) as you navigate the main window."""
        if not self.g_source_caption.isEnabled():
            return
        existing = getattr(self, "_source_popout", None)
        if existing is not None:
            existing.raise_()
            existing.activateWindow()
            self._update_source_popout()
            return
        dlg = SourcePopout(self, self.prev_image, self.next_image)
        dlg.setWindowTitle("Source caption")
        dlg.setModal(False)
        dlg.setAttribute(Qt.WA_DeleteOnClose, True)
        dlg.resize(420, 600)
        v = QVBoxLayout(dlg)
        thumb = QLabel()
        thumb.setObjectName("Panel")
        thumb.setAlignment(Qt.AlignCenter)
        # Fixed-height box so the image (whatever its aspect) centers inside a
        # constant frame — the nav bar and everything below never shift.
        thumb.setFixedHeight(self._SC_THUMB_H)
        v.addWidget(thumb)
        nav = QHBoxLayout()
        prev_btn = QToolButton()
        prev_btn.setIcon(lucide_icon("chevron-left", self.theme.text_secondary, 18))
        prev_btn.setToolTip("Previous image")
        prev_btn.clicked.connect(self.prev_image)
        next_btn = QToolButton()
        next_btn.setIcon(lucide_icon("chevron-right", self.theme.text_secondary, 18))
        next_btn.setToolTip("Next image")
        next_btn.clicked.connect(self.next_image)
        name_lab = QLabel()
        name_lab.setObjectName("Hint")
        name_lab.setAlignment(Qt.AlignCenter)
        name_lab.setWordWrap(True)
        nav.addWidget(prev_btn)
        nav.addWidget(name_lab, 1)
        nav.addWidget(next_btn)
        v.addLayout(nav)
        status_lab = QLabel()
        status_lab.setAlignment(Qt.AlignCenter)
        v.addWidget(status_lab)
        use_row = QHBoxLayout()
        use_lab = QLabel("Use this file's .txt caption")
        use_lab.setObjectName("Hint")
        use_tog = ToggleSwitch()
        use_tog.setToolTip("Off = caption this file from the media alone.")
        use_tog.toggled.connect(self._on_popout_use_toggled)
        use_row.addWidget(use_lab)
        use_row.addStretch(1)
        use_row.addWidget(use_tog)
        v.addLayout(use_row)
        text = QPlainTextEdit()
        text.setReadOnly(True)
        text.setObjectName("GuidanceBoxRO")
        text.installEventFilter(dlg)  # let plain Left/Right navigate even when text is focused
        v.addWidget(text, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        dlg._sc_thumb, dlg._sc_name, dlg._sc_status, dlg._sc_text = thumb, name_lab, status_lab, text
        dlg._sc_prev, dlg._sc_next, dlg._sc_use = prev_btn, next_btn, use_tog
        dlg.destroyed.connect(lambda *_: setattr(self, "_source_popout", None))
        self._source_popout = dlg
        self._update_source_popout()
        dlg.show()

    def _update_source_popout(self) -> None:
        dlg = getattr(self, "_source_popout", None)
        if dlg is None:
            return
        if self.current is not None:
            pm = self.preview_pixmap(self.current)
            if pm.isNull():
                dlg._sc_thumb.setPixmap(QPixmap())
                dlg._sc_thumb.setText("(no preview available)")
            else:
                dlg._sc_thumb.setPixmap(pm.scaled(380, self._SC_THUMB_H, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            total = len(self.images)
            idx = (self.images.index(self.current) + 1) if self.current in self.images else 0
            dlg._sc_name.setText(f"{self.current.name}   ({idx} / {total})" if idx else self.current.name)
        else:
            dlg._sc_thumb.setPixmap(QPixmap())
            dlg._sc_thumb.setText("(no file selected)")
            dlg._sc_name.setText("")
        # Nav mirrors the main window: disabled at the ends, and while a batch has
        # navigation locked (so the pop-out can't move the selection mid-run).
        prev_btn = getattr(dlg, "_sc_prev", None)
        next_btn = getattr(dlg, "_sc_next", None)
        if prev_btn is not None and next_btn is not None:
            total = len(self.images)
            pos = (self.images.index(self.current)) if (self.current in self.images) else -1
            locked = getattr(self, "_nav_locked", False)
            prev_btn.setEnabled(not locked and pos > 0)
            next_btn.setEnabled(not locked and 0 <= pos < total - 1)
        info = self._current_source_caption()
        use_tog = getattr(dlg, "_sc_use", None)
        has_txt = bool(self._convert_active() and self.store is not None and self.current is not None
                       and self.store.has_source_text(self.current))
        if use_tog is not None:
            use_tog.setEnabled(has_txt)
            use_tog.blockSignals(True)
            use_tog.setChecked(has_txt and not self.project.is_convert_omitted(self.current.name))
            use_tog.blockSignals(False)
        if info is None:
            dlg._sc_text.setPlainText("")
            dlg._sc_status.setText("")
            self._apply_source_strikethrough(dlg._sc_text, False)
        else:
            text, status, color, placeholder, omitted = info
            dlg._sc_text.setPlainText(text)
            dlg._sc_text.setPlaceholderText(placeholder)
            dlg._sc_status.setText(status)
            dlg._sc_status.setStyleSheet(f"color:{color}; font-size:11px;")
            self._apply_source_strikethrough(dlg._sc_text, omitted)

    def _on_popout_use_toggled(self, checked: bool) -> None:
        if self.store is None or self.current is None:
            return
        self._set_image_omit(self.current.name, omit=not checked)

    def _close_source_popout(self) -> None:
        dlg = getattr(self, "_source_popout", None)
        if dlg is not None:
            dlg.close()  # WA_DeleteOnClose + destroyed handler clears the reference

    def commit_guidance(self) -> None:
        self.project.folder_guidance = self.g_folder.toPlainText()
        # Only write the folder flag when there's text (toggle enabled); otherwise
        # leave the stored value so clearing + refilling re-applies it ("auto-on").
        if self.g_folder_enabled.isEnabled():
            self.project.folder_guidance_enabled = self.g_folder_enabled.isChecked()
        self.project.creative_json = MODE_TO_CREATIVE.get(self.g_mode.currentText())
        if self.current is not None:
            name = self.current.name
            text = self.g_per_file.toPlainText()
            if text.strip():
                self.project.per_file[name] = text
                self.project.per_file_enabled[name] = self.g_per_file_enabled.isChecked()
            else:
                self.project.per_file.pop(name, None)
                self.project.per_file_enabled.pop(name, None)

    def persist_guidance_if_dirty(self) -> None:
        if self.store is None or not self._guidance_dirty:
            return
        self.commit_guidance()
        try:
            self.store.save_project(self.project)
        except OSError as exc:
            self._set_status(f"Could not save guidance: {exc}")
            return
        self._guidance_dirty = False

    def _build_caption_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        inner = QWidget()
        inner.setObjectName("Panel")
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(4)

        self.cap_high_level = QPlainTextEdit()
        self.cap_high_level.setFixedHeight(96)
        self.cap_aesthetics = QLineEdit()
        self.cap_lighting = QLineEdit()
        self.cap_medium = QLineEdit()
        self.style_mode = QComboBox()
        self.style_mode.addItems(["photo", "art_style"])
        self.cap_style_detail = QLineEdit()
        self.cap_background = QPlainTextEdit()
        self.cap_background.setFixedHeight(72)

        lay.addWidget(self._field_label("High-level description"))
        lay.addWidget(self._attach_expand(self.cap_high_level, "High-level description"))
        lay.addSpacing(6)
        lay.addWidget(self._field_label("Style mode"))
        lay.addWidget(self.style_mode)
        lay.addWidget(self._field_label("Aesthetics"))
        lay.addWidget(self._attach_expand(self.cap_aesthetics, "Aesthetics", single_line=True))
        lay.addWidget(self._field_label("Lighting"))
        lay.addWidget(self._attach_expand(self.cap_lighting, "Lighting", single_line=True))
        lay.addWidget(self._field_label("Medium"))
        lay.addWidget(self._attach_expand(self.cap_medium, "Medium", single_line=True))
        self.style_detail_label = self._field_label("Photo")
        lay.addWidget(self.style_detail_label)
        lay.addWidget(self._attach_expand(self.cap_style_detail, "Style detail", single_line=True))
        lay.addSpacing(6)
        lay.addWidget(self._field_label("Background"))
        lay.addWidget(self._attach_expand(self.cap_background, "Background"))
        lay.addSpacing(8)
        # Raw JSON lives here rather than the toolbar: it's a per-caption inspector,
        # and it only means anything for structured presets.
        self.rawjson_btn = QPushButton("View raw JSON")
        self.rawjson_btn.setToolTip("Show the raw caption JSON for this file (Ctrl+J)")
        self.rawjson_btn.clicked.connect(self._toggle_raw_json)
        lay.addWidget(self.rawjson_btn)
        lay.addStretch(1)

        for w in (self.cap_high_level, self.cap_background):
            w.textChanged.connect(self._mark_dirty)
        for w in (self.cap_aesthetics, self.cap_lighting, self.cap_medium, self.cap_style_detail):
            w.textChanged.connect(self._mark_dirty)
        self.style_mode.currentTextChanged.connect(self._on_style_mode_changed)

        scroll.setWidget(inner)
        return scroll

    def _build_plain_tab(self) -> QWidget:
        """Single free-text caption editor, used by plain-text presets. The caption
        *is* the .txt sidecar's contents, so there's nothing to deconstruct."""
        w = QWidget()
        w.setObjectName("Panel")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(4)
        lay.addWidget(self._field_label("Caption"))
        self.cap_plain = QPlainTextEdit()
        # "file", not "image": the same box captions clips.
        self.cap_plain.setPlaceholderText(
            "One caption for this file \u2014 saved as a .txt next to it."
        )
        lay.addWidget(self._attach_expand(self.cap_plain, "Caption"), 1)
        self.plain_count = QLabel("")
        self.plain_count.setStyleSheet(f"color: {self.theme.text_secondary}; font-size: 11px;")
        lay.addWidget(self.plain_count)
        self.cap_plain.textChanged.connect(self._mark_dirty)
        self.cap_plain.textChanged.connect(self._update_plain_count)
        return w

    def _build_preset_strip(self) -> QWidget:
        """Preset selector across the top of the right panel. The choice is saved
        per-folder, so a dataset always reopens in the format it was captioned in."""
        bar = QFrame()
        bar.setObjectName("PresetBar")
        bar.setStyleSheet(
            f"#PresetBar {{ background: {self.theme.surface_2}; "
            f"border-bottom: 1px solid {self.theme.border}; }}"
        )
        # One row per selector. The right panel isn't wide enough to share: preset
        # names run long ("MiniMax H3 — Official Prompt Structure"), and pairing the
        # other two on one line collided their labels. A grid keeps the labels
        # right-aligned so the combos start at a common left edge.
        rows = QVBoxLayout(bar)
        rows.setContentsMargins(8, 6, 8, 6)
        rows.setSpacing(4)
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        grid.setColumnStretch(1, 1)
        rows.addLayout(grid)

        def _row(index: int, text: str) -> QLabel:
            label = QLabel(text)
            label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(label, index, 0)
            return label

        _row(0, "Preset:")
        self.preset_combo = QComboBox()
        for key, preset in self.available_presets().items():
            self.preset_combo.addItem(preset.label, key)
        self.preset_combo.setToolTip("Caption format for this folder")
        # The longest label ("MiniMax H3 — Official Prompt Structure") needs about
        # 300px; below that Qt elides it, so the tooltip carries the full name and
        # its blurb.
        self.preset_combo.setMinimumWidth(240)
        self.preset_combo.currentIndexChanged.connect(self._on_preset_changed)
        grid.addWidget(self.preset_combo, 0, 1)
        # Stills and clips need different captioning guidance, and in a mixed folder
        # you usually know which dataset you're building — so this is an explicit
        # choice rather than something inferred per file.
        self.media_label = _row(1, "Captioning:")
        self.media_combo = QComboBox()
        self.media_combo.addItem("Auto (follow file)", "auto")
        self.media_combo.addItem("Photos", "image")
        self.media_combo.addItem("Videos", "video")
        self.media_combo.setToolTip(
            "Which kind of dataset these captions are for. Changes the instructions "
            "sent to the model.")
        self.media_combo.setMinimumWidth(240)   # match the preset row's width
        self.media_combo.currentIndexChanged.connect(self._on_media_mode_changed)
        grid.addWidget(self.media_combo, 1, 1)

        # What this dataset trains. Sits beside the format preset because the two
        # are orthogonal: format decides the file's shape, the goal decides which
        # details the caption omits.
        _row(2, "Training:")
        self.goal_combo = QComboBox()
        for key, goal in self.available_goals().items():
            self.goal_combo.addItem(goal.label, key)
            self.goal_combo.setItemData(
                self.goal_combo.count() - 1, goal.summary, Qt.ToolTipRole)
        self.goal_combo.setMinimumWidth(240)
        self.goal_combo.currentIndexChanged.connect(self._on_goal_changed)
        grid.addWidget(self.goal_combo, 2, 1)

        self.preset_ext_label = QLabel("")
        self.preset_ext_label.setStyleSheet(
            f"color: {self.theme.text_secondary}; font-size: 11px;")
        grid.addWidget(self.preset_ext_label, 0, 2)
        outer = QWidget()
        col = QVBoxLayout(outer)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(2)
        col.addWidget(bar)
        # The quick reference: the omit/describe rule in one line, where you're
        # working, rather than buried in the system-prompt dialog.
        self.goal_hint = QLabel("")
        self.goal_hint.setObjectName("Hint")
        self.goal_hint.setWordWrap(True)
        self.goal_hint.setContentsMargins(12, 0, 12, 4)
        col.addWidget(self.goal_hint)
        return outer

    def _build_elements_tab(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Panel")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(4)

        self.elements_list = QListWidget()
        self.elements_list.setFixedHeight(150)
        self.elements_list.currentRowChanged.connect(self._on_element_row_changed)
        lay.addWidget(self.elements_list)

        btn_row = QHBoxLayout()
        self.el_duplicate_btn = QPushButton("Duplicate")
        self.el_duplicate_btn.clicked.connect(self._duplicate_element)
        self.el_remove_btn = QPushButton("Remove")
        self.el_remove_btn.clicked.connect(self._remove_element)
        btn_row.addWidget(self.el_duplicate_btn)
        btn_row.addWidget(self.el_remove_btn)
        lay.addLayout(btn_row)

        lay.addSpacing(6)
        self.el_editor = QWidget()
        ed = QVBoxLayout(self.el_editor)
        ed.setContentsMargins(0, 0, 0, 0)
        ed.setSpacing(4)

        self.el_type = QComboBox()
        self.el_type.addItems(["obj", "text"])
        self.el_type.currentTextChanged.connect(self._on_el_type_changed)
        ed.addWidget(self._field_label("Type"))
        ed.addWidget(self.el_type)

        ed.addWidget(self._field_label("Description"))
        self.el_desc = QPlainTextEdit()
        self.el_desc.setFixedHeight(76)
        self.el_desc.textChanged.connect(self._on_el_desc_changed)
        ed.addWidget(self._attach_expand(self.el_desc, "Description", with_tags=True))

        self.el_text_label = self._field_label("Text content")
        self.el_text = QLineEdit()
        self.el_text.textChanged.connect(self._mark_dirty)
        self.el_text_container = self._attach_expand(self.el_text, "Text content", single_line=True, with_tags=True)
        ed.addWidget(self.el_text_label)
        ed.addWidget(self.el_text_container)

        self.el_has_box = QCheckBox("Has bounding box")
        self.el_has_box.toggled.connect(self._on_has_box_changed)
        ed.addWidget(self.el_has_box)

        coords = QHBoxLayout()
        self.el_y1 = self._coord_spin()
        self.el_x1 = self._coord_spin()
        self.el_y2 = self._coord_spin()
        self.el_x2 = self._coord_spin()
        for tag, spin in (("y1", self.el_y1), ("x1", self.el_x1), ("y2", self.el_y2), ("x2", self.el_x2)):
            cell = QVBoxLayout()
            cell.setSpacing(2)
            cell.addWidget(self._field_label(tag))
            cell.addWidget(spin)
            coords.addLayout(cell)
        ed.addLayout(coords)

        hint = QLabel("Coordinates are 0–1000. Drag on the canvas in a later stage.")
        hint.setObjectName("Hint")
        hint.setWordWrap(True)
        ed.addWidget(hint)

        lay.addWidget(self.el_editor)
        lay.addStretch(1)
        self._set_element_editor_enabled(False)
        return w

    def _coord_spin(self) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(0, 1000)
        spin.setSingleStep(1)
        spin.valueChanged.connect(self._on_coord_changed)
        return spin

    def _build_canvas_toolstrip(self) -> QWidget:
        strip = QWidget()
        strip.setObjectName("ToolStrip")
        strip.setFixedWidth(40)
        lay = QVBoxLayout(strip)
        lay.setContentsMargins(5, 6, 5, 6)
        lay.setSpacing(4)
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        self._tool_buttons = {}
        for mode, tip in (
            ("select", "Select / move / resize"),
            ("draw", "Draw a new box for the selected element"),
            ("delete", "Delete a box"),
            ("pan", "Pan"),
        ):
            if mode == "delete":
                # a non-mode action button: add a centered box as a new obj element
                self._add_box_btn = QToolButton()
                self._add_box_btn.setToolTip("Add a centered bounding box (new object)")
                self._add_box_btn.setFixedSize(30, 30)
                self._add_box_btn.setIconSize(QSize(20, 20))
                self._add_box_btn.clicked.connect(self.add_bbox_element)
                lay.addWidget(self._add_box_btn)
            btn = QToolButton()
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setFixedSize(30, 30)
            btn.setIconSize(QSize(20, 20))
            if mode == "select":
                btn.setChecked(True)
            btn.clicked.connect(lambda _checked, m=mode: self.set_canvas_mode(m))
            self.tool_group.addButton(btn)
            lay.addWidget(btn)
            self._tool_buttons[mode] = btn
        self._refresh_tool_icons()
        return strip

    def set_canvas_mode(self, mode: str) -> None:
        self.view.set_mode(mode)
        self._set_status(f"Canvas: {mode}")

    def _reposition_toolstrip(self) -> None:
        ts = getattr(self, "_toolstrip", None)
        if ts is None:
            return
        ts.adjustSize()
        margin = 12
        y = max(margin, (self.view.height() - ts.height()) // 2)
        ts.move(margin, y)
        ts.raise_()

    def _build_nav_bar(self) -> QWidget:
        bar = QFrame()
        bar.setObjectName("NavBar")
        bar.setFixedHeight(38)
        lay = QHBoxLayout(bar)
        lay.setContentsMargins(12, 4, 12, 4)
        lay.setSpacing(8)

        pill = QFrame()
        pill.setObjectName("NavPill")
        pl = QHBoxLayout(pill)
        pl.setContentsMargins(4, 2, 4, 2)
        pl.setSpacing(2)
        prev_btn = QToolButton()
        prev_btn.setObjectName("NavBtn")
        prev_btn.setIcon(lucide_icon("chevron-left", self.theme.text_secondary, 16))
        prev_btn.setToolTip("Previous image (Ctrl+[)")
        prev_btn.clicked.connect(self.prev_image)
        self._nav_count = QLabel("0 / 0")
        self._nav_count.setObjectName("NavCount")
        self._nav_count.setAlignment(Qt.AlignCenter)
        self._nav_count.setMinimumWidth(56)
        next_btn = QToolButton()
        next_btn.setObjectName("NavBtn")
        next_btn.setIcon(lucide_icon("chevron-right", self.theme.text_secondary, 16))
        next_btn.setToolTip("Next image (Ctrl+])")
        next_btn.clicked.connect(self.next_image)
        pl.addWidget(prev_btn)
        pl.addWidget(self._nav_count)
        pl.addWidget(next_btn)

        self._zoom_label = QLabel("")
        self._zoom_label.setObjectName("Hint")
        self._zoom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._zoom_label.setMinimumWidth(44)

        lay.addStretch(1)
        lay.addWidget(pill)
        lay.addStretch(1)
        lay.addWidget(self._zoom_label)
        return bar

    def _update_zoom_label(self) -> None:
        label = getattr(self, "_zoom_label", None)
        if label is None:
            return
        if self.pixmap_item is None:
            label.setText("")
            return
        label.setText(f"{round(self.view.transform().m11() * 100)}%")

    def _activate_tool(self, mode: str) -> None:
        """Set the canvas mode and check the matching tool-strip button."""
        btn = self._tool_buttons.get(mode)
        if btn is not None:
            btn.setChecked(True)
        self.set_canvas_mode(mode)

    _TOOL_GLYPHS = {
        "select": "mouse-pointer-2",
        "draw": "square-dashed",
        "delete": "trash-2",
        "pan": "move",
        "plus": "square-plus",
    }

    def _tool_icon(self, mode: str, color: str) -> QIcon:
        """Lucide glyph for a canvas tool, recolored to the given token color."""
        return lucide_icon(self._TOOL_GLYPHS.get(mode, "square-dashed"), color, 20)

    def _refresh_tool_icons(self) -> None:
        if not hasattr(self, "_tool_buttons"):
            return
        color = self.theme.text_secondary
        for mode, btn in self._tool_buttons.items():
            btn.setIcon(self._tool_icon(mode, color))
        add_btn = getattr(self, "_add_box_btn", None)
        if add_btn is not None:
            add_btn.setIcon(self._tool_icon("plus", color))

    def apply_appearance(self, settings: CaptioningSettings) -> None:
        # Live in Qt: fonts + colors apply immediately, no restart (unlike Tk).
        app = QApplication.instance()
        if app is not None:
            font = QFont()
            if settings.ui_font_family:
                font.setFamily(settings.ui_font_family)
            font.setPointSize(settings.ui_font_size if settings.ui_font_size > 0 else 10)
            app.setFont(font)
        self.theme = Theme(settings)
        self.setStyleSheet(build_stylesheet(settings))
        self._refresh_tool_icons()

    def open_preferences(self, page: str | None = None) -> None:
        same = self.qsettings.value("bbox_same_as_caption", False, bool)
        dialog = PreferencesDialog(
            self, self.settings, bbox_same_as_caption=same, default_tags=self._default_tags
        )
        if page and isinstance(page, str):
            match = dialog.nav.findItems(page, Qt.MatchExactly)
            if match:
                dialog.nav.setCurrentRow(dialog.nav.row(match[0]))
        if dialog.exec() and dialog.result is not None:
            self._apply_preferences_result(dialog)

    def _apply_preferences_result(self, dialog) -> None:
        """Consume a PreferencesDialog's collected result and apply it live. Shared
        by the dialog's Save (on close) and Apply (without closing) actions."""
        if dialog.result is None:
            return
        self.settings = dialog.result
        self.qsettings.setValue("bbox_same_as_caption", dialog.bbox_same_as_caption)
        if dialog.tags_result is not None and dialog.tags_result != self._default_tags:
            self._default_tags = dialog.tags_result
            self._save_default_tags(self._default_tags)
            self._refresh_tags_used()
        try:
            path = save_settings(self.settings)
        except OSError as exc:
            QMessageBox.critical(self, "Preferences not saved", str(exc))
            return
        self.apply_appearance(self.settings)
        self._update_locate_button()
        monitor = getattr(self, "_server_monitor", None)
        if monitor is not None:
            monitor.update_target(self.settings.base_url, self.settings.api_key)
        self._set_status(f"Saved preferences to {path.name}.")

    def _build_ai_actions(self) -> QWidget:
        w = QWidget()
        w.setObjectName("Panel")
        lay = QVBoxLayout(w)
        lay.setContentsMargins(10, 10, 10, 8)
        lay.setSpacing(6)

        self.btn_run_captioning = QPushButton("Run JSON Captioning")
        self.btn_run_captioning.setObjectName("Primary")
        self.btn_run_captioning.setToolTip(
            "Generate the Ideogram JSON from the image. Choose one image or the whole folder."
        )
        self.btn_run_captioning.clicked.connect(self.run_json_captioning)
        self.btn_refine = QPushButton("Refine JSON")
        self.btn_refine.setToolTip(
            "Re-run the model over the current JSON using your refinement instructions "
            "(found in Preferences → Pipeline)."
        )
        self.btn_refine.clicked.connect(lambda: self.run_ai_job("refine"))
        self.btn_locate = QPushButton()
        self.btn_locate.clicked.connect(lambda: self.run_ai_job("bboxes"))
        self._update_locate_button()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setObjectName("Danger")
        self.btn_cancel.setToolTip(
            "Stop the running job. The in-flight request finishes before it halts."
        )
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_ai_job)

        self.ai_buttons = [self.btn_run_captioning, self.btn_refine, self.btn_locate]
        for b in self.ai_buttons:
            lay.addWidget(b)
        lay.addWidget(self.btn_cancel)

        self.chk_autosave = ToggleSwitch()
        self.chk_autosave.setChecked(False)
        self.chk_autosave.toggled.connect(self._on_autosave_toggled)
        lay.addWidget(self._toggle_row(
            "Autosave (no confirmation)", self.chk_autosave,
            "Write edits to disk immediately instead of buffering them until you save.",
        ))
        return w

    def _on_autosave_toggled(self, checked: bool) -> None:
        self._autosave = checked
        self.qsettings.setValue("autosave", checked)
        if checked:
            # flush any buffered edits now, and keep autosave on from here
            self.save_all()
            self._set_status("Autosave on — edits save without confirmation.")
        else:
            self._set_status("Autosave off — edits are buffered until you save.")

    def _restore_autosave_pref(self) -> None:
        saved = self.qsettings.value("autosave", False, bool)
        self.chk_autosave.blockSignals(True)
        self.chk_autosave.setChecked(bool(saved))
        self.chk_autosave.blockSignals(False)
        self._autosave = bool(saved)

    def _update_locate_button(self) -> None:
        """Label/tooltip reflect the bbox mode: fill-missing (default) vs overwrite-all."""
        if getattr(self.settings, "overwrite_bboxes", False):
            self.btn_locate.setText("Regenerate all boxes")
            self.btn_locate.setToolTip(
                "Re-locate boxes for every described element, replacing existing ones "
                "(Overwrite existing boxes is ON in Preferences → Pipeline). Generate JSON first."
            )
        else:
            self.btn_locate.setText("Locate missing boxes")
            self.btn_locate.setToolTip(
                "Find boxes for described elements that don't have one yet; existing boxes are kept "
                "(toggle Overwrite existing boxes in Preferences → Pipeline). Generate JSON first."
            )

    def _set_ai_running(self, running: bool) -> None:
        self._job_running = running
        for b in self.ai_buttons:
            b.setEnabled(not running)
        self.btn_cancel.setEnabled(running)
        # Any AI job (single or batch) freezes the caption/elements fields and the
        # canvas so an in-flight result can't be clobbered, and boxes can't be moved.
        self._set_read_only(running)

    def run_ai_job(self, operation: str) -> None:
        if self.current is not None and is_video(self.current):
            if operation in ("plain", "plain_video") and self.preset.is_plain:
                operation = "plain_video"
            else:
                # Structured Ideogram JSON (and refine/bbox passes) are image
                # pipelines; a clip has no single layout to ground boxes against.
                QMessageBox.information(
                    self, "Video captioning",
                    "Videos caption with the plain-text presets (Plain text or the "
                    "MiniMax H3 presets), which sample frames across the clip. The "
                    "Ideogram JSON preset is image-only.")
                return
        if self._job_running:
            return
        if self.store is None or self.current is None:
            self._set_status("Open a folder and select a file first.")
            return
        # flush pending edits so the operation works on the latest caption
        self.commit_caption_fields()
        self.commit_element_fields()
        self.persist_guidance_if_dirty()
        caption_copy = copy.deepcopy(self.current_caption)

        # Guidance applies to every operation that *generates* a caption — the
        # structured JSON pass and the plain-text ones alike. (Refine and bbox
        # passes take their instructions from elsewhere, so they stay excluded.)
        # This list originally held only "json_image"; the plain ops were added
        # later and silently got no guidance at all.
        generates_caption = operation in ("json_image", "plain", "plain_video")
        guidance = (self.project.resolved_for(self.current.name)
                    if generates_caption else "")
        # Convert mode: feed this image's .txt sidecar (if any) as the source caption.
        # Running the image always overwrites the in-editor caption, so no extra
        # confirmation is needed for a single run.
        source_caption = ""
        if operation == "json_image" and self._image_uses_source(self.current):
            source_caption = self.store.load_source_text(self.current)
        self._job_operation = operation
        self._job_guidance = guidance
        if generates_caption:
            # Stamped onto the caption so "guidance changed since this was
            # generated" stays accurate for plain presets too.
            self._job_guidance_folder = self.project.effective_folder_guidance()
            self._job_guidance_image = self.project.effective_image_guidance(self.current.name)
        else:
            self._job_guidance_folder = ""
            self._job_guidance_image = ""
        job_settings = self.settings
        if self.project.creative_json is not None:
            job_settings = replace(self.settings, creative_json=self.project.creative_json)
        image_path = self.current
        self._preflight_server_or_warn(
            lambda: self._ensure_local_binary_then(
                lambda: self._start_ai_job(operation, job_settings, caption_copy, guidance, image_path, source_caption)
            ),
            batch=False,
        )

    def _start_ai_job(self, operation, job_settings, caption_copy, guidance, image_path, source_caption="") -> None:
        if not self._ensure_model_configured():
            return
        if getattr(self, "_force_autostart", False):
            job_settings = replace(job_settings, auto_start_server=True)
            self._force_autostart = False
        if not self._confirm_model_download():
            self._set_status("Cancelled.")
            return
        self._job_cancelled = False
        self._set_ai_running(True)
        self._set_job_progress(f"Running {operation}…", busy=True)

        thread = AiJobThread(
            operation=operation,
            settings=job_settings,
            image_path=image_path,
            caption=caption_copy,
            guidance=guidance,
            source_caption=source_caption,
            instructions=self.settings.json_refine_instructions,
            system_prompt=self.effective_system_prompt(),
        )
        if operation == "plain_video":
            # Caption the span shown in the trim bar, so what's described is what
            # an Apply would keep. A full-clip selection is simply (0, duration).
            stage = getattr(self, "video_stage", None)
            if stage is not None and stage._path == image_path and stage._slider.duration() > 0:
                in_ms, out_ms = stage._slider.trim()
                thread.span = (in_ms / 1000.0, out_ms / 1000.0)
            thread.frame_count = max(2, int(getattr(
                self.settings, "video_caption_frames", 6)))
            thread.include_audio = self.audio_captioning_enabled()
        thread.progress.connect(self._on_job_progress)
        thread.done.connect(self._on_job_done)
        thread.error.connect(self._on_job_error)
        thread.finished.connect(self._on_job_finished)
        thread.server_started.connect(self._on_server_started)
        self._ai_thread = thread
        thread.start()

    def _preflight_server_or_warn(self, proceed, *, batch: bool) -> None:
        """Before a run, check the server is usable and, if not, show one tailored
        notice instead of letting it fail mid-request. Calls proceed() when the
        server is up (or the user opts to try anyway)."""
        settings = self.settings
        mode = settings.server_start_mode
        if mode == "local":
            running = self._server_is_running()
            model_less = running and getattr(self, "_server_modelless", False)
            if running and not model_less:
                proceed()
                return
            binary = find_llama_server()
            configured = binary is not None and has_model_config(settings, "caption")
            if not configured:
                box = QMessageBox(self)
                box.setIcon(QMessageBox.Information)
                box.setWindowTitle("No captioning server configured")
                box.setText(
                    "There's nothing set up to generate captions yet. Open Preferences to "
                    "pick a built-in llama.cpp model, or point at a server you already run."
                )
                prefs = box.addButton("Open Preferences", QMessageBox.AcceptRole)
                box.addButton(QMessageBox.Cancel)
                box.exec()
                if box.clickedButton() is prefs:
                    self.open_preferences("Connection/Server")
                return
            # Configured but not running (or up without a model loaded).
            count = len(self.images) if (batch and self.images) else 0
            tail = f" and caption all {count} files?" if count else "?"
            relaunch = " (it's running without a model, so it needs to reload)" if model_less else ""
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Start the captioning server?")
            box.setText(
                f"Captioning uses the built-in llama.cpp server, but it isn't ready yet{relaunch}. "
                f"Start it and load the captioning model{tail}\n\n"
                "The model loads into VRAM — make sure enough is free."
            )
            start = box.addButton("Start && caption", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(start)
            box.exec()
            if box.clickedButton() is start:
                # Honour the start even if auto-start is off, for this run only. The
                # worker's job_settings was captured before this gate, so flag it and
                # let _start_ai_job/_start_batch_job apply the override.
                if not settings.auto_start_server:
                    self._force_autostart = True
                proceed()
            return
        # Remote / custom server.
        if not (settings.base_url or "").strip():
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Information)
            box.setWindowTitle("No captioning server configured")
            box.setText(
                "No server address is configured. Open Preferences to set the server URL "
                "and the model to request, or switch to the built-in llama.cpp server."
            )
            prefs = box.addButton("Open Preferences", QMessageBox.AcceptRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            if box.clickedButton() is prefs:
                self.open_preferences("Connection/Server")
            return
        if self._server_reachable:
            proceed()
            return
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("Server not responding")
        box.setText(
            f"A remote captioning server is configured ({settings.base_url}) but isn't "
            "responding right now. Make sure it's running, has the model from your Model "
            "preferences loaded, and is accepting connections — then try again."
        )
        anyway = box.addButton("Run anyway", QMessageBox.AcceptRole)
        prefs = box.addButton("Open Preferences", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(prefs)
        box.exec()
        clicked = box.clickedButton()
        if clicked is anyway:
            proceed()
        elif clicked is prefs:
            self.open_preferences("Connection/Server")

    def _ensure_local_binary_then(self, proceed) -> None:
        """Pre-flight for local mode: if we're set to auto-launch a local server but
        have no binary, offer to fetch one first and continue on success. Otherwise
        proceed immediately."""
        settings = self.settings
        if settings.server_start_mode != "local" or not settings.auto_start_server:
            proceed()
            return
        if find_llama_server() is not None:
            proceed()
            return
        # No binary yet — installing happens in Preferences (with progress), not
        # silently from here. Send the user there rather than starting a download.
        self._set_status("No local server is set up yet \u2014 install llama.cpp in Settings.")
        self.open_preferences("Connection/Server")

    def run_json_captioning(self) -> None:
        if self._job_running:
            return
        if self.store is None or self.current is None:
            self._set_status("Open a folder and select a file first.")
            return
        if self.current is not None and is_video(self.current) and not self.preset.is_plain:
            QMessageBox.information(
                self, "Video captioning",
                "The Ideogram JSON preset is image-only. Switch to Plain text or a "
                "MiniMax H3 preset to caption clips.")
            return
        plain = self.preset.is_plain
        noun = "File" if plain else "Image"
        box = QMessageBox(self)
        box.setWindowTitle("Run Captioning" if plain else "Run JSON Captioning")
        box.setText(f"Caption the current {noun.lower()}, or the whole folder?")
        if plain and self.current is not None and is_video(self.current):
            frames = max(2, int(getattr(self.settings, "video_caption_frames", 6)))
            audio_ok, why = self.audio_status()
            if audio_ok:
                box.setInformativeText(
                    f"Clips are captioned from {frames} sampled frames plus the "
                    "clip's audio, so dialogue and sound can be described.")
            else:
                box.setInformativeText(
                    f"Clips are captioned from {frames} sampled frames only \u2014 "
                    f"no audio.\n\n{why}")
        single_btn = box.addButton(f"Caption Single {noun}", QMessageBox.AcceptRole)
        all_btn = box.addButton(f"Caption All {noun}s", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is single_btn:
            self.run_ai_job("plain" if plain else "json_image")   # video re-routes inside
        elif clicked is all_btn:
            self.run_batch_caption()

    def _image_has_caption(self, path: Path) -> bool:
        if self.store is None:
            return False
        try:
            cp = self.store.caption_path(path)
            return cp.exists() and cp.stat().st_size > 2
        except OSError:
            return False

    def _set_filmstrip_locked(self, locked: bool) -> None:
        """Freeze image navigation while a batch runs so the selection can't move
        out from under the per-file reloads."""
        self._nav_locked = locked
        self.filmstrip.setEnabled(not locked)

    def _set_panel_editable(self, editable: bool) -> None:
        """Make the caption/elements fields read-only (but still readable and tab-
        switchable) rather than disabling the whole tab widget. Programmatic reloads
        still populate read-only fields, so a completing batch item can refresh them."""
        ro = not editable
        for f in (self.cap_high_level, self.cap_background, self.cap_aesthetics,
                  self.cap_lighting, self.cap_medium, self.cap_style_detail,
                  self.el_desc, self.el_text,
                  self.el_y1, self.el_x1, self.el_y2, self.el_x2):
            f.setReadOnly(ro)
        for c in (self.style_mode, self.el_type):
            c.setEnabled(editable)
        self.el_has_box.setEnabled(editable)
        self.el_duplicate_btn.setEnabled(editable)
        self.el_remove_btn.setEnabled(editable)

    def _set_canvas_locked(self, locked: bool) -> None:
        """Freeze box editing on the canvas: boxes can't be moved (flag cleared),
        resized, drawn, or deleted, and the tool strip is disabled. Selecting a box to
        view it (and panning/zooming) still works."""
        for it in getattr(self, "box_items", []):
            it.setFlag(QGraphicsItem.ItemIsMovable, not locked)
        ts = getattr(self, "_toolstrip", None)
        if ts is not None:
            ts.setEnabled(not locked)

    def _set_read_only(self, on: bool) -> None:
        """Batch read-only mode: navigation stays live (review as captions land), but
        the caption/elements fields and the canvas are frozen so a completing item
        can't clobber edits — while tabs stay switchable for read-only review."""
        self._read_only = on
        self._set_panel_editable(not on)
        self._set_canvas_locked(on)
        if hasattr(self, "crop_action"):
            self.crop_action.setEnabled(not on and self.current is not None)
        if hasattr(self, "batch_resize_action"):
            self.batch_resize_action.setEnabled(not on)
        if hasattr(self, "_readonly_banner"):
            self._readonly_banner.setVisible(on)

    def _server_is_running(self) -> bool:
        proc = getattr(self, "_server_proc", None)
        return proc is not None and proc.poll() is None

    def _show_server_popover(self) -> None:
        if self._server_popover is None:
            self._server_popover = ServerPopover(
                self.theme,
                on_settings=lambda: self.open_preferences("Connection/Server"),
                on_start=self._start_local_server,
                on_stop=self._stop_local_server,
                on_start_nomodel=self._start_local_server_no_model,
                parent=self,
            )
        ok = getattr(self, "_server_reachable", None)
        local = self.settings.server_start_mode == "local"
        running = self._server_is_running()
        binary = find_llama_server() if local else None
        ready = local and (binary is not None) and has_model_config(self.settings, "caption")
        show_startstop = running or ready
        # Offer a model-less launch when the build supports router mode (cached probe).
        show_nomodel = (local and not running and binary is not None
                        and llama_server_supports_router(binary))
        if running and getattr(self, "_server_modelless", False):
            dot, text = "#3ddc84", "Server up \u2014 no model loaded"
        elif local and not ready and not running:
            dot, text = "#9aa4b6", "No server configured"
        elif ok is None:
            dot, text = "#9aa4b6", "Checking server\u2026"
        elif ok:
            dot, text = "#3ddc84", "Server connected"
        else:
            dot, text = "#ff5a52", "Server offline"
        status_html = (f'<span style="color:{dot}">\u25cf</span> '
                       f'<span style="color:#c8cdd6">{text}</span>')
        self._server_popover.configure(
            status_html=status_html, show_startstop=show_startstop,
            running=running, show_nomodel=show_nomodel,
        )
        self._server_popover.show_above(self._server_status_label)

    def _start_local_server_no_model(self) -> None:
        """Launch the server with no model resident (router mode) — a quick check
        that the binary runs and the server answers, with no download."""
        self._launch_local_server(model_less=True)

    def _start_local_server(self) -> None:
        """Bring the local server up on demand. With no model configured, send the
        user to the Models page rather than failing with a server error."""
        if not self._ensure_model_configured():
            return
        # acquire a binary first if we don't have one, then launch
        self._ensure_local_binary_then(self._launch_local_server)

    def _ensure_model_configured(self) -> bool:
        """True when a model is set for captioning. In local mode with nothing
        configured, show a popup that takes the user to Model settings (instead of
        a download prompt for a model they never chose), and return False. In
        existing/custom-server mode the loaded model is the server's concern, so
        this never blocks."""
        if self.settings.server_start_mode != "local":
            return True
        if has_model_config(self.settings, "caption"):
            return True
        QMessageBox.information(
            self, "No model set yet",
            "No captioning model is configured yet.\n\nOpen Model settings to pick "
            "a model (or point at one you've already downloaded), then start again.",
        )
        self.open_preferences("LLM Models")
        return False

    def _confirm_model_download(self) -> bool:
        """Nothing should download without a yes. If launching the configured model
        would fetch files from Hugging Face, confirm first (naming the model so it's
        clearly the one you set). Returns True to proceed."""
        settings = self.settings
        if settings.server_start_mode != "local" or not settings.auto_start_server:
            return True
        try:
            missing = missing_model_files(settings, "caption")
        except Exception:
            missing = []
        if not missing:
            return True
        label = (profile_label_from_id("caption", settings.caption_profile_id) or "").strip()
        if label.lower().startswith("download:"):
            label = label.split(":", 1)[1].strip()
        name = label or "the selected model"
        listing = "\n".join(f"  \u2022 {fn}" for fn in missing)
        # Name the destination. Without it you can't tell whether a download went
        # to your models folder or the shared Hugging Face cache, which is exactly
        # the confusion that makes a finished download look like it did nothing.
        target = getattr(settings, "model_download_target", "") or ""
        if target == MODEL_TARGET_HF:
            where = ("the shared Hugging Face cache\n(~/.cache/huggingface/hub, or "
                     "wherever HF_HOME points)")
        else:
            where = str(Path(settings.models_dir).expanduser())
        box = QMessageBox(self)
        box.setWindowTitle("Download model files?")
        box.setIcon(QMessageBox.Question)
        box.setText(f"Starting the server will fetch {name} from Hugging Face.")
        box.setInformativeText(
            f"These files aren't downloaded yet:\n\n{listing}\n\n"
            f"They'll be saved to:\n{where}\n\n"
            "Change this with 'Model download location' on the LLM Models page.")
        yes = box.addButton("Download", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(yes)
        box.exec()
        return box.clickedButton() is yes

    def _launch_local_server(self, model_less: bool = False) -> None:
        if self._server_is_running():
            self._set_status("Local server is already running.")
            return
        if not model_less and not self._confirm_model_download():
            self._set_status("Server start cancelled.")
            return
        self._server_modelless_pending = model_less
        self._set_job_progress(
            "Starting server (no model)\u2026" if model_less else "Starting local server\u2026",
            busy=True,
        )
        self._server_thread = LlamaServerThread(self.settings, model_less=model_less)
        self._server_thread.progress.connect(self._set_status)
        self._server_thread.started_proc.connect(self._on_local_server_launched)
        self._server_thread.error.connect(self._on_local_server_error)
        self._server_thread.start()

    def _on_local_server_launched(self, proc) -> None:
        self._set_job_progress("")
        if proc is not None:
            self._on_server_started(proc)
            self._server_modelless = getattr(self, "_server_modelless_pending", False)
            self._set_status("Local server started (no model loaded)."
                             if self._server_modelless else "Local server started.")
        else:
            self._set_status("A server is already running.")

    def _on_local_server_error(self, message: str) -> None:
        self._set_job_progress("")
        if self._maybe_offer_launch_rollback(message):
            return
        QMessageBox.warning(self, "Couldn't start server", message)

    def _stop_local_server(self) -> None:
        if self._server_is_running():
            self._shutdown_server()
            self._server_modelless = False
            self._set_status("Local server stopped.")
        else:
            self._set_status("No local server is running.")

    def _on_server_started(self, proc) -> None:
        """A job launched a local llama-server; hold the handle so we can shut it
        down on exit. Replaces (and stops) any earlier handle we were tracking."""
        if self._server_proc is not None and self._server_proc is not proc:
            stop_server_process(self._server_proc)
        self._server_proc = proc

    # ---- managed llama.cpp: background update check + acquire flow -----------

    def _maybe_check_llama_update(self) -> None:
        """Once-a-day, metadata-only check for a newer build of the binary we have
        installed. Gated on the user's toggle; silent and best-effort."""
        if not getattr(self.settings, "llama_auto_update_check", True):
            return
        record = read_installed_llama()
        if record is None or not record.source:
            return  # nothing installed -> nothing to compare against
        last = self.qsettings.value("llama_latest_check_ts")
        try:
            last_ts = float(last) if last is not None else 0.0
        except (TypeError, ValueError):
            last_ts = 0.0
        if time.time() - last_ts < 24 * 3600:
            return
        self._llama_check_thread = LlamaUpdateCheckThread(record.source)
        self._llama_check_thread.result.connect(self._on_llama_update_checked)
        self._llama_check_thread.start()

    def _on_llama_update_checked(self, build: int) -> None:
        self.qsettings.setValue("llama_latest_check_ts", time.time())
        if build and build > 0:
            self.qsettings.setValue("llama_latest_build", int(build))

    def run_batch_caption(self) -> None:
        if self._job_running or self.store is None or not self.images:
            return
        total = len(self.images)
        # Commit live guidance first so "changed since last caption" is accurate.
        self.commit_guidance()
        if not self.preset.is_plain:
            videos = [img for img in self.images if is_video(img)]
            if videos and len(videos) == total:
                QMessageBox.information(
                    self, "Caption all files",
                    "This folder contains only videos, and the Ideogram JSON preset "
                    "is image-only. Switch to Plain text or a MiniMax H3 preset to "
                    "caption clips.")
                return
            if videos:
                self._set_status(
                    f"{len(videos)} video(s) will be skipped \u2014 the Ideogram JSON "
                    "preset is image-only.")
        already = [img for img in self.images if self._image_has_caption(img)]
        already_set = set(already)
        new_imgs = [img for img in self.images if img not in already_set]
        stale = [img for img in already if self.project.guidance_changed(img.name)]
        # Bypassed files are out of the dataset, so they're out of the batch —
        # captioning them individually is still allowed.
        work = [img for img in self.images if not self.store.is_bypassed(img)]
        if not self.preset.is_plain:
            work = [img for img in work if not is_video(img)]
        total = len(work)
        convert_note = ""
        if self.project.convert_txt_to_json:
            with_txt = sum(1 for img in self.images if self.store.has_source_text(img))
            omitted = sum(1 for img in self.images
                          if self.store.has_source_text(img)
                          and self.project.is_convert_omitted(img.name))
            using = with_txt - omitted
            convert_note = (
                f"Convert mode: {using} of {total} file(s) will use a matching .txt "
                "source caption; the rest fall back to image-only captioning."
            )
            if omitted:
                convert_note += f" ({omitted} with a .txt marked media-only.)"
        if already:
            box = QMessageBox(self)
            box.setWindowTitle("Caption all files")
            msg = f"{len(already)} of {total} image(s) already have a caption."
            if stale:
                msg += (f"\n{len(stale)} of those have guidance changes since they "
                        "were last captioned.")
            if convert_note:
                msg += "\n\n" + convert_note
            box.setText(msg)
            box.setInformativeText("What would you like to run?")
            new_btn = box.addButton(f"Only new ({len(new_imgs)})", QMessageBox.AcceptRole)
            changed_btn = None
            if stale:
                changed_btn = box.addButton(
                    f"Changed + new ({len(stale) + len(new_imgs)})", QMessageBox.AcceptRole
                )
            all_btn = box.addButton("Re-caption all", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(changed_btn or new_btn)
            box.exec()
            clicked = box.clickedButton()
            if clicked is None or box.buttonRole(clicked) == QMessageBox.RejectRole:
                return
            if clicked is new_btn:
                work = list(new_imgs)
            elif changed_btn is not None and clicked is changed_btn:
                wanted = set(new_imgs) | set(stale)
                work = [img for img in self.images if img in wanted]   # keep folder order
            elif clicked is all_btn:
                work = list(self.images)
            if not work:
                self._set_status("Nothing to do.")
                return
        else:
            extra = ("\n\n" + convert_note) if convert_note else ""
            resp = QMessageBox.question(
                self, "Caption all files",
                f"Generate JSON for all {total} image(s)?\n\n"
                "Images are processed one at a time through your configured server."
                + extra,
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if resp != QMessageBox.Yes:
                return
        # flush buffered edits and pending guidance before the run starts
        self.commit_caption_fields()
        self.commit_element_fields()
        self.persist_guidance_if_dirty()
        job_settings = self.settings
        if self.project.creative_json is not None:
            job_settings = replace(self.settings, creative_json=self.project.creative_json)
        items = [
            (img, self.project.resolved_for(img.name),
             self.store.load_source_text(img) if self._image_uses_source(img) else "")
            for img in work
        ]
        # remember the guidance actually sent, to stamp each caption on completion
        self._batch_guidance = {str(img): g for img, g, _sc in items}
        # the same, split by scope, so a later change can be attributed folder vs per-file
        self._batch_guidance_folder = {
            str(img): self.project.effective_folder_guidance() for img in work}
        self._batch_guidance_image = {
            str(img): self.project.effective_image_guidance(img.name) for img in work}
        # health/dup tracking for this run: serialized caption -> first filename seen
        self._batch_caption_hashes = {}
        self._batch_flagged = {}
        n = len(items)
        delay = int(self.qsettings.value("batch_delay_ms", 0, int) or 0)
        self._preflight_server_or_warn(
            lambda: self._ensure_local_binary_then(
                lambda: self._start_batch_job(job_settings, items, n, delay)
            ),
            batch=True,
        )

    def _start_batch_job(self, job_settings, items, n, delay) -> None:
        if not self._ensure_model_configured():
            return
        if getattr(self, "_force_autostart", False):
            job_settings = replace(job_settings, auto_start_server=True)
            self._force_autostart = False
        if not self._confirm_model_download():
            self._set_status("Cancelled.")
            return
        self._job_cancelled = False
        self._batch_abort_shown = False
        self._set_ai_running(True)
        self._set_job_progress(f"Captioning 0/{n}…", value=0, total=n)
        thread = BatchCaptionThread(
            job_settings, items, delay_ms=delay,
            system_prompt=self.effective_system_prompt() if self.preset.is_plain else "")
        thread.frame_count = max(2, int(getattr(self.settings, "video_caption_frames", 6)))
        thread.include_audio = self.audio_captioning_enabled()
        thread.item_progress.connect(self._on_batch_progress)
        thread.item_done.connect(self._on_batch_item_done)
        thread.item_error.connect(self._on_batch_item_error)
        thread.batch_finished.connect(self._on_batch_finished)
        thread.server_started.connect(self._on_server_started)
        self._ai_thread = thread
        thread.start()

    def _on_batch_progress(self, idx: int, total: int, message: str) -> None:
        if not self._job_cancelled:
            self._set_job_progress(message, value=idx, total=total)

    def _on_batch_item_done(self, image_path_str: str, caption: object) -> None:
        if self.store is None:
            return
        path = Path(image_path_str)
        try:
            if self.preset.is_plain:
                self.store.save_plain_caption(path, self.postprocess_caption(str(caption)))
            else:
                self.store.save_caption(path, caption)
        except Exception as exc:
            self._set_status(f"Save failed for {path.name}: {exc}")
            return
        self._pending.pop(image_path_str, None)
        guidance = getattr(self, "_batch_guidance", {}).get(
            image_path_str, self.project.resolved_for(path.name)
        )
        folder_part = getattr(self, "_batch_guidance_folder", {}).get(
            image_path_str, self.project.effective_folder_guidance())
        image_part = getattr(self, "_batch_guidance_image", {}).get(
            image_path_str, self.project.effective_image_guidance(path.name))
        self.project.mark_generated(path.name, guidance, folder_part, image_part)   # persisted at batch end
        # Schema health is structured-only, but the duplicate-caption check catches
        # context bleed in both formats, so it stays.
        if self.preset.is_plain:
            issues = []
            key = str(caption).strip() or None
        else:
            issues = caption_health(caption)
            try:
                key = serialize_caption(caption)
            except Exception:
                key = None
        if key:
            prior = getattr(self, "_batch_caption_hashes", {}).get(key)
            if prior and prior != path.name:
                issues = issues + [f"identical caption to {prior} (possible context bleed)"]
            else:
                self._batch_caption_hashes[key] = path.name
        self.project.set_flags(path.name, issues)
        if issues:
            getattr(self, "_batch_flagged", {})[path.name] = issues
        self._refresh_thumb_marker(path)
        # Live-refresh the view if we're sitting on this image — but never clobber
        # unsaved edits (e.g. edits made before the batch was launched).
        if self.current is not None and str(self.current) == image_path_str and not self._dirty:
            self.load_caption_for(self.current)
            self._dirty = False

    def _on_batch_item_error(self, image_path_str: str, message: str) -> None:
        sev, text = self._diagnose_run_failure(message)
        self._set_status(f"Failed {Path(image_path_str).name}: {text}")
        # A dead/OOM'd server fails every remaining image identically — stop the run
        # and say why, once, instead of grinding through the whole folder.
        if sev == "fatal_server" and not getattr(self, "_batch_abort_shown", False):
            self._batch_abort_shown = True
            if self._ai_thread is not None:
                self._ai_thread.requestInterruption()
            QMessageBox.critical(
                self, "Server stopped — batch halted",
                f"{text}\n\nThe rest of the batch was stopped so it doesn't fail every remaining image.")

    def _on_batch_finished(self, success: int, fail: int, cancelled: bool) -> None:
        self._set_ai_running(False)
        self._ai_thread = None
        # persist the per-file guidance stamps gathered during the run
        if self.store is not None:
            try:
                self.store.save_project(self.project)
            except OSError:
                pass
        self._refresh_stale_state()
        if cancelled:
            self._set_job_progress(f"Batch cancelled — {success} captioned, {fail} failed.")
        else:
            self._set_job_progress(f"Batch complete — {success} captioned, {fail} failed.")
        flagged = getattr(self, "_batch_flagged", {})
        if fail and not cancelled:
            QMessageBox.warning(self, "Batch finished", f"{success} captioned, {fail} failed. See status for the last error.")
        if flagged:
            lines = []
            for name in sorted(flagged):
                lines.append(f"• {name}\n    – " + "\n    – ".join(flagged[name]))
            n = len(flagged)
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Warning)
            box.setWindowTitle("Captions flagged for review")
            box.setText(f"{n} caption{'s' if n != 1 else ''} may be corrupt or off-schema "
                        f"(red dot on the thumbnail). These were still saved — review and re-run as needed.")
            box.setDetailedText("\n".join(lines))
            box.exec()

    def _on_job_progress(self, message: str) -> None:
        if not self._job_cancelled:
            self._set_job_progress(message, busy=True)

    def _on_job_done(self, caption: object) -> None:
        if self._job_cancelled or self.store is None or self.current is None:
            return
        try:
            if self.preset.is_plain:
                # A plain preset's caption is text; save_caption would serialise it
                # as JSON into the .txt sidecar.
                self.store.save_plain_caption(self.current, self.postprocess_caption(str(caption)))
            else:
                self.store.save_caption(self.current, caption)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save result", str(exc))
            return
        # the AI result is now on disk; drop any buffered edits and reload it
        self._pending.pop(str(self.current), None)
        if getattr(self, "_job_operation", "") in ("json_image", "plain", "plain_video"):
            self.project.mark_generated(
                self.current.name, getattr(self, "_job_guidance", ""),
                getattr(self, "_job_guidance_folder", ""),
                getattr(self, "_job_guidance_image", ""))
            # Schema health only means something for the structured preset.
            if not self.preset.is_plain:
                self.project.set_flags(self.current.name, caption_health(caption))
            try:
                self.store.save_project(self.project)
            except OSError:
                pass
        self.load_caption_for(self.current)
        self._dirty = False
        self._refresh_thumb_marker(self.current)
        self._refresh_guidance_changes()
        self._set_job_progress("AI job complete.")

    def _diagnose_run_failure(self, message: str):
        """Map a raw job error to (severity, user_text). severity 'fatal_server'
        means the server died/OOM'd (so a batch should stop); '' means pass through."""
        low = message.lower()
        proc = getattr(self, "_server_proc", None)
        local = self.settings.server_start_mode == "local"
        log = server_log_path(self.settings)
        # Confirmed crash of the server we launched.
        if proc is not None and proc.poll() is not None:
            cat, hint = diagnose_server_log(log)
            if hint:
                if cat == "oom":
                    hint = hint + " " + BUILTIN_OOM_HINT
                return "fatal_server", (
                    f"The built-in llama.cpp server stopped during the run. {hint}\n\nLog: {log}")
            return "fatal_server", (
                f"The built-in llama.cpp server crashed during the run (exit {proc.returncode}). "
                f"The log should have the cause.\n\nLog: {log}")
        # Connection lost mid-request (server crashed/closed/hung, or remote went away).
        looks_conn = ("connection" in low or "stopped responding" in low
                      or "did not become ready" in low or "remote end closed" in low
                      or "incomplete" in low or "broken pipe" in low)
        if looks_conn:
            if local:
                cat, hint = diagnose_server_log(log)
                if cat == "oom":
                    return "fatal_server", (
                        f"The built-in server ran out of VRAM and dropped the connection. "
                        f"{hint} {BUILTIN_OOM_HINT}")
                if hint:
                    return "fatal_server", (
                        f"Lost the connection to the built-in server. {hint}\n\nLog: {log}")
                return "fatal_server", (
                    "Lost the connection to the built-in server mid-request — it may have crashed "
                    f"or run out of VRAM.\n\nLog: {log}")
            return "fatal_server", (
                "The captioning server stopped responding — it may have crashed, run out of memory, "
                "or closed the connection. Make sure it's still running with the right model loaded, "
                "then try again.")
        return "", message

    def _on_job_error(self, message: str) -> None:
        if self._job_cancelled:
            return
        _sev, text = self._diagnose_run_failure(message)
        QMessageBox.critical(self, "AI job failed", text)
        self._set_job_progress("AI job failed.")
        if self._maybe_offer_launch_rollback(message):
            return
        self._maybe_offer_arch_update(message)

    def _maybe_offer_launch_rollback(self, message: str) -> bool:
        """If a just-launched server failed to come up and we have a backup binary,
        offer to roll back to it (a freshly-installed build that won't start)."""
        if not has_llama_backup():
            return False
        low = message.lower()
        if "did not become ready" in low or "exited during startup" in low:
            roll = QMessageBox.question(
                self, "Server didn't start",
                f"The llama-server didn't start:\n\n{message}\n\n"
                "Roll back to the previously installed build?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if roll == QMessageBox.Yes and rollback_llama():
                self._set_status("Rolled back to the previous llama.cpp build.")
                return True
        return False

    def _maybe_offer_arch_update(self, message: str) -> None:
        """If a job failed because the model needs a newer llama.cpp, offer to
        update — the strongest update signal, surfaced exactly when it matters."""
        if not is_model_arch_error(message):
            return
        record = read_installed_llama()
        if record is None:
            return  # not using a managed binary; nothing we can update
        build = f"b{record.build}" if record.build else "your build"
        resp = QMessageBox.question(
            self, "Update llama.cpp?",
            "This model looks like it needs a newer llama.cpp than your installed "
            f"build ({build}). Open Settings to update it?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if resp == QMessageBox.Yes:
            self.open_preferences("Connection/Server")

    def _on_job_finished(self) -> None:
        self._set_ai_running(False)
        if self._job_cancelled:
            self._set_job_progress("AI job cancelled.")
        self._ai_thread = None

    def cancel_ai_job(self) -> None:
        if self._ai_thread is not None and self._job_running:
            self._job_cancelled = True
            self._ai_thread.requestInterruption()
            self._set_job_progress("Cancelling… (current request will finish)", busy=True)

    def _build_body(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self.top_bar)

        splitter = QSplitter(Qt.Horizontal)

        # Left: folder + per-file guidance.
        self.left_panel = self._build_guidance_panel()
        self.left_panel.setMinimumWidth(190)

        # Center: tool strip + image/bbox editor view.
        self.scene = QGraphicsScene(self)
        self.view = CanvasView(self.scene, self)
        self.view.setObjectName("Stage")
        self.view.setFrameShape(QFrame.NoFrame)
        self.pixmap_item: QGraphicsPixmapItem | None = None

        center = QWidget()
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(0)
        # Images and videos each get a full-size stage in the same slot, so video
        # controls are inline in the main window rather than behind a popup.
        self.video_stage = VideoStage(self)
        self.center_stack = QStackedWidget()
        # The image page carries its edit controls beneath the canvas, matching the
        # video stage — editing a photo shouldn't mean hunting for a modal when
        # editing a clip is inline.
        image_page = QWidget()
        image_lay = QVBoxLayout(image_page)
        image_lay.setContentsMargins(0, 0, 0, 0)
        image_lay.setSpacing(0)
        image_lay.addWidget(self.view, 1)
        image_lay.addWidget(self._build_image_edit_bar())
        self.center_stack.addWidget(image_page)         # 0: image canvas
        self.center_stack.addWidget(self.video_stage)   # 1: video
        center_lay.addWidget(self.center_stack, 1)
        center_lay.addWidget(self._build_nav_bar())
        # Floating tool strip: an overlay child of the view (NOT the viewport — the
        # viewport scrolls its children when panning, which would drag the strip).
        self._toolstrip = self._build_canvas_toolstrip()
        self._toolstrip.setParent(self.view)
        self._toolstrip.raise_()

        # Right: preset selector + AI actions above the editor for the active preset.
        # Structured presets get the Caption/Elements tabs; plain presets get a single
        # free-text field, so the Ideogram-specific UI simply isn't present.
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self._build_caption_tab(), "Caption")
        self.right_tabs.addTab(self._build_elements_tab(), "Elements")
        self.editor_stack = QStackedWidget()
        self.editor_stack.addWidget(self._build_plain_tab())   # index 0: plain
        self.editor_stack.addWidget(self.right_tabs)           # index 1: structured

        right_container = QWidget()
        right_container.setObjectName("Panel")
        right_lay = QVBoxLayout(right_container)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)
        self._readonly_banner = QLabel(
            "Captioning in progress — editing paused (read-only). Browse and flag files "
            "for manual review — press F or right-click a thumbnail to flag."
        )
        self._readonly_banner.setObjectName("ReadOnlyBanner")
        self._readonly_banner.setWordWrap(True)
        self._readonly_banner.setVisible(False)
        self._readonly_banner.setStyleSheet(
            f"#ReadOnlyBanner {{ background: {self.theme.warning}; color: {self.theme.surface_0}; "
            f"padding: 7px 10px; font-size: 12px; }}"
        )
        right_lay.addWidget(self._readonly_banner)
        right_lay.addWidget(self._build_preset_strip())
        right_lay.addWidget(self._build_ai_actions())
        right_lay.addWidget(self.editor_stack, 1)
        right_container.setMinimumWidth(290)

        self.json_panel = self._build_json_panel()

        splitter.addWidget(self.left_panel)
        splitter.addWidget(center)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([210, 1000, 390])
        self.splitter = splitter
        splitter.splitterMoved.connect(
            lambda *_: self.qsettings.setValue("splitter_state_v2", self.splitter.saveState())
        )

        body = QWidget()
        body_lay = QHBoxLayout(body)
        body_lay.setContentsMargins(0, 0, 0, 0)
        body_lay.setSpacing(0)
        body_lay.addWidget(self.rail)
        body_lay.addWidget(splitter, 1)
        self._json_tab = VerticalTab("LLM INSTRUCTIONS")
        self._json_tab.setToolTip("Show the instructions sent to the LLM (Ctrl+J)")
        self._json_tab.clicked.connect(self.json_action.toggle)
        body_lay.addWidget(self._json_tab)
        self._body = body
        # Raw-JSON is a right slide-over overlaying the body, not a splitter pane.
        self.json_panel.setParent(body)
        self.json_panel.hide()
        outer.addWidget(body, 1)

        # Bottom: thumbnail filmstrip.
        self.filmstrip = QListWidget()
        self.filmstrip.setObjectName("Panel")
        self.filmstrip.setFlow(QListWidget.LeftToRight)
        self.filmstrip.setWrapping(False)
        self.filmstrip.setViewMode(QListWidget.IconMode)
        self.filmstrip.setIconSize(QSize(THUMB, THUMB))
        self.filmstrip.setFixedHeight(THUMB + 48)
        self.filmstrip.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.filmstrip.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.filmstrip.setMovement(QListWidget.Static)
        self.filmstrip.setItemDelegate(FilmstripDelegate(self))
        self.filmstrip.currentItemChanged.connect(self._on_thumb_changed)
        self.filmstrip.setContextMenuPolicy(Qt.CustomContextMenu)
        self.filmstrip.customContextMenuRequested.connect(self._filmstrip_context_menu)
        self.filmstrip.viewport().setMouseTracking(True)
        self.filmstrip.viewport().installEventFilter(self)
        # App-level filter so A/D move between images from anywhere in the window
        # (60% keyboards often lack arrow keys), without stealing letters while typing.
        QApplication.instance().installEventFilter(self)
        self._hover_item = None
        self._preview_cache: dict[str, QPixmap] = {}
        self._hover_preview = FilmstripPreview(self.theme, None)
        outer.addWidget(self.filmstrip, 0)

        self.setCentralWidget(central)

    # ---- behavior --------------------------------------------------------
    def _ensure_left_anim(self) -> None:
        if getattr(self, "_left_anim", None) is not None:
            return
        # A "ghost" snapshot of the panel does the sliding, so the splitter can swap
        # in a single hidden frame and the canvas never animates its width.
        self._panel_ghost = QLabel(self._body)
        self._panel_ghost.setObjectName("PanelGhost")
        self._panel_ghost.hide()
        self._panel_ghost_op = QGraphicsOpacityEffect(self._panel_ghost)
        self._panel_ghost.setGraphicsEffect(self._panel_ghost_op)
        self._lp_geom = QPropertyAnimation(self._panel_ghost, b"geometry", self)
        self._lp_fade = QPropertyAnimation(self._panel_ghost_op, b"opacity", self)
        for a in (self._lp_geom, self._lp_fade):
            a.setDuration(MOTION_MED)
        self._left_anim = QParallelAnimationGroup(self)
        self._left_anim.addAnimation(self._lp_geom)
        self._left_anim.addAnimation(self._lp_fade)
        self._left_anim.finished.connect(self._on_left_slide_done)
        self._left_collapsing = False
        self._left_panel_width = None
        self._split_saved = None
        self._panel_effect = None

    def _panel_body_rect(self) -> QRect:
        tl = self.left_panel.mapTo(self._body, QPoint(0, 0))
        return QRect(tl.x(), tl.y(), self.left_panel.width(), self.left_panel.height())

    def _start_ghost_slide(self, frm: QRect, to: QRect, op_from: float, op_to: float,
                           ease: "QEasingCurve.Type") -> None:
        self._panel_ghost.raise_()
        self._lp_geom.setEasingCurve(ease)
        self._lp_geom.setStartValue(frm)
        self._lp_geom.setEndValue(to)
        self._lp_fade.setEasingCurve(ease)
        self._lp_fade.setStartValue(op_from)
        self._lp_fade.setEndValue(op_to)
        self._left_anim.start()

    def _expand_left_to(self, target: int) -> None:
        # Give pane 0 exactly `target`, taking the difference from the center pane.
        sizes = self.splitter.sizes()
        if len(sizes) < 2:
            return
        pool = sizes[0] + sizes[1]
        new = list(sizes)
        new[0] = int(target)
        new[1] = max(0, pool - int(target))
        self.splitter.setSizes(new)

    def _on_left_slide_done(self) -> None:
        self._panel_ghost.hide()
        self._panel_ghost.clear()
        self._left_collapsing = False

    def toggle_left_panel(self) -> None:
        self._ensure_left_anim()
        self._left_anim.stop()
        collapsing = self.left_panel.isVisible()
        going_visible = not collapsing
        if collapsing:
            rect = self._panel_body_rect()
            self._left_panel_width = self.left_panel.width()
            self._split_saved = self.splitter.saveState()
            self._panel_ghost.setPixmap(self.left_panel.grab())
            self._panel_ghost.setGeometry(rect)
            self._panel_ghost_op.setOpacity(1.0)
            self._panel_ghost.show()
            # Swap the layout in one hidden frame: the canvas takes the full width
            # *underneath* the ghost, then the ghost slides off to the left.
            self.left_panel.setVisible(False)
            self._left_collapsing = True
            off = QRect(rect.x() - rect.width(), rect.y(), rect.width(), rect.height())
            self._start_ghost_slide(rect, off, 1.0, 0.0, QEasingCurve.InCubic)
        else:
            # Expand: force the width explicitly rather than trusting restoreState,
            # which the show-relayout can override into a sliver on some platforms.
            self.left_panel.setMinimumWidth(190)  # re-assert the floor
            self.left_panel.setVisible(True)
            target = max(190, int(self._left_panel_width or 210))
            self._expand_left_to(target)
            # Re-apply once the show/relayout has settled so it sticks.
            QTimer.singleShot(0, lambda t=target: self._expand_left_to(t))
        ic = self.theme.text_secondary
        self.panels_action.setIcon(
            lucide_icon("panel-left-close" if going_visible else "panel-left-open", ic)
        )
        self.panels_action.setToolTip(
            "Collapse guidance panel (Ctrl+\\)" if going_visible else "Expand guidance panel (Ctrl+\\)"
        )

    def _build_json_panel(self) -> QWidget:
        panel = QWidget()
        panel.setObjectName("JsonSlideOver")
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        header = QLabel("LLM instructions")
        header.setObjectName("SectionLabel")
        lay.addWidget(header)
        self._instr_sub = QLabel("")
        self._instr_sub.setWordWrap(True)
        self._instr_sub.setStyleSheet(
            f"color: {self.theme.text_secondary}; font-size: 11px;")
        lay.addWidget(self._instr_sub)
        self.json_view = QPlainTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setFont(QFont(self.settings.mono_font_family or "Monospace"))
        lay.addWidget(self.json_view, 1)
        row = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip("Copy these instructions to the clipboard.")
        copy_btn.clicked.connect(self._copy_json)
        edit_btn = QPushButton("Edit…")
        edit_btn.setToolTip("Edit the system prompt for this preset.")
        edit_btn.clicked.connect(self.open_system_prompt)
        row.addWidget(copy_btn)
        row.addWidget(edit_btn)
        row.addStretch(1)
        lay.addLayout(row)
        return panel

    def _ensure_json_anim(self) -> None:
        if getattr(self, "_json_anim", None) is not None:
            return
        eff = QGraphicsOpacityEffect(self.json_panel)
        eff.setOpacity(1.0)
        self.json_panel.setGraphicsEffect(eff)
        self._json_opacity = eff
        self._json_geom_anim = QPropertyAnimation(self.json_panel, b"geometry", self)
        self._json_fade_anim = QPropertyAnimation(eff, b"opacity", self)
        for a in (self._json_geom_anim, self._json_fade_anim):
            a.setDuration(MOTION_MED)
        self._json_anim = QParallelAnimationGroup(self)
        self._json_anim.addAnimation(self._json_geom_anim)
        self._json_anim.addAnimation(self._json_fade_anim)
        self._json_closing = False
        self._json_anim.finished.connect(self._on_json_anim_done)

    def _json_docked_rect(self) -> QRect:
        body = self._body
        tab_w = self._json_tab.width() if getattr(self, "_json_tab", None) else 26
        width = min(380, max(280, body.width() - tab_w - 360))
        x = max(0, body.width() - tab_w - width)
        return QRect(x, 0, width, body.height())

    def _json_offscreen_rect(self, docked: QRect) -> QRect:
        # Parked just past the body's right edge so it slides in from outside.
        return QRect(self._body.width(), docked.y(), docked.width(), docked.height())

    def _on_json_anim_done(self) -> None:
        if getattr(self, "_json_closing", False):
            self.json_panel.hide()
            self._json_closing = False

    def toggle_json_panel(self, checked: bool) -> None:
        self._ensure_json_anim()
        self._json_anim.stop()
        docked = self._json_docked_rect()
        off = self._json_offscreen_rect(docked)
        if checked:
            self._json_closing = False
            self.json_panel.setGeometry(off)
            self._json_opacity.setOpacity(0.0)
            self.json_panel.show()
            self.json_panel.raise_()
            self._refresh_json_view()
            self._json_geom_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._json_geom_anim.setStartValue(off)
            self._json_geom_anim.setEndValue(docked)
            self._json_fade_anim.setEasingCurve(QEasingCurve.OutCubic)
            self._json_fade_anim.setStartValue(0.0)
            self._json_fade_anim.setEndValue(1.0)
            self._json_anim.start()
        elif self.json_panel.isVisible():
            self._json_closing = True
            self._json_geom_anim.setEasingCurve(QEasingCurve.InCubic)
            self._json_geom_anim.setStartValue(self.json_panel.geometry())
            self._json_geom_anim.setEndValue(off)
            self._json_fade_anim.setEasingCurve(QEasingCurve.InCubic)
            self._json_fade_anim.setStartValue(self._json_opacity.opacity())
            self._json_fade_anim.setEndValue(0.0)
            self._json_anim.start()
        if getattr(self, "_json_tab", None) is not None:
            self._json_tab.set_on(checked)
            self._json_tab.setToolTip(
                "Hide the LLM instructions (Ctrl+J)" if checked
                else "Show the instructions sent to the LLM (Ctrl+J)"
            )
        self.json_action.setToolTip(
            "Hide the LLM instructions (Ctrl+J)" if checked
            else "Show the instructions sent to the LLM (Ctrl+J)"
        )

    def _reposition_json_overlay(self) -> None:
        body = getattr(self, "_body", None)
        panel = getattr(self, "json_panel", None)
        if body is None or panel is None:
            return
        # Don't fight an in-flight slide; the animation already targets the docked rect.
        anim = getattr(self, "_json_anim", None)
        if anim is not None and anim.state() == QAbstractAnimation.Running:
            return
        if panel.isVisible():
            panel.setGeometry(self._json_docked_rect())
            panel.raise_()

    def _refresh_json_view(self) -> None:
        """Render what would actually be sent to the model right now: the preset's
        system prompt assembled with the guidance in effect for the current image."""
        if not getattr(self, "json_panel", None) or not self.json_panel.isVisible():
            return
        text = self.current_llm_instructions()
        sb = self.json_view.verticalScrollBar()
        pos = sb.value()
        self.json_view.setPlainText(text)
        sb.setValue(min(pos, sb.maximum()))
        if hasattr(self, "_instr_sub"):
            name = self.current.name if self.current is not None else "no image selected"
            self._instr_sub.setText(f"{self.preset.label} \u00b7 {name}")

    def current_llm_instructions(self) -> str:
        """The system message the captioning run would send for the current image and
        preset — assembled from the same helpers the run path uses, so what's shown
        here is what's actually sent, not a paraphrase."""
        guidance = ""
        if self.project is not None and self.current is not None:
            try:
                guidance = self.project.resolved_for(self.current.name) or ""
            except Exception:
                guidance = ""
        if self.preset.is_plain:
            parts = [self.effective_system_prompt().strip()]
            if guidance.strip():
                parts.append("Additional guidance for this file:\n" + guidance.strip())
            return "\n\n".join(p for p in parts if p)
        # Structured presets build their prompt from the template files, exactly as
        # the run path does.
        try:
            prompts = load_prompts()
            convert = (self.current is not None
                       and self._image_uses_source(self.current))
            key = "image_to_json_convert_system" if convert else "image_to_json_system"
            if key not in prompts:
                key = "image_to_json_system"
            text = json_system_prompt(prompts, key, self.settings, guidance=guidance)
        except Exception as exc:
            return f"(Could not assemble the prompt: {exc})"
        extra = self.system_prompt_for_preset().strip()
        if extra:
            text += "\n\n" + extra
        return text

    def _copy_json(self) -> None:
        QApplication.clipboard().setText(self.json_view.toPlainText())
        self._set_status("LLM instructions copied to clipboard.")

    def _save_json_as(self) -> None:
        if self.current is not None:
            start = str(self.current.parent / (self.current.stem + ".json"))
        else:
            start = "caption.json"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save caption JSON", start, "JSON files (*.json);;All files (*)"
        )
        if not path:
            return
        try:
            Path(path).write_text(self.json_view.toPlainText(), encoding="utf-8")
            self._set_status(f"Saved JSON to {Path(path).name}.")
        except OSError as exc:
            QMessageBox.critical(self, "Could not save JSON", str(exc))

    def open_folder(self) -> None:
        if not self.confirm_discard_video_edits("open another folder"):
            return
        start_dir = self.qsettings.value("last_folder", "", str)
        if start_dir and not Path(start_dir).is_dir():
            start_dir = ""
        folder = QFileDialog.getExistingDirectory(self, "Open image folder", start_dir)
        if not folder:
            return
        self.load_folder_path(Path(folder))

    def load_folder_path(self, folder: Path) -> None:
        """Open a specific folder. Split out of open_folder so callers that already
        know the path (duplicating a dataset, a future recent-folders list) don't
        have to go through a file dialog to get there."""
        folder = str(folder)
        self.qsettings.setValue("last_folder", folder)
        try:
            # Read the project first: the folder's saved preset decides which
            # sidecar extension the store reads and writes.
            probe = CaptionStore(Path(folder), self.settings_caption_ext())
            project = probe.load_project()
            # Custom presets are resolved through the merged map; get_preset alone
            # only knows the built-ins and would silently fall back to plain text.
            self.preset = self.available_presets().get(
                project.preset, get_preset(project.preset))
            self.store = CaptionStore(Path(folder), self.preset.extension)
            # Clear anything an interrupted render left behind before listing.
            self.store.sweep_work_files()
            self.images = self.store.images()
            self.project = project
            self._apply_preset_ui()
            self._load_folder_tags()
        except Exception as exc:  # Tier 2: surface failures readably.
            QMessageBox.critical(self, "Could not open folder", str(exc))
            return

        self.load_project_into_ui()
        self._rebuild_filmstrip()
        self.filmstrip.setCurrentRow(0)
        self._set_status(f"{len(self.images)} images in {Path(folder).name}")
        self._update_count_label()

    def settings_caption_ext(self) -> str:
        return self.preset.extension

    def preview_pixmap(self, path: Path) -> QPixmap:
        """Full-size preview for a file, using a clip's poster frame.

        QPixmap can't open an .mp4, so any dialog that previewed with QPixmap
        directly showed "(cannot load image)" for every video.
        """
        if is_video(path):
            poster = self._video_poster(path)
            pixmap = QPixmap(str(poster)) if poster else QPixmap()
            return pixmap if not pixmap.isNull() else self._video_placeholder()
        return QPixmap(str(path))

    def _thumb_pixmap(self, path: Path) -> QPixmap:
        if is_video(path):
            pm = QPixmap(str(self._video_poster(path))) if self._video_poster(path) else QPixmap()
            if pm.isNull():
                pm = self._video_placeholder()
        else:
            pm = QPixmap(str(path))
        if pm.isNull():
            return QPixmap(THUMB, THUMB)
        return pm.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def _poster_cache_dir(self) -> Path | None:
        if self.store is None:
            return None
        return self.store.project_dir() / "thumbs"

    def _video_poster(self, path: Path) -> Path | None:
        """Poster frame for a video, extracted once and cached in .captioner/thumbs
        (keyed by name + mtime, so an edited clip regenerates). None without ffmpeg."""
        cache = self._poster_cache_dir()
        if cache is None:
            return None
        try:
            stamp = int(path.stat().st_mtime)
        except OSError:
            stamp = 0
        out = cache / f"{path.stem}.{stamp}.jpg"
        if out.exists():
            return out
        # drop stale posters for this clip
        for old in cache.glob(f"{path.stem}.*.jpg"):
            old.unlink(missing_ok=True)
        if not ffmpeg_available():
            return None
        return out if extract_poster(path, out) else None

    def _video_placeholder(self) -> QPixmap:
        """Neutral film-strip tile shown when ffmpeg isn't available yet."""
        pm = QPixmap(THUMB, THUMB)
        pm.fill(QColor(self.theme.surface_2))
        painter = QPainter(pm)
        painter.drawPixmap(
            (THUMB - 28) // 2, (THUMB - 28) // 2,
            lucide_pixmap("film", self.theme.text_secondary, 28))
        painter.end()
        return pm

    def _maybe_wasd_navigate(self, event) -> bool:
        """A = previous image, D = next image — from anywhere in the main window or
        the source pop-out, unless the user is typing or editing a value. (W/S are
        reserved for bbox nudging on the canvas, handled by the view itself.)"""
        if event.modifiers() != Qt.NoModifier:
            return False
        key = event.key()
        if key == Qt.Key_A:
            delta = -1
        elif key == Qt.Key_D:
            delta = 1
        else:
            return False
        # Only when our window (or the pop-out) is active — never over a dialog.
        active = QApplication.activeWindow()
        if active is not self and active is not getattr(self, "_source_popout", None):
            return False
        fw = QApplication.focusWidget()
        # Don't steal letters from an editable text field or a value editor.
        if isinstance(fw, (QLineEdit, QPlainTextEdit)) and not fw.isReadOnly():
            return False
        if isinstance(fw, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return False
        # On the canvas, A/D nudge the selected box — let the view handle WASD.
        view = getattr(self, "view", None)
        if view is not None and fw in (view, view.viewport()):
            return False
        (self.prev_image if delta < 0 else self.next_image)()
        return True

    def eventFilter(self, obj, event):
        if event.type() == QEvent.KeyPress and self._maybe_wasd_navigate(event):
            return True
        if obj is getattr(self, "_gchg_box", None):
            et = event.type()
            if et == QEvent.Enter:
                self._show_gdiff_popup()
            elif et in (QEvent.Leave, QEvent.Hide):
                self._hide_gdiff_popup()
        if obj is getattr(self, "_used_tags_collapsed", None):
            et = event.type()
            if et == QEvent.Enter:
                self._show_tags_popup()
            elif et in (QEvent.Leave, QEvent.Hide):
                self._hide_tags_popup()
        fs = getattr(self, "filmstrip", None)
        if fs is not None and obj is fs.viewport():
            et = event.type()
            if et == QEvent.MouseMove:
                item = self.filmstrip.itemAt(event.position().toPoint())
                if item is not self._hover_item:
                    self._hover_item = item
                    if item is not None:
                        self._show_hover_preview(item)   # instant, no dwell
                    else:
                        self._hide_preview()
            elif et in (QEvent.Leave, QEvent.Wheel, QEvent.Hide):
                self._hover_item = None
                self._hide_preview()
        return super().eventFilter(obj, event)

    def _hide_preview(self) -> None:
        self._hover_preview.hide()

    def _preview_pixmap(self, path: Path) -> QPixmap:
        key = str(path)
        pm = self._preview_cache.get(key)
        if pm is None:
            src = QPixmap(key)
            pm = QPixmap() if src.isNull() else src.scaled(
                PREVIEW_IMG_W, PREVIEW_IMG_H, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self._preview_cache[key] = pm
        return pm

    def _show_hover_preview(self, item: QListWidgetItem) -> None:
        path_str = item.data(Qt.UserRole)
        if not path_str:
            self._hide_preview()
            return
        pm = self._preview_pixmap(Path(path_str))
        if pm.isNull():
            self._hide_preview()
            return
        path = Path(path_str)
        try:
            idx_text = f"{self.images.index(path) + 1} / {len(self.images)}"
        except ValueError:
            idx_text = ""
        self._hover_preview.set_content(pm, path.name, idx_text, self._thumb_banners(path))

        vp = self.filmstrip.viewport()
        rect = self.filmstrip.visualItemRect(item)
        thumb_center = vp.mapToGlobal(rect.center())
        thumb_top = vp.mapToGlobal(rect.topLeft())
        win = self._hover_preview
        m = win._margin
        ww, wh = win.width(), win.height()
        screen = self.screen().availableGeometry()
        # card bottom (and arrow tip) sits PREVIEW_GAP above the thumbnail top
        y = thumb_top.y() - PREVIEW_GAP - (m + win.card.height() + PREVIEW_ARROW)
        # centre the card on the thumbnail, then clamp to the screen
        left = thumb_center.x() - (m + PREVIEW_W // 2)
        left = max(screen.left() + 4, min(left, screen.right() - ww - 4))
        y = max(screen.top() + 4, y)
        # keep the arrow under the thumbnail centre even after clamping
        arrow_x = thumb_center.x() - left
        arrow_x = max(m + 10, min(arrow_x, m + PREVIEW_W - 10))
        self._anchor_popup_to_window(win)
        win.show_at(QPoint(left, y), int(arrow_x))

    def _on_thumb_changed(self, current: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        if current is None:
            return
        if self.current is not None:
            self._commit_active_caption()
            if self._autosave:
                if self._dirty:
                    self.save_current(silent=True)
            elif self._dirty:
                # keep edits in memory; do NOT write to disk until the user saves
                self._pending[str(self.current)] = (
                    self.current_text if self.preset.is_plain else self.current_caption)
                self._refresh_thumb_marker(self.current)
        self.persist_guidance_if_dirty()
        path = Path(current.data(Qt.UserRole))
        self.show_image(path)
        self.load_caption_for(path)
        self.load_per_file_guidance(path.name)
        self._sync_flag_action()

    def show_image(self, path: Path) -> None:
        self.current = path
        if is_video(path):
            self.video_stage.load(path, self._video_info.get(str(path)))
            self.center_stack.setCurrentIndex(1)
            if getattr(self, "_toolstrip", None) is not None:
                self._toolstrip.setVisible(False)
            self._show_video_meta(path)
            self._refresh_json_view()
            return
        # leaving a video: release the file and its audio device
        if self.center_stack.currentIndex() == 1:
            self.video_stage.stop()
            self.center_stack.setCurrentIndex(0)
        if getattr(self, "_toolstrip", None) is not None:
            self._toolstrip.setVisible(self.preset.has_boxes)
        pm = QPixmap(str(path))
        self.scene.clear()
        self.pixmap_item = None
        self.box_items = []
        if pm.isNull():
            self._set_status(f"Could not open {path.name}")
            self.crop_action.setEnabled(False)
            return
        self._image_rotation = 0
        self._image_crop_item = None
        if hasattr(self, "img_crop_btn") and self.img_crop_btn.isChecked():
            self.img_crop_btn.blockSignals(True)
            self.img_crop_btn.setChecked(False)
            self.img_crop_btn.blockSignals(False)
            self.img_aspect.setVisible(False)
        self.pixmap_item = self.scene.addPixmap(pm)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self._user_zoomed = False
        self.view.resetTransform()
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._refresh_title()
        self._set_status(f"{path.name}  \u00b7  {pm.width()}\u00d7{pm.height()}")
        self._update_count_label()
        self._update_zoom_label()
        self.crop_action.setEnabled(not getattr(self, "_read_only", False))
        if hasattr(self, "image_edit_bar"):
            editable = not getattr(self, "_read_only", False)
            for widget in (self.img_crop_btn, self.img_rot_ccw, self.img_rot_cw,
                           self.img_reset_btn):
                widget.setEnabled(editable)
            self._seed_size_fields()
            # Apply follows whether there's anything to apply, so it's set last.
            self._refresh_image_edit_label()

    def _build_image_edit_bar(self) -> QWidget:
        """Crop, rotate and resize for the picture on screen — the same shape as the
        video edit bar, so both media types are edited the same way."""
        t = self.theme
        bar = QFrame()
        bar.setObjectName("ImageEditBar")
        bar.setStyleSheet(
            f"#ImageEditBar {{ background: {t.surface_1}; "
            f"border-top: 1px solid {t.border}; }}")
        row = QHBoxLayout(bar)
        row.setContentsMargins(12, 6, 12, 6)
        row.setSpacing(8)

        self.img_crop_btn = QPushButton("Crop")
        self.img_crop_btn.setCheckable(True)
        self.img_crop_btn.setToolTip("Draw a crop region on the image")
        self.img_crop_btn.toggled.connect(self.set_image_crop_enabled)
        row.addWidget(self.img_crop_btn)

        self.img_aspect = QComboBox()
        for label, _ratio in ASPECT_PRESETS:
            self.img_aspect.addItem(label)
        self.img_aspect.setToolTip("Constrain the crop to an aspect ratio")
        self.img_aspect.setVisible(False)
        self.img_aspect.currentIndexChanged.connect(self._on_image_aspect)
        row.addWidget(self.img_aspect)

        self.img_rot_ccw = QToolButton()
        self.img_rot_ccw.setObjectName("NavBtn")
        self.img_rot_ccw.setIcon(lucide_icon("rotate-ccw", t.text_secondary, 15))
        self.img_rot_ccw.setToolTip("Rotate 90\u00b0 anticlockwise")
        self.img_rot_ccw.clicked.connect(lambda: self.rotate_image_by(-90))
        row.addWidget(self.img_rot_ccw)
        self.img_rot_cw = QToolButton()
        self.img_rot_cw.setObjectName("NavBtn")
        self.img_rot_cw.setIcon(lucide_icon("rotate-cw", t.text_secondary, 15))
        self.img_rot_cw.setToolTip("Rotate 90\u00b0 clockwise")
        self.img_rot_cw.clicked.connect(lambda: self.rotate_image_by(90))
        row.addWidget(self.img_rot_cw)

        # Resize inline, with no enabling checkbox: the boxes hold the current size,
        # so changing them IS the request. A checkbox was a second thing to click
        # that only ever said "I meant it".
        row.addWidget(QLabel("Size:"))
        self.img_w = QSpinBox()
        self.img_w.setRange(1, 32768)
        self.img_w.setMaximumWidth(80)
        self.img_w.setToolTip("Output width")
        self.img_w.valueChanged.connect(lambda v: self._on_size_typed("w", v))
        row.addWidget(self.img_w)
        times = QLabel("\u00d7")
        times.setStyleSheet(f"color: {t.text_secondary};")
        row.addWidget(times)
        self.img_h = QSpinBox()
        self.img_h.setRange(1, 32768)
        self.img_h.setMaximumWidth(80)
        self.img_h.setToolTip("Output height")
        self.img_h.valueChanged.connect(lambda v: self._on_size_typed("h", v))
        row.addWidget(self.img_h)
        px = QLabel("px")
        px.setStyleSheet(f"color: {t.text_secondary};")
        row.addWidget(px)
        self.img_lock = QToolButton()
        self.img_lock.setObjectName("NavBtn")
        self.img_lock.setCheckable(True)
        self.img_lock.setChecked(True)
        self.img_lock.setIcon(lucide_icon("link", t.text_secondary, 15))
        self.img_lock.setToolTip("Keep the aspect ratio when resizing")
        row.addWidget(self.img_lock)

        self.img_reset_btn = QPushButton("Reset")
        self.img_reset_btn.setToolTip("Drop the crop and rotation")
        self.img_reset_btn.clicked.connect(self.reset_image_edit)
        row.addWidget(self.img_reset_btn)

        self.img_edit_label = QLabel("")
        self.img_edit_label.setObjectName("Hint")
        row.addWidget(self.img_edit_label, 1)

        self.img_revert_btn = QPushButton("Restore original")
        self.img_revert_btn.setToolTip(
            "Undo every edit: put the pre-edit file back and drop the backup")
        self.img_revert_btn.clicked.connect(self.restore_original_media)
        row.addWidget(self.img_revert_btn)

        more = QPushButton("More\u2026")
        more.setToolTip("Open the full crop/resize dialog for exact pixel sizes")
        more.clicked.connect(self.open_crop_dialog)
        row.addWidget(more)

        self.img_apply_btn = QPushButton("Apply edit\u2026")
        self.img_apply_btn.setToolTip("Write the crop and rotation to the file")
        self.img_apply_btn.clicked.connect(self.apply_image_edit)
        row.addWidget(self.img_apply_btn)
        self._image_rotation = 0
        self._image_crop_item = None
        self.image_edit_bar = bar
        return bar

    # ---- image editing ----

    def set_image_crop_enabled(self, on: bool) -> None:
        self.img_aspect.setVisible(on)
        if on:
            if self.pixmap_item is None:
                self.img_crop_btn.setChecked(False)
                return
            bounds = self.pixmap_item.boundingRect()
            self._image_crop_item = CropRectItem(bounds, self._refresh_image_edit_label)
            self.scene.addItem(self._image_crop_item)
            self._on_image_aspect(self.img_aspect.currentIndex())
        elif self._image_crop_item is not None:
            self.scene.removeItem(self._image_crop_item)
            self._image_crop_item = None
        self._refresh_image_edit_label()

    def _on_image_aspect(self, idx: int) -> None:
        if self._image_crop_item is not None and 0 <= idx < len(ASPECT_PRESETS):
            self._image_crop_item.set_aspect(ASPECT_PRESETS[idx][1])
            self._refresh_image_edit_label()

    def image_crop_box(self) -> tuple[int, int, int, int] | None:
        if self._image_crop_item is None or self.pixmap_item is None:
            return None
        r = self._image_crop_item.rect()
        return (max(0, int(round(r.x()))), max(0, int(round(r.y()))),
                max(1, int(round(r.width()))), max(1, int(round(r.height()))))

    def _seed_size_fields(self) -> None:
        """Put the picture's real size in the boxes, so they read as current state
        rather than an empty form."""
        if self.pixmap_item is None:
            return
        rect = self.pixmap_item.boundingRect()
        for box, value in ((self.img_w, int(rect.width())),
                           (self.img_h, int(rect.height()))):
            box.blockSignals(True)
            box.setValue(value)
            box.blockSignals(False)
        self._base_size = (int(rect.width()), int(rect.height()))

    def _on_size_typed(self, which: str, value: int) -> None:
        if self.img_lock.isChecked() and getattr(self, "_base_size", None):
            bw, bh = self._base_size
            if bw and bh:
                other, target = ((self.img_h, round(value * bh / bw)) if which == "w"
                                 else (self.img_w, round(value * bw / bh)))
                other.blockSignals(True)
                other.setValue(max(1, target))
                other.blockSignals(False)
        self._refresh_image_edit_label()

    def output_size(self) -> tuple[int, int]:
        return int(self.img_w.value()), int(self.img_h.value())

    def rotate_image_by(self, degrees: int) -> None:
        """Turn the view so the crop rect is drawn against the final orientation."""
        if self.pixmap_item is None:
            return
        self._image_rotation = (self._image_rotation + degrees) % 360
        self.view.resetTransform()
        if self._image_rotation:
            self.view.rotate(self._image_rotation)
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._refresh_image_edit_label()

    def reset_image_edit(self) -> None:
        self._image_rotation = 0
        self._seed_size_fields()
        self.view.resetTransform()
        if self.img_crop_btn.isChecked():
            self.img_crop_btn.setChecked(False)
        if self.pixmap_item is not None:
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._refresh_image_edit_label()

    def _refresh_image_edit_label(self) -> None:
        bits = []
        crop = self.image_crop_box()
        if crop and self.pixmap_item is not None:
            full = self.pixmap_item.boundingRect()
            if (crop[2], crop[3]) != (int(full.width()), int(full.height())):
                bits.append(f"crop {crop[2]}\u00d7{crop[3]}")
        if self._image_rotation:
            bits.append(f"rotated {self._image_rotation}\u00b0")
        if getattr(self, "_base_size", None) and self.output_size() != self._base_size:
            w, h = self.output_size()
            bits.append(f"resize to {w}\u00d7{h}")
        self.img_edit_label.setText(
            "Unapplied: " + ", ".join(bits) if bits else "")
        self.img_edit_label.setStyleSheet(
            f"color: {self.theme.warning};" if bits
            else f"color: {self.theme.text_secondary};")
        self.img_apply_btn.setEnabled(bool(bits))

    def apply_image_edit(self) -> None:
        """Write crop + rotation in place, keeping the untouched original."""
        if self.current is None or self.pixmap_item is None:
            return
        crop = self.image_crop_box()
        rotation = self._image_rotation % 360
        full = self.pixmap_item.boundingRect()
        cropping = bool(crop and (crop[2], crop[3]) != (int(full.width()),
                                                        int(full.height())))
        out_w, out_h = self.output_size()
        resizing = bool(getattr(self, "_base_size", None)
                        and (out_w, out_h) != self._base_size)
        if not cropping and not rotation and not resizing:
            self._set_status("Nothing to apply.")
            return
        changes = []
        if cropping:
            changes.append(f"crop to {crop[2]}\u00d7{crop[3]}")
        if rotation:
            changes.append(f"rotate {rotation}\u00b0")
        if resizing:
            changes.append(f"resize to {out_w}\u00d7{out_h}")
        if QMessageBox.question(
            self, "Apply image edit",
            f"{', '.join(changes).capitalize()} on {self.current.name}?\n\n"
            "The untouched original is kept in .original/.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return
        from PIL import Image
        try:
            if self.store is not None:
                self.store.backup_original(self.current)
            with Image.open(self.current) as im:
                im.load()
                if cropping:
                    x, y, cw, ch = crop
                    im = im.crop((x, y, x + cw, y + ch))
                if rotation:
                    im = im.rotate(-rotation, expand=True)
                if resizing and not rotation:
                    # After a quarter turn the requested size describes the
                    # pre-rotation frame, so it no longer applies.
                    im = im.resize((out_w, out_h), Image.LANCZOS)
                im.save(self.current)
        except Exception as exc:
            QMessageBox.critical(self, "Edit failed", str(exc))
            return
        self.reset_image_edit()
        self._refresh_edited_image_cached(self.current)
        self.show_image(self.current)
        self._set_status(f"{self.current.name}: {', '.join(changes)}.")

    def toggle_video_playback(self) -> None:
        if self.center_stack.currentIndex() == 1:
            self.video_stage.toggle_play()

    def _show_video_meta(self, path: Path) -> None:
        """Title/status/action state for a video, matching what show_image does for
        an image. Crop is Pillow-based and image-only; video editing is its own tool."""
        self._refresh_title()
        info = self._video_info.get(str(path))
        if info is not None:
            self._set_status(
                f"{path.name}  \u00b7  {info.width}\u00d7{info.height}  \u00b7  "
                f"{info.duration_label}  \u00b7  {info.fps:g} fps  \u00b7  {info.codec}")
        else:
            self._set_status(f"{path.name}  \u00b7  video (install ffmpeg for details)")
        self._update_count_label()
        self._update_zoom_label()
        self.crop_action.setEnabled(False)

    def _refresh_edited_image_cached(self, path: Path) -> None:
        """Refresh one image's cached thumbnail/preview after its pixels changed on
        disk, without touching the canvas. Used per-item during a batch so the
        filmstrip updates live but the view isn't rebuilt hundreds of times."""
        key = str(path)
        self._preview_cache.pop(key, None)
        self._thumb_base[key] = self._thumb_pixmap(path)
        item = self._thumb_items.get(key)
        if item is not None:
            item.setIcon(self._decorated_thumb(path))

    def _refresh_current_after_batch(self) -> None:
        """Re-show the current image once a batch finishes, so the canvas reflects
        any new pixel dimensions."""
        if self.current is not None and self.current.exists():
            self.show_image(self.current)
            self.rebuild_boxes()

    def open_batch_resize(self) -> None:
        if self.store is None or not self.images:
            QMessageBox.information(self, "Batch resize", "Open a folder first.")
            return
        if getattr(self, "_read_only", False):
            return
        BatchResizeDialog(self, self.theme).exec()

    # ---- caption presets ----

    def _sync_preset_model_target(self) -> None:
        """Arm the trim controls with the preset's model, so choosing 'MiniMax H3
        video' also conforms clips to H3 rather than leaving that a separate step."""
        stage = getattr(self, "video_stage", None)
        if stage is None or not hasattr(stage, "_target_combo"):
            return
        key = self.preset.model_target
        if not key or key not in self.model_targets:
            return
        idx = stage._target_combo.findData(key)
        if idx >= 0 and stage._target_combo.currentIndex() != idx:
            stage._target_combo.setCurrentIndex(idx)

    def _apply_preset_ui(self) -> None:
        """Show the editor the active preset needs and hide what it doesn't use."""
        plain = self.preset.is_plain
        self._sync_preset_model_target()
        self._clear_stale_review_flags()
        self._refresh_spec_markers()
        if hasattr(self, "editor_stack"):
            self.editor_stack.setCurrentIndex(0 if plain else 1)
        if hasattr(self, "preset_ext_label"):
            self.preset_ext_label.setText(self.preset.extension)
        if hasattr(self, "preset_combo"):
            # Recoverable when the label is elided.
            tip = self.preset.label
            if self.preset.blurb:
                tip += f"\n\n{self.preset.blurb}"
            self.preset_combo.setToolTip(tip)
        if hasattr(self, "goal_combo"):
            key = getattr(self.project, "training_goal", DEFAULT_GOAL) if self.project else DEFAULT_GOAL
            idx = self.goal_combo.findData(key)
            if idx >= 0 and self.goal_combo.currentIndex() != idx:
                self.goal_combo.blockSignals(True)
                self.goal_combo.setCurrentIndex(idx)
                self.goal_combo.blockSignals(False)
            self._refresh_goal_hint()
        if hasattr(self, "media_combo"):
            # Only meaningful when the preset actually has separate guidance.
            show = self.preset.has_media_variants
            self.media_combo.setVisible(show)
            self.media_label.setVisible(show)
            mode = getattr(self.project, "media_mode", "auto") if self.project else "auto"
            idx = self.media_combo.findData(mode)
            if idx >= 0 and self.media_combo.currentIndex() != idx:
                self.media_combo.blockSignals(True)
                self.media_combo.setCurrentIndex(idx)
                self.media_combo.blockSignals(False)
        if hasattr(self, "preset_combo"):
            idx = self.preset_combo.findData(self.preset.key)
            if idx >= 0 and self.preset_combo.currentIndex() != idx:
                self.preset_combo.blockSignals(True)
                self.preset_combo.setCurrentIndex(idx)
                self.preset_combo.blockSignals(False)
        # Bounding boxes are Ideogram-only: hide the canvas tools and clear any boxes.
        ts = getattr(self, "_toolstrip", None)
        if ts is not None:
            ts.setVisible(self.preset.has_boxes)
        if plain:
            for it in list(getattr(self, "box_items", [])):
                if it.scene() is not None:
                    it.scene().removeItem(it)
            self.box_items = []
        self._sync_preset_actions()

    def _sync_preset_actions(self) -> None:
        """Show only the AI actions the active preset actually supports, and label them
        for that preset. Refine and Locate are JSON-schema operations, so they have no
        meaning for a plain-text caption; the convert-from-.txt row is likewise hidden,
        since there a .txt sidecar is the app's own output, not a source to convert."""
        structured = not self.preset.is_plain
        run = getattr(self, "btn_run_captioning", None)
        if run is not None:
            run.setText("Run JSON Captioning" if structured else "Run Captioning")
            run.setToolTip(
                "Generate the Ideogram JSON from the image. Choose one image or the "
                "whole folder."
                if structured else
                "Write a caption for the image. Choose one image or the whole folder."
            )
        for attr in ("btn_refine", "btn_locate"):
            btn = getattr(self, attr, None)
            if btn is not None:
                btn.setVisible(structured)
        conv = getattr(self, "_convert_row", None)
        if conv is not None:
            conv.setVisible(structured)
        # The LLM-instructions bar is deliberately preset-agnostic — it always shows
        # what would be sent for the active preset — so it stays enabled and visible.
        # Only the raw-JSON inspector (a structured-caption view) is gated.
        if hasattr(self, "rawjson_btn"):
            self.rawjson_btn.setVisible(structured)

    def _on_preset_changed(self, _idx: int) -> None:
        key = self.preset_combo.currentData()
        if not key or key == self.preset.key:
            return
        new = self.available_presets().get(key, get_preset(key))
        if self.store is not None and self.images and new.extension != self.preset.extension:
            resp = QMessageBox.question(
                self, "Change caption preset",
                f"Switch this folder to \u201c{new.label}\u201d?\n\n"
                f"Captions will be read from and written to {new.extension} files. "
                f"Existing {self.preset.extension} captions are left untouched on disk.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if resp != QMessageBox.Yes:
                self._apply_preset_ui()   # revert the combo
                return
        if self.current is not None:
            self._commit_active_caption()
        self.preset = new
        if self.project is not None:
            self.project.preset = new.key
        if self.store is not None:
            self.store.extension = new.extension
            try:
                self.store.save_project(self.project)
            except OSError:
                pass
        self._pending = {}
        self._apply_preset_ui()
        self._refresh_json_view()
        if self.current is not None:
            self.load_caption_for(self.current)
        self._refresh_all_markers()
        self._set_status(f"Preset: {new.label} ({new.extension})")

    def _refresh_all_markers(self) -> None:
        for path in getattr(self, "images", []):
            self._refresh_thumb_marker(path)

    def _update_plain_count(self) -> None:
        if not hasattr(self, "plain_count"):
            return
        text = self.cap_plain.toPlainText().strip()
        words = len(text.split()) if text else 0
        self.plain_count.setText(f"{words} words \u00b7 {len(text)} characters")

    def _commit_active_caption(self) -> None:
        """Fold live edits into the in-memory caption for whichever editor is active."""
        if self.preset.is_plain:
            self.current_text = self.cap_plain.toPlainText().strip()
        else:
            self.commit_caption_fields()
            self.commit_element_fields()

    def _toggle_raw_json(self) -> None:
        """Raw caption JSON in its own popup — an inspector for the current caption,
        separate from the persistent LLM-instructions bar."""
        if self.preset.is_plain:
            return
        self._commit_active_caption()
        try:
            pretty = json.dumps(json.loads(serialize_caption(self.current_caption)),
                                indent=2, ensure_ascii=False)
        except Exception as exc:
            pretty = f"(Could not render the caption as JSON: {exc})"
        name = self.current.name if self.current is not None else ""
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Raw caption JSON \u2014 {name}" if name else "Raw caption JSON")
        dlg.resize(720, 560)
        lay = QVBoxLayout(dlg)
        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.NoWrap)
        view.setFont(QFont(self.settings.mono_font_family or "Monospace"))
        view.setPlainText(pretty)
        lay.addWidget(view, 1)
        row = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(
            lambda: QApplication.clipboard().setText(view.toPlainText()))
        row.addWidget(copy_btn)
        save_btn = QPushButton("Save as\u2026")
        save_btn.clicked.connect(self._save_json_as)
        row.addWidget(save_btn)
        row.addStretch(1)
        close = QPushButton("Close")
        close.setDefault(True)
        close.clicked.connect(dlg.accept)
        row.addWidget(close)
        lay.addLayout(row)
        dlg.exec()

    def open_system_prompt(self) -> None:
        """Edit the system prompt the LLM is given for the active preset. Stored per
        preset in QSettings so switching presets brings back its own prompt."""
        media = self.current_media_kind()
        current = self.system_prompt_for_preset(media)
        dlg = QDialog(self)
        dlg.setWindowTitle(f"System prompt \u2014 {self.preset.label}")
        dlg.setMinimumWidth(620)
        dlg.resize(720, 520)
        lay = QVBoxLayout(dlg)
        hint = QLabel(
            "Sent to the model as the system message when captioning with this preset. "
            "Photos and clips have separate prompts \u2014 switch between them below."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet(f"color: {self.theme.text_secondary};")
        lay.addWidget(hint)

        media_row = QHBoxLayout()
        media_row.addWidget(QLabel("Editing prompt for:"))
        media_combo = QComboBox()
        media_combo.addItem("Photos", "image")
        media_combo.addItem("Videos", "video")
        media_combo.setCurrentIndex(1 if media == "video" else 0)
        media_row.addWidget(media_combo, 1)
        media_row.addStretch(1)
        lay.addLayout(media_row)
        edit = QPlainTextEdit()
        edit.setPlainText(current)
        lay.addWidget(edit, 1)
        if not self.preset.is_plain:
            note = QLabel(
                "This preset builds its prompt from the structured templates in "
                "captioner_prompts/; edits here are used as an extra instruction."
            )
            note.setWordWrap(True)
            note.setStyleSheet(f"color: {self.theme.warning}; font-size: 11px;")
            lay.addWidget(note)
        pending: dict[str, str] = {}

        def _on_media_changed(_idx: int) -> None:
            nonlocal media
            pending[media] = edit.toPlainText()      # keep unsaved edits per media
            media = media_combo.currentData()
            edit.setPlainText(pending.get(media, self.system_prompt_for_preset(media)))

        media_combo.currentIndexChanged.connect(_on_media_changed)

        btns = QHBoxLayout()
        reset = QPushButton("Reset to default")
        reset.setToolTip("Restore this preset's built-in prompt for the media above")
        reset.clicked.connect(
            lambda: edit.setPlainText(self.default_system_prompt_for_preset(media)))
        btns.addWidget(reset)
        btns.addStretch(1)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(dlg.reject)
        btns.addWidget(cancel)
        save = QPushButton("Save")
        save.setDefault(True)
        save.clicked.connect(dlg.accept)
        btns.addWidget(save)
        lay.addLayout(btns)
        if dlg.exec() == QDialog.Accepted:
            pending[media] = edit.toPlainText()
            for kind, text in pending.items():
                self.qsettings.setValue(self._prompt_settings_key(kind), text)
            self._set_status(
                f"System prompt saved ({', '.join(sorted(pending))}).")

    def audio_captioning_enabled(self) -> bool:
        """Audio goes out only when the user wants it AND the model can actually
        hear — a vision-only model errors on an audio part rather than ignoring it.

        The profile's flag states intent; the mmproj states fact, so when the
        projector is on disk it wins. That matters for community Omni conversions:
        they can register under a vision architecture and ship a projector with no
        audio tower, which would otherwise fail only at request time.
        """
        return self.audio_status()[0]

    def audio_status(self) -> tuple[bool, str]:
        """(will audio be sent, why not). Captioning a clip without sound is a
        legitimate choice, but it should never be a surprise — a vision-only model
        describes lips moving and invents nothing, which reads like a broken
        caption unless you know audio was never sent."""
        if not getattr(self.settings, "send_clip_audio", True):
            return False, ("'Send clip audio' is off in Preferences \u2192 Pipeline.")
        try:
            config = runtime_config_for_task(self.settings, "caption")
        except Exception:
            return False, "No captioning model is configured."
        # Profile labels are picker text ("Download: … (~20GB)"); strip the framing
        # so the name reads naturally inside a sentence.
        label = (config.label or config.api_model or "the selected model")
        label = re.sub(r"^(Download|Use|Server):\s*", "", label).strip()
        label = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip() or "The selected model"
        detected = self._detected_mmproj_audio()
        if detected is False:
            return False, (f"{label} has no audio encoder \u2014 its projector is "
                           "vision-only. Choose an Omni profile to caption sound.")
        if detected is None and not config.supports_audio:
            return False, (f"{label} is a vision-only model. Choose an Omni profile "
                           "(Qwen3-Omni) to include dialogue and sound.")
        return True, ""

    def _detected_mmproj_audio(self) -> bool | None:
        """True/False from the resolved projector, or None when it isn't on disk
        yet (nothing downloaded, or it can't be read)."""
        try:
            path = existing_mmproj_path(self.settings, "caption")
        except Exception:
            return None
        if path is None:
            return None
        cached = getattr(self, "_mmproj_audio_cache", {})
        key = str(path)
        if key not in cached:
            cached[key] = mmproj_has_audio_encoder(path)
            self._mmproj_audio_cache = cached
        return cached[key]

    def available_goals(self) -> dict:
        """Built-in goals plus any the user defined, so one added in Preferences
        appears in the strip without a restart."""
        return load_goals(app_base_dir())

    def current_goal(self) -> "TrainingGoal":
        key = getattr(self.project, "training_goal", None) if self.project else None
        return self.available_goals().get(key or "", get_goal(key))

    def goal_rules_text(self) -> str:
        """The active goal's policy text, or "" for General."""
        goal = self.current_goal()
        return goal.rules.strip() if goal.has_rules else ""

    def effective_system_prompt(self, media: str | None = None) -> str:
        """What the run actually sends: the preset's format prompt, then the goal's
        content policy.

        Kept apart from system_prompt_for_preset() on purpose — that one is the
        *editable* prompt shown in the System prompt dialog, and folding the goal
        into it would bake a policy into the saved text where it couldn't be
        changed by switching goals.
        """
        parts = [self.system_prompt_for_preset(media).strip(), self.goal_rules_text()]
        return "\n\n".join(p for p in parts if p)

    def current_media_kind(self) -> str:
        """Which guidance to use: the folder's explicit choice, or the selected
        file's own kind when that's left on Auto."""
        mode = getattr(self.project, "media_mode", "auto") if self.project else "auto"
        if mode in ("image", "video"):
            return mode
        if self.current is not None and is_video(self.current):
            return "video"
        return "image"

    def _on_goal_changed(self, _idx: int) -> None:
        key = self.goal_combo.currentData() or DEFAULT_GOAL
        if self.project is None or key == getattr(self.project, "training_goal", None):
            return
        self.project.training_goal = key
        if self.store is not None:
            try:
                self.store.save_project(self.project)
            except OSError:
                pass
        self._refresh_goal_hint()
        self._refresh_json_view()
        # A different goal means materially different captions, so anything
        # captioned under the old one is stale in exactly the way edited guidance is.
        self._refresh_guidance_changes()
        label = self.available_goals().get(key, get_goal(key)).label
        self._set_status(f"Training goal: {label}. "
                         "Captions made under the previous goal are marked changed.")

    def _refresh_goal_hint(self) -> None:
        hint = getattr(self, "goal_hint", None)
        if hint is None:
            return
        goal = self.current_goal()
        hint.setText(goal.summary)
        hint.setStyleSheet(
            f"color: {self.theme.text_secondary}; font-size: 11px;")
        hint.setToolTip(goal.rules or goal.summary)

    def _on_media_mode_changed(self, _idx: int) -> None:
        mode = self.media_combo.currentData() or "auto"
        if self.project is None or mode == self.project.media_mode:
            return
        self.project.media_mode = mode
        if self.store is not None:
            try:
                self.store.save_project(self.project)
            except OSError:
                pass
        self._refresh_json_view()          # instructions bar follows immediately
        label = {"auto": "each file's own type", "image": "photos",
                 "video": "videos"}[mode]
        self._set_status(f"Captioning instructions now written for {label}.")

    def default_system_prompt_for_preset(self, media: str | None = None) -> str:
        return self.preset.prompt_for(media or self.current_media_kind())

    def _prompt_settings_key(self, media: str) -> str:
        # Per preset AND per media: an edited photo prompt must not overwrite the
        # clip prompt for the same preset.
        return f"system_prompt/{self.preset.key}/{media}"

    def system_prompt_for_preset(self, media: str | None = None) -> str:
        media = media or self.current_media_kind()
        return self.qsettings.value(
            self._prompt_settings_key(media),
            self.default_system_prompt_for_preset(media), str)

    # ---- video support ----

    def apply_video_edit(self, path: Path, info: VideoInfo, start_s: float,
                         end_s: float, target, crop=None, rotate: int = 0) -> None:
        """Re-encode a clip in place, backing up the untouched original first.

        Same contract as image crop: the dataset keeps one file per clip (which is
        what the trainers want), and .original/ holds the pristine source so the
        edit is always reversible.
        """
        if self.store is None:
            return
        if target is not None:
            plan = plan_for_target(info, target, start_s, end_s, crop=crop)
            plan = replace(plan, rotate=rotate)
        else:
            plan = VideoEditPlan(start_s=start_s, end_s=end_s, crop=crop,
                                 rotate=rotate)
        changes = plan.changes(info)
        if not changes:
            self._set_status("Nothing to apply \u2014 the clip already matches.")
            return
        detail = "\n".join(f"  \u2022 {c}" for c in changes)
        warning = ""
        if target is not None and plan.frame_limit:
            if plan.frame_limit < target.min_frames():
                warning = (
                    f"\n\nWarning: {plan.frame_limit} frames is "
                    f"{target.seconds_for_frames(plan.frame_limit):.2f}s, below "
                    f"{target.label}'s minimum of "
                    f"{target.seconds_for_frames(target.min_frames()):.2f}s. The clip "
                    "is too short for this target and may be rejected or padded in "
                    "training.")
            elif target.exact_fps and abs((info.fps or 0) - target.fps) > 0.01:
                warning = (f"\n\n{target.label} needs exactly {target.fps:g}.000 fps; "
                           f"this clip will be resampled from {info.fps:g}.")
        already = self.store.has_original_backup(path)
        note = ("\n\nThe original is already backed up in .original/ from an earlier "
                "edit; that backup is kept." if already else
                "\n\nThe untouched original will be copied to .original/ first.")
        if QMessageBox.question(
            self, "Apply video edit",
            f"Re-encode {path.name} with:\n\n{detail}{warning}{note}",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return
        # The player must let go of the file before we can replace it (Windows
        # especially will refuse otherwise).
        self.video_stage.release()
        try:
            self.store.backup_original(path)
        except OSError as exc:
            QMessageBox.critical(self, "Could not back up", str(exc))
            return
        progress = QProgressDialog(
            f"Re-encoding {path.name} \u2026", None, 0, 0, self)
        progress.setWindowTitle("Applying video edit")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        thread = _VideoEditThread(path, plan)
        thread.start()
        while thread.isRunning():
            QApplication.processEvents()
            thread.wait(50)
        thread.wait()
        progress.close()
        if not thread.ok:
            QMessageBox.critical(self, "Edit failed",
                                 thread.message or "ffmpeg failed.")
            self._reload_video_metadata()
            return
        self.video_stage.clear_pending(path)
        self._refresh_video_edit_marker(path)
        self._purge_poster_cache(path)
        self._reload_video_metadata()
        self._refresh_spec_markers()
        self._set_status(f"{path.name}: {', '.join(changes)}.")

    def _refresh_video_edit_marker(self, path: Path | None) -> None:
        if path is None:
            return
        item = getattr(self, "_thumb_items", {}).get(str(path))
        if item is None:
            return
        stage = getattr(self, "video_stage", None)
        pending = bool(stage and stage.has_pending_edits(path))
        item.setData(VIDEO_EDIT_ROLE, pending)
        item.setToolTip(self._thumb_tooltip(path))
        strip = getattr(self, "filmstrip", None)
        if strip is not None:
            strip.viewport().update()

    def pending_video_edits(self) -> list[Path]:
        stage = getattr(self, "video_stage", None)
        if stage is None:
            return []
        return [Path(p) for p in stage._pending_edits]

    def confirm_discard_video_edits(self, action: str) -> bool:
        """Ask before an action that would drop unapplied clip edits.

        Pending edits survive switching between clips, so this only fires where they
        genuinely can't: leaving the folder or closing the app.
        """
        pending = self.pending_video_edits()
        if not pending:
            return True
        names = ", ".join(p.name for p in pending[:4])
        if len(pending) > 4:
            names += f", and {len(pending) - 4} more"
        return QMessageBox.question(
            self, "Unapplied video edits",
            f"{len(pending)} clip(s) have edits that haven't been written:\n\n"
            f"{names}\n\nThey'll be lost if you {action}. Apply them first?",
            QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Cancel,
        ) == QMessageBox.Discard

    def _purge_poster_cache(self, path: Path) -> None:
        cache = self._poster_cache_dir()
        if cache is None:
            return
        for old in cache.glob(f"{path.stem}.*.jpg"):
            old.unlink(missing_ok=True)

    def _offer_ffmpeg_install(self) -> None:
        """Folder has videos but no ffmpeg: offer the one-click managed download,
        mirroring the llama.cpp flow. Declining still lists the videos (placeholder
        thumbs, no metadata)."""
        resp = QMessageBox.question(
            self, "ffmpeg needed for videos",
            "This folder contains videos, but ffmpeg isn't installed yet.\n\n"
            "ffmpeg powers video thumbnails, metadata and editing. Download a "
            "portable build now (~100 MB, kept inside the app folder)?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        if resp != QMessageBox.Yes:
            self._set_status("Videos listed without ffmpeg \u2014 thumbnails and "
                             "editing need it (Preferences any time).")
            return
        progress = QProgressDialog("Downloading ffmpeg \u2026", None, 0, 0, self)
        progress.setWindowTitle("Installing ffmpeg")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        thread = _FfmpegInstallThread()
        thread.status.connect(progress.setLabelText)
        thread.start()
        while thread.isRunning():
            QApplication.processEvents()
            thread.wait(50)
        thread.wait()
        progress.close()
        if thread.ok:
            self._set_status("ffmpeg installed.")
            self._reload_video_metadata()   # posters/badges fill in
        else:
            QMessageBox.critical(self, "ffmpeg install failed", thread.message)

    def _reload_video_metadata(self) -> None:
        for path in self.images:
            if not is_video(path):
                continue
            info = probe_video(path)
            if info is not None:
                self._video_info[str(path)] = info
            item = self._thumb_items.get(str(path))
            if item is not None:
                item.setData(DURATION_ROLE, info.duration_label if info else "video")
                self._thumb_base[str(path)] = self._thumb_pixmap(path)
                item.setIcon(self._decorated_thumb(path))
        if self.current is not None and is_video(self.current):
            self.show_image(self.current)

    def open_crop_dialog(self) -> None:
        """Crop/resize the current image. Backed by .original/ (first backup wins);
        after an edit, every cached view of the image is refreshed in place."""
        if self.current is None or getattr(self, "_read_only", False):
            return
        try:
            dlg = CropResizeDialog(self, self.current, self.theme)
        except ValueError as exc:
            QMessageBox.critical(self, "Crop / Resize", str(exc))
            return
        if dlg.exec() != QDialog.Accepted:
            return
        self._refresh_edited_image(self.current)

    def _refresh_edited_image(self, path: Path) -> None:
        """Reload one image everywhere after its pixels changed on disk: filmstrip
        thumbnail (re-decorated), hover-preview cache, and the main canvas."""
        key = str(path)
        self._preview_cache.pop(key, None)
        self._thumb_base[key] = self._thumb_pixmap(path)
        item = self._thumb_items.get(key)
        if item is not None:
            item.setIcon(self._decorated_thumb(path))
        self.show_image(path)
        # the caption didn't change, but show_image cleared the canvas boxes
        self.rebuild_boxes()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reposition_json_overlay()
        if self.pixmap_item is not None and not self._user_zoomed:
            self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self._update_zoom_label()

    # ---- caption read/write ---------------------------------------------
    def _mark_dirty(self, *args) -> None:
        if not self._loading:
            self._touch_dirty()

    def _touch_dirty(self) -> None:
        self._dirty = True
        self._refresh_thumb_marker(self.current)
        # Live-update the raw-JSON panel (debounced) when it's open.
        timer = getattr(self, "_json_live_timer", None)
        if timer is not None and getattr(self, "json_panel", None) and self.json_panel.isVisible():
            timer.start()

    def _live_json_refresh(self) -> None:
        if not getattr(self, "json_panel", None) or not self.json_panel.isVisible():
            return
        # Sync current field values into the caption dict (no disk write), then the
        # commit calls refresh the JSON view from the updated dict.
        self.commit_caption_fields()
        self.commit_element_fields()

    def _has_unsaved(self, path: Path | None) -> bool:
        if path is None:
            return False
        if str(path) in self._pending:
            return True
        return self.current == path and self._dirty

    def _thumb_label(self, path: Path) -> str:
        # red shadowed text + glow now signal "unsaved"; keep the dot for guidance
        prefix = "• " if self.project.has_per_file_guidance(path.name) else ""
        return prefix + path.name[:14]

    def _decorated_thumb(self, path: Path) -> QIcon:
        # The unsaved indicator is now an amber corner dot drawn by the
        # FilmstripDelegate, so the thumbnail itself stays unaltered.
        base = self._thumb_base.get(str(path))
        return QIcon(base) if base is not None else QIcon()

    def _thumb_banners(self, path: Path) -> list[tuple]:
        """Marker banners for the hover-preview stack, in display order (the two red
        states on top), each as (text, bg_color, fg_color, tooltip). Colors match the
        filmstrip dots; the problems banner carries the specific issues on its tooltip."""
        name = path.name
        dark = self.theme.surface_0
        violet = STALE_COLOR  # guidance family (stale + omit share this colour)
        out: list[tuple] = []
        if self.project.is_review_marked(name):
            out.append(("Flagged for review", FLAG_COLOR, "#ffffff", ""))
        stage = getattr(self, "video_stage", None)
        if stage is not None and stage.has_pending_edits(path):
            out.append(("Unapplied edits", self.theme.warning, dark,
                        "Trim, crop or rotation set but not written to the file yet "
                        "\u2014 open the clip and press Apply edit."))
        spec = self.spec_issues_for(path)
        if spec:
            out.append(("Doesn't meet the model's specs", SPEC_COLOR, dark,
                        "\u2022 " + "\n\u2022 ".join(spec)))
        issues = self.project.caption_issues(name)
        if issues:
            tip = "\u2022 " + "\n\u2022 ".join(issues)
            out.append(("Caption may have problems", REVIEW_COLOR, "#ffffff", tip))
        if self._has_unsaved(path):
            out.append(("Unsaved changes", self.theme.warning, dark, ""))
        if self.project.guidance_changed(name):
            folder_ch = self.project.folder_guidance_changed(name)
            image_ch = self.project.image_guidance_changed(name)
            if not folder_ch and not image_ch:
                # caption predates split-stamping — can't attribute the scope
                out.append(("Guidance changed", violet, dark, ""))
            else:
                if folder_ch:
                    out.append(("Folder guidance changed", violet, dark, ""))
                if image_ch:
                    out.append(("This file's guidance changed", violet, dark, ""))
        if self._image_is_omit_marked(path):
            out.append((".txt caption omitted", OMIT_COLOR, dark, ""))
        return out

    def spec_issues_for(self, path: Path) -> list[str]:
        """How this clip fails the active preset's model target, or [] if it's fine.

        Deliberately narrow: only the things a trainer will NOT fix for you.
        Trainers bucket by aspect ratio and resize during latent caching, so an
        arbitrary source resolution is a non-issue and flagging it was noise. What
        they don't handle is the temporal axis — LTX silently truncates or pads an
        off-grid frame count, and H3 wants exactly 24.000 fps.

        Checked live against the target rather than at import, because the answer
        changes the moment the preset changes — the same clip is legal for LTX and
        illegal for Wan. Only clips are checked; stills have no fps or frame grid.
        """
        target = self._preset_model_target()
        if target is None or not is_video(path):
            return []
        info = self._video_info.get(str(path))
        if info is None:
            return []
        issues: list[str] = []
        if target.exact_fps and abs(info.fps - target.fps) > 0.01:
            issues.append(
                f"{info.fps:g} fps \u2014 {target.label} needs exactly "
                f"{target.fps:g}.000 fps")
        elif abs(info.fps - target.fps) > 0.01:
            issues.append(f"{info.fps:g} fps \u2014 {target.label} trains at "
                          f"{target.fps:g}")
        frames_at_target = int(round(info.duration_s * target.fps))
        if frames_at_target > target.max_frames():
            issues.append(
                f"{info.duration_s:.2f}s is over {target.label}'s maximum of "
                f"{target.seconds_for_frames(target.max_frames()):.2f}s")
        elif frames_at_target < target.min_frames():
            issues.append(
                f"{info.duration_s:.2f}s is under {target.label}'s minimum of "
                f"{target.seconds_for_frames(target.min_frames()):.2f}s")
        elif not target.is_legal_frames(frames_at_target):
            issues.append(
                f"{frames_at_target} frames isn't on {target.label}'s "
                f"{target.frame_modulus}n+{target.frame_remainder} grid")
        return issues

    def _preset_model_target(self):
        """The model the spec flag judges clips against.

        The trim target wins when one is armed: the preset seeds it, but choosing a
        different one in the video stage is an explicit statement of what you're
        building for, and the amber warning should agree with it rather than keep
        answering for the preset.
        """
        stage = getattr(self, "video_stage", None)
        if stage is not None:
            chosen = stage.current_target()
            if chosen is not None:
                return chosen
        key = self.preset.model_target
        return self.model_targets.get(key) if key else None

    def postprocess_caption(self, text: str) -> str:
        """Preset-specific tidy-up applied to a freshly generated caption.

        H3 reads a newline as a shot change, so wrapped prose invents cuts. The
        prompt asks for single-line fields, but instruction-following varies by
        model, so the output is normalised rather than trusted.
        """
        if text and self.preset.key.startswith("minimax_h3"):
            return normalise_h3_caption(text)
        return text

    def _video_editing_active(self) -> bool:
        """True when a clip is on screen, so the video shortcuts should win."""
        stack = getattr(self, "center_stack", None)
        return stack is not None and stack.currentIndex() == 1

    def keyPressEvent(self, event) -> None:
        if self._video_editing_active() and self._video_key(event):
            event.accept()
            return
        super().keyPressEvent(event)

    def _video_key(self, event) -> bool:
        """Editing keys for a clip. Returns True when the key was consumed."""
        stage = getattr(self, "video_stage", None)
        if stage is None:
            return False
        key, mods = event.key(), event.modifiers()
        shift = bool(mods & Qt.ShiftModifier)
        plain = mods in (Qt.NoModifier, Qt.ShiftModifier)
        if not plain:
            return False
        step = 10 if shift else 1          # Shift jumps ten frames at a time
        if key in (Qt.Key_Left, Qt.Key_A):
            stage.step_frames(-step)
            return True
        if key in (Qt.Key_Right, Qt.Key_D):
            stage.step_frames(step)
            return True
        if key == Qt.Key_Space:
            stage.toggle_playback()
            return True
        if key == Qt.Key_I:
            stage.set_in_at_playhead()
            return True
        if key == Qt.Key_O:
            stage.set_out_at_playhead()
            return True
        if key == Qt.Key_V:
            stage.set_tool("playhead")
            return True
        if key == Qt.Key_H:
            stage.set_tool("select")
            return True
        if key == Qt.Key_Home:
            stage.seek_to_trim("in")
            return True
        if key == Qt.Key_End:
            stage.seek_to_trim("out")
            return True
        return False

    def available_presets(self) -> dict:
        """Built-ins plus any the user defined, so a preset added in Preferences
        shows up in the strip without a restart."""
        return all_presets(app_base_dir())

    def caption_issues_for(self, path: Path) -> list[str]:
        """Schema health for a caption, or [] when the active preset has no schema.

        The checks parse the file as Ideogram JSON, so running them against a plain
        .txt caption reports "corrupt — could not parse JSON" on every well-formed
        file. Only the validating preset gets flagged; everyone else is unflagged
        rather than wrongly flagged.
        """
        if self.store is None or not self.preset.validates:
            return []
        return self.store.caption_file_issues(path)

    def _refresh_spec_markers(self) -> None:
        """Re-evaluate every clip against the active preset's target.

        Runs on preset change because the verdict follows the target, not the file:
        a 30fps clip is fine under no target, wrong under Wan, and wrong differently
        under H3.
        """
        items = getattr(self, "_thumb_items", {})
        if not items:
            return
        flagged = 0
        for path in getattr(self, "images", []):
            item = items.get(str(path))
            if item is None:
                continue
            issues = self.spec_issues_for(path)
            item.setData(SPEC_ROLE, bool(issues))
            item.setToolTip(self._thumb_tooltip(path))
            flagged += 1 if issues else 0
        strip = getattr(self, "filmstrip", None)
        if strip is not None:
            strip.viewport().update()
        target = self._preset_model_target()
        if target is not None and flagged:
            self._set_status(
                f"{flagged} clip(s) don't meet {target.label}'s specs \u2014 hover the "
                "amber triangle, or use Fit to target and Apply edit.")

    def _clear_stale_review_flags(self) -> None:
        """Drop review flags recorded under a validating preset when the active one
        has no schema, so old red dots don't linger after a preset switch."""
        if self.project is None or self.preset.validates:
            return
        changed = False
        for path in getattr(self, "images", []):
            if self.project.caption_issues(path.name):
                self.project.set_flags(path.name, [])
                changed = True
        if changed and self.store is not None:
            try:
                self.store.save_project(self.project)
            except OSError:
                pass

    def _thumb_tooltip(self, path: Path) -> str:
        """Plain-text tooltip naming every marker on this thumbnail.

        The hover preview shows the same states as coloured banners, but a dot in a
        corner with no tooltip is unreadable — you have to already know the code to
        look it up. Built from _thumb_banners so the wording can't drift.
        """
        lines: list[str] = []
        for text, _bg, _fg, tip in self._thumb_banners(path):
            marker = {
                "Flagged for review": "\U0001F6A9 bottom-right flag",
                "Doesn't meet the model's specs": "\u25b2 amber triangle (right edge)",
                "Unapplied edits": "\u25dc amber arc (bottom-left)",
                "Caption may have problems": "\u25cf red dot (bottom-left)",
                "Unsaved changes": "\u25cf amber dot (top-right)",
                "Guidance changed": "\u25cf violet dot (top-left)",
                "Folder guidance changed": "\u25cf violet dot (top-left)",
                "This file's guidance changed": "\u25cf violet dot (top-left)",
                ".txt caption omitted": "\u25cf violet slashed dot (left)",
            }.get(text, "")
            line = f"{text} \u2014 {marker}" if marker else text
            if tip:
                line += "\n" + "\n".join(f"   {t}" for t in tip.splitlines())
            lines.append(line)
        if not lines:
            return path.name
        return f"{path.name}\n\n" + "\n".join(lines)

    def _rebuild_filmstrip(self) -> None:
        """Populate the strip from self.images, inserting the bypass divider."""
        self.filmstrip.clear()
        self._pending = {}
        self._thumb_items = {}
        self._thumb_base = {}
        for _a in self._dirty_dot_anims.values():
            _a.stop()
        self._dirty_dot_anims = {}
        self._dirty_dot = {}
        self._preview_cache = {}
        self._hover_item = None
        self._hover_preview.hide()
        if not self.images:
            # The folder comes from the store: this runs on any re-list (a bypass,
            # a delete, an import), not only from open_folder where a local existed.
            where = self.store.folder if self.store is not None else "this folder"
            self._set_status(f"No files found in {where}")
            self._update_count_label()
            return

        self._video_info: dict[str, VideoInfo] = {}
        has_videos = any(is_video(p) for p in self.images)
        if has_videos and not ffmpeg_available():
            self._offer_ffmpeg_install()
        divider_added = False
        for path in self.images:
            self._thumb_base[str(path)] = self._thumb_pixmap(path)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, str(path))
            if is_video(path):
                info = probe_video(path) if ffmpeg_available() else None
                if info is not None:
                    self._video_info[str(path)] = info
                item.setData(DURATION_ROLE, info.duration_label if info else "video")
            item.setData(UNSAVED_ROLE, False)
            item.setData(STALE_ROLE, self.project.guidance_changed(path.name))
            # Re-validate the caption file on disk (catches hand-edits / corruption
            # since the last run), not just the flags stamped at generation time.
            _issues = self.caption_issues_for(path)
            self.project.set_flags(path.name, _issues)
            _flagged = self.project.is_review_marked(path.name)
            item.setData(REVIEW_ROLE, bool(_issues))
            item.setData(FLAG_ROLE, _flagged)
            item.setData(OMIT_ROLE, self._image_is_omit_marked(path))
            item.setData(SPEC_ROLE, bool(self.spec_issues_for(path)))
            bypassed = self.store.is_bypassed(path)
            item.setData(BYPASS_ROLE, bypassed)
            if bypassed and not divider_added:
                divider = QListWidgetItem()
                divider.setData(SEPARATOR_ROLE, True)
                divider.setFlags(Qt.NoItemFlags)   # not selectable, not a file
                divider.setSizeHint(QSize(46, self.filmstrip.iconSize().height() + 44))
                self.filmstrip.addItem(divider)
                divider_added = True
            item.setToolTip(self._thumb_tooltip(path))
            self.filmstrip.addItem(item)
            self._thumb_items[str(path)] = item
            item.setIcon(self._decorated_thumb(path))
            item.setText(self._thumb_label(path))

    # ---- adding media ----

    def open_duplicate_dataset(self) -> None:
        """Copy this dataset somewhere else, choosing what comes along.

        Two jobs in one: a backup (take everything) and a variant for a training run
        (media only, or media without captions to re-caption differently). Which one
        you're doing is expressed by the toggles rather than by two separate tools.
        """
        if self.store is None or not self.images:
            self._set_status("Open a folder with some files first.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Duplicate / back up dataset")
        dlg.setMinimumWidth(560)
        lay = QVBoxLayout(dlg)

        intro = QLabel(
            f"Copies {len(self.images)} file(s) from {self.store.folder.name} to a "
            "new folder. Nothing in this folder is changed.")
        intro.setObjectName("Hint")
        intro.setWordWrap(True)
        lay.addWidget(intro)

        dest_row = QHBoxLayout()
        dest_row.addWidget(QLabel("Copy to:"))
        dest_edit = QLineEdit(str(self.store.folder.parent /
                                  f"{self.store.folder.name}-copy"))
        dest_row.addWidget(dest_edit, 1)
        browse = QPushButton("Browse\u2026")

        def pick():
            chosen = QFileDialog.getExistingDirectory(
                dlg, "Choose destination", str(self.store.folder.parent))
            if chosen:
                dest_edit.setText(chosen)
        browse.clicked.connect(pick)
        dest_row.addWidget(browse)
        lay.addLayout(dest_row)
        lay.addSpacing(8)

        keep_captions = QCheckBox("Keep captions")
        keep_captions.setChecked(True)
        keep_captions.setToolTip(
            "Copy each file's caption sidecar. Turn off to duplicate the media for "
            "captioning a different way.")
        keep_settings = QCheckBox("Keep captioner settings (preset, guidance, flags)")
        keep_settings.setChecked(True)
        keep_settings.setToolTip(
            "Copy .captioner/project.json, so the copy opens with the same preset, "
            "training goal and guidance.")
        keep_originals = QCheckBox("Keep original backups (.original/)")
        keep_originals.setChecked(False)
        keep_originals.setToolTip(
            "Copy the pre-edit files kept by crop, resize, rotate and mute. Useful "
            "for a true backup; dead weight for a training copy.")
        keep_bypassed = QCheckBox("Keep bypassed files (.bypass/)")
        keep_bypassed.setChecked(True)
        keep_bypassed.setToolTip(
            "Copy files you've excluded from the dataset. They stay excluded in "
            "the copy.")
        for box in (keep_captions, keep_settings, keep_originals, keep_bypassed):
            lay.addWidget(box)

        lay.addSpacing(8)
        buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
        go = buttons.addButton("Copy", QDialogButtonBox.AcceptRole)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        lay.addWidget(buttons)

        if dlg.exec() != QDialog.Accepted:
            return
        dest = Path(dest_edit.text().strip()).expanduser()
        if not self._duplicate_destination_ok(dest):
            return
        self._run_duplicate(dest, keep_captions.isChecked(), keep_settings.isChecked(),
                            keep_originals.isChecked(), keep_bypassed.isChecked())

    def _duplicate_destination_ok(self, dest: Path) -> bool:
        """Refuse the destinations that would damage or endlessly recurse."""
        if not str(dest):
            return False
        source = self.store.folder.resolve()
        try:
            resolved = dest.resolve()
        except OSError:
            resolved = dest
        if resolved == source:
            QMessageBox.warning(self, "Choose a different folder",
                                "That's the folder you're copying from.")
            return False
        if source in resolved.parents:
            QMessageBox.warning(
                self, "Choose a folder outside this one",
                "Copying a folder into itself would recurse. Pick a destination "
                "beside it instead.")
            return False
        if resolved.is_dir() and any(resolved.iterdir()):
            return QMessageBox.question(
                self, "Folder isn't empty",
                f"{resolved.name} already has files in it. Copy into it anyway?\n\n"
                "Files with matching names will be overwritten.",
                QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
            ) == QMessageBox.Yes
        return True

    def _run_duplicate(self, dest: Path, captions: bool, settings: bool,
                       originals: bool, bypassed: bool) -> None:
        bar = QProgressDialog("Copying\u2026", "Cancel", 0, 100, self)
        bar.setWindowTitle("Duplicate dataset")
        bar.setWindowModality(Qt.WindowModal)
        bar.setMinimumDuration(0)
        bar.setValue(0)

        def tick(done: int, total: int, name: str) -> bool:
            bar.setMaximum(max(1, total))
            bar.setValue(done)
            bar.setLabelText(f"Copying {name}\u2026")
            QApplication.processEvents()
            return not bar.wasCanceled()

        try:
            counts = self.store.duplicate_to(
                dest, keep_captions=captions, keep_settings=settings,
                keep_originals=originals, keep_bypassed=bypassed, progress=tick)
        except OSError as exc:
            bar.close()
            QMessageBox.critical(self, "Could not copy", str(exc))
            return
        bar.close()
        parts = [f"{counts[k]} {k}" for k in
                 ("media", "captions", "originals", "bypassed", "settings")
                 if counts.get(k)]
        summary = ", ".join(parts) or "nothing"
        if counts.get("cancelled"):
            QMessageBox.information(
                self, "Copy cancelled",
                f"Stopped partway. {summary} had already been copied to "
                f"{dest}.\n\nThe partial copy was left in place.")
            self._set_status(f"Copied {summary} to {dest}.")
            return
        note = (f"\n\n{counts['skipped']} file(s) could not be copied."
                if counts.get("skipped") else "")
        # Which folder you want to be in afterwards depends on why you copied:
        # a backup means stay, a variant means switch. Only the user knows which,
        # so ask rather than guess — and default to staying, since that's the
        # non-destructive answer if the dialog is dismissed.
        box = QMessageBox(self)
        box.setWindowTitle("Copy finished")
        box.setIcon(QMessageBox.Information)
        box.setText(f"Copied {summary} to {dest.name}.{note}")
        box.setInformativeText(
            "Open the copy now, or keep working in the original folder?")
        switch_btn = box.addButton("Open the copy", QMessageBox.AcceptRole)
        stay_btn = box.addButton("Stay here", QMessageBox.RejectRole)
        box.setDefaultButton(stay_btn)
        box.exec()
        if box.clickedButton() is switch_btn:
            self.load_folder_path(dest)
            self._set_status(f"Copied {summary} \u2014 now working in {dest.name}.")
            return
        self._set_status(f"Copied {summary} to {dest}.")

    def open_add_media(self) -> None:
        """File picker for adding media to the dataset."""
        if self.store is None:
            self._set_status("Open a folder first.")
            return
        exts = sorted(set(IMAGE_EXTENSIONS) | VIDEO_EXTENSIONS)
        pattern = " ".join(f"*{e}" for e in exts)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add media to this dataset", "",
            f"Images and video ({pattern});;All files (*)")
        if paths:
            self.add_media([Path(p) for p in paths])

    def add_media(self, sources: list[Path]) -> None:
        """Copy files in, re-list, and select the first one added."""
        if self.store is None or not sources:
            return
        if getattr(self, "_read_only", False):
            self._set_status("Read-only mode \u2014 nothing was added.")
            return
        try:
            added, skipped = self.store.import_media(sources)
        except Exception as exc:
            QMessageBox.critical(self, "Could not add media", str(exc))
            return
        if not added:
            QMessageBox.information(
                self, "Nothing added",
                "None of those could be added:\n\n" + "\n".join(skipped[:10])
                if skipped else "No images or videos were found.")
            return
        self.images = self.store.images()
        self._rebuild_filmstrip()
        row = self._row_for_path(added[0])
        if row is not None:
            self.filmstrip.setCurrentRow(row)
        message = f"Added {len(added)} file(s)."
        if skipped:
            message += f" Skipped {len(skipped)}: {skipped[0]}"
            if len(skipped) > 1:
                message += f" (+{len(skipped) - 1} more)"
        self._set_status(message)

    def _media_from_mime(self, mime) -> list[Path]:
        if not mime.hasUrls():
            return []
        return [Path(url.toLocalFile()) for url in mime.urls()
                if url.isLocalFile() and url.toLocalFile()]

    def dragEnterEvent(self, event) -> None:
        if self.store is not None and self._media_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        if self.store is not None and self._media_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        paths = self._media_from_mime(event.mimeData())
        if self.store is None or not paths:
            super().dropEvent(event)
            return
        event.acceptProposedAction()
        # A single dropped folder with no folder open would more likely mean "open
        # this" than "copy its contents into the current dataset".
        self.add_media(paths)

    def restore_original_media(self, path: Path | None = None) -> None:
        """Revert a file to its pre-edit state and forget the backup."""
        path = path or self.current
        if self.store is None or path is None:
            return
        if not self.store.has_original_backup(path):
            self._set_status(f"{path.name} has no backup \u2014 it hasn't been edited.")
            return
        if QMessageBox.question(
            self, "Restore original",
            f"Undo every edit to {path.name} and put the original back?\n\n"
            "The backup is removed, so this can't be undone again.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        ) != QMessageBox.Yes:
            return
        if not self.store.revert_to_original(path):
            QMessageBox.critical(self, "Could not restore", "The backup has gone.")
            return
        self._refresh_edited_image_cached(path)
        if is_video(path):
            self._reload_video_metadata()
            self._refresh_spec_markers()
        else:
            self.show_image(path)
        self._set_status(f"{path.name} restored to its original.")

    def remove_media(self, path: Path | None = None) -> None:
        """Offer the two ways to take a file out: reversibly, or for good."""
        path = path or self.current
        if self.store is None or path is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Remove from dataset")
        box.setIcon(QMessageBox.Question)
        box.setText(f"Take {path.name} out of the dataset?")
        box.setInformativeText(
            "Bypass keeps the file and its caption in .bypass/, so you can bring "
            "them back.\n\nDelete removes the file, its caption and any backup "
            "from disk permanently.")
        bypass_btn = box.addButton("Bypass", QMessageBox.AcceptRole)
        delete_btn = box.addButton("Delete permanently", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(bypass_btn)
        box.exec()
        clicked = box.clickedButton()
        if clicked is bypass_btn:
            self.toggle_bypass(path)
            return
        if clicked is not delete_btn:
            return
        # Second confirmation: this one can't be undone.
        if QMessageBox.question(
            self, "Delete permanently",
            f"Permanently delete {path.name}, its caption and any backup?\n\n"
            "This cannot be undone.",
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        removed = self.store.delete_media(path)
        self.images = self.store.images()
        self._rebuild_filmstrip()
        if self.images:
            self.filmstrip.setCurrentRow(0)
        self._set_status(f"Deleted {len(removed)} file(s) for {path.name}.")

    def toggle_bypass(self, path: Path | None = None) -> None:
        """Move a file out of the dataset (or bring it back).

        The file physically moves to .bypass/ with its caption, because the trainer
        reads the folder — a flag in project.json wouldn't actually exclude it.
        """
        path = path or self.current
        if self.store is None or path is None:
            return
        bypassed = self.store.is_bypassed(path)
        try:
            new_path = (self.store.unbypass(path) if bypassed
                        else self.store.bypass(path))
        except OSError as exc:
            QMessageBox.critical(self, "Could not move the file", str(exc))
            return
        verb = "restored to the dataset" if bypassed else "bypassed"
        self._reload_folder_keeping(new_path)
        self._set_status(f"{new_path.name} {verb}.")

    def _reload_folder_keeping(self, path: Path) -> None:
        """Re-list the folder and reselect the file at its new location.

        A full re-list rather than patching one row: the move changes ordering (the
        bypassed group sorts last) and inserts or removes the divider, so the row
        indices shift.
        """
        if self.store is None:
            return
        self.images = self.store.images()
        self._rebuild_filmstrip()
        row = self._row_for_path(path)
        if row is not None:
            self.filmstrip.setCurrentRow(row)

    def _row_for_path(self, path: Path) -> int | None:
        """Filmstrip row for a file. Rows and image indices diverge once the divider
        is present, so look the item up rather than assuming they match."""
        for row in range(self.filmstrip.count()):
            item = self.filmstrip.item(row)
            if item is not None and item.data(Qt.UserRole) == str(path):
                return row
        return None

    def _toggle_review_flag(self, *args) -> None:
        if self.store is None or self.current is None:
            return
        marked = self.project.toggle_review_mark(self.current.name)
        try:
            self.store.save_project(self.project)
        except OSError:
            pass
        self._refresh_thumb_marker(self.current)
        self._sync_flag_action()
        self._set_status("Flagged for review" if marked else "Review flag cleared")

    def _toggle_review_flag_for(self, path: Path) -> None:
        if self.store is None:
            return
        self.project.toggle_review_mark(path.name)
        try:
            self.store.save_project(self.project)
        except OSError:
            pass
        self._refresh_thumb_marker(path)
        if path == self.current:
            self._sync_flag_action()

    def _sync_flag_action(self) -> None:
        action = getattr(self, "flag_action", None)
        if action is not None:
            marked = self.current is not None and self.project.is_review_marked(self.current.name)
            action.setChecked(bool(marked))

    def _next_flagged_image(self, *args) -> None:
        count = self.filmstrip.count()
        if count == 0:
            return
        flagged = [
            i for i in range(count)
            if self.project.is_review_marked(Path(self.filmstrip.item(i).data(Qt.UserRole)).name)
        ]
        if not flagged:
            self._set_status("No files flagged for review")
            return
        cur = self.filmstrip.currentRow()
        nxt = next((r for r in flagged if r > cur), flagged[0])  # wrap to first
        self.filmstrip.setCurrentRow(nxt)
        self._set_status(f"Flagged for review ({flagged.index(nxt) + 1}/{len(flagged)})")

    def _filmstrip_context_menu(self, pos) -> None:
        item = self.filmstrip.itemAt(pos)
        if item is None:
            return
        path = Path(item.data(Qt.UserRole))
        menu = QMenu(self)
        marked = self.project.is_review_marked(path.name)
        flag_act = menu.addAction("Clear review flag" if marked else "Flag for review")
        bypassed = self.store.is_bypassed(path)
        bypass_act = menu.addAction(
            "Restore to dataset" if bypassed else "Bypass (exclude from dataset)")
        revert_act = None
        if self.store.has_original_backup(path):
            revert_act = menu.addAction("Restore original (undo edits)")
        menu.addSeparator()
        remove_act = menu.addAction("Remove\u2026")
        # Editing lives on an unlabelled icon in the left rail, which is easy to
        # miss — offer it where the file actually is.
        edit_act = None
        if not is_video(path):
            menu.addSeparator()
            edit_act = menu.addAction("Crop / rotate / resize\u2026")
        chosen = menu.exec(self.filmstrip.mapToGlobal(pos))
        if chosen == flag_act:
            self._toggle_review_flag_for(path)
        elif chosen == bypass_act:
            self.toggle_bypass(path)
        elif revert_act is not None and chosen == revert_act:
            self.restore_original_media(path)
        elif chosen == remove_act:
            self.remove_media(path)
        elif edit_act is not None and chosen == edit_act:
            if self.current != path and path in self.images:
                self.filmstrip.setCurrentRow(self.images.index(path))
            self.open_crop_dialog()

    def _refresh_thumb_marker(self, path: Path | None) -> None:
        if path is None:
            return
        item = self._thumb_items.get(str(path))
        if item is not None:
            was = bool(item.data(UNSAVED_ROLE))
            now = self._has_unsaved(path)
            item.setData(UNSAVED_ROLE, now)
            item.setData(STALE_ROLE, self.project.guidance_changed(path.name))
            issues = self.project.caption_issues(path.name)
            flagged = self.project.is_review_marked(path.name)
            item.setData(REVIEW_ROLE, bool(issues))
            item.setData(FLAG_ROLE, flagged)
            item.setData(OMIT_ROLE, self._image_is_omit_marked(path))
            item.setData(SPEC_ROLE, bool(self.spec_issues_for(path)))
            item.setToolTip(self._thumb_tooltip(path))
            item.setText(self._thumb_label(path))
            item.setIcon(self._decorated_thumb(path))
            if now != was:
                self._animate_dirty_dot(str(path), 1.0 if now else 0.0)
        if path == self.current:
            self._refresh_title()

    def _refresh_title(self) -> None:
        """Window filename header. Turns amber and appends an unsaved marker when
        the current image has uncommitted edits."""
        path = self.current
        if path is None:
            self.title_label.setText("")
            self.title_label.setStyleSheet("")
            return
        dot = "● " if self.project.has_per_file_guidance(path.name) else ""
        if self._has_unsaved(path):
            self.title_label.setText(f"{dot}{path.name} - Unsaved Changes")
            self.title_label.setStyleSheet(f"color: {self.theme.warning};")
        else:
            self.title_label.setText(f"{dot}{path.name}")
            self.title_label.setStyleSheet("")

    def _schedule_stale_refresh(self, *args) -> None:
        if not self._loading and getattr(self, "_stale_timer", None) is not None:
            self._stale_timer.start()

    def _refresh_stale_state(self) -> None:
        """Recompute guidance staleness for every thumbnail and the current pane.
        Commits the live editor text into the in-memory project first so a folder
        or per-file edit flags immediately, before it's persisted to disk."""
        if self.store is None:
            self._refresh_guidance_changes()
            return
        if getattr(self, "_used_tags_flow", None) is not None:
            self.commit_guidance()
        for key, item in self._thumb_items.items():
            stale = self.project.guidance_changed(Path(key).name)
            if bool(item.data(STALE_ROLE)) != stale:
                item.setData(STALE_ROLE, stale)
                item.setToolTip(self._thumb_tooltip(path))
                self._repaint_thumb(key)
        self._refresh_guidance_changes()

    def _refresh_guidance_changes(self) -> None:
        """Show/hide the compact 'guidance changed' section for the current image.
        The full diff is shown on hover via the pop-out, so nothing to render inline."""
        if not hasattr(self, "_gchg_box"):
            return
        name = self.current.name if self.current is not None else None
        changed = name is not None and self.project.guidance_changed(name)
        self._gchg_box.setVisible(changed)
        if not changed:
            self._hide_gdiff_popup()

    @staticmethod
    def _guidance_diff_html(prev: str, curr: str) -> str:
        """Line-level diff: removed lines struck through and muted, added lines in
        the violet stale color. Unchanged lines are omitted to keep it compact."""
        rows: list[str] = []
        for line in difflib.ndiff(prev.splitlines(), curr.splitlines()):
            code, body = line[:2], html.escape(line[2:]).strip()
            if not body:
                continue
            if code == "+ ":
                rows.append(f'<span style="color:{STALE_COLOR}">+ {body}</span>')
            elif code == "- ":
                rows.append(
                    f'<span style="color:#6C737C;text-decoration:line-through">'
                    f'− {body}</span>'
                )
        if not rows:
            return '<span style="color:#6C737C">Guidance text changed.</span>'
        return "<br>".join(rows)

    def _anchor_popup_to_window(self, pop) -> None:
        """Wayland refuses to map a frameless popup/tooltip surface unless it has a
        transient parent (KDE/Plasma: 'Failed to create popup ... has a transientParent
        set'). Qt otherwise falls back to the currently-active window, which is
        unreliable — e.g. right after the guidance dialog closes, which is exactly when
        users saw the hover popups stop appearing. Pin every hover popup to the main
        window's surface before it's shown. Harmless on X11/Windows/macOS."""
        try:
            host = self.windowHandle()
            if host is None:
                return
            if pop.windowHandle() is None:
                pop.winId()  # force native surface so windowHandle() exists
            ph = pop.windowHandle()
            if ph is not None:
                ph.setTransientParent(host)
        except Exception:
            pass

    def _ensure_gdiff_popup(self) -> "GuidanceDiffPopup":
        pop = getattr(self, "_gdiff_popup", None)
        if pop is None:
            pop = GuidanceDiffPopup(self.theme, None)
            self._gdiff_popup = pop
        return pop

    def _show_gdiff_popup(self) -> None:
        box = getattr(self, "_gchg_box", None)
        if box is None or box.isHidden() or self.current is None:
            return
        name = self.current.name
        if not self.project.guidance_changed(name):
            return
        prev = self.project.last_run_guidance(name) or ""
        curr = self.project.resolved_for(name)
        html = self._guidance_diff_html(prev, curr)
        pop = self._ensure_gdiff_popup()
        target = box.mapToGlobal(QPoint(box.width(), 0))
        try:
            screen = self.screen().availableGeometry()
        except Exception:
            screen = None
        self._anchor_popup_to_window(pop)
        pop.show_diff(html, target, screen)

    def _hide_gdiff_popup(self) -> None:
        pop = getattr(self, "_gdiff_popup", None)
        if pop is not None:
            pop.hide()

    def _ensure_tags_popup(self) -> "TagListPopup":
        pop = getattr(self, "_tags_popup", None)
        if pop is None:
            pop = TagListPopup(self.theme, None)
            self._tags_popup = pop
        return pop

    def _show_tags_popup(self) -> None:
        pill = getattr(self, "_used_tags_collapsed", None)
        used = getattr(self, "_used_tags_used", [])
        if pill is None or pill.isHidden() or not used:
            return
        pop = self._ensure_tags_popup()
        target = pill.mapToGlobal(QPoint(pill.width(), 0))
        try:
            screen = self.screen().availableGeometry()
        except Exception:
            screen = None
        self._anchor_popup_to_window(pop)
        pop.show_tags(self._make_used_pill, used, target, screen)

    def _hide_tags_popup(self) -> None:
        pop = getattr(self, "_tags_popup", None)
        if pop is not None:
            pop.hide()

    def _repaint_thumb(self, key: str) -> None:
        item = self._thumb_items.get(key)
        if item is None:
            return
        self.filmstrip.viewport().update(self.filmstrip.visualItemRect(item))

    def _animate_dirty_dot(self, key: str, target: float) -> None:
        """Scale + fade the unsaved corner dot in (120ms OutCubic) or out (90ms
        OutQuad). Driven per-item; the delegate reads progress from _dirty_dot."""
        old = self._dirty_dot_anims.pop(key, None)
        if old is not None:
            old.stop()
        start = self._dirty_dot.get(key, 0.0 if target > 0 else 1.0)
        anim = QVariantAnimation(self)
        anim.setStartValue(float(start))
        anim.setEndValue(float(target))
        anim.setDuration(DOT_APPEAR if target > 0 else DOT_DISAPPEAR)
        anim.setEasingCurve(QEasingCurve.OutCubic if target > 0 else QEasingCurve.OutQuad)

        def on_value(v) -> None:
            self._dirty_dot[key] = float(v)
            self._repaint_thumb(key)

        def on_done() -> None:
            if target <= 0:
                self._dirty_dot.pop(key, None)
            else:
                self._dirty_dot[key] = 1.0
            self._repaint_thumb(key)
            self._dirty_dot_anims.pop(key, None)

        anim.valueChanged.connect(on_value)
        anim.finished.connect(on_done)
        self._dirty_dot_anims[key] = anim
        anim.start()

    def _apply_style_mode_label(self, mode: str) -> None:
        self.style_detail_label.setText("Art style" if mode == "art_style" else "Photo")

    def _on_style_mode_changed(self, mode: str) -> None:
        self._apply_style_mode_label(mode)
        self._mark_dirty()

    def load_caption_for(self, path: Path) -> None:
        if self.store is None:
            return
        key = str(path)
        if self.preset.is_plain:
            if key in self._pending:
                text, message, pending = self._pending[key], None, True
            else:
                text, message = self.store.load_plain_caption(path)
                pending = False
            self.current_text = text
            self._loading = True
            try:
                self.cap_plain.setPlainText(text)
            finally:
                self._loading = False
            self._update_plain_count()
            self._dirty = pending
            if message:
                self._set_status(message)
            return
        if key in self._pending:
            caption, message, pending = self._pending[key], None, True
        else:
            caption, message = self.store.load_caption(path)
            pending = False
        self.current_caption = caption
        self._assign_color_ids()
        self.populate_caption_fields()
        self.populate_elements_list()
        self.rebuild_boxes()
        if self.selected_element_index is not None:
            self._select_box_for_element(self.selected_element_index)
        # buffered images carry unsaved edits; freshly-loaded ones are clean
        self._dirty = pending
        # On-disk loads: re-validate the file so an external edit/corruption updates the
        # review marker (buffered edits are validated when generated/saved instead).
        if not pending:
            self.project.set_flags(path.name, self.caption_issues_for(path))
        self._refresh_thumb_marker(path)
        self._refresh_json_view()
        if message:
            self._set_status(message)

    def populate_caption_fields(self) -> None:
        self._loading = True
        try:
            cap = self.current_caption
            self.cap_high_level.setPlainText(cap.get("high_level_description", ""))
            style = cap.get("style_description", {}) or {}
            self.cap_aesthetics.setText(style.get("aesthetics", ""))
            self.cap_lighting.setText(style.get("lighting", ""))
            mode = "art_style" if ("art_style" in style and "photo" not in style) else "photo"
            self.style_mode.setCurrentText(mode)
            self._apply_style_mode_label(mode)
            default_medium = "illustration" if mode == "art_style" else "photograph"
            self.cap_medium.setText(style.get("medium", default_medium) or default_medium)
            self.cap_style_detail.setText(style.get(mode, ""))
            comp = cap.get("compositional_deconstruction", {}) or {}
            self.cap_background.setPlainText(comp.get("background", ""))
        finally:
            self._loading = False
            self._dirty = False

    def commit_caption_fields(self) -> None:
        cap = self.current_caption
        cap["high_level_description"] = self.cap_high_level.toPlainText().strip()
        style = cap.setdefault("style_description", {})
        style["aesthetics"] = self.cap_aesthetics.text().strip()
        style["lighting"] = self.cap_lighting.text().strip()
        style["medium"] = self.cap_medium.text().strip()
        mode = self.style_mode.currentText()
        detail = self.cap_style_detail.text().strip()
        if mode == "art_style":
            style["art_style"] = detail
            style.pop("photo", None)
        else:
            style["photo"] = detail
            style.pop("art_style", None)
        comp = cap.setdefault("compositional_deconstruction", {})
        comp["background"] = self.cap_background.toPlainText().strip()
        self._refresh_json_view()

    def save_current(self, *args, silent: bool = False) -> None:
        if self.store is None or self.current is None:
            return
        self._commit_active_caption()
        try:
            if self.preset.is_plain:
                path = self.store.save_plain_caption(self.current, self.current_text)
            else:
                path = self.store.save_caption(self.current, self.current_caption)
        except Exception as exc:  # Tier 2: readable save failure.
            QMessageBox.critical(self, "Could not save caption", str(exc))
            return
        self._dirty = False
        self._pending.pop(str(self.current), None)
        # A manual save means the human reviewed it — clear any corrupt-output flag.
        if self.project.is_flagged(self.current.name):
            self.project.clear_flag(self.current.name)
            try:
                self.store.save_project(self.project)
            except OSError:
                pass
        self._refresh_thumb_marker(self.current)
        self.persist_guidance_if_dirty()
        if not silent:
            self._set_status(f"Saved {path.name}")

    def save_all(self, *args) -> int:
        if self.store is None:
            return 0
        # fold the current image's live edits into the buffer first
        if self.current is not None:
            self._commit_active_caption()
            if self._dirty:
                self._pending[str(self.current)] = (
                    self.current_text if self.preset.is_plain else self.current_caption)
        saved = 0
        failed = []
        for key, caption in list(self._pending.items()):
            try:
                if self.preset.is_plain:
                    self.store.save_plain_caption(Path(key), caption)
                else:
                    self.store.save_caption(Path(key), caption)
                saved += 1
            except Exception as exc:
                failed.append(f"{Path(key).name}: {exc}")
        self._pending.clear()
        self._dirty = False
        for path in self.images:
            self._refresh_thumb_marker(path)
        self.persist_guidance_if_dirty()
        if failed:
            QMessageBox.critical(self, "Some captions could not be saved", "\n".join(failed))
        self._set_status(f"Saved {saved} edited caption(s).")
        return saved

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self._reposition_toolstrip()
        self._reposition_json_overlay()
        # Restore splitter sizes on first show — doing it during __init__ is too
        # early (the splitter has no real width yet, so proportions get lost).
        if not getattr(self, "_splitter_restored", False):
            self._splitter_restored = True
            state = self.qsettings.value("splitter_state_v2")
            if state is not None and getattr(self, "splitter", None) is not None:
                try:
                    self.splitter.restoreState(state)
                except Exception:
                    pass

    def _shutdown_server(self) -> None:
        """Stop the llama-server this app launched (local mode), if any."""
        proc = getattr(self, "_server_proc", None)
        if proc is not None:
            stop_server_process(proc)
            self._server_proc = None

    def closeEvent(self, event) -> None:
        if not self.confirm_discard_video_edits("close the app"):
            event.ignore()
            return
        self.qsettings.setValue("window_geometry", self.saveGeometry())
        if getattr(self, "splitter", None) is not None:
            self.qsettings.setValue("splitter_state_v2", self.splitter.saveState())
        if self.current is not None:
            self.commit_caption_fields()
            self.commit_element_fields()
            if self._autosave and self._dirty:
                self.save_current(silent=True)
            elif self._dirty:
                self._pending[str(self.current)] = self.current_caption
        if self._autosave or not self._pending:
            self._shutdown_server()
            event.accept()
            return
        box = QMessageBox(self)
        box.setWindowTitle("Unsaved edits")
        box.setText(f"You have unsaved edits to {len(self._pending)} image(s).")
        box.setInformativeText("Save them to disk before closing?")
        box.setStandardButtons(QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        box.setDefaultButton(QMessageBox.Save)
        choice = box.exec()
        if choice == QMessageBox.Save:
            self.save_all()
            self._shutdown_server()
            event.accept()
        elif choice == QMessageBox.Discard:
            self._shutdown_server()
            event.accept()
        else:
            event.ignore()

    # ---- elements read/write --------------------------------------------
    def _elements(self) -> list:
        comp = self.current_caption.setdefault("compositional_deconstruction", {})
        return comp.setdefault("elements", [])

    def _assign_color_ids(self) -> None:
        # Runtime-only stable color identity per element. Stripped on save by
        # the schema's normalize step, so it never reaches the caption JSON.
        els = self._elements()
        for i, el in enumerate(els):
            el["_color_id"] = i
        self._next_color_id = len(els)

    def _element_label(self, el: dict) -> str:
        etype = el.get("type", "obj")
        if etype == "text":
            name = (el.get("text", "") or el.get("desc", "")).strip() or "(text)"
        else:
            name = el.get("desc", "").strip() or "(obj)"
        if len(name) > 28:
            name = name[:27] + "…"
        label = f"{etype} · {name}"
        if not el.get("bbox"):
            label += "  · no box"
        return label

    def _element_name(self, el: dict) -> str:
        etype = el.get("type", "obj")
        if etype == "text":
            return (el.get("text", "") or el.get("desc", "")).strip() or "(text)"
        return el.get("desc", "").strip() or "(obj)"

    def _elide(self, text: str, px: int) -> str:
        return QFontMetrics(self.font()).elidedText(text, Qt.ElideRight, px)

    def _make_element_row(self, index: int, el: dict) -> ElementRow:
        row = ElementRow(index)
        h = QHBoxLayout(row)
        h.setContentsMargins(6, 3, 6, 3)
        h.setSpacing(5)

        up = QToolButton()
        up.setIcon(lucide_icon("chevron-up", self.theme.text_secondary, 14))
        up.setFixedSize(16, 16)
        up.setToolTip("Move up")
        up.clicked.connect(lambda _c, i=index: self._move_element(i, -1))
        down = QToolButton()
        down.setIcon(lucide_icon("chevron-down", self.theme.text_secondary, 14))
        down.setFixedSize(16, 16)
        down.setToolTip("Move down")
        down.clicked.connect(lambda _c, i=index: self._move_element(i, +1))

        dot = QLabel()
        dot.setFixedSize(12, 12)
        pill = QLabel()
        pill.setObjectName("TypePill")
        pill.setAlignment(Qt.AlignCenter)
        pill.setFixedWidth(34)
        lbl = QLabel()

        delete = QToolButton()
        delete.setIcon(lucide_icon("x", self.theme.text_muted, 14))
        delete.setFixedSize(16, 16)
        delete.setToolTip("Delete element")
        delete.clicked.connect(lambda _c, i=index: self._remove_element_at(i))

        h.addWidget(up)
        h.addWidget(down)
        h.addWidget(dot)
        h.addWidget(pill)
        h.addWidget(lbl, 1)
        h.addWidget(delete)

        row.dot, row.pill, row.lbl = dot, pill, lbl
        self._update_row_visuals(row, el, index)
        row.clicked.connect(self._select_element_row)
        return row

    def _update_row_visuals(self, row: ElementRow, el: dict, index: int) -> None:
        etype = el.get("type", "obj")
        row.pill.setText("TXT" if etype == "text" else "OBJ")
        if el.get("bbox"):
            color = box_color_for(el.get("_color_id", index))
            row.dot.setStyleSheet(f"background:{color}; border-radius:6px;")
            row.dot.setToolTip("")
        else:
            row.dot.setStyleSheet("background:transparent; border:1px solid #555; border-radius:6px;")
            row.dot.setToolTip("no box")
        name = self._element_name(el)
        row.lbl.setText(self._elide(name, 130))
        row.lbl.setToolTip(name)

    def _update_row_active_styles(self) -> None:
        t = self.theme
        for i, row in enumerate(getattr(self, "_element_rows", [])):
            if i == self.selected_element_index:
                row.setStyleSheet(
                    f"#ElementRow {{ background: {t.accent_subtle}; "
                    f"border:1px solid {t.accent_subtle_border}; border-radius:6px; }}"
                )
            else:
                row.setStyleSheet(
                    f"#ElementRow {{ background: {t.surface_2}; "
                    f"border:1px solid {t.border}; border-radius:6px; }}"
                )

    def _select_element_row(self, index: int) -> None:
        self.elements_list.setCurrentRow(index)

    def _move_element(self, index: int, delta: int) -> None:
        els = self._elements()
        target = index + delta
        if target < 0 or target >= len(els):
            return
        self.commit_element_fields()
        els[index], els[target] = els[target], els[index]
        self._touch_dirty()
        self.populate_elements_list()
        self.rebuild_boxes()
        self.elements_list.setCurrentRow(target)

    def _duplicate_element(self) -> None:
        idx = self.selected_element_index
        els = self._elements()
        if idx is None or idx < 0 or idx >= len(els):
            return
        self.commit_element_fields()
        new = copy.deepcopy(els[idx])
        if isinstance(new.get("bbox"), list) and len(new["bbox"]) == 4:
            new["bbox"] = [min(v + 30, 1000) for v in new["bbox"]]
        new["_color_id"] = self._next_color_id
        self._next_color_id += 1
        els.insert(idx + 1, new)
        self._touch_dirty()
        self.populate_elements_list()
        self.rebuild_boxes()
        self.elements_list.setCurrentRow(idx + 1)

    def _remove_element_at(self, index: int) -> None:
        els = self._elements()
        if index < 0 or index >= len(els):
            return
        name = self._element_name(els[index])
        confirm = QMessageBox.question(
            self,
            "Delete element",
            f"Delete the “{name}” element?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        els.pop(index)
        self._touch_dirty()
        self.selected_element_index = None
        self.populate_elements_list()
        self.rebuild_boxes()

    def populate_elements_list(self) -> None:
        self._loading = True
        try:
            self.elements_list.clear()
            self._element_rows = []
            for i, el in enumerate(self._elements()):
                row = self._make_element_row(i, el)
                item = QListWidgetItem()
                item.setSizeHint(row.sizeHint())
                self.elements_list.addItem(item)
                self.elements_list.setItemWidget(item, row)
                self._element_rows.append(row)
        finally:
            self._loading = False
        self.selected_element_index = None
        self._set_element_editor_enabled(False)
        if self._elements():
            self.elements_list.setCurrentRow(0)

    def _on_element_row_changed(self, row: int) -> None:
        if self._loading:
            return
        self.commit_element_fields()
        if row is None or row < 0 or row >= len(self._elements()):
            self.selected_element_index = None
            self._set_element_editor_enabled(False)
            self._update_row_active_styles()
            return
        self.selected_element_index = row
        self.populate_element_editor()
        self._select_box_for_element(row)
        self._update_row_active_styles()

    def populate_element_editor(self) -> None:
        idx = self.selected_element_index
        els = self._elements()
        if idx is None or idx < 0 or idx >= len(els):
            self._set_element_editor_enabled(False)
            return
        self._set_element_editor_enabled(True)
        el = els[idx]
        self._loading = True
        try:
            etype = el.get("type", "obj")
            if etype not in ("obj", "text"):
                etype = "obj"
            self.el_type.setCurrentText(etype)
            self._apply_el_type_visibility(etype)
            self.el_desc.setPlainText(el.get("desc", ""))
            self.el_text.setText(el.get("text", ""))
            bbox = el.get("bbox")
            has = isinstance(bbox, (list, tuple)) and len(bbox) == 4
            self.el_has_box.setChecked(bool(has))
            y1, x1, y2, x2 = (bbox if has else (0, 0, 0, 0))
            self.el_y1.setValue(int(y1))
            self.el_x1.setValue(int(x1))
            self.el_y2.setValue(int(y2))
            self.el_x2.setValue(int(x2))
            self._set_coords_enabled(bool(has))
        finally:
            self._loading = False
        self._select_box_for_element(idx)

    def commit_element_fields(self) -> None:
        idx = self.selected_element_index
        els = self._elements()
        if idx is None or idx < 0 or idx >= len(els):
            return
        el = els[idx]
        etype = self.el_type.currentText()
        el["type"] = etype
        el["desc"] = self.el_desc.toPlainText().strip()
        if etype == "text":
            el["text"] = self.el_text.text().strip()
        else:
            el.pop("text", None)
        if self.el_has_box.isChecked():
            el["bbox"] = [self.el_y1.value(), self.el_x1.value(), self.el_y2.value(), self.el_x2.value()]
        else:
            el.pop("bbox", None)
        self._refresh_json_view()

    def add_bbox_element(self) -> None:
        if self.store is None or self.current is None:
            self._set_status("Open a folder and select a file first.")
            return
        self.commit_element_fields()
        els = self._elements()
        # centered box in 0–1000 space: [y1, x1, y2, x2]
        new = {"type": "obj", "desc": "", "bbox": [250, 250, 750, 750], "_color_id": self._next_color_id}
        self._next_color_id += 1
        els.append(new)
        self._touch_dirty()
        self.right_tabs.setCurrentIndex(1)  # Elements tab
        self.populate_elements_list()
        self.elements_list.setCurrentRow(len(els) - 1)
        self.rebuild_boxes()
        # drop into select mode so the new box can be moved/resized right away
        self._activate_tool("select")
        self._set_status("Added a centered bounding box.")

    def _add_element(self, etype: str) -> None:
        if self.store is None or self.current is None:
            return
        self.commit_element_fields()
        els = self._elements()
        new = {"type": etype, "desc": "", "_color_id": self._next_color_id}
        self._next_color_id += 1
        if etype == "text":
            new["text"] = ""
        els.append(new)
        self._touch_dirty()
        self._loading = True
        try:
            self.elements_list.addItem(self._element_label(new))
        finally:
            self._loading = False
        self.elements_list.setCurrentRow(len(els) - 1)
        self.rebuild_boxes()

    def _remove_element(self) -> None:
        if self.selected_element_index is not None:
            self._remove_element_at(self.selected_element_index)

    def _on_el_type_changed(self, etype: str) -> None:
        self._apply_el_type_visibility(etype)
        self._refresh_current_element_label()
        self._mark_dirty()

    def _on_el_desc_changed(self) -> None:
        self._refresh_current_element_label()
        self._mark_dirty()

    def _on_has_box_changed(self, checked: bool) -> None:
        self._set_coords_enabled(checked)
        if self._loading:
            return
        if checked and all(s.value() == 0 for s in (self.el_y1, self.el_x1, self.el_y2, self.el_x2)):
            self.el_y1.setValue(250)
            self.el_x1.setValue(250)
            self.el_y2.setValue(750)
            self.el_x2.setValue(750)
        self._refresh_current_element_label()
        self._mark_dirty()
        self.rebuild_boxes()
        if self.selected_element_index is not None:
            self._select_box_for_element(self.selected_element_index)

    def _apply_el_type_visibility(self, etype: str) -> None:
        is_text = etype == "text"
        self.el_text_label.setVisible(is_text)
        self.el_text_container.setVisible(is_text)

    def _set_coords_enabled(self, enabled: bool) -> None:
        for spin in (self.el_y1, self.el_x1, self.el_y2, self.el_x2):
            spin.setEnabled(enabled)

    def _set_element_editor_enabled(self, enabled: bool) -> None:
        self.el_editor.setEnabled(enabled)
        self.el_remove_btn.setEnabled(enabled)

    def _refresh_current_element_label(self) -> None:
        if self._loading:
            return
        idx = self.selected_element_index
        els = self._elements()
        if idx is None or idx < 0 or idx >= len(els):
            return
        self.commit_element_fields()
        rows = getattr(self, "_element_rows", [])
        if 0 <= idx < len(rows):
            self._update_row_visuals(rows[idx], els[idx], idx)

    # ---- canvas <-> model sync ------------------------------------------
    def _norm_to_scene(self, bbox) -> QRectF | None:
        sr = self.scene.sceneRect()
        W, H = sr.width(), sr.height()
        if W <= 0 or H <= 0:
            return None
        y1, x1, y2, x2 = bbox
        left = x1 / 1000.0 * W
        top = y1 / 1000.0 * H
        right = x2 / 1000.0 * W
        bottom = y2 / 1000.0 * H
        return QRectF(left, top, max(MIN_BOX_PX, right - left), max(MIN_BOX_PX, bottom - top))

    def _scene_to_norm(self, rect: QRectF):
        sr = self.scene.sceneRect()
        W, H = sr.width(), sr.height()
        if W <= 0 or H <= 0:
            return None
        y1 = int(round(rect.top() / H * 1000))
        x1 = int(round(rect.left() / W * 1000))
        y2 = int(round(rect.bottom() / H * 1000))
        x2 = int(round(rect.right() / W * 1000))
        top, bottom = sorted((_clamp(y1, 0, 1000), _clamp(y2, 0, 1000)))
        left, right = sorted((_clamp(x1, 0, 1000), _clamp(x2, 0, 1000)))
        return [top, left, bottom, right]

    def rebuild_boxes(self) -> None:
        for it in self.box_items:
            if it.scene() is not None:
                self.scene.removeItem(it)
        self.box_items = []
        if self.pixmap_item is None:
            return
        for i, el in enumerate(self._elements()):
            bbox = el.get("bbox")
            if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
                continue
            rect = self._norm_to_scene(bbox)
            if rect is None:
                continue
            item = BBoxItem(rect, i, self, color=box_color_for(el.get("_color_id", i)))
            item.set_label(self._element_label(el))
            self.scene.addItem(item)
            self.box_items.append(item)
        if self.selected_element_index is not None:
            self._select_box_for_element(self.selected_element_index)
        self._set_canvas_locked(getattr(self, "_read_only", False))

    def _select_box_for_element(self, idx: int | None) -> None:
        for it in self.box_items:
            it.setSelected(it.element_index == idx)

    def on_box_pressed(self, item: "BBoxItem") -> None:
        idx = item.element_index
        if idx is None:
            return
        self.right_tabs.setCurrentIndex(1)
        if self.selected_element_index != idx:
            self.elements_list.setCurrentRow(idx)

    def on_box_geometry_live(self, item: "BBoxItem") -> None:
        idx = item.element_index
        els = self._elements()
        if idx is None or idx < 0 or idx >= len(els):
            return
        bbox = self._scene_to_norm(item.mapRectToScene(item.rect()))
        if bbox is None:
            return
        els[idx]["bbox"] = bbox
        self._touch_dirty()
        if idx == self.selected_element_index:
            self._syncing = True
            try:
                self.el_has_box.setChecked(True)
                self._set_coords_enabled(True)
                self.el_y1.setValue(bbox[0])
                self.el_x1.setValue(bbox[1])
                self.el_y2.setValue(bbox[2])
                self.el_x2.setValue(bbox[3])
            finally:
                self._syncing = False
        list_item = self.elements_list.item(idx)
        if list_item is not None:
            list_item.setText(self._element_label(els[idx]))
            item.set_label(self._element_label(els[idx]))

    def nudge_selected_box(self, dx: int, dy: int) -> bool:
        """Move the selected element's box by (dx, dy) in 0–1000 units, keeping its
        size fixed and clamped in-bounds. Returns True if there was a box to nudge."""
        if getattr(self, "_read_only", False):
            return False
        idx = self.selected_element_index
        els = self._elements()
        if idx is None or idx < 0 or idx >= len(els):
            return False
        bbox = els[idx].get("bbox")
        if not (isinstance(bbox, (list, tuple)) and len(bbox) == 4):
            return False
        top, left, bottom, right = (int(v) for v in bbox)
        # clamp the shift so the box keeps its size and stays within 0–1000
        dx = _clamp(dx, -left, 1000 - right)
        dy = _clamp(dy, -top, 1000 - bottom)
        if dx == 0 and dy == 0:
            return True  # selected but pinned against the edge — still consume the key
        new_bbox = [top + dy, left + dx, bottom + dy, right + dx]
        els[idx]["bbox"] = new_bbox
        self._touch_dirty()
        item = next((it for it in self.box_items if it.element_index == idx), None)
        if item is not None:
            rect = self._norm_to_scene(new_bbox)
            if rect is not None:
                # setPos would fire itemChange -> on_box_geometry_live mid-update and
                # clobber the box we just computed; move silently, then restore.
                item.setFlag(QGraphicsItem.ItemSendsGeometryChanges, False)
                item.set_scene_rect(rect)
                item.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
                item.setSelected(True)
        # keep the coordinate fields in step (same path as a mouse drag)
        self._syncing = True
        try:
            # block signals: the element already has a box, and firing
            # _on_has_box_changed here would reset it to a default rectangle.
            self.el_has_box.blockSignals(True)
            self.el_has_box.setChecked(True)
            self.el_has_box.blockSignals(False)
            self._set_coords_enabled(True)
            self.el_y1.setValue(new_bbox[0])
            self.el_x1.setValue(new_bbox[1])
            self.el_y2.setValue(new_bbox[2])
            self.el_x2.setValue(new_bbox[3])
        finally:
            self._syncing = False
        return True

    def apply_drawn_box(self, scene_rect: QRectF) -> None:
        if scene_rect.width() < MIN_BOX_PX or scene_rect.height() < MIN_BOX_PX:
            return
        idx = self.selected_element_index
        els = self._elements()
        if idx is None or idx < 0 or idx >= len(els):
            self._set_status("Select an element first, then draw its box.")
            return
        bbox = self._scene_to_norm(scene_rect)
        if bbox is None:
            return
        els[idx]["bbox"] = bbox
        self._touch_dirty()
        self.populate_element_editor()
        self.rebuild_boxes()
        self._select_box_for_element(idx)
        list_item = self.elements_list.item(idx)
        if list_item is not None:
            list_item.setText(self._element_label(els[idx]))

    def _remove_box_for_element(self, idx: int, *, confirm: bool = True) -> bool:
        """Remove the bbox from element idx, with the usual confirm + editor/list/
        canvas refresh. Returns True if a box was actually removed."""
        els = self._elements()
        if not (0 <= idx < len(els)):
            return False
        if not isinstance(els[idx].get("bbox"), (list, tuple)):
            return False
        if confirm:
            name = self._element_name(els[idx])
            if QMessageBox.question(
                self,
                "Delete box",
                f"Remove the bounding box from “{name}”?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            ) != QMessageBox.Yes:
                return False
        els[idx].pop("bbox", None)
        self._touch_dirty()
        if idx == self.selected_element_index:
            self.populate_element_editor()
        rows = getattr(self, "_element_rows", [])
        if 0 <= idx < len(rows):
            self._update_row_visuals(rows[idx], els[idx], idx)
        self.rebuild_boxes()
        return True

    def delete_selected_box(self) -> bool:
        """Delete the box of the currently selected element. Both the Delete key and
        the delete tool route here, so deletion always targets the box the user
        selected — never a larger box that merely overlaps the click point."""
        if getattr(self, "_read_only", False):
            return False
        idx = self.selected_element_index
        els = self._elements()
        if idx is None or not (0 <= idx < len(els)):
            self._set_status("Select a box first, then delete it.")
            return False
        if not isinstance(els[idx].get("bbox"), (list, tuple)):
            self._set_status("The selected element has no box to delete.")
            return False
        return self._remove_box_for_element(idx, confirm=True)

    def _on_coord_changed(self, *args) -> None:
        if self._loading:
            return
        self._touch_dirty()
        if self._syncing:
            return
        self._update_selected_box_from_spinboxes()

    def _update_selected_box_from_spinboxes(self) -> None:
        idx = self.selected_element_index
        els = self._elements()
        if idx is None or idx < 0 or idx >= len(els):
            return
        if not self.el_has_box.isChecked():
            return
        bbox = [self.el_y1.value(), self.el_x1.value(), self.el_y2.value(), self.el_x2.value()]
        els[idx]["bbox"] = bbox
        rect = self._norm_to_scene(bbox)
        existing = next((it for it in self.box_items if it.element_index == idx), None)
        if existing is not None and rect is not None:
            existing.set_scene_rect(rect)
        else:
            self.rebuild_boxes()
            self._select_box_for_element(idx)
        list_item = self.elements_list.item(idx)
        if list_item is not None:
            list_item.setText(self._element_label(els[idx]))

    def _set_status(self, text: str) -> None:
        self.statusBar().showMessage(text)

    def _update_count_label(self) -> None:
        total = len(self.images) if getattr(self, "images", None) else 0
        nav = getattr(self, "_nav_count", None)
        if total == 0:
            self._count_label.setText("")
            if nav is not None:
                nav.setText("0 / 0")
        elif self.current is not None and self.current in self.images:
            idx = self.images.index(self.current) + 1
            # "File", not "Image": a folder can hold clips as well as stills.
            self._count_label.setText(f"File {idx} / {total}")
            if nav is not None:
                nav.setText(f"{idx} / {total}")
        else:
            self._count_label.setText(f"{total} files")
            if nav is not None:
                nav.setText(f"— / {total}")

    def _build_server_status(self) -> None:
        self._count_label = QLabel()
        self._count_label.setObjectName("CountStatus")
        self.statusBar().addPermanentWidget(self._count_label)

        # Dedicated job-progress indicator, kept as permanent status-bar widgets
        # so transient messages (e.g. selecting another image) never clobber it.
        self._job_progress_label = QLabel()
        self._job_progress_label.setObjectName("JobProgress")
        self._job_progress_label.setVisible(False)
        self._job_progress_bar = QProgressBar()
        self._job_progress_bar.setObjectName("JobProgressBar")
        self._job_progress_bar.setTextVisible(False)
        self._job_progress_bar.setFixedSize(120, 12)
        self._job_progress_bar.setVisible(False)
        self.statusBar().addPermanentWidget(self._job_progress_label)
        self.statusBar().addPermanentWidget(self._job_progress_bar)

        self._resource_label = QLabel()
        self._resource_label.setObjectName("ResourceMonitor")
        self._resource_label.setToolTip("System RAM" + (" · VRAM · GPU usage"))
        self.statusBar().addPermanentWidget(self._resource_label)
        self._server_status_label = ClickableLabel()
        self._server_status_label.setObjectName("ServerStatus")
        self._server_status_label.setCursor(Qt.PointingHandCursor)
        self._server_status_label.setToolTip("Server status & controls")
        self._server_status_label.clicked.connect(self._show_server_popover)
        self.statusBar().addPermanentWidget(self._server_status_label)
        self._set_server_status(None)  # "checking" until the first ping returns

    def _set_job_progress(self, text: str = "", *, value=None, total=None, busy: bool = False) -> None:
        """Show job/batch progress in its own permanent widget (not the transient
        status message). Empty text clears it."""
        if not text:
            self._job_progress_label.clear()
            self._job_progress_label.setVisible(False)
            self._job_progress_bar.setVisible(False)
            return
        fm = self._job_progress_label.fontMetrics()
        self._job_progress_label.setText(fm.elidedText(text, Qt.ElideRight, 340))
        self._job_progress_label.setToolTip(text)
        self._job_progress_label.setVisible(True)
        if total:
            self._job_progress_bar.setRange(0, total)
            self._job_progress_bar.setValue(value or 0)
            self._job_progress_bar.setVisible(True)
        elif busy:
            self._job_progress_bar.setRange(0, 0)  # indeterminate marquee
            self._job_progress_bar.setVisible(True)
        else:
            self._job_progress_bar.setVisible(False)

    def _set_server_status(self, ok) -> None:
        self._server_reachable = ok
        if ok is None:
            dot, text = "#9aa4b6", "Checking server…"
        elif ok:
            dot, text = "#3ddc84", "Server connected"
        else:
            dot, text = "#ff5a52", "Server offline"
        self._server_status_label.setText(
            f'<span style="color:{dot}">●</span> '
            f'<span style="color:#9aa4b6">{text}</span>'
        )

    def _start_server_monitor(self) -> None:
        self._server_monitor = ServerStatusMonitor(self.settings.base_url, self.settings.api_key)
        self._server_monitor.status.connect(self._set_server_status)
        self._resource_monitor = ResourceMonitor(self)
        self._resource_monitor.sampled.connect(self._resource_label.setText)
        app = QApplication.instance()
        if app is not None:
            app.aboutToQuit.connect(self._stop_server_monitor)
        self._server_monitor.start()
        self._resource_monitor.start()

    def _stop_server_monitor(self) -> None:
        mon = getattr(self, "_server_monitor", None)
        if mon is not None and mon.isRunning():
            mon.requestInterruption()
            mon.wait(2000)
        res = getattr(self, "_resource_monitor", None)
        if res is not None and res.isRunning():
            res.requestInterruption()
            res.wait(2000)


def main() -> None:
    app = QApplication(sys.argv)
    # The window manager appends this to every window and dialog title, which is
    # why they all still read "Ideogram4 …" long after the tool stopped being
    # Ideogram-only. APP_TITLE is the single source of truth.
    app.setApplicationName(APP_TITLE)
    app.setApplicationDisplayName(APP_TITLE)
    icon = app_icon()
    app.setWindowIcon(icon)
    window = MainWindow()
    window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
