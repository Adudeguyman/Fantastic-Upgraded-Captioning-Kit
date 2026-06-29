"""Stage 0/1 of the PySide6 port: a launchable window with the locked layout
bones (toolbar, collapsible guidance left, image view center, tabbed
Caption/Elements panel right, thumbnail filmstrip bottom), wired to real
folder-open and image display. Editing/AI behavior arrives in later stages.
"""

from __future__ import annotations

import copy
import difflib
import html
import json
import os
import re
import subprocess
import sys
import time
import webbrowser
from dataclasses import replace
from pathlib import Path

try:
    from PySide6.QtCore import Qt, QSize, QSettings, QRect, QRectF, QPoint, QPointF, QMimeData, QEvent, QThread, Signal, QByteArray, QTimer, QPropertyAnimation, QEasingCurve, Property, QParallelAnimationGroup, QAbstractAnimation, QVariantAnimation
    from PySide6.QtGui import QAction, QBrush, QColor, QFont, QFontDatabase, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap, QIcon, QTextCharFormat, QTextCursor, QTextFormat, QDrag, QPolygonF
    from PySide6.QtSvg import QSvgRenderer
    from PySide6.QtWidgets import (
        QApplication,
        QFileDialog,
        QFrame,
        QLayout,
        QGraphicsPixmapItem,
        QGraphicsOpacityEffect,
        QGraphicsDropShadowEffect,
        QGraphicsScene,
        QGraphicsView,
        QGraphicsItem,
        QGraphicsRectItem,
        QHBoxLayout,
        QButtonGroup,
        QAbstractButton,
        QCheckBox,
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
        "Activate the captioner environment and install requirements:\n\n"
        "    conda activate id4caption\n"
        "    pip install -r requirements-qt.txt\n\n"
        f"(original import error: {exc})\n"
    )
    raise SystemExit(1)

# Shared backend — imported unchanged from the existing Tkinter app.
from .store import CaptionStore, ProjectConfig
from .llm_captioning import (
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
    discover_local_gguf_models,
    estimate_gguf_vram_gb,
    guess_mmproj_for,
    CUSTOM_LOCAL_PROFILE,
    profile_label_from_id,
    is_server_ready,
    server_model_ids,
    ensure_server_running,
    stop_server_process,
    find_llama_server,
    detect_gpu,
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
from .schema import caption_health, default_caption, serialize_caption


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
    "ellipsis": "<circle cx='12' cy='12' r='1' /> <circle cx='19' cy='12' r='1' /> <circle cx='5' cy='12' r='1' />",
    "flag": "<path d='M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z' /> <path d='M4 22v-7' />",
    "folder-open": "<path d='m6 14 1.5-2.9A2 2 0 0 1 9.24 10H20a2 2 0 0 1 1.94 2.5l-1.54 6a2 2 0 0 1-1.95 1.5H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h3.9a2 2 0 0 1 1.69.9l.81 1.2a2 2 0 0 0 1.67.9H18a2 2 0 0 1 2 2v2' />",
    "info": "<circle cx='12' cy='12' r='10' /> <path d='M12 16v-4' /> <path d='M12 8h.01' />",
    "lock": "<rect width='18' height='11' x='3' y='11' rx='2' ry='2' /> <path d='M7 11V7a5 5 0 0 1 10 0v4' />",
    "lock-open": "<rect width='18' height='11' x='3' y='11' rx='2' ry='2' /> <path d='M7 11V7a5 5 0 0 1 9.9-1' />",
    "maximize": "<path d='M8 3H5a2 2 0 0 0-2 2v3' /> <path d='M21 8V5a2 2 0 0 0-2-2h-3' /> <path d='M3 16v3a2 2 0 0 0 2 2h3' /> <path d='M16 21h3a2 2 0 0 0 2-2v-3' />",
    "maximize-2": "<path d='M15 3h6v6' /> <path d='m21 3-7 7' /> <path d='m3 21 7-7' /> <path d='M9 21H3v-6' />",
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

    QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
        background: {t.surface_2}; color: {t.text_primary};
        border: 1px solid {t.border_strong}; border-radius: 6px; padding: 5px 10px;
        selection-background-color: {t.accent_subtle}; selection-color: {t.text_primary};
    }}
    QLineEdit:focus, QPlainTextEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {t.accent}; }}
    QLineEdit:disabled, QPlainTextEdit:disabled, QComboBox:disabled {{ background: {t.surface_1}; border-color: {t.border}; color: {t.text_disabled}; }}
    QComboBox QAbstractItemView {{ background: {t.surface_2}; color: {t.text_primary}; border: 1px solid {t.border_strong}; selection-background-color: {t.accent_subtle}; selection-color: {t.text_primary}; }}
    QComboBox::drop-down {{ border: none; width: 20px; }}

    QCheckBox {{ background: transparent; color: {t.text_primary}; spacing: 8px; }}
    QCheckBox::indicator {{ width: 16px; height: 16px; border: 1px solid {t.border_strong}; border-radius: 4px; background: {t.surface_2}; }}
    QCheckBox::indicator:hover {{ border-color: {t.accent}; }}
    QCheckBox::indicator:checked {{ background: {t.accent}; border-color: {t.accent}; }}

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
_PRESET_ART_STYLE = (
    "For the high_level_description section append a suffix of  in the style of "
    "my_art_style.\n"
    "For the art_style prepend my_art_style,  in front of the regular art style "
    "description."
)
_PRESET_SINGLE_CHARACTER = (
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
    "color) but add a bounding box for them with a short description of their pose."
)

FOLDER_GUIDANCE_PRESETS: list[tuple[str, str]] = [
    ("Art Style", _PRESET_ART_STYLE),
    ("Single Character", _PRESET_SINGLE_CHARACTER),
]
IMAGE_GUIDANCE_PRESETS: list[tuple[str, str]] = [
    ("Multi-Character", _PRESET_MULTI_CHARACTER),
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
REVIEW_COLOR = "#E24B4A"             # red — "needs review: caption may be corrupt"
FLAG_ROLE = int(Qt.UserRole) + 4     # per-item flag: user manually flagged for review
FLAG_COLOR = "#E5484D"               # red flag — "you flagged this for manual review"

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
        # Arrow keys nudge the selected box (Shift = ×10). Falls through to the
        # default view behaviour (scroll) when no box is selected.
        arrows = {Qt.Key_Left: (-1, 0), Qt.Key_Right: (1, 0),
                  Qt.Key_Up: (0, -1), Qt.Key_Down: (0, 1)}
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
        if self.mode == "draw" and event.button() == Qt.LeftButton:
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
    'bboxes'. Progress strings and the result/error come back as signals so the
    main thread can update widgets safely.
    """

    progress = Signal(str)
    done = Signal(object)
    error = Signal(str)
    server_started = Signal(object)

    def __init__(self, operation, settings, image_path, caption, guidance, source_caption, instructions):
        super().__init__()
        self.operation = operation
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
                    self.settings, self.image_path, progress=prog, guidance=self.guidance
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

    def __init__(self, settings, items, delay_ms: int = 0):
        super().__init__()
        self.settings = settings
        self.items = items  # list of (Path, guidance)
        self.delay_ms = max(0, int(delay_ms))

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
        for i, (image_path, guidance) in enumerate(self.items, start=1):
            if self.isInterruptionRequested():
                cancelled = True
                break
            self.item_progress.emit(i, total, f"Captioning {i}/{total}: {image_path.name}")

            def prog(msg: str, _i=i, _t=total) -> None:
                self.item_progress.emit(_i, _t, f"[{_i}/{_t}] {msg}")

            try:
                caption = generate_json_from_image(
                    self.settings, image_path, progress=prog, guidance=guidance
                )
                if self.settings.add_bboxes_after_json and not self.isInterruptionRequested():
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
        "Local (llama.cpp)": ("http://127.0.0.1:8000/v1", "llama-cpp", "local"),
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
            (None, "Detected GPU", "_gpustatus", None),
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
        ("Models", ()),
        ("Pipeline", (
            ("creative_json", "Creative JSON (off = faithful)", "bool", None),
            ("disable_thinking", "Disable thinking", "bool", None),
            ("add_bboxes_after_json", "Auto-locate boxes after JSON", "bool", None),
            ("overwrite_bboxes", "Overwrite existing boxes", "bool", None),
            ("filter_bbox_targets", "Filter bbox targets", "bool", None),
            ("vision_image_format", "Vision image format", "choice", ("auto", "jpeg", "png")),
            ("max_tokens_caption", "Max tokens — caption", "int", (1, 200000)),
            ("max_tokens_json", "Max tokens — JSON", "int", (1, 200000)),
            ("max_tokens_bboxes", "Max tokens — bboxes", "int", (1, 200000)),
            ("context_chars", "Context chars", "int", (0, 100000)),
            ("max_targets_per_call", "Max bbox targets / call (0 = all)", "int", (0, 10000)),
            ("json_refine_instructions", "Refine instructions", "multiline", None),
        )),
        ("Tags", ()),
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
        self._qsettings = QSettings("IdeogramCaptioner", "QtApp")
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
            if name == "Models":
                page = self._build_models_page()
            elif name == "Tags":
                page = self._build_tags_page()
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
                    if kind == "_gpustatus":
                        try:
                            summary = detect_gpu().summary
                        except Exception:
                            summary = "detection unavailable"
                        status = QLabel(summary)
                        status.setObjectName("GpuStatus")
                        status.setWordWrap(True)
                        status.setStyleSheet("color: #9aa3ad; font-style: italic;")
                        status.setToolTip(
                            "GPU detected on this machine. Determines which llama.cpp "
                            "build (CUDA / Vulkan / CPU) the app would download for local mode."
                        )
                        form.addRow(QLabel(label), status)
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

        for task, title in (("caption", "Caption / JSON model"), ("bbox", "BBox VLM")):
            head_row = QHBoxLayout()
            head_row.setContentsMargins(0, 0, 0, 0)
            section = QLabel(title)
            section.setObjectName("SectionLabel")
            head_row.addWidget(section)
            head_row.addStretch(1)
            if task == "caption":
                rec_btn = QPushButton("Browse models\u2026")
                rec_btn.setToolTip("Browse all models with VRAM fit estimates for your card.")
                rec_btn.clicked.connect(lambda: self._open_model_picker("caption"))
                head_row.addWidget(rec_btn, 0, Qt.AlignRight)
            lay.addLayout(head_row)

            if task == "bbox":
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
        if item is not None and item.text() == "Models" and self._profile_combos:
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
        lbl.setText(f"<b>{short}</b>{badge}")

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
        CONTENT_W = 660

        dlg = QDialog(self)
        dlg.setWindowTitle("Choose a model")
        dlg.resize(720, 560)
        dlg.setMinimumWidth(700)
        lay = QVBoxLayout(dlg)

        if vram:
            gpu_name = detect_gpu().name or "your GPU"
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

        def add_row(row):
            row.setFixedWidth(CONTENT_W)
            row.ensurePolished()
            h = row.sizeHint().height()
            lay_ = row.layout()
            if lay_ is not None and lay_.hasHeightForWidth():
                h = max(h, lay_.heightForWidth(CONTENT_W))
            h = max(h, 42)            # never shorter than a Use/HF button row
            item = QListWidgetItem(listing)
            item.setSizeHint(QSize(CONTENT_W, h))
            listing.addItem(item)
            listing.setItemWidget(item, row)

        def add_section(text):
            hdr = QLabel(text)
            hdr.setObjectName("SectionLabel")
            hdr.setContentsMargins(8, 10, 8, 2)
            item = QListWidgetItem(listing)
            item.setFlags(Qt.NoItemFlags)
            item.setSizeHint(QSize(CONTENT_W, hdr.sizeHint().height() + 8))
            listing.addItem(item)
            listing.setItemWidget(item, hdr)

        def add_hint(text):
            e = QLabel(text)
            e.setObjectName("Hint")
            e.setWordWrap(True)
            e.setContentsMargins(8, 2, 8, 6)
            e.setFixedWidth(CONTENT_W - 4)
            h = e.heightForWidth(CONTENT_W - 4)
            if h <= 0:
                h = e.sizeHint().height()
            item = QListWidgetItem(listing)
            item.setFlags(Qt.NoItemFlags)
            item.setSizeHint(QSize(CONTENT_W, h + 4))
            listing.addItem(item)
            listing.setItemWidget(item, e)

        def make_profile_row(p, *, allow_use, show_link=False):
            name = p.label.split(":", 1)[1].strip() if p.label.lower().startswith("download:") else p.label
            row = QWidget()
            outer = QVBoxLayout(row)
            outer.setContentsMargins(8, 6, 8, 6)
            outer.setSpacing(2)
            top = QHBoxLayout()
            top.setContentsMargins(0, 0, 0, 0)
            top.setSpacing(8)
            star = "\u2605 " if p.id == rec_id else ""
            title = QLabel(f"{star}{name}")
            title.setWordWrap(True)
            title.setMaximumWidth(300)
            top.addWidget(title, 1)
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
                use_btn.setCursor(Qt.PointingHandCursor)
                use_btn.clicked.connect(lambda _c, prof=p: self._pick_model(task, prof, dlg))
                top.addWidget(use_btn, 0)
            if p.kind == "hf" and p.hf_repo and not show_link:
                hf_btn = QPushButton("HF")
                hf_btn.setToolTip(f"https://huggingface.co/{p.hf_repo}")
                hf_btn.clicked.connect(lambda _c, repo=p.hf_repo: webbrowser.open(f"https://huggingface.co/{repo}"))
                top.addWidget(hf_btn, 0)
            outer.addLayout(top)
            if show_link and p.kind == "hf" and p.hf_repo:
                link = QLabel(f'<a href="https://huggingface.co/{p.hf_repo}" style="color:#6cb6ff;">huggingface.co/{p.hf_repo}</a>')
                link.setOpenExternalLinks(True)
                link.setObjectName("Hint")
                link.setMaximumWidth(CONTENT_W - 20)
                outer.addWidget(link)
            if p.note:
                note = QLabel(p.note)
                note.setObjectName("Hint")
                note.setWordWrap(True)
                note.setStyleSheet("color:#A78BFA;")
                note.setMaximumWidth(CONTENT_W - 20)
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
            fname = QLabel(path.name)
            fname.setWordWrap(True)
            fname.setMaximumWidth(CONTENT_W - 250)
            where = QLabel(self._short_dir(path.parent))
            where.setObjectName("Hint")
            where.setWordWrap(True)
            where.setMaximumWidth(CONTENT_W - 250)
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
                add_hint("No downloaded GGUF files found. Add folders on the Models page "
                         "(Browse\u2026 / Detect model folders), or download one below.")
            else:
                for path in found:
                    add_row(make_detected_row(path))
            # 2) Custom / existing-server options (secondary)
            others = [p for p in profiles_for_task(task)
                      if p.kind in ("server", "custom_hf", "custom_local")]
            if others:
                add_section("Custom & server options")
                for p in others:
                    add_row(make_profile_row(p, allow_use=True))
            # 3) Recommended models to download (heaviest action) at the bottom
            add_section("Recommended to download")
            for p in hf_profiles:
                add_row(make_profile_row(p, allow_use=True))
        else:
            add_section("Detected models")
            if not found:
                add_hint("No downloaded GGUF files found. Add your server's model folders on "
                         "the Models page (Browse\u2026 / Detect model folders).")
            else:
                for path in found:
                    add_row(make_detected_row(path))
            add_section("Recommended models")
            add_hint("Download and configure these in your external server, then select the "
                     "model there or enter its name. Links go to Hugging Face.")
            for p in hf_profiles:
                add_row(make_profile_row(p, allow_use=False, show_link=True))

        box = QDialogButtonBox(QDialogButtonBox.Close)
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

    def _detected_vram(self) -> float | None:
        """Total VRAM in GB, detected once and cached for the session."""
        if not hasattr(self, "_vram_gb_cache"):
            try:
                self._vram_gb_cache = detect_gpu().vram_total_gb
            except Exception:
                self._vram_gb_cache = None
        return self._vram_gb_cache

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
        target = QSettings("IdeogramCaptioner", "QtApp").value("last_server_preset")
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
        QSettings("IdeogramCaptioner", "QtApp").setValue("last_server_preset", name)
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
            val = QSettings("IdeogramCaptioner", "QtApp").value("llama_latest_build")
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
            button.setProperty("wants_latest", False)
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
        button.setProperty("wants_latest", True)

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
            items = self.nav.findItems("Models", Qt.MatchExactly)
            if items:
                self.nav.setCurrentRow(self.nav.row(items[0]))
            return
        else:
            main._launch_local_server()   # binary present — launch directly
        QTimer.singleShot(500, self._refresh_server_panel)

    def _acquire_llama(self) -> None:
        button = self._llama_action_btn
        latest = bool(button.property("wants_latest"))
        button.setEnabled(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            plan = plan_llama_acquisition(self.settings, latest=latest)
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
        # review dot: red circle at the BOTTOM-LEFT — caption failed a health check.
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

        # Amber banner shown only when the previewed image has unsaved edits.
        self.unsaved_bar = QLabel("Unsaved changes", self.card)
        self.unsaved_bar.setAlignment(Qt.AlignCenter)
        self.unsaved_bar.setStyleSheet(
            f"background: {theme.warning}; color: {theme.surface_0};"
            f" font-family: {self._MONO}; font-size: 10px; font-weight: 600;"
            f" border-radius: 4px; padding: 3px 0; margin-bottom: 5px;"
        )
        self.unsaved_bar.setVisible(False)
        self._bar_shown = False
        cl.addWidget(self.unsaved_bar)

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
                    unsaved: bool = False) -> None:
        fm = QFontMetrics(self.name.font())
        self.name.setText(fm.elidedText(name, Qt.ElideRight, PREVIEW_IMG_W - 60))
        self.idx.setText(index_text)
        if not pixmap.isNull():
            self.image.setPixmap(pixmap)
        if self._bar_shown != unsaved:
            self._bar_shown = unsaved
            self.unsaved_bar.setVisible(unsaved)
            self._resize_to_card()

    def show_at(self, final_pos: QPoint, arrow_x: int) -> None:
        self._arrow_x = arrow_x
        self.move(final_pos)
        self.setWindowOpacity(1.0)
        self.show()
        self.update()


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
        mime.setData(TriggerTextEdit.TRIGGER_MIME, self.text().encode("utf-8"))
        mime.setText(self.text())
        drag = QDrag(self)
        drag.setMimeData(mime)
        pm = self.grab()
        drag.setPixmap(pm)
        drag.setHotSpot(QPoint(pm.width() // 2, pm.height() // 2))
        drag.exec(Qt.CopyAction)


class TriggerTextEdit(QPlainTextEdit):
    """Per-image guidance editor where inserted triggers act as draggable chips."""

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


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.settings = load_settings()
        self.theme = Theme(self.settings)
        self.qsettings = QSettings("IdeogramCaptioner", "QtApp")
        self._default_tags = self._load_default_tags()
        self.store: CaptionStore | None = None
        self.images: list[Path] = []
        self.current: Path | None = None
        self.current_caption: dict = default_caption()
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

        self.setWindowTitle("Ideogram4 Fantastic Upgraded Captioning Kit")
        # Restore the last window size/position (and maximized/screen state);
        # fall back to a sensible default on first run or if the saved blob is bad.
        geo = self.qsettings.value("window_geometry")
        if not (geo is not None and self.restoreGeometry(geo)):
            self.resize(1400, 960)
        self.apply_appearance(self.settings)

        self._build_toolbar()
        self._build_body()
        self._restore_autosave_pref()
        self._load_guidance_presets()
        self._folder_tags: list[str] = []
        self._build_server_status()
        self._start_server_monitor()
        self._maybe_check_llama_update()
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

        self.json_action = QAction(lucide_icon("braces", ic), "Raw JSON", self)
        self.json_action.setCheckable(True)
        self.json_action.setShortcut("Ctrl+J")
        self.json_action.setToolTip("Show raw caption JSON (Ctrl+J)")
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
        self.flag_action.setToolTip("Flag this image for manual review (F)")
        self.flag_action.triggered.connect(self._toggle_review_flag)

        next_flag_action = QAction("Next flagged image", self)
        next_flag_action.setShortcut("Shift+F")
        next_flag_action.setToolTip("Jump to the next image flagged for review (Shift+F)")
        next_flag_action.triggered.connect(self._next_flagged_image)

        about_action = QAction(lucide_icon("info", ic), "About", self)
        about_action.triggered.connect(self.show_about)

        for act in (open_action, guidance_action, self.panels_action, fit_action,
                    save_action, save_all_action, self.json_action,
                    prev_action, next_action, self.flag_action, next_flag_action,
                    prefs_action, about_action):
            self.addAction(act)

        # ---- left icon rail ----
        rail = QFrame()
        rail.setObjectName("Rail")
        rail.setFixedWidth(50)
        rlay = QVBoxLayout(rail)
        rlay.setContentsMargins(7, 10, 7, 10)
        rlay.setSpacing(6)
        for act in (open_action, guidance_action, fit_action, self.panels_action,
                    self.json_action, self.flag_action):
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
            "Ideogram captioner",
            "Ideogram JSON Captioner — Qt edition\n\n"
            "A local tool for editing and generating structured JSON captions "
            "for Ideogram 4 dataset preparation.\n\n"
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
        current = field.toPlainText() if isinstance(field, QPlainTextEdit) else field.text()
        use_tags = with_tags and self.store is not None
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
            v.addWidget(editor, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        v.addWidget(buttons)
        editor.setFocus()
        if dlg.exec():
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
                "Just this image — name the specific characters or objects you want called out."
            )
        else:
            editor.setPlaceholderText(
                "Applied to every image here — art style, lighting, composition, "
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

        dlg = GuidanceDialog(self)
        dlg.setWindowTitle("Custom Caption Guidance")
        outer = QVBoxLayout(dlg)
        body = QHBoxLayout()
        outer.addLayout(body, 1)

        left = QVBoxLayout()
        body.addLayout(left, 3)
        folder_initial = self.project.folder_guidance if has_images else self.g_folder.toPlainText()
        folder_ed = self._build_popup_scope(left, "folder", "Folder · all images", folder_initial)

        if has_images:
            start_idx = self.images.index(self.current) if self.current in self.images else 0
            # work on a copy of the per-image guidance so edits can be discarded
            work_per_image = dict(self.project.per_image)
            image_initial = work_per_image.get(self.images[start_idx].name, "")
        else:
            start_idx = 0
            work_per_image = {}
            image_initial = ""
        image_ed = self._build_popup_scope(left, "image", "This image", image_initial)
        state = {"idx": start_idx}

        original_folder = folder_initial
        original_per_image = {k: v for k, v in work_per_image.items() if v.strip()}
        original_image = image_initial

        def save_image_field() -> None:
            if not has_images:
                return
            self._commit_pending(image_ed)
            img = self.images[state["idx"]]
            text = image_ed.toPlainText()
            if text.strip():
                work_per_image[img.name] = text
            else:
                work_per_image.pop(img.name, None)

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
                pm = QPixmap(str(img))
                if pm.isNull():
                    preview.setText("(cannot load image)")
                else:
                    preview.setPixmap(pm.scaled(440, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                name_label.setText(f"{img.name}   ({state['idx'] + 1} / {len(self.images)})")
                prev_btn.setEnabled(state["idx"] > 0)
                next_btn.setEnabled(state["idx"] < len(self.images) - 1)

            def go(delta: int) -> None:
                save_image_field()
                state["idx"] = max(0, min(state["idx"] + delta, len(self.images) - 1))
                image_ed._suppress = True
                image_ed.setPlainText(work_per_image.get(self.images[state["idx"]].name, ""))
                image_ed._suppress = False
                image_ed._pending = False
                image_ed._accept_btn.setEnabled(False)
                image_ed._reject_btn.setEnabled(False)
                image_ed.rescan()
                refresh_preview()

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
                current = {k: v for k, v in work_per_image.items() if v.strip()}
                return current != original_per_image
            return image_ed.toPlainText() != original_image

        def apply() -> None:
            self._commit_pending(folder_ed)
            save_image_field()
            folder_text = folder_ed.toPlainText()
            if has_images:
                self.project.folder_guidance = folder_text
                self.project.per_image = dict(work_per_image)
                self.g_folder.setPlainText(folder_text)
                if self.current is not None:
                    self.load_per_image_guidance(self.current.name)
                for img in self.images:
                    self._refresh_thumb_marker(img)
                self._guidance_dirty = True
                self.persist_guidance_if_dirty()
            else:
                self.g_folder.setPlainText(folder_text)
                image_text = image_ed.toPlainText()
                if image_text.strip():
                    self.g_per_image.setPlainText(image_text)
            nonlocal original_folder, original_per_image, original_image
            original_folder = folder_text
            original_per_image = {k: v for k, v in work_per_image.items() if v.strip()}
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
            "When off, the folder guidance is ignored for every image. Enabled once you add folder guidance.",
        )
        self.g_per_image_enabled = ToggleSwitch()
        self.g_per_image_enabled.setEnabled(False)
        self._per_image_enabled_row = self._toggle_row(
            "Apply this-image guidance", self.g_per_image_enabled,
            "When off, this image's guidance is kept but not applied. Enabled once you add per-image guidance.",
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
        self.g_per_image = QPlainTextEdit()
        self.g_per_image.setObjectName("GuidanceBoxRO")
        self.g_per_image.setReadOnly(True)
        self.g_per_image.setFixedHeight(120)
        self.g_per_image.setPlaceholderText("No guidance for this image — edit in Guidance Settings.")
        self.g_per_image.setToolTip(
            "Per-image guidance (read-only here). Edit it in Guidance Settings. Added on top "
            "of the folder guidance for this image only."
        )

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
            "Open the full editor — edit folder & per-image guidance, browse images, "
            "and manage presets."
        )
        settings_btn.clicked.connect(self._open_guidance_expand)
        lay.addWidget(settings_btn)

        # ---- Global (applies to every image) ----
        lay.addWidget(self._folder_enabled_row)
        mode_label = self._field_label("Mode")
        mode_label.setToolTip("How closely generation should follow the image (applies to the whole folder).")
        lay.addWidget(mode_label)
        lay.addWidget(self.g_mode)
        folder_label = self._field_label("Folder · all images")
        folder_label.setToolTip("Guidance applied to every image in this folder.")
        lay.addWidget(folder_label)
        lay.addWidget(self.g_folder)

        divider = QFrame()
        divider.setObjectName("PanelDivider")
        divider.setFrameShape(QFrame.HLine)
        lay.addWidget(divider)

        # ---- This image ----
        lay.addWidget(self._per_image_enabled_row)
        image_label = self._field_label("This image")
        image_label.setToolTip("Guidance applied only to the currently selected image.")
        lay.addWidget(image_label)
        lay.addWidget(self.g_per_image)

        # Tags used — read-only reflection of which palette tags appear in THIS
        # image's per-image guidance. Editing happens only in Guidance Settings.
        used_label = self._field_label("Tags used")
        used_label.setToolTip("Trigger tags referenced in this image's guidance.")
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

        # Guidance-changed banner — shown when THIS image's effective guidance has
        # changed since its caption was generated. Same violet as the filmstrip
        # flag, with a compact line-diff of what changed since the last run.
        self._gchg_box = QWidget()
        gv = QVBoxLayout(self._gchg_box)
        gv.setContentsMargins(0, 10, 0, 0)
        gv.setSpacing(4)
        self._gchg_head = QLabel("Guidance changed since last caption")
        self._gchg_head.setWordWrap(True)
        self._gchg_head.setStyleSheet(
            f"color: {STALE_COLOR}; font-weight: 600; font-size: 11px;"
        )
        self._gchg_diff = QLabel()
        self._gchg_diff.setObjectName("Hint")
        self._gchg_diff.setWordWrap(True)
        self._gchg_diff.setTextFormat(Qt.RichText)
        gv.addWidget(self._gchg_head)
        gv.addWidget(self._gchg_diff)
        self._gchg_box.setVisible(False)
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
        self.g_per_image.textChanged.connect(self._mark_guidance_dirty)
        self.g_per_image.textChanged.connect(self._refresh_tags_used)
        self.g_per_image.textChanged.connect(self._sync_per_image_toggle)
        self.g_per_image.textChanged.connect(self._schedule_stale_refresh)
        self.g_folder_enabled.toggled.connect(self._on_folder_enabled_toggled)
        self.g_per_image_enabled.toggled.connect(self._on_per_image_enabled_toggled)
        self.g_folder_enabled.toggled.connect(self._schedule_stale_refresh)
        self.g_per_image_enabled.toggled.connect(self._schedule_stale_refresh)
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

    def _refresh_tags_used(self) -> None:
        if not hasattr(self, "_used_tags_flow"):
            return
        self._clear_layout(self._used_tags_flow)
        text = self.g_per_image.toPlainText()
        known = list(getattr(self, "_folder_tags", [])) + [
            t for t in self._default_tags if t not in getattr(self, "_folder_tags", [])
        ]
        used = [t for t in known if self._tag_used_in(text, t)]
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

    def _sync_per_image_toggle(self, *args) -> None:
        """Per-image toggle: interactive only when this image has guidance; reflects
        the stored per-image override (default on)."""
        has = bool(self.g_per_image.toPlainText().strip())
        sw = self.g_per_image_enabled
        sw.setEnabled(has)
        proj = getattr(self, "project", None)
        name = self.current.name if self.current is not None else None
        active = proj.per_image_active(name) if (proj is not None and name) else True
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

    def _on_per_image_enabled_toggled(self, checked: bool) -> None:
        if self._loading or self.current is None:
            return
        self.project.per_image_enabled[self.current.name] = checked
        self._guidance_dirty = True

    def load_project_into_ui(self) -> None:
        self._loading = True
        try:
            self.g_folder.setPlainText(self.project.folder_guidance)
            self.g_mode.setCurrentText(CREATIVE_TO_MODE.get(self.project.creative_json, "Inherit"))
            self._sync_folder_toggle()
        finally:
            self._loading = False
        self._guidance_dirty = False

    def load_per_image_guidance(self, filename: str) -> None:
        self._loading = True
        try:
            self.g_per_image.setPlainText(self.project.per_image_guidance(filename))
            self._sync_per_image_toggle()
        finally:
            self._loading = False
        self._refresh_guidance_changes()

    def commit_guidance(self) -> None:
        self.project.folder_guidance = self.g_folder.toPlainText()
        # Only write the folder flag when there's text (toggle enabled); otherwise
        # leave the stored value so clearing + refilling re-applies it ("auto-on").
        if self.g_folder_enabled.isEnabled():
            self.project.folder_guidance_enabled = self.g_folder_enabled.isChecked()
        self.project.creative_json = MODE_TO_CREATIVE.get(self.g_mode.currentText())
        if self.current is not None:
            name = self.current.name
            text = self.g_per_image.toPlainText()
            if text.strip():
                self.project.per_image[name] = text
                self.project.per_image_enabled[name] = self.g_per_image_enabled.isChecked()
            else:
                self.project.per_image.pop(name, None)
                self.project.per_image_enabled.pop(name, None)

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
        lay.addStretch(1)

        for w in (self.cap_high_level, self.cap_background):
            w.textChanged.connect(self._mark_dirty)
        for w in (self.cap_aesthetics, self.cap_lighting, self.cap_medium, self.cap_style_detail):
            w.textChanged.connect(self._mark_dirty)
        self.style_mode.currentTextChanged.connect(self._on_style_mode_changed)

        scroll.setWidget(inner)
        return scroll

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

    def run_ai_job(self, operation: str) -> None:
        if self._job_running:
            return
        if self.store is None or self.current is None:
            self._set_status("Open a folder and select an image first.")
            return
        # flush pending edits so the operation works on the latest caption
        self.commit_caption_fields()
        self.commit_element_fields()
        self.persist_guidance_if_dirty()
        caption_copy = copy.deepcopy(self.current_caption)

        # resolved guidance (folder + per-image) applies to image->JSON generation;
        # the per-project creative/faithful override, when set, wins over the global.
        guidance = self.project.resolved_for(self.current.name) if operation == "json_image" else ""
        self._job_operation = operation
        self._job_guidance = guidance
        job_settings = self.settings
        if self.project.creative_json is not None:
            job_settings = replace(self.settings, creative_json=self.project.creative_json)
        image_path = self.current
        self._ensure_local_binary_then(
            lambda: self._start_ai_job(operation, job_settings, caption_copy, guidance, image_path)
        )

    def _start_ai_job(self, operation, job_settings, caption_copy, guidance, image_path) -> None:
        if not self._ensure_model_configured():
            return
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
            source_caption="",
            instructions=self.settings.json_refine_instructions,
        )
        thread.progress.connect(self._on_job_progress)
        thread.done.connect(self._on_job_done)
        thread.error.connect(self._on_job_error)
        thread.finished.connect(self._on_job_finished)
        thread.server_started.connect(self._on_server_started)
        self._ai_thread = thread
        thread.start()

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
            self._set_status("Open a folder and select an image first.")
            return
        box = QMessageBox(self)
        box.setWindowTitle("Run JSON Captioning")
        box.setText("Caption the current image, or the whole folder?")
        single_btn = box.addButton("Caption Single Image", QMessageBox.AcceptRole)
        all_btn = box.addButton("Caption All Images", QMessageBox.AcceptRole)
        box.addButton(QMessageBox.Cancel)
        box.exec()
        clicked = box.clickedButton()
        if clicked is single_btn:
            self.run_ai_job("json_image")
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
        out from under the per-image reloads."""
        self._nav_locked = locked
        self.filmstrip.setEnabled(not locked)

    def _set_read_only(self, on: bool) -> None:
        """Batch read-only mode: navigation stays live (review as captions land),
        but the caption editor is paused so a completing item can't clobber edits."""
        self._read_only = on
        if hasattr(self, "right_tabs"):
            self.right_tabs.setEnabled(not on)
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
        self.open_preferences("Models")
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
        resp = QMessageBox.question(
            self, "Download model files?",
            f"Starting the server will fetch {name} from Hugging Face. "
            f"These files aren't downloaded yet:\n\n{listing}\n\nDownload them now?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        )
        return resp == QMessageBox.Yes

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
        already = [img for img in self.images if self._image_has_caption(img)]
        already_set = set(already)
        new_imgs = [img for img in self.images if img not in already_set]
        stale = [img for img in already if self.project.guidance_changed(img.name)]
        work = list(self.images)
        if already:
            box = QMessageBox(self)
            box.setWindowTitle("Caption all images")
            msg = f"{len(already)} of {total} image(s) already have a caption."
            if stale:
                msg += (f"\n{len(stale)} of those have guidance changes since they "
                        "were last captioned.")
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
            resp = QMessageBox.question(
                self, "Caption all images",
                f"Generate JSON for all {total} image(s)?\n\n"
                "Images are processed one at a time through your configured server.",
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
        items = [(img, self.project.resolved_for(img.name)) for img in work]
        # remember the guidance actually sent, to stamp each caption on completion
        self._batch_guidance = {str(img): g for img, g in items}
        # health/dup tracking for this run: serialized caption -> first filename seen
        self._batch_caption_hashes = {}
        self._batch_flagged = {}
        n = len(items)
        delay = int(self.qsettings.value("batch_delay_ms", 0, int) or 0)
        self._ensure_local_binary_then(
            lambda: self._start_batch_job(job_settings, items, n, delay)
        )

    def _start_batch_job(self, job_settings, items, n, delay) -> None:
        if not self._ensure_model_configured():
            return
        if not self._confirm_model_download():
            self._set_status("Cancelled.")
            return
        self._job_cancelled = False
        self._set_ai_running(True)
        self._set_read_only(True)
        self._set_job_progress(f"Captioning 0/{n}…", value=0, total=n)
        thread = BatchCaptionThread(job_settings, items, delay_ms=delay)
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
            self.store.save_caption(path, caption)
        except Exception as exc:
            self._set_status(f"Save failed for {path.name}: {exc}")
            return
        self._pending.pop(image_path_str, None)
        guidance = getattr(self, "_batch_guidance", {}).get(
            image_path_str, self.project.resolved_for(path.name)
        )
        self.project.mark_generated(path.name, guidance)   # persisted at batch end
        # Tier 0-2 health check: flag corrupt/off-schema captions for review.
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
        self._set_status(f"Failed {Path(image_path_str).name}: {message}")

    def _on_batch_finished(self, success: int, fail: int, cancelled: bool) -> None:
        self._set_ai_running(False)
        self._set_read_only(False)
        self._ai_thread = None
        # persist the per-image guidance stamps gathered during the run
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
            self.store.save_caption(self.current, caption)
        except Exception as exc:
            QMessageBox.critical(self, "Could not save result", str(exc))
            return
        # the AI result is now on disk; drop any buffered edits and reload it
        self._pending.pop(str(self.current), None)
        if getattr(self, "_job_operation", "") == "json_image":
            self.project.mark_generated(self.current.name, getattr(self, "_job_guidance", ""))
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

    def _on_job_error(self, message: str) -> None:
        if self._job_cancelled:
            return
        QMessageBox.critical(self, "AI job failed", message)
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

        # Left: folder + per-image guidance.
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
        center_lay.addWidget(self.view, 1)
        center_lay.addWidget(self._build_nav_bar())
        # Floating tool strip: an overlay child of the view (NOT the viewport — the
        # viewport scrolls its children when panning, which would drag the strip).
        self._toolstrip = self._build_canvas_toolstrip()
        self._toolstrip.setParent(self.view)
        self._toolstrip.raise_()

        # Right: AI actions above a tabbed Caption / Elements panel.
        self.right_tabs = QTabWidget()
        self.right_tabs.addTab(self._build_caption_tab(), "Caption")
        self.right_tabs.addTab(self._build_elements_tab(), "Elements")

        right_container = QWidget()
        right_container.setObjectName("Panel")
        right_lay = QVBoxLayout(right_container)
        right_lay.setContentsMargins(0, 0, 0, 0)
        right_lay.setSpacing(0)
        self._readonly_banner = QLabel(
            "Captioning in progress — editing paused (read-only). Browse and flag images "
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
        right_lay.addWidget(self._build_ai_actions())
        right_lay.addWidget(self.right_tabs, 1)
        right_container.setMinimumWidth(290)

        self.json_panel = self._build_json_panel()

        splitter.addWidget(self.left_panel)
        splitter.addWidget(center)
        splitter.addWidget(right_container)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([210, 820, 360])
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
        self._json_tab = VerticalTab("RAW JSON")
        self._json_tab.setToolTip("Show raw caption JSON (Ctrl+J)")
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
        header = QLabel("Raw caption JSON")
        header.setObjectName("SectionLabel")
        lay.addWidget(header)
        self.json_view = QPlainTextEdit()
        self.json_view.setReadOnly(True)
        self.json_view.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.json_view.setFont(QFont(self.settings.mono_font_family or "Monospace"))
        lay.addWidget(self.json_view, 1)
        row = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        copy_btn.setToolTip("Copy the JSON to the clipboard.")
        copy_btn.clicked.connect(self._copy_json)
        save_btn = QPushButton("Save as…")
        save_btn.setToolTip("Save the JSON to a file.")
        save_btn.clicked.connect(self._save_json_as)
        row.addWidget(copy_btn)
        row.addWidget(save_btn)
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
                "Hide raw caption JSON (Ctrl+J)" if checked else "Show raw caption JSON (Ctrl+J)"
            )
        self.json_action.setToolTip(
            "Hide raw caption JSON (Ctrl+J)" if checked else "Show raw caption JSON (Ctrl+J)"
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
        if not getattr(self, "json_panel", None) or not self.json_panel.isVisible():
            return
        try:
            pretty = json.dumps(json.loads(serialize_caption(self.current_caption)),
                                indent=2, ensure_ascii=False)
        except Exception:
            pretty = ""
        sb = self.json_view.verticalScrollBar()
        pos = sb.value()
        self.json_view.setPlainText(pretty)
        sb.setValue(min(pos, sb.maximum()))

    def _copy_json(self) -> None:
        QApplication.clipboard().setText(self.json_view.toPlainText())
        self._set_status("Caption JSON copied to clipboard.")

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
        start_dir = self.qsettings.value("last_folder", "", str)
        if start_dir and not Path(start_dir).is_dir():
            start_dir = ""
        folder = QFileDialog.getExistingDirectory(self, "Open image folder", start_dir)
        if not folder:
            return
        self.qsettings.setValue("last_folder", folder)
        try:
            self.store = CaptionStore(Path(folder), self.settings_caption_ext())
            self.images = self.store.images()
            self.project = self.store.load_project()
            self._load_folder_tags()
        except Exception as exc:  # Tier 2: surface failures readably.
            QMessageBox.critical(self, "Could not open folder", str(exc))
            return

        self.load_project_into_ui()
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
            self._set_status(f"No images found in {folder}")
            self._update_count_label()
            return

        for path in self.images:
            self._thumb_base[str(path)] = self._thumb_pixmap(path)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, str(path))
            item.setData(UNSAVED_ROLE, False)
            item.setData(STALE_ROLE, self.project.guidance_changed(path.name))
            _issues = self.project.caption_issues(path.name)
            _flagged = self.project.is_review_marked(path.name)
            item.setData(REVIEW_ROLE, bool(_issues))
            item.setData(FLAG_ROLE, _flagged)
            item.setToolTip(self._thumb_tooltip(_issues, _flagged))
            self.filmstrip.addItem(item)
            self._thumb_items[str(path)] = item
            item.setIcon(self._decorated_thumb(path))
            item.setText(self._thumb_label(path))
        self.filmstrip.setCurrentRow(0)
        self._set_status(f"{len(self.images)} images in {Path(folder).name}")
        self._update_count_label()

    def settings_caption_ext(self) -> str:
        return ".json"

    def _thumb_pixmap(self, path: Path) -> QPixmap:
        pm = QPixmap(str(path))
        if pm.isNull():
            return QPixmap(THUMB, THUMB)
        return pm.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def eventFilter(self, obj, event):
        if obj is self.filmstrip.viewport():
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
        self._hover_preview.set_content(pm, path.name, idx_text, self._has_unsaved(path))

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
        win.show_at(QPoint(left, y), int(arrow_x))

    def _on_thumb_changed(self, current: QListWidgetItem | None, _prev: QListWidgetItem | None) -> None:
        if current is None:
            return
        if self.current is not None:
            self.commit_caption_fields()
            self.commit_element_fields()
            if self._autosave:
                if self._dirty:
                    self.save_current(silent=True)
            elif self._dirty:
                # keep edits in memory; do NOT write to disk until the user saves
                self._pending[str(self.current)] = self.current_caption
                self._refresh_thumb_marker(self.current)
        self.persist_guidance_if_dirty()
        path = Path(current.data(Qt.UserRole))
        self.show_image(path)
        self.load_caption_for(path)
        self.load_per_image_guidance(path.name)
        self._sync_flag_action()

    def show_image(self, path: Path) -> None:
        self.current = path
        pm = QPixmap(str(path))
        self.scene.clear()
        self.pixmap_item = None
        self.box_items = []
        if pm.isNull():
            self._set_status(f"Could not open {path.name}")
            return
        self.pixmap_item = self.scene.addPixmap(pm)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self._user_zoomed = False
        self.view.resetTransform()
        self.view.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._refresh_title()
        self._set_status(f"{path.name}  ·  {pm.width()}×{pm.height()}")
        self._update_count_label()
        self._update_zoom_label()

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
        prefix = "• " if self.project.has_per_image_guidance(path.name) else ""
        return prefix + path.name[:14]

    def _decorated_thumb(self, path: Path) -> QIcon:
        # The unsaved indicator is now an amber corner dot drawn by the
        # FilmstripDelegate, so the thumbnail itself stays unaltered.
        base = self._thumb_base.get(str(path))
        return QIcon(base) if base is not None else QIcon()

    def _thumb_tooltip(self, issues: list[str], flagged: bool) -> str:
        parts = []
        if flagged:
            parts.append("Flagged for review")
        if issues:
            parts.append("Possible problems:\n• " + "\n• ".join(issues))
        return "\n\n".join(parts)

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
            self._set_status("No images flagged for review")
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
        act = menu.addAction("Clear review flag" if marked else "Flag for review")
        if menu.exec(self.filmstrip.mapToGlobal(pos)) == act:
            self._toggle_review_flag_for(path)

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
            item.setToolTip(self._thumb_tooltip(issues, flagged))
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
        dot = "● " if self.project.has_per_image_guidance(path.name) else ""
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
        or per-image edit flags immediately, before it's persisted to disk."""
        if self.store is None:
            self._refresh_guidance_changes()
            return
        if getattr(self, "_used_tags_flow", None) is not None:
            self.commit_guidance()
        for key, item in self._thumb_items.items():
            stale = self.project.guidance_changed(Path(key).name)
            if bool(item.data(STALE_ROLE)) != stale:
                item.setData(STALE_ROLE, stale)
                self._repaint_thumb(key)
        self._refresh_guidance_changes()

    def _refresh_guidance_changes(self) -> None:
        """Show/hide the read-only 'guidance changed' banner for the current image,
        with a compact diff of what changed since its caption was last generated."""
        if not hasattr(self, "_gchg_box"):
            return
        name = self.current.name if self.current is not None else None
        if name is None or not self.project.guidance_changed(name):
            self._gchg_box.setVisible(False)
            return
        prev = self.project.last_run_guidance(name) or ""
        curr = self.project.resolved_for(name)
        self._gchg_diff.setText(self._guidance_diff_html(prev, curr))
        self._gchg_box.setVisible(True)

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
        self.commit_caption_fields()
        self.commit_element_fields()
        try:
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
            self.commit_caption_fields()
            self.commit_element_fields()
            if self._dirty:
                self._pending[str(self.current)] = self.current_caption
        saved = 0
        failed = []
        for key, caption in list(self._pending.items()):
            try:
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
            self._set_status("Open a folder and select an image first.")
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
            self._count_label.setText(f"Image {idx} / {total}")
            if nav is not None:
                nav.setText(f"{idx} / {total}")
        else:
            self._count_label.setText(f"{total} images")
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
    app.setApplicationName("Ideogram4 Fantastic Upgraded Captioning Kit")
    app.setApplicationDisplayName("Ideogram4 Fantastic Upgraded Captioning Kit")
    icon = app_icon()
    app.setWindowIcon(icon)
    window = MainWindow()
    window.setWindowIcon(icon)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
