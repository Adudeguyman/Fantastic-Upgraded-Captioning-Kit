# Filmstrip — Hover Preview & Unsaved Indicator

Exact specs pulled from the mockup. Values are literal. All from the dark-theme token set (no light theme in this app).

## Tokens involved
| token | hex | role |
|---|---|---|
| surface-1 | `#171A1F` | filmstrip background / thumb cutout ring |
| surface-2 | `#1E2227` | popup body fill |
| surface-image | `#15181C` | image area / thumbnail base |
| border-strong-hover | `#454C56` | popup border + arrow border |
| thumb-border | `#2E343B` | thumbnail default border |
| thumb-border-hover | `#3A4654` | thumbnail hover border |
| accent | `#4C8DFF` | selected thumbnail border + filename |
| warning | `#E0A33B` | unsaved indicator + unsaved filename |
| text-primary | `#ECEEF1` | popup filename |
| text-muted | `#6C737C` | popup index |
| text-secondary | `#A6ADB6` | default filename |

Geometry/motion: radius 6px (thumb), 8px (popup), 5px (popup image). Spacing 5/6/8px. Motion vars: hover-delay 350ms, show 140ms OutCubic, hide 90ms OutQuad.

Thumbnail base size: **72 × 52px**, radius 6px. (The striped `repeating-linear-gradient` background is a placeholder for the real QPixmap — ignore it.)

---

## 1. Hover-preview popup

### Interaction
On hover, after **350ms** dwell, the preview **fades in (opacity 0→1) + scales (0.96→1) + lifts (translateY 4px→0)** over **140ms** `ease-out` (cubic-bezier(0.215,0.61,0.355,1)), transform-origin bottom center. On leave it **fades out over 90ms** `ease-in` (no scale/translate) and the dwell timer resets immediately.

### HTML/CSS (as built)
```html
<!-- anchor: position:relative on the filmstrip row; popup is absolutely placed -->
<div class="thumb-preview">
  <div class="preview-img"></div>
  <div class="preview-meta">
    <span class="preview-name">Golden-retriever-pups-08.jpg</span>
    <span class="preview-idx">6 / 18</span>
  </div>
  <span class="preview-arrow"></span>
</div>
```
```css
.thumb-preview{
  position:absolute;
  bottom:100%;            /* sits above the filmstrip row */
  margin-bottom:8px;      /* gap between popup and thumb */
  left:366px;             /* = thumb's left within the row; center popup over thumb in code */
  z-index:30;
  width:208px;
  background:#1E2227;
  border:1px solid #454C56;
  border-radius:8px;
  padding:6px;
  box-shadow:0 12px 32px rgba(0,0,0,.55);   /* DECORATIVE — see Qt note */
}
.preview-img{
  width:196px; height:147px;   /* 4:3; = popup width minus 2×6 padding */
  border-radius:5px;
  background:#15181C;           /* placeholder; draw the QPixmap here */
}
.preview-meta{
  display:flex; align-items:center; justify-content:space-between;
  gap:8px; padding:7px 4px 2px;
}
.preview-name{                 /* filename, truncates */
  font:400 11px 'IBM Plex Mono'; color:#ECEEF1;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.preview-idx{                  /* position in set */
  flex:0 0 auto;
  font:400 10px 'IBM Plex Mono'; color:#6C737C;
}
.preview-arrow{                /* 9px diamond pointer, centered under popup */
  position:absolute; left:99px; bottom:-5px;   /* left ≈ popup centre − 4.5 */
  width:9px; height:9px; background:#1E2227;
  border-right:1px solid #454C56; border-bottom:1px solid #454C56;
  transform:rotate(45deg);
}
```

### Anchor / offset
- Anchored to the **hovered thumbnail**, floating **above** it (`bottom:100%` + `margin-bottom:8px` → 8px gap).
- Horizontally **centered on the thumb**: in Qt, `popup_x = thumb_center_x − popup_width/2`, clamped to the filmstrip bounds so edge thumbs don't overflow. Move the arrow's x to stay under the thumb center after clamping.
- Pointer: a 9px square rotated 45° (a diamond), same fill as the body with only its right+bottom edges bordered so it reads as a tail. Centered under the popup (`left ≈ width/2 − 4.5`).

### Contents
- **Image** 196×147 (the only required element).
- **Filename** — IBM Plex Mono 11px / 400, `#ECEEF1`, ellipsis-truncated.
- **Index** — IBM Plex Mono 10px / 400, `#6C737C`, e.g. `6 / 18`. (We did not include pixel dimensions; add as a third muted span if you want them.)

### Qt notes
- `box-shadow:0 12px 32px rgba(0,0,0,.55)` is **web-only** → use `QGraphicsDropShadowEffect(blurRadius=32, xOffset=0, yOffset=12, color=rgba(0,0,0,140))` on the popup widget, or drop it and lean on the 1px `#454C56` border.
- Easiest as a frameless `QWidget`/`QToolTip`-style popup. Animate `windowOpacity` (0→1) with `QPropertyAnimation`, 140ms `OutCubic`; for the scale/translate use a `QGraphicsOpacityEffect` + geometry tween or skip them (fade alone is fine and the cheapest faithful version). Dwell delay = a 350ms `QTimer` started on enter, cancelled on leave.

---

## 2. Unsaved-changes indicator (the clean version)

We dropped the red glow. The replacement is an **amber dot badge** in the **top-right corner** of the thumbnail, plus the **filename text turns amber**. That's it — no border glow, no tint.

### What it is
- **Dot:** 10×10px circle, `background:#E0A33B` (warning), with a **2px ring in the filmstrip background color `#171A1F`** so it punches cleanly off both the thumb and any selected-border. Positioned `top:-5px; right:-5px` (straddling the top-right corner).
- **Filename:** same 10px Mono, color switches from `#A6ADB6` → `#E0A33B`.

### HTML/CSS (as built)
```html
<div class="thumb-cell">
  <div class="thumb">                                  <!-- add 'is-selected' when active -->
    <span class="thumb-dot" title="Unsaved changes"></span>  <!-- only when dirty -->
  </div>
  <div class="thumb-name is-dirty">Beagle-…</div>      <!-- is-dirty / is-selected toggles color -->
</div>
```
```css
.thumb{
  position:relative;
  width:72px; height:52px; border-radius:6px;
  background:#15181C;            /* QPixmap goes here */
  border:1px solid #2E343B;
}
.thumb.is-selected{ border:2px solid #4C8DFF; }   /* selected state */
.thumb-dot{
  position:absolute; top:-5px; right:-5px;
  width:10px; height:10px; border-radius:50%;
  background:#E0A33B;
  border:2px solid #171A1F;     /* ring = filmstrip bg, so dot reads on any border */
}
.thumb-name{ font:400 10px 'IBM Plex Mono'; color:#A6ADB6;
  width:72px; text-align:center; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.thumb-name.is-dirty{ color:#E0A33B; }
.thumb-name.is-selected{ color:#4C8DFF; }     /* selected wins on the filename color */
```

### Coexistence with other states
- **Selected + unsaved:** the **blue 2px border** (selected) and the **amber dot** (unsaved) live on different layers — border on the edge, dot at the corner with its own `#171A1F` ring — so they never collide. For the **filename**, selected color (`#4C8DFF`) takes precedence over dirty (`#E0A33B`); the dot still carries the unsaved signal, so no information is lost.
- **Per-image guidance dot** (if shown): put it **top-LEFT** (`top:-5px; left:-5px`), same 10px + ring treatment in a different hue, so the two corner badges never overlap.

### Appear / disappear animation
- Appear: dot **scales 0→1** (transform-origin center) + fade, **120ms `OutCubic`**. Filename color cross-fades over the same 120ms.
- Disappear (on save): **fade + scale 1→0 over 90ms `OutQuad`**.
- No pulsing/looping — it's a persistent state marker, not an alert.

### Qt notes
- Pure solid fill + 1px/2px borders + border-radius → **translates 1:1, no web-only effects.** The corner ring replaces what a glow/shadow would have done.
- Animate with `QPropertyAnimation` on a small badge `QWidget` (geometry for scale, or a `QGraphicsOpacityEffect` for fade). Dark-only — no light variant needed.

---

## Web-only effects to watch (whole filmstrip)
| effect | where | Qt approximation |
|---|---|---|
| `box-shadow:0 12px 32px rgba(0,0,0,.55)` | hover popup | `QGraphicsDropShadowEffect` (blur 32, y 12, alpha 140) or drop |
| scale/translate on popup-in | hover popup | optional; fade-only is faithful and cheapest |
| corner-overhang badge (`top/right:-5px`) | unsaved dot | none needed — absolute child widget, allow it to overflow the thumb rect |

No backdrop-blur, CSS filters, or layered/inset shadows are used in either component.
