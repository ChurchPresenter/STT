---
name: STT
description: Self-hosted real-time speech transcription platform for live event operators.
colors:
  dark:
    bg: "#0D0E11"
    side: "#111318"
    surface: "#16181E"
    surface-2: "#1E2129"
    inset: "#0F1115"
    border: "#272B34"
    text-primary: "#ECEEF3"
    text-secondary: "#A3AAB7"
    text-muted: "#848B99"
    accent: "#5B5EF0"
    accent-text: "#AEB0FF"
    success: "#3DD68C"
    warning: "#F2B042"
    danger: "#F87171"
    danger-solid: "#D93A3A"
  light:
    bg: "#F4F5F7"
    side: "#FBFBFC"
    surface: "#FFFFFF"
    surface-2: "#F2F3F6"
    inset: "#F7F8FA"
    border: "#E2E4E9"
    text-primary: "#14161B"
    text-secondary: "#555B68"
    text-muted: "#676D7B"
    accent: "#4F46E5"
    accent-text: "#4338CA"
    success: "#0B7A50"
    warning: "#8F5600"
    danger: "#B42323"
    danger-solid: "#D13434"
typography:
  family: "Geist, Inter, Segoe UI, -apple-system, sans-serif"
  mono: "Geist Mono, ui-monospace, Menlo, Consolas, monospace"
  page-title: { fontSize: "1.625rem", fontWeight: 600, letterSpacing: "-0.02em" }
  card-title: { fontSize: "0.9375rem", fontWeight: 600 }
  body: { fontSize: "0.875rem", fontWeight: 400, lineHeight: 1.45 }
  label: { fontSize: "0.8125rem", fontWeight: 500 }
  group-label: { fontSize: "0.6875rem", fontWeight: 600, letterSpacing: "0.08em", textTransform: "uppercase" }
rounded:
  sm: 8px
  md: 10px
  lg: 14px
---

# STT design

The operator console for live events: calm, dense, readable across a room. Two themes, **dark** and **light**, share one layout. The caption display (`templates/index.html`) is a separate register with its own palette and is never restyled by this system.

## Tokens

All colours live as CSS variables in `templates/base.html` (`:root` for dark, `:root[data-theme="light"]` for light, plus a no-JS `prefers-color-scheme` fallback). Pages use the variables, never hard-coded hex.

- **Text-coloured status** (`--success`, `--warning`, `--danger`) is tuned per theme for text on the surface. Use the `-bg` variants for tinted backgrounds.
- **Solid buttons** use `--success-solid`, `--danger-solid`, `--warning-solid` (and `-hover`), which hold white-text contrast in both themes. Never put white text on `--success` or `--danger`.
- The theme is stored in `localStorage['stt-theme']`; the head script applies it before first paint and falls back to the OS setting.

## Layout

A sticky 248px **sidebar** (brand, grouped navigation, server and disk status, Light/Dark switch) beside a main column with an eyebrow label naming the group, then the page title. Navigation groups: **Operate** (Live, Corrections, File transcription), **Configure** (Live settings, Translation, Highlighting, Display & URLs), **Manage** (Models, Files, Server). Below 900px the sidebar becomes a top bar with a Menu button.

## Components

- **Cards**: surface background, 1px border, 14px radius, no shadow (a faint one in light only), no coloured side stripes.
- **Buttons**: 10px radius, 600 weight; primary is the accent, status buttons use the solid tokens, ghost buttons are transparent with a border. No lift on hover.
- **Inputs**: inset background, 1px border, 8px radius, accent focus ring.
- **Segmented control and toggles**: 3px track padding, active segment on the surface colour; toggle is a 40×24 pill.
- **Nav item**: 38px tall, 9px radius, active item on `--accent-subtle` with `--accent-text`.
- **Monospace** (Geist Mono) for times, ports, thresholds and numeric values.

## Rules

- Contrast: body text 4.5:1 in both themes, including faint labels.
- No gradient text, no glow cards, no glassmorphism.
- Motion is 150–200ms with `--ease-out`, honours `prefers-reduced-motion`, and nothing on the live view moves for decoration.
- Fonts are self-hosted from `static/fonts/` (Geist, Geist Mono, Inter fallback); the UI must work with no network.
- New UI follows this file; the source of the design is the "STT UI Redesign" canvas.
