"""
Stilisert illustrasjon av flaskehals mellom to budområder.

Viser hvordan begrenset overføringskapasitet mellom et område med kraftoverskudd
(vannkraft og vindkraft) og et område med kraftunderskudd (industri og
husholdning) gir opphav til en flaskehals — og dermed adskilte prisområder.

Brukes som illustrasjon i teorikapittelet (Flaskehalser / Budområder).
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon, PathPatch
from matplotlib.path import Path as MplPath

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import apply_style, C1, C2, C3, C4, C5, lighten  # noqa: E402

apply_style()

# ── Figuroppsett ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 7.4))
ax.set_xlim(0, 13)
ax.set_ylim(0, 8)
ax.set_aspect("equal")
ax.axis("off")

BG = "#F7F7F5"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)

# ── Bakgrunnssoner: to tydelige budområder ───────────────────────────────────
# Venstre område (kraftoverskudd) — lys teal
AREA_LEFT  = lighten(C1, 0.18)  # veldig lys teal
AREA_RIGHT = lighten(C4, 0.15)  # veldig lys mørkegrønn

# Bruk avrundede rektangler som "soner"
zone_left = FancyBboxPatch(
    (0.25, 0.7), 5.7, 6.6,
    boxstyle="round,pad=0.0,rounding_size=0.35",
    fc=AREA_LEFT, ec=C1, lw=1.2, alpha=0.85, zorder=0,
)
zone_right = FancyBboxPatch(
    (7.05, 0.7), 5.7, 6.6,
    boxstyle="round,pad=0.0,rounding_size=0.35",
    fc=AREA_RIGHT, ec=C4, lw=1.2, alpha=0.85, zorder=0,
)
ax.add_patch(zone_left)
ax.add_patch(zone_right)

# Områdeetiketter — tydelig plassert øverst i hver sone
ax.text(3.1, 6.85, "Budområde A",
        fontsize=22, fontweight="bold", color=C5,
        ha="center", va="center", zorder=10)
ax.text(3.1, 6.35, "Kraftoverskudd",
        fontsize=17, color="#444444", style="italic",
        ha="center", va="center", zorder=10)

ax.text(9.9, 6.85, "Budområde B",
        fontsize=22, fontweight="bold", color=C5,
        ha="center", va="center", zorder=10)
ax.text(9.9, 6.35, "Kraftunderskudd",
        fontsize=17, color="#444444", style="italic",
        ha="center", va="center", zorder=10)


# ── Ikoner ───────────────────────────────────────────────────────────────────
def icon_card(ax, x, y, w=1.2, h=1.2, fc="white", ec="#CCCCCC"):
    """Hvit avrundet boks som holder et ikon."""
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.14",
                       fc=fc, ec=ec, lw=1.2, zorder=5)
    ax.add_patch(p)


def hydro_icon(ax, x, y, color):
    """Vannkraftverk: demning + reservoar + turbin-utløp."""
    icon_card(ax, x, y)
    # Reservoar (vann bak demningen) — lys blå-grønn for å vise vann
    water_top = lighten(C1, 0.50)
    ax.add_patch(Polygon(
        [(x-0.50, y+0.05), (x-0.10, y+0.05),
         (x-0.10, y+0.40), (x-0.50, y+0.40)],
        closed=True, fc=water_top, ec="none", zorder=6,
    ))
    # Bølger på reservoaret
    for dy in [0.18, 0.30]:
        ax.plot([x-0.46, x-0.38, x-0.30, x-0.22, x-0.14],
                [y+dy, y+dy+0.025, y+dy, y+dy+0.025, y+dy],
                color=color, lw=1.0, zorder=7)
    # Demning (skrå mur)
    ax.add_patch(Polygon(
        [(x-0.10, y-0.40), (x+0.02, y-0.40),
         (x-0.04, y+0.42), (x-0.10, y+0.42)],
        closed=True, fc=color, ec=color, zorder=7,
    ))
    # Utløp / kraftstasjon nederst
    ax.add_patch(Rectangle((x+0.02, y-0.40), 0.46, 0.22,
                           fc="#888888", ec="#888888", zorder=6))
    # Vannstrøm ut (zig-zag)
    ax.plot([x-0.04, x+0.06, x+0.16, x+0.26, x+0.36],
            [y-0.18, y-0.10, y-0.18, y-0.10, y-0.18],
            color=water_top, lw=2.0, zorder=7)
    # Bakke (linje under)
    ax.plot([x-0.50, x+0.50], [y-0.40, y-0.40],
            color="#999999", lw=1.0, zorder=6)


def wind_icon(ax, x, y, color):
    """Stilisert vindturbin: tårn + tre rotorblader."""
    icon_card(ax, x, y)
    # Tårn (smalere på toppen)
    ax.add_patch(Polygon(
        [(x-0.05, y-0.40), (x+0.05, y-0.40),
         (x+0.03, y+0.10), (x-0.03, y+0.10)],
        closed=True, fc="#888888", ec="#888888", zorder=6,
    ))
    # Nav
    ax.add_patch(plt.Circle((x, y+0.13), 0.06,
                            fc=color, ec=color, zorder=8))
    # Tre blader (120°)
    for theta in [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]:
        dx, dy = 0.36*np.cos(theta), 0.36*np.sin(theta)
        ax.plot([x, x+dx], [y+0.13, y+0.13+dy],
                color=color, lw=3.0, solid_capstyle="round", zorder=7)


def factory_icon(ax, x, y, color):
    """Stilisert fabrikk: bygning + sagtannstak + skorstein."""
    icon_card(ax, x, y)
    # Bygningskropp
    ax.add_patch(Rectangle((x-0.42, y-0.38), 0.84, 0.50,
                           fc="#888888", ec="#888888", zorder=6))
    # Sagtannstak
    for x0 in [-0.40, -0.20, 0.00, 0.20]:
        ax.add_patch(Polygon(
            [(x+x0, y+0.12), (x+x0+0.20, y+0.12),
             (x+x0+0.10, y+0.30)],
            closed=True, fc=color, ec=color, zorder=7,
        ))
    # Vinduer
    for wx in [-0.28, -0.08, 0.12, 0.32]:
        ax.add_patch(Rectangle((x+wx-0.05, y-0.22), 0.10, 0.10,
                               fc=color, ec=color, zorder=7))
    # Bakke
    ax.plot([x-0.50, x+0.50], [y-0.38, y-0.38],
            color="#999999", lw=1.0, zorder=6)


def house_icon(ax, x, y, color):
    """Stilisert husholdning."""
    icon_card(ax, x, y)
    # Vegger
    ax.add_patch(Rectangle((x-0.34, y-0.36), 0.68, 0.50,
                           fc="#888888", ec="#888888", zorder=6))
    # Tak
    ax.add_patch(Polygon(
        [(x-0.42, y+0.14), (x+0.42, y+0.14), (x, y+0.40)],
        closed=True, fc=color, ec=color, zorder=7,
    ))
    # Dør
    ax.add_patch(Rectangle((x-0.08, y-0.36), 0.16, 0.28,
                           fc=color, ec=color, zorder=7))
    # Vinduer
    for wx in [-0.20, 0.20]:
        ax.add_patch(Rectangle((x+wx-0.06, y-0.10), 0.12, 0.12,
                               fc=color, ec=color, zorder=7))
    # Bakke
    ax.plot([x-0.50, x+0.50], [y-0.36, y-0.36],
            color="#999999", lw=1.0, zorder=6)


# ── Plassering av komponenter ────────────────────────────────────────────────
# Venstre sone (Budområde A) — produksjon
prod_top1 = (1.6, 4.7)   # vannkraft øverst
prod_top2 = (3.7, 5.2)   # vannkraft midt
prod_bot  = (2.3, 2.4)   # vindkraft

# Høyre sone (Budområde B) — forbruk
cons_top = (11.0, 4.7)   # industri
cons_bot = (11.0, 2.2)   # husholdning

# Flaskehals-knutepunkt
hub = (6.5, 3.7)
hub_w, hub_h = 0.95, 0.95


# ── Kraftlinjer som myke kurver (Bezier) ─────────────────────────────────────
def curved_line(ax, p_from, p_to, color, lw=4.0, sag=0.45, zorder=3):
    """
    Tegn en myk kurve mellom to punkter ved hjelp av en kvadratisk Bezier.
    `sag` styrer hvor mye kurven bøyer (positiv = bøyer nedover).
    """
    x0, y0 = p_from
    x1, y1 = p_to
    # Kontrollpunkt: midt mellom + vertikalt offset
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2 - sag
    verts = [(x0, y0), (cx, cy), (x1, y1)]
    codes = [MplPath.MOVETO, MplPath.CURVE3, MplPath.CURVE3]
    path = MplPath(verts, codes)
    patch = PathPatch(path, fc="none", ec=color, lw=lw,
                      capstyle="round", zorder=zorder)
    ax.add_patch(patch)


# Produksjonssegmenter (lys teal, hengende)
curved_line(ax, prod_top1, (hub[0]-0.2, hub[1]+0.1), color=C1, lw=5.0, sag=0.55)
curved_line(ax, prod_top2, (hub[0]-0.1, hub[1]+0.25), color=C2, lw=5.0, sag=0.50)
curved_line(ax, prod_bot,  (hub[0]-0.25, hub[1]-0.1), color=C1, lw=5.0, sag=0.35)

# Forbrukssegmenter (mørk teal/grønn)
curved_line(ax, (hub[0]+0.2, hub[1]+0.15), cons_top, color=C4, lw=5.0, sag=0.40)
curved_line(ax, (hub[0]+0.2, hub[1]-0.15), cons_bot, color=C5, lw=5.0, sag=0.55)


# ── Flaskehals-knutepunkt ────────────────────────────────────────────────────
HIGHLIGHT = lighten(C1, 0.55)
ax.add_patch(FancyBboxPatch(
    (hub[0] - hub_w/2, hub[1] - hub_h/2), hub_w, hub_h,
    boxstyle="round,pad=0.02,rounding_size=0.20",
    fc="white", ec=C1, lw=3.2, zorder=8,
))
ax.add_patch(FancyBboxPatch(
    (hub[0] - hub_w/2 + 0.12, hub[1] - hub_h/2 + 0.12),
    hub_w - 0.24, hub_h - 0.24,
    boxstyle="round,pad=0.0,rounding_size=0.12",
    fc=HIGHLIGHT, ec="none", alpha=0.55, zorder=8,
))


# ── Tegn ikoner ──────────────────────────────────────────────────────────────
hydro_icon(ax, *prod_top1, color=C1)
hydro_icon(ax, *prod_top2, color=C2)
wind_icon(ax,  *prod_bot,  color=C1)
factory_icon(ax, *cons_top, color=C4)
house_icon(ax,   *cons_bot, color=C5)


# ── Flaskehals-etikett med pil ───────────────────────────────────────────────
ax.annotate(
    "Flaskehals",
    xy=(hub[0], hub[1] + hub_h/2),
    xytext=(hub[0] - 1.6, hub[1] + 2.3),
    fontsize=22, fontweight="bold", color=C1, ha="center", va="center",
    arrowprops=dict(arrowstyle="-|>", color=C1, lw=2.0,
                    mutation_scale=20,
                    connectionstyle="arc3,rad=0.25",
                    shrinkA=2, shrinkB=10),
    zorder=11,
)

plt.tight_layout()

# ── Lagre ────────────────────────────────────────────────────────────────────
OUT = ROOT.parent / "figures" / "illustrasjoner"
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"flaskehals.{ext}",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())

print(f"Lagret: {OUT / 'flaskehals.pdf'}")
print(f"Lagret: {OUT / 'flaskehals.png'}")
