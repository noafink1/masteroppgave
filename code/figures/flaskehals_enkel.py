"""
Enkel illustrasjon av flaskehals mellom to budområder.

Forenklet versjon av flaskehals.py. Kraftlinjer går kontinuerlig fra produsent
(vannkraft, vindkraft) direkte til forbruker (fabrikk, hus), hver med sin egen
farge. Flaskehalsen er en boks som ligger oppå linjene der budområdet endrer
seg — den representerer overføringsbegrensningen, ikke et fysisk knutepunkt.
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
fig, ax = plt.subplots(figsize=(12, 6.5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.set_aspect("equal")
ax.axis("off")

BG = "#F7F7F5"
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)


# ── Ikonbyggere ──────────────────────────────────────────────────────────────
def icon_card(ax, x, y, w=1.1, h=1.1):
    p = FancyBboxPatch((x - w/2, y - h/2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.14",
                       fc="white", ec="#CCCCCC", lw=1.2, zorder=5)
    ax.add_patch(p)


def hydro_icon(ax, x, y, color):
    icon_card(ax, x, y)
    water_top = lighten(C1, 0.50)
    ax.add_patch(Polygon(
        [(x-0.46, y+0.05), (x-0.10, y+0.05),
         (x-0.10, y+0.36), (x-0.46, y+0.36)],
        closed=True, fc=water_top, ec="none", zorder=6))
    for dy in [0.16, 0.27]:
        ax.plot([x-0.42, x-0.34, x-0.26, x-0.18, x-0.12],
                [y+dy, y+dy+0.022, y+dy, y+dy+0.022, y+dy],
                color=color, lw=1.0, zorder=7)
    ax.add_patch(Polygon(
        [(x-0.10, y-0.36), (x+0.02, y-0.36),
         (x-0.04, y+0.38), (x-0.10, y+0.38)],
        closed=True, fc=color, ec=color, zorder=7))
    ax.add_patch(Rectangle((x+0.02, y-0.36), 0.42, 0.20,
                           fc="#888888", ec="#888888", zorder=6))
    ax.plot([x-0.04, x+0.06, x+0.16, x+0.26, x+0.36],
            [y-0.16, y-0.08, y-0.16, y-0.08, y-0.16],
            color=water_top, lw=2.0, zorder=7)


def wind_icon(ax, x, y, color):
    icon_card(ax, x, y)
    ax.add_patch(Polygon(
        [(x-0.05, y-0.36), (x+0.05, y-0.36),
         (x+0.03, y+0.08), (x-0.03, y+0.08)],
        closed=True, fc="#888888", ec="#888888", zorder=6))
    ax.add_patch(plt.Circle((x, y+0.11), 0.055,
                            fc=color, ec=color, zorder=8))
    for theta in [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]:
        dx, dy = 0.32*np.cos(theta), 0.32*np.sin(theta)
        ax.plot([x, x+dx], [y+0.11, y+0.11+dy],
                color=color, lw=2.8, solid_capstyle="round", zorder=7)


def factory_icon(ax, x, y, color):
    icon_card(ax, x, y)
    ax.add_patch(Rectangle((x-0.38, y-0.34), 0.76, 0.46,
                           fc="#888888", ec="#888888", zorder=6))
    for x0 in [-0.36, -0.18, 0.00, 0.18]:
        ax.add_patch(Polygon(
            [(x+x0, y+0.12), (x+x0+0.18, y+0.12),
             (x+x0+0.09, y+0.28)],
            closed=True, fc=color, ec=color, zorder=7))
    for wx in [-0.25, -0.07, 0.11, 0.29]:
        ax.add_patch(Rectangle((x+wx-0.04, y-0.20), 0.08, 0.09,
                               fc=color, ec=color, zorder=7))


def house_icon(ax, x, y, color):
    icon_card(ax, x, y)
    ax.add_patch(Rectangle((x-0.30, y-0.32), 0.60, 0.46,
                           fc="#888888", ec="#888888", zorder=6))
    ax.add_patch(Polygon(
        [(x-0.38, y+0.14), (x+0.38, y+0.14), (x, y+0.38)],
        closed=True, fc=color, ec=color, zorder=7))
    ax.add_patch(Rectangle((x-0.07, y-0.32), 0.14, 0.26,
                           fc=color, ec=color, zorder=7))
    for wx in [-0.18, 0.18]:
        ax.add_patch(Rectangle((x+wx-0.05, y-0.08), 0.10, 0.10,
                               fc=color, ec=color, zorder=7))


# ── Plassering ───────────────────────────────────────────────────────────────
# Produsenter (over stiplet linje — kraftoverskudd)
prod_top = (1.8, 6.0)    # vannkraft øverst
prod_mid = (3.5, 5.5)    # vannkraft midt
prod_bot = (1.8, 4.5)    # vindkraft

# Forbrukere (under stiplet linje — kraftunderskudd)
cons_top = (8.7, 2.4)    # fabrikk
cons_bot = (10.3, 1.0)   # hus

# Flaskehals-knutepunkt
hub = (6.0, 3.4)
hub_w, hub_h = 1.0, 1.0


# ── Kontinuerlig linje fra produsent til forbruker (én farge hele veien) ─────
def power_line(ax, p_from, p_to, color, lw=5.0, zorder=3):
    """
    Cubic Bezier som styres mot flaskehalsen i midten — slik at alle linjer
    samles visuelt der budområdene møtes, men hver beholder sin egen farge
    fra start til slutt.
    """
    x0, y0 = p_from
    x1, y1 = p_to
    hx, hy = hub
    cp1 = ((x0 + hx) / 2, (y0 + hy) / 2)
    cp2 = ((x1 + hx) / 2, (y1 + hy) / 2)
    path = MplPath(
        [(x0, y0), cp1, cp2, (x1, y1)],
        [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4],
    )
    ax.add_patch(PathPatch(path, fc="none", ec=color, lw=lw,
                           capstyle="round", zorder=zorder))


# ── Stiplet skille mellom budområder ─────────────────────────────────────────
# Venstre del: høyt (over fabrikk-nivå), slik at produsenter ligger over linja
# Høyre del: lavt, slik at fabrikk og hus ligger under linja
ax.plot([0, hub[0] - hub_w/2], [3.9, 3.9], "--",
        color="#999999", lw=1.4, alpha=0.8, zorder=1)
ax.plot([hub[0] + hub_w/2, 12], [2.9, 2.9], "--",
        color="#999999", lw=1.4, alpha=0.8, zorder=1)
ax.plot([hub[0] - hub_w/2, hub[0] + hub_w/2], [3.9, 2.9], "--",
        color="#999999", lw=1.4, alpha=0.5, zorder=1)


# ── Kraftlinjer (kontinuerlige, hver med sin egen farge) ─────────────────────
power_line(ax, prod_top, cons_top, color=C2, lw=5.0)   # vannkraft → fabrikk
power_line(ax, prod_mid, cons_top, color=C3, lw=5.0)   # vannkraft → fabrikk
power_line(ax, prod_mid, cons_bot, color=C2, lw=5.0)   # vannkraft → hus
power_line(ax, prod_bot, cons_bot, color=C1, lw=5.0)   # vindkraft → hus


# ── Områdeetiketter ──────────────────────────────────────────────────────────
ax.text(0.4, 6.7, "Budområde", fontsize=18, fontweight="bold", color=C5,
        ha="left", va="center")
ax.text(0.4, 6.30, "Kraftoverskudd", fontsize=15, color="#555555",
        ha="left", va="center")

ax.text(11.6, 0.55, "Budområde", fontsize=18, fontweight="bold", color=C5,
        ha="right", va="center")
ax.text(11.6, 0.15, "Kraftunderskudd", fontsize=15, color="#555555",
        ha="right", va="center")


# ── Ikoner (tegnes etter linjene så de havner oppå endepunktene) ─────────────
hydro_icon(ax, *prod_top, color=C2)
hydro_icon(ax, *prod_mid, color=C3)
wind_icon(ax,  *prod_bot, color=C1)
factory_icon(ax, *cons_top, color=C4)
house_icon(ax,   *cons_bot, color=C5)


# ── Flaskehals-boks (overlay på toppen av linjene) ───────────────────────────
HIGHLIGHT = lighten(C1, 0.55)
ax.add_patch(FancyBboxPatch(
    (hub[0] - hub_w/2, hub[1] - hub_h/2), hub_w, hub_h,
    boxstyle="round,pad=0.02,rounding_size=0.20",
    fc="white", ec=C1, lw=3.5, zorder=8))
ax.add_patch(FancyBboxPatch(
    (hub[0] - hub_w/2 + 0.12, hub[1] - hub_h/2 + 0.12),
    hub_w - 0.24, hub_h - 0.24,
    boxstyle="round,pad=0.0,rounding_size=0.12",
    fc=HIGHLIGHT, ec="none", alpha=0.6, zorder=8))


# ── Flaskehals-etikett med ledelinje (oppe til høyre, klar av fabrikken) ─────
label_x, label_y = hub[0] + 1.8, hub[1] + 2.5
ax.plot([hub[0] + hub_w/2 - 0.1, label_x - 0.6, label_x - 0.1],
        [hub[1] + hub_h/2 - 0.1, label_y, label_y],
        color=C1, lw=2.0, zorder=11)
ax.text(label_x, label_y, "Flaskehals",
        fontsize=22, fontweight="bold", color=C1,
        ha="left", va="center", zorder=11)


plt.tight_layout()

OUT = ROOT.parent / "figures" / "illustrasjoner"
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"flaskehals_enkel.{ext}",
                dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())

print(f"Lagret: {OUT / 'flaskehals_enkel.pdf'}")
print(f"Lagret: {OUT / 'flaskehals_enkel.png'}")
