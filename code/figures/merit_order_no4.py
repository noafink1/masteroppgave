"""
Stilisert merit order-figur for NO4.

Viser den konvekse prisresponsen i et hydrokraftdominert system:
- Flatt nedre segment (regulerbar vannkraft med lave vannverdier)
- Bratt øvre segment (magasinrestriksjoner + bindende overføringskapasitet)
- Samme 230 MW etterspørselssjokk gir liten ΔP ved lav etterspørsel,
  stor ΔP ved høy etterspørsel.

Brukes som illustrasjon i diskusjonskapittelet (sec:ekstrempriser).
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import apply_style, C1, C3, C5  # noqa: E402

apply_style()

# ── Tilbudskurve (stilisert) ─────────────────────────────────────────────────
def supply_curve(q):
    """Stilisert merit order for NO4: flatt nedre, bratt øvre segment."""
    base = 180 + 0.02 * q
    knekk = 3500.0
    bratt = np.where(q > knekk,
                     0.00025 * (q - knekk) ** 2 + 0.1 * (q - knekk),
                     0.0)
    return base + bratt


q_grid = np.linspace(0, 5200, 600)
p_grid = supply_curve(q_grid)

# ── Etterspørselsnivåer + Stargate-sjokk ─────────────────────────────────────
stargate = 230.0
q_lav, q_hoy = 2500.0, 4400.0
q_lav_s, q_hoy_s = q_lav + stargate, q_hoy + stargate

p_lav   = float(supply_curve(np.array([q_lav])))
p_lav_s = float(supply_curve(np.array([q_lav_s])))
p_hoy   = float(supply_curve(np.array([q_hoy])))
p_hoy_s = float(supply_curve(np.array([q_hoy_s])))

dp_lav = p_lav_s - p_lav
dp_hoy = p_hoy_s - p_hoy

# ── Figur ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 7.2))

# Hovedkurve
ax.plot(q_grid, p_grid, color=C5, lw=2.8, zorder=3)

# Punkter på kurven (før/etter sjokk)
ax.scatter([q_lav, q_lav_s], [p_lav, p_lav_s],
           color=C1, s=85, zorder=5, edgecolor="white", linewidths=1.5)
ax.scatter([q_hoy, q_hoy_s], [p_hoy, p_hoy_s],
           color=C3, s=85, zorder=5, edgecolor="white", linewidths=1.5)

# Subtile ledelinjer ned til x-aksen (for å vise etterspørselsnivå)
for q, p, c in [(q_lav, p_lav, C1), (q_hoy, p_hoy, C3)]:
    ax.vlines(q, 0, p, color=c, lw=1.0, ls=":", alpha=0.55, zorder=1)

# ── Annotasjoner: ΔP for lavpris ─────────────────────────────────────────────
# Boks oppe til venstre for lavpris-punktet, med tynn rett pil
ax.annotate(
    f"Lavpris-time\nΔP ≈ {dp_lav:.0f} NOK/MWh",
    xy=((q_lav + q_lav_s) / 2, (p_lav + p_lav_s) / 2),
    xytext=(2150, 430),
    fontsize=17, color=C1, ha="left", va="center",
    arrowprops=dict(arrowstyle="-|>", color=C1, lw=1.6,
                    mutation_scale=18,
                    connectionstyle="arc3,rad=-0.2",
                    shrinkA=4, shrinkB=6),
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C1, lw=1.2, alpha=0.95),
)

# ── Annotasjoner: ΔP for høypris ─────────────────────────────────────────────
# Boks oppe til venstre for høypris-punktet, nær kurven men ikke oppå
ax.annotate(
    f"Høypris-time\nΔP ≈ {dp_hoy:.0f} NOK/MWh",
    xy=((q_hoy + q_hoy_s) / 2, (p_hoy + p_hoy_s) / 2),
    xytext=(3100, 820),
    fontsize=17, color=C3, ha="left", va="center",
    arrowprops=dict(arrowstyle="-|>", color=C3, lw=1.6,
                    mutation_scale=18,
                    connectionstyle="arc3,rad=-0.2",
                    shrinkA=4, shrinkB=6),
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=C3, lw=1.2, alpha=0.95),
)

# ── Segment-etiketter (over kurven, i tomme områder) ─────────────────────────
ax.text(1100, 480, "Flatt nedre segment\n(regulerbar vannkraft)",
        ha="center", va="center", fontsize=16, color=C5, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

ax.text(4200, 1200, "Bratt øvre segment\n(magasin- og\noverføringsrestriksjoner)",
        ha="center", va="center", fontsize=16, color=C5, style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85))

# ── +230 MW skift-indikatorer på x-aksen ─────────────────────────────────────
# Tynne piler rett over x-aksen for å vise forskyvningen
for q, q_s, c in [(q_lav, q_lav_s, C1), (q_hoy, q_hoy_s, C3)]:
    arrow = FancyArrowPatch((q, 60), (q_s, 60),
                            arrowstyle="->", color=c, lw=1.8,
                            mutation_scale=14, zorder=2)
    ax.add_patch(arrow)
    ax.text((q + q_s) / 2, 105, "+230 MW",
            ha="center", va="bottom", fontsize=15, color=c)

# ── Akser og ramme ───────────────────────────────────────────────────────────
ax.set_xlim(0, 5200)
ax.set_ylim(0, 1500)
ax.set_xlabel("Kumulativ produksjon (MW)")
ax.set_ylabel("Marginalkostnad (NOK/MWh)")
ax.set_title("Merit-order-kurven i NO4", pad=18, fontsize=26)

# Forenklet legend
legend_elements = [
    Line2D([0], [0], color=C5, lw=2.8, label="Tilbudskurve"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C1,
           markersize=10, markeredgecolor="white", label="Lavpris-time (før/etter sjokk)"),
    Line2D([0], [0], marker="o", color="w", markerfacecolor=C3,
           markersize=10, markeredgecolor="white", label="Høypris-time (før/etter sjokk)"),
]
ax.legend(handles=legend_elements, loc="upper left", frameon=True, framealpha=0.95)

plt.tight_layout()

# ── Lagre ────────────────────────────────────────────────────────────────────
OUT = ROOT.parent / "figures" / "illustrasjoner"
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"merit_order_no4.{ext}", dpi=150, bbox_inches="tight")

print(f"Lagret: {OUT / 'merit_order_no4.pdf'}")
print(f"Lagret: {OUT / 'merit_order_no4.png'}")
print(f"ΔP lav  = {dp_lav:.1f} NOK/MWh")
print(f"ΔP høy  = {dp_hoy:.1f} NOK/MWh")
print(f"Forhold = {dp_hoy / dp_lav:.1f}x")
