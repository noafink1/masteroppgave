"""
Stilisert "Merit Order-effekt"-figur (klassisk lærebokversjon).

Gjenskaper figuren fra German Renewable Energies Agency (2011), men i
thesis-stil (Georgia, 5-fargers palett #038D7F → #00271E).

Viser hvordan vind- og solkraft med marginalkostnad ≈ 0 skyver
tilbudskurven mot høyre, slik at de dyreste fossile teknologiene
ikke lenger settes som marginal pris ved gjennomsnittlig etterspørsel.
"""

from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.config import apply_style, PALETTE, C1, C2, C3, C4, C5, lighten  # noqa: E402

apply_style()

# ── Søylene (energikilder, marginalkostnad) ──────────────────────────────────
# (label, x_start, bredde, marginalkostnad)
bars = [
    ("Sol",         0.0,  1.2, 0.0),
    ("Vind",        1.2,  1.2, 0.0),
    ("Kjernekraft", 2.4,  1.2, 0.15),
    ("Vannkraft",   3.6,  1.2, 0.30),
    ("Kull",        4.8,  1.2, 0.55),
    ("Gass",        6.0,  1.2, 0.75),
    ("Bioenergi",   7.2,  1.2, 1.00),
]

# ── Tilbudskurve (glatt, eksponentielt stigende) ─────────────────────────────
x = np.linspace(0, 8.6, 400)
# Konveks kurve som ligner originalens grønne linje
y_curve = 0.02 + 0.012 * np.exp(0.55 * x)

# ── Etterspørsel og priser ───────────────────────────────────────────────────
q_demand   = 7.2          # gjennomsnittlig etterspørsel
p_avg      = 0.02 + 0.012 * np.exp(0.55 * q_demand)   # opprinnelig pris
p_new      = 0.02 + 0.012 * np.exp(0.55 * 5.4)        # ny pris etter merit order-effekt

# ── Figur ────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 7))

# Gradient fra lys teal (fornybar, marginalkostnad ≈ 0) til mørk grønn
# (dyre fossile teknologier). Unngår å bruke C1 i søylene så
# tilbudskurven (også C1) ikke blander seg inn.
bar_colors = [
    lighten(C2, 0.35),
    lighten(C2, 0.55),
    lighten(C3, 0.55),
    lighten(C3, 0.80),
    C3,
    C4,
    C5,
]

# Søylehøyder ≈ marginalkostnad (samsvarer med kurven)
for (label, x0, w, _), col in zip(bars, bar_colors):
    h = 0.02 + 0.012 * np.exp(0.55 * (x0 + w/2))
    ax.bar(x0, h, width=w, align="edge", color=col,
           edgecolor="white", linewidth=1.0, zorder=1)
    ax.text(x0 + w/2, -0.05, label, ha="center", va="top",
            fontsize=18, color=C5)

# Tilbudskurve
ax.plot(x, y_curve, color=C1, lw=3.2, zorder=4)

# Vertikal etterspørselslinje (legges over søylen som halvtransparent)
ax.vlines(q_demand, 0, 1.55, color=C5, lw=1.4, ls=":", zorder=5, alpha=0.7)
# Etikett sentrert over linjen
ax.text(q_demand, 1.58, "Gjennomsnittlig etterspørsel",
        ha="center", va="bottom", fontsize=17, color=C4, style="italic")

# Horisontale prislinjer
ax.hlines(p_avg, 0, q_demand, color=C4, lw=1.4, ls=":", zorder=3)
ax.hlines(p_new, 0, 5.4,      color=C1, lw=1.4, ls=":", zorder=3)

# Vertikal linje for skjæringspunkt ny pris
ax.vlines(5.4, 0, p_new, color=C1, lw=1.4, ls=":", zorder=3)

# Skjæringspunkter (prikker)
ax.scatter([q_demand], [p_avg], s=110, color=C4, zorder=6,
           edgecolor="white", linewidths=1.5)
ax.scatter([5.4], [p_new], s=110, color=C1, zorder=6,
           edgecolor="white", linewidths=1.5)

# Pristekster (plasseres til høyre for de stiplede linjene, ikke overlapping)
ax.text(0.15, p_avg + 0.04, "Opprinnelig strømpris",
        ha="left", va="bottom", fontsize=17, color=C4)
ax.text(0.15, p_new + 0.04, "Ny strømpris",
        ha="left", va="bottom", fontsize=17, color=C1, weight="bold")

# Merit order-effekt-pil (nedover, mellom de to prisene)
ax.annotate("", xy=(2.4, p_new + 0.05), xytext=(2.4, p_avg - 0.05),
            arrowprops=dict(arrowstyle="-|>", color=C1, lw=2.6,
                            mutation_scale=22))
ax.text(2.65, (p_avg + p_new) / 2, "Merit-order-\neffekten",
        ha="left", va="center", fontsize=19, color=C1, weight="bold")

# Etikett over vind- og solsøylene (plasseres mellom søylene og Ny strømpris-linjen)
ax.text(0.55, 0.07, "Strøm fra vind og sol\n(marginalkostnad ≈ 0)",
        ha="left", va="bottom", fontsize=16, color=C3, style="italic")


# Akser og ramme
ax.set_xlim(-0.05, 8.8)
ax.set_ylim(0, 1.75)
ax.set_xlabel("Tilbud (etter energikilde, økende marginalkostnad)", labelpad=35)
ax.set_ylabel("Pris (NOK/MWh)")
ax.set_title("Merit-order-effekten", pad=18, fontsize=26)

# Skjul tickverdier (kvalitativ figur)
ax.set_xticks([])
ax.set_yticks([])
ax.grid(False)
for spine in ("top", "right"):
    ax.spines[spine].set_visible(False)

plt.tight_layout()

# ── Lagre ────────────────────────────────────────────────────────────────────
OUT = ROOT.parent / "figures" / "illustrasjoner"
OUT.mkdir(parents=True, exist_ok=True)
for ext in ("pdf", "png"):
    fig.savefig(OUT / f"merit_order_effekten.{ext}", dpi=150, bbox_inches="tight")

print(f"Lagret: {OUT / 'merit_order_effekten.pdf'}")
print(f"Lagret: {OUT / 'merit_order_effekten.png'}")
