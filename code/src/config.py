"""
Sentral konfigurasjon for hele analysepipelinen.
Endre parametere her — de propagerer automatisk til alle notebooks.
"""

import os
import matplotlib.pyplot as plt

# ── Regimeklassifisering ──────────────────────────────────────────────────────
PRICE_TOL = 10   # NOK/MWh — absolutt pristoleranse for regimeklassifisering
FLOW_TOL  = 10   # MW      — flyttoleranse for regimeklassifisering

# ── Stargate-scenarier ────────────────────────────────────────────────────────
STARGATE_MW       = 230   # fase 1
STARGATE_TOTAL_MW = 520   # fase 1 + fase 2

# ── Tidsperioder ─────────────────────────────────────────────────────────────
ALL_YEARS   = list(range(2020, 2027))
TRAIN_YEARS = [2020, 2021, 2022, 2023]
TEST_YEARS  = [2024, 2025, 2026]

# ── Filstier (relative til code/) ────────────────────────────────────────────
DATA_DIR         = "../data/modell-data/"
FIG_DIR          = "../figures/notebook/"
INTERMEDIATE_DIR = "intermediate/"

# ── Produksjonsdata — kolonnemapping ────────────────────────────────────────
PROD_COL_MAP = {
    "NO4 Fossil Gas (MW)":                      "prod_fossil_NO4",
    "NO4 Hydro Run Of River And Poundage (MW)":  "prod_hydro_ror_NO4",
    "NO4 Hydro Water Reservoir (MW)":            "prod_hydro_res_NO4",
    "NO4 Other Renewable (MW)":                  "prod_other_renew_NO4",
    "NO4 Wind Onshore (MW)":                     "prod_wind_NO4",
    "NO4 Total (MW)":                            "prod_total_NO4",
}

# ── Modellfeatures: rå nivåer + magasinavvik + tidsdummyer ──────────────────
BASE_FEATURES  = ["cons_NO4", "fill_avvik", "prod_wind_NO4"]
MONTH_DUMMIES  = [f"D_m{m}" for m in range(2, 13)]   # jan = referanse
HOUR_DUMMIES   = [f"D_h{h}" for h in range(1, 24)]   # time 0 = referanse
MODEL_FEATURES = BASE_FEATURES + MONTH_DUMMIES + HOUR_DUMMIES
OLS_FEATURES   = MODEL_FEATURES
XGB_FEATURES   = MODEL_FEATURES
MSDR_FEATURES  = BASE_FEATURES

# ── 2SLS: instrumenter og endogen variabel ──────────────────────────────────
IV_ENDOG       = "cons_NO4"
IV_INSTRUMENTS = ["temp_NO4", "precip_NO4"]
IV_EXOG        = [f for f in MODEL_FEATURES if f != IV_ENDOG]
TSLS_FEATURES  = MODEL_FEATURES

# ── Kvantilregresjon ──────────────────────────────────────────────────────────
QUANTILES         = [0.5, 0.9]
QUANTREG_FEATURES = MODEL_FEATURES

# ── StandardScaler for ISO-andel logit (NB05) ──────────────────────────────
SCALE_COLS = ["cons_NO4", "fill_rate", "prod_wind_NO4", "net_position"]
SCALED_NAME_MAP = {
    "cons_NO4":      "cons_c",
    "fill_rate":     "fill_c",
    "prod_wind_NO4": "wind_c",
    "net_position":  "net_pos_c",
}

# Sesongdummyer (brukes i deskriptiv analyse, ikke i regresjon)
SEASON_DUMMIES = ["Spring", "Summer", "Winter"]

# Målvariabel (rå)
TARGET = "price_NO4"

# ── Modelletiketter ──────────────────────────────────────────────────────────
# Rekkefølge: hovedmodell først, så naiv baseline, så komplementmodeller
MODEL_COLS = ["TSLS", "OLS", "XGB", "MSDR", "Q50", "Q90", "Lag1", "Lag24"]
MODEL_LABELS = {
    "TSLS":  "2SLS (HAC)",
    "OLS":   "OLS (HAC)",
    "XGB":   "XGBoost",
    "MSDR":  "MS-DR (2 reg.)",
    "Q50":   "QuantReg Q50",
    "Q90":   "QuantReg Q90",
    "Lag1":  "Baseline lag-1",
    "Lag24": "Baseline lag-24",
}

# Modellsett brukt i notebook 04-06
ALL_MODEL_COLS = MODEL_COLS
ALL_MODEL_LABELS = MODEL_LABELS

# Norske sesongnavn
SEASON_ORDER = {"Winter": "Vinter", "Spring": "Vår", "Summer": "Sommer", "Autumn": "Høst"}

# ── Fargepalett (thesis) ──────────────────────────────────────────────────────
PALETTE = ["#038D7F", "#007265", "#00574C", "#003E34", "#00271E"]
C1, C2, C3, C4, C5 = PALETTE   # lys → mørk

# ── Regime-etiketter ─────────────────────────────────────────────────────────
REGIME_LABELS_NO = {
    "ISO":         "ISO",
    "SYS":         "SYS",
    "CONG":        "CONG",
    "PRICE_TAKER": "Pristaker",
    "UNCERTAIN":   "Usikker",
}
STACK_ORDER = ["ISO", "SYS", "CONG", "PRICE_TAKER", "UNCERTAIN"]


# ── Figurstil ─────────────────────────────────────────────────────────────────
def apply_style():
    """Sett matplotlib rcParams for thesis-figurer."""
    plt.rcParams.update({
        "figure.dpi":       150,
        "font.size":        18,
        "font.family":      "georgia",
        "axes.titlesize":   22,
        "axes.labelsize":   20,
        "xtick.labelsize":  18,
        "ytick.labelsize":  18,
        "legend.fontsize":  16,
        "axes.grid":        True,
        "grid.alpha":       0.3,
    })


def lighten(hex_col, factor):
    """Bland hex-farge med hvit. factor=1 → original, factor=0 → hvit."""
    r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
    return "#{:02X}{:02X}{:02X}".format(
        int(r * factor + 255 * (1 - factor)),
        int(g * factor + 255 * (1 - factor)),
        int(b * factor + 255 * (1 - factor)),
    )


def hex_to_rgb(hex_str):
    """Konverter hex-farge til (r, g, b) tuple (0–255)."""
    hex_str = hex_str.lstrip("#")
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))


def interp_color(val, min_val, max_val, cmap=None):
    """Interpoler farge mellom C1 og C5 basert på normalisert verdi."""
    if cmap is None:
        cmap = [C1, C5]
    t = (val - min_val) / (max_val - min_val + 1e-9)
    t = max(0, min(1, t))
    r1, g1, b1 = hex_to_rgb(cmap[0])
    r2, g2, b2 = hex_to_rgb(cmap[-1])
    return "#{:02X}{:02X}{:02X}".format(
        int(r1 + t * (r2 - r1)),
        int(g1 + t * (g2 - g1)),
        int(b1 + t * (b2 - b1)),
    )


# Regime-farger for stacked bar charts
STACK_COLORS = {
    "ISO":         lighten(C1, 0.40),
    "SYS":         C1,
    "CONG":        C2,
    "PRICE_TAKER": C4,
    "UNCERTAIN":   C5,
}
