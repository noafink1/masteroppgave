"""
Regimeklassifisering for NO4 basert på priser og flytverdier.
Kopiert fra modell_andreas.ipynb og gjort til importerbar modul.
"""

from .config import PRICE_TOL, FLOW_TOL

# ── Regimegrupper ─────────────────────────────────────────────────────────────
ISO_REGIMES         = {"ISO_EXP", "ISO_LOC"}
CONG_REGIMES        = {"CONG_FI", "CONG_NO3", "CONG_SE1", "CONG_SE2", "CONG_MULTI"}
PRICE_TAKER_REGIMES = {"FI", "NO3", "SE1", "SE2"}


def near_boundary_sys(p, sys_price, price_tol, rel_tol):
    """Hjelpefunksjon for systemgrense-sjekk utenfor klassen."""
    diff = abs(p - sys_price)
    return price_tol * 0.5 < diff < price_tol * 1.5


def classify_regime(row, price_tol=PRICE_TOL, flow_tol=FLOW_TOL, rel_tol=0.05):
    """
    Klassifiserer prisregime for NO4 basert på priser og flytverdier.

    Regimer:
        SYS        - NO4 integrert med systemprisen
        FI         - NO4 pristaker for Finland (importerer fra FI)
        NO3        - NO4 pristaker for NO3
        SE1        - NO4 pristaker for SE1
        SE2        - NO4 pristaker for SE2
        CONG_FI    - Bindende snitt mot Finland
        CONG_NO3   - Bindende snitt mot NO3
        CONG_SE1   - Bindende snitt mot SE1
        CONG_SE2   - Bindende snitt mot SE2
        CONG_MULTI - Bindende snitt mot flere soner samtidig
        ISO_EXP    - NO4 isolert og eksporterer (lokal pris høyere, setter pris selv)
        ISO_LOC    - NO4 isolert uten klar kobling (uklart regime)
        UNCERTAIN  - Timer nær toleransegrenser (ustabil klassifisering)
    """
    net = {
        "FI":  row.flow_FI_NO4  - row.flow_NO4_FI,
        "NO3": row.flow_NO3_NO4 - row.flow_NO4_NO3,
        "SE1": row.flow_SE1_NO4 - row.flow_NO4_SE1,
        "SE2": row.flow_SE2_NO4 - row.flow_NO4_SE2,
    }
    prices = {
        "FI":  row.price_FI,
        "NO3": row.price_NO3,
        "SE1": row.price_SE1,
        "SE2": row.price_SE2,
    }
    p = row.price_NO4

    def prices_equal(p1, p2):
        return abs(p1 - p2) < price_tol or abs(p1 - p2) / (max(abs(p1), abs(p2)) + 1e-6) < rel_tol

    def is_congested(zone):
        return (p > prices[zone] + price_tol) and (net[zone] > flow_tol)

    def is_coupled(zone):
        return prices_equal(p, prices[zone])

    def near_boundary(zone):
        diff = abs(p - prices[zone])
        return price_tol * 0.5 < diff < price_tol * 1.5

    # 1) Systemintegrert
    if prices_equal(p, row.system_price):
        return "SYS"

    # 2) Bindende snitt
    congested_zones = [z for z in prices if is_congested(z)]
    if len(congested_zones) > 1:
        return "CONG_MULTI"
    if len(congested_zones) == 1:
        return f"CONG_{congested_zones[0]}"

    # 3) Koblede soner (lik pris)
    coupled_zones = [z for z in prices if is_coupled(z)]
    if coupled_zones:
        dominant = max(coupled_zones, key=lambda z: abs(net[z]))
        if net[dominant] > flow_tol:
            return dominant
        elif net[dominant] < -flow_tol:
            return "ISO_EXP"
        else:
            return "SYS"

    # 4) Nær toleransegrense
    if any(near_boundary(z) for z in prices) or near_boundary_sys(p, row.system_price, price_tol, rel_tol):
        return "UNCERTAIN"

    # 5) Lokalt isolert
    return "ISO_LOC"


def regime_group(r):
    """Forenklet gruppering: ISO / CONG / PRICE_TAKER / SYS / UNCERTAIN."""
    if r in ISO_REGIMES:         return "ISO"
    if r in CONG_REGIMES:        return "CONG"
    if r in PRICE_TAKER_REGIMES: return "PRICE_TAKER"
    return r  # SYS, UNCERTAIN
