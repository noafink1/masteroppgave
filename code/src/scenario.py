"""
Scenariobygging: legg til forbrukssjokk for priseffekt-simulering.
"""

import pandas as pd


def build_scenario(X_base, delta_mw):
    """
    Bygg scenariodatasett ved å legge delta_mw MW på rå cons_NO4.

    fill_avvik, prod_wind_NO4 og alle tidsdummyer holdes uendret —
    disse er eksogene og påvirkes ikke av Stargate-sjokket.

    Parametere:
        X_base:   DataFrame med MODEL_FEATURES-kolonner (basescenario)
        delta_mw: MW å legge til forbruket (230 eller 520)

    Returnerer:
        X_scenario: DataFrame med oppdatert cons_NO4
    """
    X_scenario = X_base.copy()
    X_scenario["cons_NO4"] = X_scenario["cons_NO4"] + delta_mw
    return X_scenario


def fill_group(fr, fill_median):
    """Klassifiser magasinfylling som Lavt/Høyt relativt til medianen."""
    return "Lavt magasin" if fr <= fill_median else "Høyt magasin"
