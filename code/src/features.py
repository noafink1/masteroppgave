"""
Feature engineering: kalenderfeatures, magasinavvik, tidsdummyer, normalisering.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import SCALE_COLS, SCALED_NAME_MAP, SEASON_DUMMIES, TRAIN_YEARS


def get_season(month):
    """Mapper måned til sesong."""
    if month in [12, 1, 2]:  return "Winter"
    if month in [3, 4, 5]:   return "Spring"
    if month in [6, 7, 8]:   return "Summer"
    return "Autumn"


def add_calendar_features(df):
    """Legg til kalenderfeatures: hour, month, weekday, season, is_weekend."""
    df["hour"]       = df.index.hour
    df["month"]      = df.index.month
    df["weekday"]    = df.index.dayofweek
    df["season"]     = df["month"].map(get_season)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def compute_fill_deviation(df, train_years=TRAIN_YEARS):
    """
    Beregn ukesbasert magasinavvik: ~m_t = m_t - median(m_w(t)).

    Medianen beregnes per ISO-uke utelukkende fra treningsårene for å unngå
    datalekkasje. Returnerer oppdatert df og weekly_median-serien.
    """
    week = df.index.isocalendar().week.astype(int)
    train_mask = df.index.year.isin(train_years)
    weekly_median = (
        df.loc[train_mask, "fill_rate"]
        .groupby(week[train_mask])
        .median()
    )
    df["fill_avvik"] = df["fill_rate"].values - week.map(weekly_median).values
    return df, weekly_median


def add_time_dummies(df):
    """
    Legg til måneds- og timedummyer.
    Referanse: januar (D_m1 utelatt) og time 0 (D_h0 utelatt).
    """
    for m in range(2, 13):
        df[f"D_m{m}"] = (df.index.month == m).astype(int)
    for h in range(1, 24):
        df[f"D_h{h}"] = (df.index.hour == h).astype(int)
    return df


def fit_scaler(df, train_years=TRAIN_YEARS, scale_cols=SCALE_COLS):
    """Fit StandardScaler på treningsår. Brukes til ISO-logit i NB05."""
    train_mask = df.index.year.isin(train_years)
    scaler = StandardScaler()
    scaler.fit(df.loc[train_mask, scale_cols])
    scaler_info = {
        "mean": dict(zip(scale_cols, scaler.mean_)),
        "std":  dict(zip(scale_cols, scaler.scale_)),
        "cols": scale_cols,
    }
    return scaler, scaler_info


def apply_scaling(df, scaler, scale_cols=SCALE_COLS):
    """Appliser scaler og lag interaksjonsledd (brukes i NB01 for logit)."""
    scaled = scaler.transform(df[scale_cols])
    scaled_names = [SCALED_NAME_MAP[c] for c in scale_cols]
    for i, name in enumerate(scaled_names):
        df[name] = scaled[:, i]
    if "cons_c" in df.columns and "fill_c" in df.columns:
        df["cons_x_fill_c"] = df["cons_c"] * df["fill_c"]
    return df


def add_season_dummies(df, season_dummies=SEASON_DUMMIES):
    """Sesongdummyer brukt i ISO-logit (NB05)."""
    for s in season_dummies:
        df[s] = (df["season"] == s).astype(int)
    return df


def flag_negative_prices(df, price_col="price_NO4"):
    """Flagg negative priser med binær kolonne."""
    df["neg_price"] = (df[price_col] < 0).astype(int)
    return df
