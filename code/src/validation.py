"""
Felles hjelpefunksjoner for modellvalidering i notebook 04.
"""

import pandas as pd

from .config import INTERMEDIATE_DIR, MODEL_COLS, MODEL_LABELS, TARGET, TEST_YEARS
from .evaluation import eval_metrics


def load_prediction_tables(intermediate_dir=INTERMEDIATE_DIR):
    """Last standard prediksjonstabeller fra notebook 03a-03e.

    Returnerer dict med ett oppslag per modellkolonne i predikasjonene:
    TSLS, OLS, XGB, MSDR, Q50, Q90.
    """
    tables = {}
    file_map = {
        "TSLS": "preds_tsls.parquet",
        "OLS":  "preds_ols.parquet",
        "XGB":  "preds_xgb.parquet",
        "MSDR": "preds_msdr.parquet",
    }
    for model, fname in file_map.items():
        tables[model] = pd.read_parquet(f"{intermediate_dir}{fname}")

    qreg = pd.read_parquet(f"{intermediate_dir}preds_quantreg.parquet")
    tables["Q50"] = qreg[["actual", "Q50", "season", "year", "month", "hour"]]
    tables["Q90"] = qreg[["actual", "Q90", "season", "year", "month", "hour"]]
    return tables


def build_validation_table(df_iso, pred_tables, target=TARGET, test_years=TEST_YEARS):
    """
    Bygg felles valideringstabell med faktiske verdier, modellprediksjoner og baselines.
    """
    test_mask = df_iso.index.year.isin(test_years)
    base = pd.DataFrame(index=df_iso.loc[test_mask].index)
    base["actual"] = df_iso.loc[test_mask, target]
    base["season"] = df_iso.loc[test_mask, "season"]
    base["year"] = df_iso.loc[test_mask].index.year
    base["month"] = df_iso.loc[test_mask].index.month
    base["hour"] = df_iso.loc[test_mask].index.hour

    for model, table in pred_tables.items():
        base[model] = table[model].reindex(base.index)

    base["Lag1"] = df_iso[target].shift(1).reindex(base.index)
    base["Lag24"] = df_iso[target].shift(24).reindex(base.index)
    return base


def metrics_table(df_preds, model_cols=MODEL_COLS, model_labels=MODEL_LABELS):
    """Beregn total evalueringsoppsummering for alle modeller."""
    rows = []
    for col in model_cols:
        metrics = eval_metrics(df_preds["actual"], df_preds[col])
        metrics["Modell"] = model_labels.get(col, col)
        rows.append(metrics)
    out = pd.DataFrame(rows)[["Modell", "MAE", "RMSE", "R²", "N"]]
    return out.sort_values(["MAE", "RMSE"]).reset_index(drop=True)


def segmented_metrics(df_preds, segment_col, model_cols=MODEL_COLS,
                      model_labels=MODEL_LABELS):
    """Beregn metrikker per segment, for eksempel år eller sesong."""
    rows = []
    for segment, df_seg in df_preds.groupby(segment_col):
        for col in model_cols:
            metrics = eval_metrics(df_seg["actual"], df_seg[col])
            metrics["Segment"] = segment
            metrics["Modell"] = model_labels.get(col, col)
            rows.append(metrics)
    return pd.DataFrame(rows)[["Segment", "Modell", "MAE", "RMSE", "R²", "N"]]


def add_residual_columns(df_preds, model_cols=MODEL_COLS):
    """Legg til residualkolonner for hver modell."""
    df = df_preds.copy()
    for col in model_cols:
        df[f"resid_{col}"] = df["actual"] - df[col]
    return df


def rolling_mae(actual, pred, window=24 * 7):
    """Rullerende MAE brukt til stabilitetssjekk over tid."""
    return (actual - pred).abs().rolling(window=window, min_periods=window // 2).mean()


def bias_table(df_preds, model_cols=MODEL_COLS, model_labels=MODEL_LABELS):
    """Beregn gjennomsnittlig og median prediksjonsfeil per modell."""
    rows = []
    for col in model_cols:
        err = df_preds[col] - df_preds["actual"]
        rows.append({
            "Modell": model_labels.get(col, col),
            "ME": round(err.mean(), 2),
            "Medianfeil": round(err.median(), 2),
            "MAPE": round((err.abs() / df_preds["actual"].abs().clip(lower=1)).mean() * 100, 2),
        })
    return pd.DataFrame(rows)


def tail_metrics_table(df_preds, quantile=0.90, model_cols=MODEL_COLS,
                       model_labels=MODEL_LABELS):
    """Evaluer modeller kun på høye pristimer over valgt kvantil."""
    cutoff = df_preds["actual"].quantile(quantile)
    tail = df_preds[df_preds["actual"] >= cutoff].copy()
    rows = []
    for col in model_cols:
        metrics = eval_metrics(tail["actual"], tail[col])
        metrics["Modell"] = model_labels.get(col, col)
        rows.append(metrics)
    out = pd.DataFrame(rows)[["Modell", "MAE", "RMSE", "R²", "N"]]
    return out.sort_values(["MAE", "RMSE"]).reset_index(drop=True), cutoff


def price_level_table(df_preds, model_cols=MODEL_COLS, model_labels=MODEL_LABELS):
    """Evaluer modeller etter prisnivå: lav, normal og høy."""
    q1 = df_preds["actual"].quantile(1 / 3)
    q2 = df_preds["actual"].quantile(2 / 3)

    def label(v):
        if v <= q1:
            return "Lav pris"
        if v <= q2:
            return "Normal pris"
        return "Høy pris"

    df = df_preds.copy()
    df["price_level"] = df["actual"].map(label)
    rows = []
    for level, df_level in df.groupby("price_level"):
        for col in model_cols:
            metrics = eval_metrics(df_level["actual"], df_level[col])
            metrics["Prisnivå"] = level
            metrics["Modell"] = model_labels.get(col, col)
            rows.append(metrics)
    out = pd.DataFrame(rows)[["Prisnivå", "Modell", "MAE", "RMSE", "R²", "N"]]
    return out, q1, q2
