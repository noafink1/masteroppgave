"""
Evalueringsverktøy: metrikker og etterspørselsnivå-klassifisering.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MODEL_COLS, MODEL_LABELS


def eval_metrics(y_true, y_pred):
    """Beregn MAE, RMSE, R² for et sett med prediksjoner."""
    mask = y_true.notna() & y_pred.notna()
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return {"MAE": np.nan, "RMSE": np.nan, "R²": np.nan, "N": 0}
    return {
        "MAE":  round(mean_absolute_error(yt, yp), 1),
        "RMSE": round(np.sqrt(mean_squared_error(yt, yp)), 1),
        "R²":   round(r2_score(yt, yp), 4),
        "N":    len(yt),
    }


def eval_segment(df_preds, segment_name="Total", model_cols=MODEL_COLS,
                 model_labels=MODEL_LABELS):
    """Evaluer alle modeller for et gitt segment."""
    rows = []
    for col in model_cols:
        if col not in df_preds.columns:
            continue
        m = eval_metrics(df_preds["actual"], df_preds[col])
        m["Modell"] = model_labels.get(col, col)
        m["Segment"] = segment_name
        rows.append(m)
    return rows


def demand_level(c, t1, t2):
    """Klassifiser forbruk i tersiler: Lav / Normal / Høy."""
    if c <= t1: return "Lav"
    if c <= t2: return "Normal"
    return "Høy"
