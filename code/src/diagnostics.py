"""
Diagnostiske tester for stasjonaritet, OLS-antagelser og endogenitet.
"""

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    het_breuschpagan,
    het_white,
    linear_reset,
)
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, adfuller, kpss


def _clean_series(series):
    """Returner numerisk serie uten manglende verdier."""
    s = pd.Series(series).astype(float)
    return s.replace([np.inf, -np.inf], np.nan).dropna()


def stationarity_test_table(series_map, regression="c", nlags="auto"):
    """
    Kjør ADF- og KPSS-tester for flere serier.

    regression:
        "c"  = konstant
        "ct" = konstant + trend
    """
    rows = []
    for name, series in series_map.items():
        s = _clean_series(series)
        if len(s) < 20:
            rows.append({
                "Serie": name,
                "N": len(s),
                "ADF-stat": np.nan,
                "ADF p-verdi": np.nan,
                "KPSS-stat": np.nan,
                "KPSS p-verdi": np.nan,
                "Vurdering": "For få observasjoner",
            })
            continue

        adf_stat, adf_pval, *_ = adfuller(s, regression=regression, autolag="AIC")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                kpss_stat, kpss_pval, *_ = kpss(s, regression=regression, nlags=nlags)
        except ValueError:
            kpss_stat, kpss_pval = np.nan, np.nan

        if adf_pval < 0.05 and (pd.isna(kpss_pval) or kpss_pval >= 0.05):
            verdict = "Taler for stasjonær"
        elif adf_pval >= 0.05 and (pd.isna(kpss_pval) or kpss_pval < 0.05):
            verdict = "Taler for ikke-stasjonær"
        elif adf_pval < 0.05 and kpss_pval < 0.05:
            verdict = "Blandet evidens"
        else:
            verdict = "Usikker"

        rows.append({
            "Serie": name,
            "N": len(s),
            "ADF-stat": adf_stat,
            "ADF p-verdi": adf_pval,
            "KPSS-stat": kpss_stat,
            "KPSS p-verdi": kpss_pval,
            "Vurdering": verdict,
        })

    return pd.DataFrame(rows)


def build_design_matrix(df, feature_cols, add_constant=True):
    """Bygg ren designmatrise uten manglende verdier i gitte features."""
    X = df.loc[:, feature_cols].astype(float).replace([np.inf, -np.inf], np.nan)
    if add_constant:
        X = sm.add_constant(X, has_constant="add")
    return X


def fit_ols(df, y_col, feature_cols, cov_type="HAC", maxlags=24):
    """Kjør standard OLS/HAC-estimering på et ferdig datasett."""
    cols = [y_col] + list(feature_cols)
    work = df.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = build_design_matrix(work, feature_cols, add_constant=True)
    y = work[y_col].astype(float)
    model = sm.OLS(y, X).fit(cov_type=cov_type, cov_kwds={"maxlags": maxlags})
    return model, work, X, y


def residual_diagnostics(model, X, bg_lags=24, acf_lags=24):
    """Samle standard residualdiagnostikk for en estimert OLS-modell."""
    resid = pd.Series(model.resid).dropna()
    bp_lm, bp_pval, bp_f, bp_f_pval = het_breuschpagan(resid, X)
    white_lm, white_pval, white_f, white_f_pval = het_white(resid, X)
    bg_lm, bg_pval, bg_f, bg_f_pval = acorr_breusch_godfrey(model, nlags=bg_lags)
    dw_stat = durbin_watson(resid)
    reset = linear_reset(model, power=2, use_f=True)
    acf_vals = acf(resid, nlags=acf_lags, fft=True)

    return pd.DataFrame([
        {"Test": "Durbin-Watson", "Statistikk": dw_stat, "p-verdi": np.nan},
        {"Test": f"Breusch-Godfrey ({bg_lags} lag)", "Statistikk": bg_lm, "p-verdi": bg_pval},
        {"Test": "Breusch-Pagan", "Statistikk": bp_lm, "p-verdi": bp_pval},
        {"Test": "White", "Statistikk": white_lm, "p-verdi": white_pval},
        {"Test": "Ramsey RESET", "Statistikk": float(reset.fvalue), "p-verdi": float(reset.pvalue)},
        {"Test": "ACF lag 1", "Statistikk": float(acf_vals[1]) if len(acf_vals) > 1 else np.nan, "p-verdi": np.nan},
        {"Test": "ACF lag 24", "Statistikk": float(acf_vals[24]) if len(acf_vals) > 24 else np.nan, "p-verdi": np.nan},
    ])


def vif_table(df, feature_cols):
    """Beregn VIF for forklaringsvariabler uten konstantledd."""
    X = df.loc[:, feature_cols].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    rows = []
    for i, col in enumerate(X.columns):
        rows.append({
            "Variabel": col,
            "VIF": variance_inflation_factor(X.values, i),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("VIF", ascending=False).reset_index(drop=True)


def first_stage_diagnostics(df, endog_col, instrument_cols, exog_cols, hac_lags=24):
    """
    Kjør førstesteg for kandidat-instrumenter og rapporter relevans.
    """
    cols = [endog_col] + list(instrument_cols) + list(exog_cols)
    work = df.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X_first = build_design_matrix(work, list(instrument_cols) + list(exog_cols), add_constant=True)
    y_first = work[endog_col].astype(float)
    model = sm.OLS(y_first, X_first).fit(cov_type="HAC", cov_kwds={"maxlags": hac_lags})

    restriction = " = 0, ".join(instrument_cols) + " = 0"
    f_test = model.f_test(restriction)
    instrument_rows = []
    for col in instrument_cols:
        instrument_rows.append({
            "Variabel": col,
            "Koeffisient": model.params[col],
            "t-verdi": model.tvalues[col],
            "p-verdi": model.pvalues[col],
        })

    summary = pd.DataFrame([{
        "Test": "Førstesteg F-test (ekskluderte instrumenter)",
        "F-statistikk": float(np.squeeze(f_test.fvalue)),
        "p-verdi": float(np.squeeze(f_test.pvalue)),
        "N": int(model.nobs),
        "R²": float(model.rsquared),
    }])
    return model, work, X_first, y_first, pd.DataFrame(instrument_rows), summary


def control_function_endogeneity_test(
    df,
    y_col,
    endog_col,
    instrument_cols,
    exog_cols,
    structural_cov_type="HAC",
    hac_lags=24,
):
    """
    Kontrollfunksjonstest for endogenitet:
    1. Førstesteg på endogen variabel
    2. Residual inn i strukturell ligning
    """
    first_stage, work, _, _, instrument_table, first_stage_summary = first_stage_diagnostics(
        df=df,
        endog_col=endog_col,
        instrument_cols=instrument_cols,
        exog_cols=exog_cols,
        hac_lags=hac_lags,
    )
    y_series = df.loc[work.index, y_col].astype(float)
    work = work.copy()
    work[y_col] = y_series
    work["first_stage_resid"] = first_stage.resid

    structural_cols = [endog_col] + list(exog_cols) + ["first_stage_resid"]
    X_struct = build_design_matrix(work, structural_cols, add_constant=True)
    y_struct = work[y_col].astype(float)
    model = sm.OLS(y_struct, X_struct).fit(
        cov_type=structural_cov_type,
        cov_kwds={"maxlags": hac_lags},
    )

    endogeneity_row = pd.DataFrame([{
        "Test": "Kontrollfunksjon (residualledd)",
        "Koeffisient": model.params["first_stage_resid"],
        "t-verdi": model.tvalues["first_stage_resid"],
        "p-verdi": model.pvalues["first_stage_resid"],
        "N": int(model.nobs),
    }])
    return {
        "first_stage_model": first_stage,
        "structural_model": model,
        "instrument_table": instrument_table,
        "first_stage_summary": first_stage_summary,
        "endogeneity_test": endogeneity_row,
        "work": work,
    }


def functional_form_comparison(df, y_col, base_features, candidate_specs, hac_lags=24):
    """
    Sammenlign lineær benchmark mot alternative spesifikasjoner.

    candidate_specs:
        {"Navn": [liste over ekstra variabler]}
    """
    rows = []
    for name, feature_cols in candidate_specs.items():
        model, work, X, _ = fit_ols(
            df=df,
            y_col=y_col,
            feature_cols=list(base_features) + list(feature_cols),
            cov_type="HAC",
            maxlags=hac_lags,
        )
        reset = linear_reset(model, power=2, use_f=True)
        rows.append({
            "Spesifikasjon": name,
            "N": int(model.nobs),
            "R²": float(model.rsquared),
            "AIC": float(model.aic),
            "BIC": float(model.bic),
            "RESET p-verdi": float(reset.pvalue),
        })
    return pd.DataFrame(rows).sort_values(["AIC", "BIC"]).reset_index(drop=True)
