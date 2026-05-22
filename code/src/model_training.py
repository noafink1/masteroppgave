"""
Felles hjelpefunksjoner for modelltrening i notebook 03a-03e.
"""

import pickle

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import INTERMEDIATE_DIR, TARGET, TRAIN_YEARS, TEST_YEARS
from .features import add_time_dummies, compute_fill_deviation


def load_prepared_iso_data(intermediate_dir=INTERMEDIATE_DIR):
    """Last ferdig klargjort ISO-datasett fra notebook 03."""
    return pd.read_parquet(f"{intermediate_dir}df_iso.parquet")


def prepare_iso_data(df, train_years=TRAIN_YEARS):
    """
    Filtrer til ISO-timer og bygg modellgrunnlaget brukt i notebook 03a-03c.
    """
    df_iso = df[df["regime_group"] == "ISO"].copy()
    df_iso, weekly_median = compute_fill_deviation(df_iso, train_years=train_years)
    df_iso = add_time_dummies(df_iso)
    return df_iso, weekly_median


def get_split_masks(df, train_years=TRAIN_YEARS, test_years=TEST_YEARS):
    """Returner boolske masker for trening og test basert på år."""
    train_mask = df.index.year.isin(train_years)
    test_mask = df.index.year.isin(test_years)
    return train_mask, test_mask


def split_features_target(df, feature_cols, target=TARGET,
                          train_years=TRAIN_YEARS, test_years=TEST_YEARS):
    """Lag standard train/test-splitt for en gitt featureliste."""
    train_mask, test_mask = get_split_masks(
        df, train_years=train_years, test_years=test_years
    )
    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, target].copy()
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, target].copy()
    return X_train, y_train, X_test, y_test, train_mask, test_mask


def make_prediction_frame(df, mask, actual, prediction_col, predictions):
    """Bygg standard prediksjonsformat for videre bruk i notebook 04-05."""
    preds = pd.DataFrame(index=df.loc[mask].index)
    preds["actual"] = actual.values
    preds[prediction_col] = np.asarray(predictions)
    preds["season"] = df.loc[mask, "season"].values
    preds["year"] = df.loc[mask].index.year
    preds["month"] = df.loc[mask].index.month
    preds["hour"] = df.loc[mask].index.hour
    return preds


def save_prepared_data(df_iso, weekly_median, intermediate_dir=INTERMEDIATE_DIR):
    """Lagre klargjort ISO-datasett og ukesmedianer fra notebook 03."""
    df_iso.to_parquet(f"{intermediate_dir}df_iso.parquet")
    with open(f"{intermediate_dir}fill_weekly_median.pkl", "wb") as f:
        pickle.dump({"weekly_median": weekly_median}, f)


def save_model_artifacts(stem, preds, payload, intermediate_dir=INTERMEDIATE_DIR):
    """Lagre standard outputs fra en modellnotebook."""
    preds.to_parquet(f"{intermediate_dir}preds_{stem}.parquet")
    with open(f"{intermediate_dir}models_{stem}.pkl", "wb") as f:
        pickle.dump(payload, f)


def stationary_regime_probabilities(transition_matrix):
    """Beregn ergodiske regimesannsynligheter fra overgangsmatrisen."""
    matrix = np.asarray(transition_matrix, dtype=float)
    eigenvalues, eigenvectors = np.linalg.eig(matrix.T)
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    probs = np.real(eigenvectors[:, idx])
    return probs / probs.sum()


def predict_msdr_ergodic(result, X, feature_cols, n_regimes=2):
    """
    Out-of-sample-prediksjon for MS-DR med felles eksogene koeffisienter.
    Regimevektene settes til ergodiske sannsynligheter.
    """
    transition = np.squeeze(result.regime_transition)
    ergodic = stationary_regime_probabilities(transition)
    beta = np.array([result.params[f"x{i + 1}"] for i in range(len(feature_cols))])
    xb = X[feature_cols].to_numpy() @ beta
    intercepts = np.array([result.params[f"const[{k}]"] for k in range(n_regimes)])
    regime_preds = xb[:, None] + intercepts[None, :]
    preds = regime_preds @ ergodic
    return preds, ergodic, transition


def fit_2sls(df, y_col, endog_col, instrument_cols, exog_cols, hac_lags=24):
    """
    Tilpass 2SLS via linearmodels.IV2SLS med HAC-standardfeil (Bartlett-kjerne).

    Returnerer:
        result : linearmodels IVResults
        work   : ren DataFrame som ble brukt (index bevart)
    """
    from linearmodels.iv import IV2SLS

    cols = [y_col, endog_col] + list(instrument_cols) + list(exog_cols)
    work = df.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = work[y_col].astype(float)
    endog = work[[endog_col]].astype(float)
    instr = work[list(instrument_cols)].astype(float)
    exog = sm.add_constant(work[list(exog_cols)].astype(float), has_constant="add")
    model = IV2SLS(y, exog, endog, instr).fit(
        cov_type="kernel", kernel="bartlett", bandwidth=hac_lags,
    )
    return model, work


def predict_2sls(result, df, endog_col, exog_cols):
    """
    Out-of-sample-prediksjon fra IV2SLS-resultat.
    Bruker faktiske verdier av endogen variabel (ikke fitted), siden
    formålet er prediksjon, ikke identifikasjon.
    """
    cols = [endog_col] + list(exog_cols)
    work = df.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    X = sm.add_constant(work[cols].astype(float), has_constant="add")
    params = result.params
    X_aligned = X.reindex(columns=params.index, fill_value=0.0)
    return pd.Series(X_aligned.values @ params.values, index=work.index)


def fit_quantreg(df, y_col, feature_cols, quantile, max_iter=5000):
    """
    Tilpass kvantilregresjon ved gitt kvantil tau.
    Asymptotiske standardfeil under iid-antakelse.
    """
    from statsmodels.regression.quantile_regression import QuantReg

    cols = [y_col] + list(feature_cols)
    work = df.loc[:, cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    y = work[y_col].astype(float)
    X = sm.add_constant(work[list(feature_cols)].astype(float), has_constant="add")
    model = QuantReg(y, X).fit(q=quantile, max_iter=max_iter)
    return model, work, X, y


def predict_msdr_out_of_sample(result, y_train, y_test, X_train, X_test,
                               probabilities="predicted"):
    """
    Lag out-of-sample-prediksjoner for testperioden fra et estimert MS-DR-resultat.
    Bygger en ny modell med hele exog-banen, men bruker kun de estimerte parameterne
    fra treningsperioden.
    """
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    all_y = pd.concat([y_train, y_test])
    all_X = pd.concat([X_train, X_test])
    pred_model = MarkovRegression(
        all_y,
        k_regimes=result.model.k_regimes,
        trend=getattr(result.model, "trend", "c"),
        exog=all_X,
        switching_variance=getattr(result.model, "switching_variance", True),
        switching_exog=getattr(result.model, "switching_exog", True),
    )
    preds = pred_model.predict(
        result.params,
        start=len(y_train),
        end=len(all_y) - 1,
        probabilities=probabilities,
    )
    transition = np.squeeze(result.regime_transition)
    ergodic = stationary_regime_probabilities(transition)
    return np.asarray(preds), ergodic, transition
