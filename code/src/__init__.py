"""
src — Importerbare moduler for masteroppgave-analysen.
"""

from .config import apply_style

from .data_loading import load_csv_years, load_all_data, load_production, load_weather
from .features import (get_season, add_calendar_features,
                        compute_fill_deviation, add_time_dummies,
                        fit_scaler, apply_scaling, add_season_dummies,
                        flag_negative_prices)
from .regime import classify_regime, near_boundary_sys, regime_group
from .evaluation import eval_metrics, eval_segment, demand_level
from .scenario import build_scenario, fill_group
from .uncertainty import (
    block_bootstrap_day_index,
    bootstrap_ci,
    bootstrap_iso_share_delta,
    bootstrap_p90_delta,
    bootstrap_extreme_share_delta,
    bootstrap_total_effect,
    p90_sensitivity,
)
from .model_training import (
    load_prepared_iso_data,
    prepare_iso_data,
    get_split_masks,
    split_features_target,
    make_prediction_frame,
    save_prepared_data,
    save_model_artifacts,
    stationary_regime_probabilities,
    predict_msdr_ergodic,
    predict_msdr_out_of_sample,
    fit_2sls,
    predict_2sls,
    fit_quantreg,
)
from .validation import (
    load_prediction_tables,
    build_validation_table,
    metrics_table,
    segmented_metrics,
    add_residual_columns,
    rolling_mae,
    bias_table,
    tail_metrics_table,
    price_level_table,
)
from .diagnostics import (
    stationarity_test_table,
    build_design_matrix,
    fit_ols,
    residual_diagnostics,
    vif_table,
    first_stage_diagnostics,
    control_function_endogeneity_test,
    functional_form_comparison,
)
