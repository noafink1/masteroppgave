"""
Usikkerhetsestimering: block-bootstrap KI for scenarioeffekter.

Veileders ønske (mai 2026): tallfest usikkerhetsrom for hovedkonklusjonene
(ΔE[P], ΔPr(ISO), ΔP90) — ikke bare kvalitativ robusthet.

Block-bootstrap på uker (7 dager) bevarer ukessyklus og autokorrelasjon i
de timesoppløste prisene. Vanlig i.i.d. bootstrap undervurderer SE i tidsserier.
"""

import numpy as np
import pandas as pd


def block_bootstrap_day_index(dt_index, block_days=7, seed=None):
    """
    Generer én bootstrap-resampling av rad-posisjoner ved å trekke blokker av dager.

    Parametere:
        dt_index:    DatetimeIndex på timesoppløst data
        block_days:  blokkstørrelse i dager (default 7 = uke)
        seed:        np.random.SeedSequence-kompatibel seed

    Returnerer:
        np.array med rad-posisjoner (ints) som kan brukes med df.iloc[...]
    """
    rng = np.random.default_rng(seed)
    days = pd.Series(dt_index.normalize().unique()).sort_values().values
    n_blocks_needed = int(np.ceil(len(days) / block_days))
    block_starts = rng.integers(0, max(1, len(days) - block_days + 1),
                                 size=n_blocks_needed)

    # Map dato → rad-posisjon for rask oppslag
    pos_by_day = {}
    for pos, ts in enumerate(dt_index):
        d = ts.normalize()
        pos_by_day.setdefault(d, []).append(pos)

    out = []
    for start in block_starts:
        for d_offset in range(block_days):
            day = days[min(start + d_offset, len(days) - 1)]
            day_ts = pd.Timestamp(day)
            if day_ts in pos_by_day:
                out.extend(pos_by_day[day_ts])
    return np.array(out, dtype=int)


def bootstrap_ci(values, alpha=0.05):
    """Returner (lo, median, hi) percentiler. alpha=0.05 ⇒ 95 % KI."""
    arr = np.asarray(values)
    lo = float(np.percentile(arr, 100 * alpha / 2))
    md = float(np.percentile(arr, 50))
    hi = float(np.percentile(arr, 100 * (1 - alpha / 2)))
    return lo, md, hi


def bootstrap_iso_share_delta(df, logit_model, scale_cols, scaled_name_map,
                               scaler, logit_features, delta_mw,
                               prod_col_for_netpos="prod_total_NO4",
                               n_boot=500, block_days=7, seed=42, verbose=False):
    """
    Block-bootstrap KI for ΔPr(ISO) ved en gitt MW-økning.

    Logit-modellen holdes fast (trent én gang på full treningsperiode);
    bare prediksjonssettet resamples. Dette gir usikkerhet fra
    sampling/komposisjon, ikke fra parameterestimering. For full
    parameterusikkerhet bør logit refittes per iterasjon, men det er
    typisk en mindre komponent enn samplingsusikkerheten her.

    Parametere:
        df:                 DataFrame med all data (full periode) + alle SCALE_COLS
        logit_model:        ferdig fittet sklearn LogisticRegression
        scale_cols:         SCALE_COLS fra config
        scaled_name_map:    SCALED_NAME_MAP fra config
        scaler:             ferdig fittet StandardScaler
        logit_features:     liste med feature-navn brukt av logit
        delta_mw:           MW å legge på cons_NO4
        n_boot:             antall bootstrap-iterasjoner
        block_days:         blokkstørrelse i dager
        seed:               RNG-seed

    Returnerer:
        dict med "estimates" (np.array av ΔPr(ISO) i prosentpoeng),
        "ci" (lo, median, hi).
    """
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)

    for b in range(n_boot):
        b_seed = int(rng.integers(0, 2**31 - 1))
        idx = block_bootstrap_day_index(df.index, block_days=block_days, seed=b_seed)
        df_b = df.iloc[idx].copy()

        # Baseline
        sc_base = scaler.transform(df_b[scale_cols])
        for i, col in enumerate(scale_cols):
            df_b[scaled_name_map[col]] = sc_base[:, i]
        p_base = logit_model.predict_proba(df_b[logit_features])[:, 1].mean()

        # Scenario
        df_s = df_b.copy()
        df_s["cons_NO4"] = df_s["cons_NO4"] + delta_mw
        if "net_position" in df_s.columns and prod_col_for_netpos in df_s.columns:
            df_s["net_position"] = df_s[prod_col_for_netpos] - df_s["cons_NO4"]
        sc_scen = scaler.transform(df_s[scale_cols])
        for i, col in enumerate(scale_cols):
            df_s[scaled_name_map[col]] = sc_scen[:, i]
        p_scen = logit_model.predict_proba(df_s[logit_features])[:, 1].mean()

        estimates[b] = (p_scen - p_base) * 100  # prosentpoeng

        if verbose and (b + 1) % 100 == 0:
            print(f"  ΔPr(ISO) bootstrap: {b+1}/{n_boot}")

    return {"estimates": estimates, "ci": bootstrap_ci(estimates)}


def bootstrap_p90_delta(prices_baseline, prices_scenario,
                         n_boot=500, block_days=7, seed=42, verbose=False):
    """
    Block-bootstrap KI for ΔP90 = P90(scenario) − P90(baseline).

    Prisene må være indeksert på samme DatetimeIndex. Bootstrap er på
    rad-posisjoner (uker), så baseline- og scenarioserien resamples i takt.

    Parametere:
        prices_baseline:  pd.Series med basis-priser (samme index som scenario)
        prices_scenario:  pd.Series med scenario-priser
        n_boot, block_days, seed: bootstrap-parametere

    Returnerer:
        dict med "estimates", "ci", "p90_base", "p90_scen".
    """
    assert prices_baseline.index.equals(prices_scenario.index), \
        "baseline og scenario må ha samme index"

    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    p90_base_boot = np.empty(n_boot)
    p90_scen_boot = np.empty(n_boot)

    base_vals = prices_baseline.values
    scen_vals = prices_scenario.values

    for b in range(n_boot):
        b_seed = int(rng.integers(0, 2**31 - 1))
        idx = block_bootstrap_day_index(prices_baseline.index,
                                         block_days=block_days, seed=b_seed)
        p_b = np.percentile(base_vals[idx], 90)
        p_s = np.percentile(scen_vals[idx], 90)
        p90_base_boot[b] = p_b
        p90_scen_boot[b] = p_s
        estimates[b] = p_s - p_b

        if verbose and (b + 1) % 100 == 0:
            print(f"  ΔP90 bootstrap: {b+1}/{n_boot}")

    return {
        "estimates":   estimates,
        "ci":          bootstrap_ci(estimates),
        "p90_base_ci": bootstrap_ci(p90_base_boot),
        "p90_scen_ci": bootstrap_ci(p90_scen_boot),
    }


def bootstrap_extreme_share_delta(prices_baseline, prices_scenario, threshold,
                                    n_boot=500, block_days=7, seed=42, verbose=False):
    """
    Block-bootstrap KI for Δandel timer med pris > terskel.

    Selv om ΔP90 = β·Δq er konstant (og dermed null usikkerhet) for lineære
    modeller, har *andelen* timer som krysser P90-terskelen meningsfull
    variasjon — fordi sammensetningen av timer i bootstrap-utvalget endrer
    hvor mange som ligger nær terskelen.

    Returnerer dict med "estimates" (Δandel i prosentpoeng), "ci",
    "share_base_ci", "share_scen_ci".
    """
    assert prices_baseline.index.equals(prices_scenario.index)
    rng = np.random.default_rng(seed)
    deltas    = np.empty(n_boot)
    base_arr  = np.empty(n_boot)
    scen_arr  = np.empty(n_boot)
    base_vals = prices_baseline.values
    scen_vals = prices_scenario.values

    for b in range(n_boot):
        b_seed = int(rng.integers(0, 2**31 - 1))
        idx = block_bootstrap_day_index(prices_baseline.index,
                                         block_days=block_days, seed=b_seed)
        share_base = float((base_vals[idx] > threshold).mean() * 100)
        share_scen = float((scen_vals[idx] > threshold).mean() * 100)
        base_arr[b] = share_base
        scen_arr[b] = share_scen
        deltas[b]   = share_scen - share_base
        if verbose and (b + 1) % 100 == 0:
            print(f"  Δshare>{threshold:.0f} bootstrap: {b+1}/{n_boot}")

    return {
        "estimates":      deltas,
        "ci":             bootstrap_ci(deltas),
        "share_base_ci":  bootstrap_ci(base_arr),
        "share_scen_ci":  bootstrap_ci(scen_arr),
    }


def bootstrap_total_effect(df, dp_iso_series, logit_model, scale_cols,
                            scaled_name_map, scaler, logit_features,
                            delta_mw, target="price_NO4",
                            prod_col_for_netpos="prod_total_NO4",
                            n_boot=500, block_days=7, seed=42, verbose=False):
    """
    Block-bootstrap KI for total ΔE[P] = Pr(ISO)·ΔE[P|I=1] + ΔPr(ISO)·ΔP_gap.

    Propagerer usikkerhet fra Pr(ISO), ΔPr(ISO) og ΔP_gap simultant ved å
    resample blokker av uker. Marginaleffekten ΔE[P|I=1] (dp_iso_series)
    holdes fast per modell — det er β × delta_mw fra hovedmodellen.

    Parametere:
        df:               DataFrame med all data (full periode), inkludert target
                          og is_iso (eller regime_group for å regne is_iso)
        dp_iso_series:    pd.Series med ΔP per ISO-time (skalar gjentatt eller
                          per-rad). Bare gjennomsnittet brukes.
        logit_model, scale_cols, scaled_name_map, scaler, logit_features:
                          som i bootstrap_iso_share_delta
        delta_mw:         MW
        target:           kolonnenavn for pris

    Returnerer:
        dict med "estimates" (total ΔE[P] NOK/MWh), "ci",
        og dekomponering "t1_ci", "t2_ci", "d_pr_ci", "gap_ci".
    """
    if "is_iso" not in df.columns:
        df = df.copy()
        df["is_iso"] = (df["regime_group"] == "ISO").astype(int)

    rng = np.random.default_rng(seed)
    totals  = np.empty(n_boot)
    t1_arr  = np.empty(n_boot)
    t2_arr  = np.empty(n_boot)
    dpr_arr = np.empty(n_boot)
    gap_arr = np.empty(n_boot)

    dp_iso_mean = float(np.mean(dp_iso_series))

    for b in range(n_boot):
        b_seed = int(rng.integers(0, 2**31 - 1))
        idx = block_bootstrap_day_index(df.index, block_days=block_days, seed=b_seed)
        df_b = df.iloc[idx].copy()

        # Pr(ISO) baseline og scenario fra logit
        sc_base = scaler.transform(df_b[scale_cols])
        for i, col in enumerate(scale_cols):
            df_b[scaled_name_map[col]] = sc_base[:, i]
        p_base = logit_model.predict_proba(df_b[logit_features])[:, 1].mean()

        df_s = df_b.copy()
        df_s["cons_NO4"] = df_s["cons_NO4"] + delta_mw
        if "net_position" in df_s.columns and prod_col_for_netpos in df_s.columns:
            df_s["net_position"] = df_s[prod_col_for_netpos] - df_s["cons_NO4"]
        sc_scen = scaler.transform(df_s[scale_cols])
        for i, col in enumerate(scale_cols):
            df_s[scaled_name_map[col]] = sc_scen[:, i]
        p_scen = logit_model.predict_proba(df_s[logit_features])[:, 1].mean()
        d_pr = p_scen - p_base

        # ΔP_gap fra bootstrap-utvalget
        iso_mask = df_b["is_iso"] == 1
        if iso_mask.sum() == 0 or (~iso_mask).sum() == 0:
            # degenerert bootstrap-trekning — hopp over
            totals[b]  = np.nan
            t1_arr[b]  = np.nan
            t2_arr[b]  = np.nan
            dpr_arr[b] = d_pr
            gap_arr[b] = np.nan
            continue
        gap = (df_b.loc[iso_mask, target].mean()
               - df_b.loc[~iso_mask, target].mean())

        t1 = p_base * dp_iso_mean
        t2 = d_pr * gap

        t1_arr[b]  = t1
        t2_arr[b]  = t2
        dpr_arr[b] = d_pr * 100  # pp
        gap_arr[b] = gap
        totals[b]  = t1 + t2

        if verbose and (b + 1) % 100 == 0:
            print(f"  Total ΔE[P] bootstrap: {b+1}/{n_boot}")

    # Filtrer ut NaN fra evt. degenererte trekninger
    valid = ~np.isnan(totals)
    return {
        "estimates": totals[valid],
        "ci":        bootstrap_ci(totals[valid]),
        "t1_ci":     bootstrap_ci(t1_arr[valid]),
        "t2_ci":     bootstrap_ci(t2_arr[valid]),
        "d_pr_ci":   bootstrap_ci(dpr_arr[valid]),
        "gap_ci":    bootstrap_ci(gap_arr[valid]),
        "n_valid":   int(valid.sum()),
    }


def p90_sensitivity(prices_train, prices_full, dp_train, dp_full):
    """
    Sensitivitet for P90-terskel definert på treningsperiode vs full periode.

    Returnerer både terskler og resulterende ΔP90 under begge definisjoner,
    for å vise hvor mye strukturbruddet (post-2024) flytter konklusjonen.

    Parametere:
        prices_train: pd.Series — råpriser i treningsperioden
        prices_full:  pd.Series — råpriser i full periode
        dp_train:     pd.Series — ΔP i scenario (samme indeks som prices_train)
        dp_full:      pd.Series — ΔP i scenario (samme indeks som prices_full)

    Returnerer:
        dict med p90_train, p90_full, og share_extreme_* tall.
    """
    p90_train = float(np.percentile(prices_train, 90))
    p90_full  = float(np.percentile(prices_full,  90))

    # Baseline-andel ekstreme timer med hver terskel (per definisjon ~10 % når
    # terskelen brukes på sitt eget utvalg, så vi rapporterer kryssanvendelse)
    share_base_train_on_full = float((prices_full > p90_train).mean() * 100)
    share_base_full_on_train = float((prices_train > p90_full).mean() * 100)

    # Scenario-priser
    pscen_full = prices_full + dp_full
    share_scen_train = float((pscen_full > p90_train).mean() * 100)
    share_scen_full  = float((pscen_full > p90_full).mean() * 100)

    return {
        "p90_train":               p90_train,
        "p90_full":                p90_full,
        "share_base_train_on_full": share_base_train_on_full,
        "share_base_full_on_train": share_base_full_on_train,
        "share_scen_train":        share_scen_train,
        "share_scen_full":         share_scen_full,
        "delta_share_train":       share_scen_train  - float((prices_full > p90_train).mean() * 100),
        "delta_share_full":        share_scen_full   - float((prices_full > p90_full).mean() * 100),
    }
