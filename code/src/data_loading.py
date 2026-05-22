"""
Datalasting: CSV-filer fra Nord Pool og Excel fra NVE.
"""

import pandas as pd

from .config import DATA_DIR, ALL_YEARS, PROD_COL_MAP


def load_csv_years(pattern, years, sep=";"):
    """
    Laster og konkatenerer rå CSV-filer for angitte år.
    `pattern` er en streng med {year} som plassholder.
    Returnerer DataFrame med 'Delivery Start (CET)' som DatetimeIndex.
    """
    frames = []
    for yr in years:
        path = pattern.format(year=yr)
        df_yr = pd.read_csv(path, sep=sep)
        frames.append(df_yr)
    df_raw = pd.concat(frames, ignore_index=True)
    df_raw["Delivery Start (CET)"] = pd.to_datetime(
        df_raw["Delivery Start (CET)"], format="%d.%m.%Y %H:%M:%S"
    )
    df_raw = df_raw.set_index("Delivery Start (CET)")
    if "Delivery End (CET)" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Delivery End (CET)"])
    return df_raw


def load_production(data_dir=DATA_DIR, years=ALL_YEARS):
    """
    Last produksjonsdata fra Nord Pool CSV-filer for NO4.

    Håndterer kolonneforskjeller mellom år:
        - 'NO4 Other (MW)' slås sammen med 'NO4 Other Renewable (MW)'
        - 'NO4 Wind Offshore (MW)' (kun 2023) droppes
    Returnerer timesresamplet DataFrame med interne kolonnenavn.
    """
    frames = []
    for yr in years:
        path = data_dir + f"production/Production_{yr}_NO4_None_MW.csv"
        df_yr = pd.read_csv(path, sep=";")

        # Harmoniser: slå sammen "NO4 Other (MW)" → "NO4 Other Renewable (MW)"
        if "NO4 Other (MW)" in df_yr.columns:
            if "NO4 Other Renewable (MW)" in df_yr.columns:
                df_yr["NO4 Other Renewable (MW)"] = (
                    df_yr["NO4 Other Renewable (MW)"]
                    .fillna(df_yr["NO4 Other (MW)"])
                )
            else:
                df_yr.rename(
                    columns={"NO4 Other (MW)": "NO4 Other Renewable (MW)"},
                    inplace=True,
                )
            if "NO4 Other (MW)" in df_yr.columns:
                df_yr = df_yr.drop(columns=["NO4 Other (MW)"])

        # Dropp Wind Offshore (kun 2023, mest tomme verdier)
        if "NO4 Wind Offshore (MW)" in df_yr.columns:
            df_yr = df_yr.drop(columns=["NO4 Wind Offshore (MW)"])

        frames.append(df_yr)

    df_raw = pd.concat(frames, ignore_index=True)
    df_raw["Delivery Start (CET)"] = pd.to_datetime(
        df_raw["Delivery Start (CET)"], format="%d.%m.%Y %H:%M:%S"
    )
    df_raw = df_raw.set_index("Delivery Start (CET)")
    if "Delivery End (CET)" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Delivery End (CET)"])

    # Behold kun mappede kolonner og gi interne navn
    cols_present = [c for c in PROD_COL_MAP if c in df_raw.columns]
    df_raw = df_raw[cols_present].rename(columns=PROD_COL_MAP)

    # Fyll NaN i komponentkolonner med 0 (sporadisk manglende verdier)
    df_raw = df_raw.fillna(0)

    # Resample til timer (håndterer 2026 15-min data)
    return df_raw.resample("h").mean()


def load_weather(data_dir=DATA_DIR):
    """
    Last daglig temperatur og nedbør fra Narvik Sentrum (met.no).

    Kildefil: data/modell-data/weather/Narvik_temperatur_and_percipitation.csv
    Format: semikolonseparert, komma som desimaltegn, datoformat DD.MM.YYYY
    Siste linje er en metadatanote og hoppes over.

    Returnerer:
        DataFrame med DatetimeIndex (daglig oppløsning) og kolonner:
            temp_NO4  — maksimumstemperatur (°C)
            precip_NO4 — nedbør (mm/dag)
    """
    path = data_dir + "weather/Narvik_temperatur_and_percipitation.csv"
    df = pd.read_csv(
        path,
        sep=";",
        decimal=",",
        encoding="utf-8-sig",   # håndterer BOM
        skipfooter=1,            # hopp over metadatanote på siste linje
        engine="python",
    )
    df["date"] = pd.to_datetime(df["Tid(norsk normaltid)"], format="%d.%m.%Y")
    df = df.rename(columns={
        "Maksimumstemperatur (døgn)": "temp_NO4",
        "Nedbør (døgn)":             "precip_NO4",
    })[["date", "temp_NO4", "precip_NO4"]]

    # Eksplisitt konvertering til float — decimal="," i engine="python"
    # konverterer ikke alltid kolonner med spesialtegn i headeren
    for col in ["temp_NO4", "precip_NO4"]:
        df[col] = (
            df[col].astype(str)
                   .str.replace(",", ".", regex=False)
                   .pipe(pd.to_numeric, errors="coerce")
        )

    df = df.set_index("date")
    return df


def load_all_data(data_dir=DATA_DIR, years=ALL_YEARS):
    """
    Last alle datakilder og slå sammen til én DataFrame.

    Datakilder:
        - Spotpriser (NO4, NO3, SE1, SE2, FI)
        - Systempris
        - Grensekryssende flyt inn/ut av NO4
        - Forbruk (NO3, NO4, FI, SE1)
        - Fyllingsgrad fra NVE (ukentlig → ffill til timer)

    Returnerer:
        df: DataFrame med DatetimeIndex, sortert, uten duplikater/NaN
    """
    # ── Spotpriser ────────────────────────────────────────────────────────────
    prices_raw = load_csv_years(
        data_dir + "pris/AuctionPrice_{year}_DayAhead_NO4,NO3,SE1,SE2,FI_NOK_None.csv",
        years
    )

    # ── Systempris ────────────────────────────────────────────────────────────
    syspris_raw = load_csv_years(
        data_dir + "system-pris/SystemPrice_{year}_DayAhead__NOK_None.csv",
        years
    )

    # ── Grensekryssende flyt ──────────────────────────────────────────────────
    flow_raw = load_csv_years(
        data_dir + "flyt/AuctionFlow_{year}_DayAhead_NO4_None_MW.csv",
        years
    )
    flow_raw = flow_raw.drop(
        columns=[c for c in ["NO4 Total Export (MW)", "NO4 Total Import (MW)",
                             "NO4 Total Net Position (MW)"]
                 if c in flow_raw.columns]
    )

    # ── Forbruk ───────────────────────────────────────────────────────────────
    cons_raw = load_csv_years(
        data_dir + "consumption/Consumption_{year}_NO3,NO4,FI,SE1_None_MW.csv",
        years
    )

    # ── Resample til timesfrekvens ────────────────────────────────────────────
    prices_h  = prices_raw.resample("h").mean()
    syspris_h = syspris_raw.resample("h").mean()
    flow_h    = flow_raw.resample("h").mean()
    cons_h    = cons_raw.resample("h").mean()

    # ── Slå sammen ────────────────────────────────────────────────────────────
    df = (
        prices_h
        .join(syspris_h, how="inner")
        .join(flow_h,    how="inner")
        .join(cons_h,    how="inner")
    )
    df = df.dropna()

    # ── Gi kolonner meningsfulle navn ─────────────────────────────────────────
    df.columns = [
        # Spotpriser
        "price_FI", "price_NO3", "price_NO4", "price_SE1", "price_SE2",
        # Systempris
        "system_price",
        # Flyt inn/ut av NO4
        "flow_NO4_FI", "flow_FI_NO4",
        "flow_NO4_NO3", "flow_NO3_NO4",
        "flow_NO4_SE1", "flow_SE1_NO4",
        "flow_NO4_SE2", "flow_SE2_NO4",
        # Forbruk
        "cons_FI", "cons_NO3", "cons_NO4", "cons_SE1",
    ]

    # ── Produksjon ────────────────────────────────────────────────────────────
    prod_h = load_production(data_dir, years)
    df = df.join(prod_h, how="inner")

    # Netto posisjon (produksjon − forbruk) for NO4
    df["net_position"] = df["prod_total_NO4"] - df["cons_NO4"]

    # ── Fyllingsgrad fra NVE (ukentlig → ffill) ──────────────────────────────
    resvr_path = data_dir + "vannmagasin/Fyllingsgrad og energiinnhold i prisområder og vassdragsområder.xlsx"
    resvr_raw = pd.read_excel(resvr_path)

    resvr_no4 = (
        resvr_raw[resvr_raw["Område"] == "NO4"][["År", "Uke", "Fyllingsgrad"]]
        .rename(columns={"År": "year", "Uke": "week", "Fyllingsgrad": "fill_rate"})
        .copy()
    )

    df["year"] = df.index.year
    df["week"] = df.index.isocalendar().week.astype(int)

    df = df.reset_index().merge(resvr_no4, on=["year", "week"], how="left")
    df = df.set_index("Delivery Start (CET)")
    df["fill_rate"] = df["fill_rate"].ffill()
    df = df.drop(columns=["year", "week"])

    # ── Værdata (daglig → timer via ffill) ───────────────────────────────────
    weather_daily = load_weather(data_dir)
    # Broadcast daglig verdi til alle 24 timer: reindekser til timesfrekvens
    hourly_idx = pd.date_range(
        weather_daily.index.min(),
        weather_daily.index.max() + pd.Timedelta(hours=23),
        freq="h",
    )
    weather_h = weather_daily.reindex(hourly_idx, method="ffill")
    weather_h.index.name = "Delivery Start (CET)"
    df = df.join(weather_h, how="left")

    # ── Rydd opp ──────────────────────────────────────────────────────────────
    df = df.sort_index()

    # Fjern duplikater
    n_dup = df.index.duplicated().sum()
    if n_dup > 0:
        df = df[~df.index.duplicated(keep="first")]

    return df
