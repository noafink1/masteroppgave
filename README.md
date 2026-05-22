# Masteroppgave — Stargate Norway og spotprisen i NO4

Dette repoet inneholder koden og dataene som ligger til grunn for masteroppgaven
*"Stargate Norway: prisvirkning og volatilitet i NO4"* (UiS, 2026).

## Innhold

- `code/` — Python-kode og Jupyter-notatbøker
  - `src/` — importerbare moduler (config, datainnlasting, regimer, features, evaluering, scenario)
  - `01_data_and_regimes.ipynb` → `06_results_summary.ipynb` — analyseflyt
- `data/` — rådata fra Nord Pool (priser, flyt, forbruk, produksjon) og NVE (magasinfylling)

## Kjørerekkefølge for notatbøkene

```
01_data_and_regimes        # last inn data, lag features og regimer → lagrer parquet
02_descriptive_analysis    # deskriptive figurer (datakapittel)
03_model_training          # variabelvalg, diagnostikk, modelltrening
04_validation              # metrikker, rullende kryssvalidering, residualanalyse
05_simulation              # scenarioanalyse (Stargate-grunnlast på 230 MW)
06_results_summary         # samlede tabeller og figurer
```

Mellomresultater (`code/intermediate/`) er utelatt fra repoet — de regenereres
ved å kjøre `01_data_and_regimes.ipynb` først, deretter de øvrige notatbøkene.

## Avhengigheter

Standard vitenskapelig Python-stack: `pandas`, `numpy`, `matplotlib`,
`scikit-learn`, `statsmodels`, `xgboost`.
