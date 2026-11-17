# egypt-survey-ml

Machine-learning models of rural agricultural wages in Egypt, built from the 2022 Labor Force Survey (LFS) and linked to irrigation command areas and agricultural, water, demographic and economic indicators.

## Approach

- **Two-part wage model** per sex: a CatBoost classifier estimates P(paid) and a CatBoost regressor estimates the wage if paid. The expected wage is `p * r`.
- **Ensemble**: survey respondents are assigned to command units at random. One classifier/regressor pair is trained per random draw (seed), and the spread over members expresses uncertainty.
- **Explanation**: SHAP values of both models are converted to EGP and combined so that they add up to the predicted expected wage. Feature importance bars show an uncertainty band across ensemble members.

## Repository layout

| Path | Contents |
|---|---|
| `src/egypt_survey_ml/` | Reusable package: dataset building (`dataset.py`, `config.py`, `data_prep.py`), training (`training.py`, `catboost.py`), ensemble prediction (`predict.py`), SHAP attribution (`explain.py`), data locations (`paths.py`) |
| `notebooks/data-gathering/` | Collect covariates (GDP, demographics, field sizes, food production, water use) |
| `notebooks/correlation/` | Assign survey data to command units and correlate indicators |
| `notebooks/training/` | Hyperparameter tuning and model training |
| `notebooks/evaluation/` | Predictions, SHAP plots, map plots |
| `ERF_Data/` | Survey cleaning notebooks and crosswalk scripts (see [ERF_Data/README.md](ERF_Data/README.md)) |
| `tests/` | Unit tests |
| `figures/`, `models/`, `output/` | Generated figures, trained models and results |

## Getting started

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run jupyter lab
```

## Data

The survey microdata is restricted and **not included** in this repository. Place it as described in [ERF_Data/README.md](ERF_Data/README.md), or use the Deltares OneDrive layout via `egypt_survey_ml.paths.DataPaths`.

## Typical workflow

1. Preprocess the LFS data (`ERF_Data/Data_Cleaning/`).
2. Gather covariates and assign them to command units (`notebooks/data-gathering/`, `notebooks/correlation/`).
3. Tune and train the models (`notebooks/training/`).
4. Predict and explain (`notebooks/evaluation/run_predictions.ipynb`).

## License

GNU GPL v3, see [LICENSE](LICENSE).
