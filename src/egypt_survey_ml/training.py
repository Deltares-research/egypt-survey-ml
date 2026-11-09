"""Train and evaluate the wage models: a CatBoost classifier (P(paid)) and a regressor (wage if paid).

:class:`ModelTrainer` trains one model per command-unit draw (seed) and sex, saves them in the
layout expected by :func:`egypt_survey_ml.predict.load_members` and writes ``model_metrics.csv``.
Tuned hyperparameters per seed are optional: without them the defaults are used.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, NamedTuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy import stats as sstats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .paths import REPO_ROOT

try:
    from tqdm.auto import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):  # progress bar is a nicety, not a dependency
        return iterable


DEFAULT_LEARNING_RATE = 0.05
DEFAULT_DEPTH = 6
SEXES = ("male", "female")


class HyperparameterWarning(UserWarning):
    """The tuned hyperparameter files do not (fully) cover what is being trained."""


# --------------------------------------------------------------------------- metrics
def evaluate(y_true_wage, y_pred_wage, is_paid_true, label=""):
    rmse = mean_squared_error(y_true_wage, y_pred_wage) ** 0.5
    metrics = {
        "label": label,
        "n": len(y_true_wage),
        "mae": mean_absolute_error(y_true_wage, y_pred_wage),
        "rmse": rmse,
        "r2": r2_score(y_true_wage, y_pred_wage),
        "spearman": sstats.spearmanr(y_true_wage, y_pred_wage).correlation,
        "mean_pred_at_true_zero": (
            y_pred_wage[is_paid_true == 0].mean()
            if (is_paid_true == 0).any()
            else np.nan
        ),
        "mean_pred_at_true_positive": (
            y_pred_wage[is_paid_true == 1].mean()
            if (is_paid_true == 1).any()
            else np.nan
        ),
        "mean_actual_at_true_positive": (
            y_true_wage[is_paid_true == 1].mean()
            if (is_paid_true == 1).any()
            else np.nan
        ),
    }
    pos = is_paid_true == 1
    if pos.sum() > 1:
        metrics["intensive_mae"] = mean_absolute_error(
            y_true_wage[pos], y_pred_wage[pos]
        )
        metrics["intensive_rmse"] = (
            mean_squared_error(y_true_wage[pos], y_pred_wage[pos]) ** 0.5
        )
        metrics["intensive_r2"] = r2_score(y_true_wage[pos], y_pred_wage[pos])
    return metrics


def evaluate_participation(y_true, p_pred, threshold=0.5, label=""):
    y_pred = (p_pred >= threshold).astype(int)

    return {
        "label": label,
        "n": len(y_true),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, p_pred),
        "pr_auc": average_precision_score(y_true, p_pred),
        "logloss": log_loss(y_true, p_pred),
        "pred_positive_rate": y_pred.mean(),
        "actual_positive_rate": y_true.mean(),
    }


def gender_gap(pred_male, pred_female, actual_male, actual_female):
    return {
        "predicted_gap": pred_male.mean() - pred_female.mean(),
        "observed_gap": actual_male.mean() - actual_female.mean(),
    }


# ------------------------------------------------------------------- model kinds
@dataclass(frozen=True)
class _Kind:
    """Everything that differs between the wage regressor and the participation classifier."""

    name: str
    target: str  # column of the model frame the model is fitted on
    model_cls: type
    model_params: dict
    predict: Callable  # (fitted model, X) -> predictions
    evaluate: Callable  # (test frame, predictions, label) -> metrics dict
    headline: str  # metric reported while training and saved in model_metrics.csv
    summary_cols: list  # metrics summarised at the end of a run
    hp_keys: tuple  # hyperparameter tables used by this kind
    record: Callable  # (rep, seed, {sex: (metrics, model path)}) -> model_metrics.csv row
    # zero-wage handling of the model frames; None follows ``DatasetConfig.zero_handling``
    zero_handling: str | None = None


def _regressor_record(rep, seed, per_sex):
    return {
        "rep": rep,
        "seed": seed,
        "male_r2": per_sex["male"][0]["r2"],
        "female_r2": per_sex["female"][0]["r2"],
        "male_model_file": str(per_sex["male"][1]),
        "female_model_file": str(per_sex["female"][1]),
    }


def _classifier_record(rep, seed, per_sex):
    return {
        "rep": rep,
        "seed": seed,
        "male_accuracy": per_sex["male"][0]["accuracy"],
        "female_accuracy": per_sex["female"][0]["accuracy"],
        "model_file": str(per_sex["female"][1]),
    }


KINDS = {
    "regressor": _Kind(
        name="regressor",
        # Fit directly on raw wage: no log1p/expm1 round-trip, which was diagnosed as the main
        # source of systematic under-prediction in train-catboost_v2_rural.ipynb.
        target="totwag_all",
        model_cls=CatBoostRegressor,
        model_params={"loss_function": "RMSE", "monotone_constraints": {"salinity_production_Wheat": 1, "producer_price": 1, "salinity_production_Maize (incl. 0067 and 0068)": 1, "salinity_production_Sugar cane": 1}},#, "monotone_constraints": {"producer_price": 1, "salinity_production_Maize (incl. 0067 and 0068)": 1,"salinity_production_Sugar cane": 1, "salinity_production_Wheat": 1, "salinity_production_Alfalfa for forage": 1, "salinity_production_Sugar beet": 1,"salinity_production_Rice, paddy": 1}},
        # a wage can't be negative; CatBoost's raw output can be
        predict=lambda model, X: np.clip(model.predict(X), 0, None),
        evaluate=lambda frame, pred, label: evaluate(
            frame["totwag_all"].to_numpy(),
            pred,
            frame["is_paid"].to_numpy(),
            label=label,
        ),
        headline="r2",
        summary_cols=["mae", "rmse", "r2", "spearman"],
        hp_keys=SEXES,
        record=_regressor_record,
    ),
    "classifier": _Kind(
        name="classifier",
        target="is_paid",
        model_cls=CatBoostClassifier,
        model_params={"monotone_constraints": {"producer_price": 1, "salinity_production_Wheat": 1, "salinity_production_Maize (incl. 0067 and 0068)": 1, "salinity_production_Sugar cane": 1}},#,{"monotone_constraints": {"producer_price": 1, "salinity_production_Maize (incl. 0067 and 0068)": 1,"salinity_production_Sugar cane": 1, "salinity_production_Wheat": 1, "salinity_production_Alfalfa for forage": 1, "salinity_production_Sugar beet": 1,"salinity_production_Rice, paddy": 1}},
        predict=lambda model, X: model.predict(X),
        evaluate=lambda frame, pred, label: evaluate_participation(
            frame["is_paid"].to_numpy(), pred, label=label
        ),
        headline="accuracy",
        summary_cols=["accuracy", "precision", "recall", "f1", "auc", "pr_auc"],
        hp_keys=("classifier",),
        record=_classifier_record,
        # the classifier learns who is paid, so it needs the zero-wage rows that
        # zero handling removes (or replaces) for the wage regressor
        zero_handling="none",
    ),
}


class TrainingResult(NamedTuple):
    metrics: pd.DataFrame  # what is written to model_metrics.csv
    runs: pd.DataFrame  # evaluation metrics of every trained model
    gaps: pd.DataFrame  # predicted and observed male-female gap per seed


def _resolve(path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_hp_table(path: Path | None) -> pd.DataFrame | None:
    """Hyperparameters indexed by seed (columns ``lr`` and ``depth``, both optional)."""
    if path is None or not path.is_file():
        return None
    table = pd.read_csv(path)
    if "seed" not in table.columns:
        raise ValueError(f"hyperparameter file {path} has no 'seed' column")
    table["seed"] = table["seed"].astype(int)
    if table["seed"].duplicated().any():
        raise ValueError(f"hyperparameter file {path} lists a seed more than once")
    return table.set_index("seed")


def _hp_key(kind: str, sex: str) -> str:
    return sex if kind == "regressor" else "classifier"


# --------------------------------------------------------------------------- trainer
class ModelTrainer:
    """Trains the wage regressors and participation classifiers of a prepared dataset.

    The feature columns of the male and female models come from the dataset
    (``ds.feature_columns(sex)``), i.e. the feature mask plus the per-sex extra/excluded
    columns of its :class:`~egypt_survey_ml.config.DatasetConfig`.

    The three hyperparameter files (``male_hp_path``, ``female_hp_path`` for the regressors,
    ``cls_hp_path`` for the classifier) are optional csv files with a ``seed`` column and
    optional ``lr`` and ``depth`` columns. Relative paths are relative to the repository
    root. Models without a (valid) file, or for a seed missing in it, use the defaults.
    """

    def __init__(
        self,
        ds,
        male_hp_path=None,
        female_hp_path=None,
        cls_hp_path=None,
    ):
        self.ds = ds

        self.MODELS = {}
        self.MODELS_CLASSIFIER = {}
        self._stores = {"regressor": self.MODELS, "classifier": self.MODELS_CLASSIFIER}

        provided = {
            "male": male_hp_path,
            "female": female_hp_path,
            "classifier": cls_hp_path,
        }
        # path as given by the user (None if not provided), and the table if it is readable
        self.hp_paths = {
            key: None if path is None else _resolve(path)
            for key, path in provided.items()
        }
        self.hp_parameters = {
            key: _read_hp_table(path) for key, path in self.hp_paths.items()
        }

    # ------------------------------------------------------------ fitting a single model
    def _fit(
        self,
        kind: _Kind,
        train_frame,
        test_frame,
        feature_cols,
        cats,
        learning_rate,
        depth,
        iterations,
        early_stopping_rounds,
        use_best_model,
        random_seed,
    ):
        X_train = train_frame[feature_cols]
        y_train = train_frame[kind.target]
        model = kind.model_cls(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=False,
            **kind.model_params,
        )
        # There is only a train and a test set (no validation set), so early stopping and
        # best-model selection deliberately monitor the training data.
        model.fit(
            X_train,
            y_train,
            cat_features=cats,
            eval_set=(X_train, y_train),
            use_best_model=use_best_model,
            early_stopping_rounds=early_stopping_rounds,
        )
        return model, kind.predict(model, test_frame[feature_cols])

    def fit_and_predict(
        self,
        train_frame,
        test_frame,
        sex,
        learning_rate=DEFAULT_LEARNING_RATE,
        depth=DEFAULT_DEPTH,
        iterations=500,
        early_stopping_rounds=None,
        use_best_model=None,
        random_seed=42,
    ):
        """Fit a wage regressor with the feature columns of ``sex`` and predict the test frame."""
        return self._fit(
            KINDS["regressor"],
            train_frame,
            test_frame,
            self.ds.feature_columns(sex),
            self.ds.categorical_columns(sex),
            learning_rate,
            depth,
            iterations,
            early_stopping_rounds,
            use_best_model,
            random_seed,
        )

    def fit_and_predict_classifier(
        self,
        train_frame,
        test_frame,
        sex,
        learning_rate=DEFAULT_LEARNING_RATE,
        depth=DEFAULT_DEPTH,
        iterations=500,
        early_stopping_rounds=None,
        use_best_model=None,
        random_seed=42,
    ):
        """Fit a participation classifier with the feature columns of ``sex``."""
        return self._fit(
            KINDS["classifier"],
            train_frame,
            test_frame,
            self.ds.feature_columns(sex),
            self.ds.categorical_columns(sex),
            learning_rate,
            depth,
            iterations,
            early_stopping_rounds,
            use_best_model,
            random_seed,
        )

    # ------------------------------------------------------------- one command-unit draw
    def _run_once(
        self,
        kind: _Kind,
        ds,
        seed,
        iterations,
        learning_rate,
        depth,
        early_stopping_rounds,
        use_best_model,
        sex_label,
        random_seed=42,
    ):
        if sex_label not in (None, *SEXES):
            raise ValueError(f"sex_label must be None, 'male' or 'female', got {sex_label!r}")
        ds = self.ds if ds is None else ds
        store = self._stores[kind.name]

        results, preds = {}, {}
        for sex in SEXES if sex_label is None else (sex_label,):
            feature_cols = ds.feature_columns(sex)
            cats = ds.categorical_columns(sex)
            frames = {
                test_data: ds.realization(
                    sex, seed, test_data=test_data, zero_handling=kind.zero_handling
                ).frame
                for test_data in (False, True)
            }
            train_frame, test_frame = frames[False], frames[True]

            model, pred = self._fit(
                kind,
                train_frame,
                test_frame,
                feature_cols,
                cats,
                learning_rate,
                depth,
                iterations,
                early_stopping_rounds,
                use_best_model,
                random_seed,
            )
            results[sex] = kind.evaluate(test_frame, pred, sex)
            y_test = test_frame[kind.target].to_numpy()
            preds[sex] = (pred, y_test)
            store[sex] = {
                "model": model,
                "X_test": test_frame[feature_cols],
                "cat_features": cats,
                "y_test": y_test,
                "y_pred": pred,
            }

        gap = (
            gender_gap(
                preds["male"][0], preds["female"][0], preds["male"][1], preds["female"][1]
            )
            if sex_label is None
            else {}
        )
        return results, gap, store

    def run_once(
        self,
        ds,
        seed,
        iterations=500,
        learning_rate=DEFAULT_LEARNING_RATE,
        depth=DEFAULT_DEPTH,
        early_stopping_rounds=None,
        use_best_model=None,
        sex_label=None,
        random_seed=42,
    ):
        """Train and evaluate the wage regressor(s) for one draw. ``ds=None`` uses ``self.ds``.

        ``random_seed`` is CatBoost's own seed, independent of the command-unit draw ``seed``.
        """
        return self._run_once(
            KINDS["regressor"],
            ds,
            seed,
            iterations,
            learning_rate,
            depth,
            early_stopping_rounds,
            use_best_model,
            sex_label,
            random_seed,
        )

    def run_once_classifier(
        self,
        ds,
        seed,
        iterations=500,
        learning_rate=DEFAULT_LEARNING_RATE,
        depth=DEFAULT_DEPTH,
        early_stopping_rounds=None,
        use_best_model=None,
        sex_label=None,
        random_seed=42,
    ):
        """Train and evaluate the participation classifier(s) for one draw."""
        return self._run_once(
            KINDS["classifier"],
            ds,
            seed,
            iterations,
            learning_rate,
            depth,
            early_stopping_rounds,
            use_best_model,
            sex_label,
            random_seed,
        )

    # ---------------------------------------------------------------- hyperparameters
    def _hyperparameters(self, kind: str, sex: str, seed: int, lr, depth):
        """Tuned ``(lr, depth)`` of this seed, or the given defaults if there are none."""
        table = self.hp_parameters[_hp_key(kind, sex)]
        if table is not None and seed in table.index:
            row = table.loc[seed]
            if "lr" in row and pd.notna(row["lr"]):
                lr = float(row["lr"])
            if "depth" in row and pd.notna(row["depth"]):
                depth = int(row["depth"])
        return lr, depth

    def _describe_hyperparameters(self, kind: _Kind, seeds, lr, depth) -> str:
        """One line on where the ``lr`` / ``depth`` of each model of ``kind`` come from."""
        parts = []
        for key in kind.hp_keys:
            table = self.hp_parameters[key]
            n_tuned = 0 if table is None else sum(s in table.index for s in seeds)
            source = (
                f"tuned for {n_tuned}/{len(seeds)} seeds, defaults for the rest"
                if n_tuned
                else "defaults"
            )
            parts.append(f"{key}: {source}")
        return f"{'; '.join(parts)} (defaults: lr={lr}, depth={depth})"

    def _tuned_seeds(self, keys) -> list[int]:
        """Seeds of all valid hyperparameter tables among ``keys``, sorted."""
        seeds = set()
        for key in keys:
            if self.hp_parameters[key] is not None:
                seeds |= set(self.hp_parameters[key].index)
        return sorted(seeds)

    def _warn_about_hyperparameters(self, keys, seeds):
        """Warn if the hyperparameter files of ``keys`` are incomplete for ``seeds``.

        Silent if none of the ``keys`` has a hyperparameter path, i.e. if the user
        deliberately trains with the default hyperparameters.
        """
        provided = [key for key in keys if self.hp_paths[key] is not None]
        if not provided:
            return

        def warn(message):
            warnings.warn(message, HyperparameterWarning, stacklevel=4)

        for key in provided:
            if self.hp_parameters[key] is None:
                warn(
                    f"hyperparameter file for '{key}' not found: {self.hp_paths[key]}; "
                    "using default hyperparameters"
                )
        not_provided = [key for key in keys if key not in provided]
        if not_provided:
            warn(
                f"no hyperparameter path given for {not_provided} while one is given for "
                f"{provided}; using default hyperparameters for {not_provided}"
            )

        valid = {k: self.hp_parameters[k] for k in keys if self.hp_parameters[k] is not None}
        if not valid:
            return
        absent_everywhere = [
            s for s in seeds if all(s not in table.index for table in valid.values())
        ]
        if absent_everywhere:
            warn(
                f"seeds {absent_everywhere} are not in the hyperparameter file(s) for "
                f"{list(valid)}; using default hyperparameters for them"
            )
        partial = {}
        for s in seeds:
            present = [k for k, table in valid.items() if s in table.index]
            if 0 < len(present) < len(valid):
                partial[s] = (present, [k for k in valid if k not in present])
        if partial:
            details = "; ".join(
                f"seed {s}: in {present}, missing in {missing}"
                for s, (present, missing) in partial.items()
            )
            warn(
                f"seeds are only in some of the hyperparameter files ({details}); "
                "using default hyperparameters where they are missing"
            )

    # ------------------------------------------------------------------ training runs
    def _train_kind(
        self,
        kind: _Kind,
        ds,
        seeds,
        model_dir: Path,
        iterations,
        lr,
        depth,
        early_stopping_rounds,
        random_seed,
    ) -> TrainingResult:
        for sex in SEXES:
            (model_dir / sex).mkdir(exist_ok=True, parents=True)

        n_reps = len(seeds)
        print_every = max(1, n_reps // 20)  # ~20 status lines, regardless of n_reps
        print(f"[{kind.name}s] hyperparameters: {self._describe_hyperparameters(kind, seeds, lr, depth)}")
        start_time = time.time()
        records, all_runs, gap_runs = [], [], []
        fit_trees = {sex: [] for sex in SEXES}  # trees kept after early stopping / best model
        fit_seconds = []

        for rep, seed in enumerate(
            tqdm(seeds, desc=f"{kind.name}s: command-unit reassignment reps")
        ):
            per_sex = {}
            for sex in SEXES:
                sex_lr, sex_depth = self._hyperparameters(
                    kind.name, sex, seed, lr, depth
                )
                model_start = time.perf_counter()
                res, _, store = self._run_once(
                    kind,
                    ds,
                    seed,
                    iterations,
                    sex_lr,
                    sex_depth,
                    early_stopping_rounds,
                    True,
                    sex,
                    random_seed,
                )
                fit_seconds.append(time.perf_counter() - model_start)
                fit_trees[sex].append(store[sex]["model"].tree_count_)
                path = model_dir / sex / f"model_{rep:03d}.cbm"
                store[sex]["model"].save_model(str(path))
                per_sex[sex] = (res[sex], path)
                all_runs.append(res[sex])

            records.append(kind.record(rep, seed, per_sex))
            models = self._stores[kind.name]
            gap_runs.append(
                gender_gap(
                    models["male"]["y_pred"],
                    models["female"]["y_pred"],
                    models["male"]["y_test"],
                    models["female"]["y_test"],
                )
            )

            if (rep + 1) % print_every == 0 or rep + 1 == n_reps:
                elapsed = time.time() - start_time
                eta_min = elapsed / (rep + 1) * (n_reps - rep - 1) / 60
                running = pd.DataFrame(all_runs)
                medians = running.groupby("label")[kind.headline].median()
                print(
                    f"[{kind.name} rep {rep + 1}/{n_reps}]  elapsed={elapsed / 60:.1f} min  "
                    f"est. remaining={eta_min:.1f} min  running median {kind.headline}: "
                    f"male={medians['male']:.3f}  female={medians['female']:.3f}  |  "
                    f"mean trees: male={np.mean(fit_trees['male']):.0f} "
                    f"female={np.mean(fit_trees['female']):.0f}  "
                    f"mean {np.mean(fit_seconds):.1f} s/model"
                )

        metrics_df = pd.DataFrame(records)
        metrics_df.to_csv(model_dir / "model_metrics.csv", index=False)
        result = TrainingResult(metrics_df, pd.DataFrame(all_runs), pd.DataFrame(gap_runs))
        self._print_summary(kind, result, n_reps)
        return result

    @staticmethod
    def _print_summary(kind: _Kind, result: TrainingResult, n_reps: int):
        print(f"\n=== {kind.name}s: {n_reps} repetitions, percentile summary ===\n")
        for sex in SEXES:
            sub = result.runs[result.runs["label"] == sex]
            print(f"--- {sex} ---")
            print(
                sub[kind.summary_cols]
                .describe(percentiles=[0.05, 0.5, 0.95])
                .loc[["5%", "50%", "95%"]]
            )
            print()
        print("--- predicted vs. observed gender gap across repetitions ---")
        print(result.gaps.describe(percentiles=[0.05, 0.5, 0.95]).loc[["5%", "50%", "95%"]])

    def run_models(
        self,
        ds=None,
        seeds=None,
        kind=None,
        model_dir=None,
        iterations=5000,
        lr=DEFAULT_LEARNING_RATE,
        depth=DEFAULT_DEPTH,
        early_stopping_rounds=200,
        random_seed=42,
    ) -> dict[str, TrainingResult]:
        """Train a model per seed and sex, save them and ``model_metrics.csv``.

        ``kind`` is ``"regressor"``, ``"classifier"`` or ``None`` for both. ``seeds=None``
        takes all seeds of the hyperparameter files of the models being trained. Each seed
        uses its tuned ``lr`` / ``depth`` if available, else the ``lr`` / ``depth`` given here.
        ``random_seed`` is CatBoost's own seed, the same for every model; ``seeds`` only set
        the command-unit draws.
        With ``kind=None`` the models are saved in ``<model_dir>/regressors`` and
        ``<model_dir>/classifiers``; the default folder is ``models/hp-tuned/<kind>s-<time>``.
        Warns (:class:`HyperparameterWarning`) if hyperparameter files are given but do not
        cover the models or seeds; no warnings if no hyperparameter paths are given.
        Returns a :class:`TrainingResult` per trained kind.
        """
        if kind is not None and kind not in KINDS:
            raise ValueError(f"kind must be None, 'regressor' or 'classifier', got {kind!r}")
        kinds = list(KINDS.values()) if kind is None else [KINDS[kind]]
        ds = self.ds if ds is None else ds
        keys = [key for k in kinds for key in k.hp_keys]

        if seeds is None:
            seeds = self._tuned_seeds(keys)
            if not seeds:
                raise ValueError(
                    "no seeds given and no valid hyperparameter file to take them from"
                )
        seeds = [int(s) for s in seeds]
        self._warn_about_hyperparameters(keys, seeds)

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        results = {}
        for k in kinds:
            if model_dir is None:
                folder = REPO_ROOT / "models/hp-tuned" / f"{k.name}s-{stamp}"
            elif len(kinds) == 1:
                folder = _resolve(model_dir)
            else:
                folder = _resolve(model_dir) / f"{k.name}s"
            results[k.name] = self._train_kind(
                k,
                ds,
                seeds,
                folder,
                iterations,
                lr,
                depth,
                early_stopping_rounds,
                random_seed,
            )
        return results
