import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from scipy import stats as sstats
from catboost import CatBoostRegressor, CatBoostClassifier

REPO_ROOT = Path.cwd().parent
# make the shared code in src/ importable even when the package is not pip-installed
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from egypt_survey_ml.config import DatasetConfig
from egypt_survey_ml.dataset import prepare_dataset

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


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


class ModelTrainer:
    def __init__(
        self, ds, feature_cols, cats, male_hp_path, female_hp_path, cls_hp_path
    ):
        self.ds = ds
        self.feature_cols = feature_cols
        self.cats = cats

        self.MODELS = {}
        self.MODELS_CLASSIFIER = {}

        hp_reg_male_df = pd.read_csv(REPO_ROOT / male_hp_path)
        hp_reg_female_df = pd.read_csv(REPO_ROOT / female_hp_path)
        hp_cls_df = pd.read_csv(REPO_ROOT / cls_hp_path)

        self.hp_parameters = {
            "male": hp_reg_male_df,
            "female": hp_reg_female_df,
            "classifier": hp_cls_df,
        }

    def fit_and_predict(
        self,
        train_frame,
        test_frame,
        learning_rate=0.05,
        depth=6,
        iterations=500,
        early_stopping_rounds=None,
        use_best_model=None,
    ):
        # Fit directly on raw wage (the fix -- see the title cell). No log1p/expm1 round-trip:
        # that round-trip was diagnosed as the main source of the systematic under-prediction
        # in train-catboost_v2_rural.ipynb (predictions ~1/5 to 1/12 of actual paid-mean wage).
        X_train = train_frame[self.feature_cols]
        y_train = train_frame["totwag_all"]
        X_test = test_frame[self.feature_cols]
        y_test = test_frame["totwag_all"]
        model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function="RMSE",
            random_seed=42,
            verbose=False,
        )
        model.fit(
            X_train,
            y_train,
            cat_features=self.cats,
            eval_set=(X_train, y_train),
            use_best_model=use_best_model,
            early_stopping_rounds=early_stopping_rounds,
        )
        pred_wage = model.predict(X_test)
        return model, np.clip(
            pred_wage, 0, None
        )  # wage can't be negative; CatBoost's raw output can be

    def fit_and_predict_classifier(
        self,
        train_frame,
        test_frame,
        learning_rate=0.05,
        depth=6,
        iterations=500,
        early_stopping_rounds=None,
        use_best_model=None,
    ):
        X_train = train_frame[self.feature_cols]
        y_train = train_frame["is_paid"]
        X_test = test_frame[self.feature_cols]
        y_test = test_frame["is_paid"]
        model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=42,
            verbose=False,
        )
        model.fit(
            X_train,
            y_train,
            cat_features=self.cats,
            eval_set=(X_train, y_train),
            use_best_model=use_best_model,
            early_stopping_rounds=early_stopping_rounds,
        )
        pred_paid = model.predict(X_test)
        return model, pred_paid

    def run_once(
        self,
        ds,
        seed,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        early_stopping_rounds=None,
        use_best_model=None,
        sex_label=None,
    ):
        if sex_label is None:
            model_options = [(1, "male"), (2, "female")]
        elif sex_label == "male":
            model_options = [(1, "male")]
        elif sex_label == "female":
            model_options = [(2, "female")]

        results = {}
        preds = {}
        for sex_code, sex_label in model_options:
            train_frame = ds.realization(sex_label, seed, test_data=False).frame
            test_frame = ds.realization(sex_label, seed, test_data=True).frame

            model, pred_wage = self.fit_and_predict(
                train_frame,
                test_frame,
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                early_stopping_rounds=early_stopping_rounds,
                use_best_model=use_best_model,
            )
            m = evaluate(
                test_frame["totwag_all"].to_numpy(),
                pred_wage,
                test_frame["is_paid"].to_numpy(),
                label=sex_label,
            )
            results[sex_label] = m
            preds[sex_label] = (pred_wage, test_frame["totwag_all"].to_numpy())
            self.MODELS[sex_label] = {
                "model": model,
                "X_test": test_frame[self.feature_cols],
                "cat_features": self.cats,
                "y_test": test_frame["totwag_all"].to_numpy(),
                "y_pred": pred_wage,
            }

        if sex_label is None:
            gap = gender_gap(
                preds["male"][0],
                preds["female"][0],
                preds["male"][1],
                preds["female"][1],
            )
        else:
            gap = {}
        return results, gap, self.MODELS

    def run_once_classifier(
        self,
        ds,
        seed,
        iterations=500,
        learning_rate=0.05,
        depth=6,
        early_stopping_rounds=None,
        use_best_model=None,
        sex_label=None,
    ):
        if sex_label is None:
            model_options = [(1, "male"), (2, "female")]
        elif sex_label == "male":
            model_options = [(1, "male")]
        elif sex_label == "female":
            model_options = [(2, "female")]

        results = {}
        preds = {}
        for sex_code, sex_label in model_options:
            train_frame = ds.realization(sex_label, seed, test_data=False).frame
            test_frame = ds.realization(sex_label, seed, test_data=True).frame

            model, pred_paid = self.fit_and_predict_classifier(
                train_frame,
                test_frame,
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                early_stopping_rounds=early_stopping_rounds,
                use_best_model=use_best_model,
            )
            m = evaluate_participation(
                test_frame["is_paid"].to_numpy(), pred_paid, label=sex_label
            )
            results[sex_label] = m
            preds[sex_label] = (pred_paid, test_frame["is_paid"].to_numpy())
            self.MODELS_CLASSIFIER[sex_label] = {
                "model": model,
                "X_test": test_frame[self.feature_cols],
                "cat_features": self.cats,
                "y_test": test_frame["is_paid"].to_numpy(),
                "y_pred": pred_paid,
            }

        if sex_label is None:
            gap = gender_gap(
                preds["male"][0],
                preds["female"][0],
                preds["male"][1],
                preds["female"][1],
            )
        else:
            gap = {}
        return results, gap, self.MODELS_CLASSIFIER

    def run_models(
        self,
        ds,
        seeds,
        sex_label=None,
        model_dir=None,
        iterations=5000,
        lr=0.05,
        depth=6,
    ):
        records = []
        all_runs = []
        gap_runs = []
        N_REPS = len(seeds)
        start_time = time.time()
        PRINT_EVERY = max(
            1, N_REPS // 20
        )  # ~20 status lines total, regardless of N_REPS
        if model_dir is None:
            MODEL_DIR = REPO_ROOT / f"models/hp-tuned/regressors-{start_time}"
        else:
            MODEL_DIR = Path(model_dir)
            if not MODEL_DIR.is_absolute():
                MODEL_DIR = REPO_ROOT / MODEL_DIR

        MALE_MODELS = MODEL_DIR / "male"
        FEMALE_MODELS = MODEL_DIR / "female"

        MALE_MODELS.mkdir(exist_ok=True, parents=True)
        FEMALE_MODELS.mkdir(exist_ok=True, parents=True)
        results = {}
        for sex_label in ["male", "female"]:
            hp_reg_df = self.hp_parameters[sex_label]
            for i, row in tqdm(
                hp_reg_df.iterrows(),
                desc="command-unit reassignment reps",
                total=len(hp_reg_df),
            ):
                seed = int(row["seed"])
                depth = row["depth"]
                lr = row["lr"] if "lr" in row else lr
                depth = row["depth"] if "depth" in row else depth

                res, gap, models = self.run_once(
                    seed=seed,
                    iterations=iterations,
                    learning_rate=lr,
                    depth=depth,
                    early_stopping_rounds=200,
                    use_best_model=True,
                    sex_label=sex_label,
                )
                model = models[sex_label]["model"]
                model_path = MODEL_DIR / sex_label / f"model_{i:03d}.cbm"
                model.save_model(model_path)

                results.setdefault(i, {})
                results[i][sex_label] = {
                    "r2": res[sex_label]["r2"],
                    "model_file": str(model_path),
                    "seed": seed,
                }

                if len(results[i]) == 2:
                    records.append(
                        {
                            "rep": i,
                            "seed": seed,
                            "male_r2": results[i]["male"]["r2"],
                            "female_r2": results[i]["female"]["r2"],
                            "male_model_file": results[i]["male"]["model_file"],
                            "female_model_file": results[i]["female"]["model_file"],
                        }
                    )
                for sex_label, m in res.items():
                    all_runs.append(m)
                gap_runs.append(gap)

                if (i + 1) % PRINT_EVERY == 0 or (i + 1) == N_REPS:
                    elapsed = time.time() - start_time
                    rate = elapsed / (i + 1)
                    eta_min = rate * (N_REPS - (i + 1)) / 60
                    running = pd.DataFrame(all_runs)
                    male_r2 = running.loc[running["label"] == "male", "r2"].median()
                    female_r2 = running.loc[running["label"] == "female", "r2"].median()
                    print(
                        f"[rep {i + 1}/{N_REPS}]  elapsed={elapsed/60:.1f} min  "
                        f"est. remaining={eta_min:.1f} min  "
                        f"running median R²: male={male_r2:.3f}  female={female_r2:.3f}"
                    )

        metrics_df = pd.DataFrame(records)
        metrics_df.to_csv(MODEL_DIR / "model_metrics.csv", index=False)

        runs_df = pd.DataFrame(all_runs)
        gaps_df = pd.DataFrame(gap_runs)


import time

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x  # progress bar is a nicety, not a dependency

records = []
all_runs = []
gap_runs = []
start_time = time.time()
PRINT_EVERY = max(1, N_REPS // 20)  # ~20 status lines total, regardless of N_REPS
MODEL_DIR = REPO_ROOT / "models/hp-tuned/no-ribasim/classifiers-100"
MALE_MODELS = MODEL_DIR / "male"
FEMALE_MODELS = MODEL_DIR / "female"

MALE_MODELS.mkdir(exist_ok=True, parents=True)
FEMALE_MODELS.mkdir(exist_ok=True, parents=True)

for i, row in tqdm(
    hp_cls_df.iterrows(), desc="command-unit reassignment reps", total=len(hp_cls_df)
):
    seed = int(row["seed"])
    depth = row["depth"]
    lr = row["lr"]

    res, gap, models = run_once_classifier(
        seed=seed,
        iterations=5000,
        learning_rate=lr,
        depth=depth,
        early_stopping_rounds=200,
        use_best_model=True,
    )
    for sex_label in ["male", "female"]:
        model = models[sex_label]["model"]
        model_path = MODEL_DIR / sex_label / f"model_{i:03d}.cbm"
        model.save_model(model_path)

    records.append(
        {
            "rep": i,
            "seed": seed,
            "male_accuracy": res["male"]["accuracy"],
            "female_accuracy": res["female"]["accuracy"],
            "model_file": str(model_path),
        }
    )
    for sex_label, m in res.items():
        all_runs.append(m)
    gap_runs.append(gap)

    if (i + 1) % PRINT_EVERY == 0 or (i + 1) == N_REPS:
        elapsed = time.time() - start_time
        rate = elapsed / (i + 1)
        eta_min = rate * (N_REPS - (i + 1)) / 60
        running = pd.DataFrame(all_runs)
        male_acc = running.loc[running["label"] == "male", "accuracy"].median()
        female_acc = running.loc[running["label"] == "female", "accuracy"].median()
        print(
            f"[rep {i + 1}/{N_REPS}]  elapsed={elapsed/60:.1f} min  "
            f"est. remaining={eta_min:.1f} min  "
            f"running median accuracy: male={male_r2:.3f}  female={female_r2:.3f}"
        )

metrics_df = pd.DataFrame(records)
metrics_df.to_csv(MODEL_DIR / "model_metrics.csv", index=False)

runs_df = pd.DataFrame(all_runs)
gaps_df = pd.DataFrame(gap_runs)

print(f"\n=== {N_REPS} repetitions, percentile summary ===\n")
for sex_label in ["male", "female"]:
    sub = runs_df[runs_df["label"] == sex_label]
    print(f"--- {sex_label} ---")
    print(
        sub[["mae", "rmse", "r2", "spearman"]]
        .describe(percentiles=[0.05, 0.5, 0.95])
        .loc[["5%", "50%", "95%"]]
    )
    print()

print("--- predicted vs. observed gender gap across repetitions ---")
print(gaps_df.describe(percentiles=[0.05, 0.5, 0.95]).loc[["5%", "50%", "95%"]])
