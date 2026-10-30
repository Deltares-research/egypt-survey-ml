"""Shared CatBoost training and evaluation helpers extracted from the tuned notebook."""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
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
from scipy import stats as sstats

LAST_MODELS = {}
LAST_MODELS_CLASSIFIER = {}


def _resolve_from_caller(name, default=None):
    frame = inspect.currentframe()
    try:
        caller = frame.f_back
        while caller is not None:
            if name in caller.f_locals:
                return caller.f_locals[name]
            if name in caller.f_globals:
                return caller.f_globals[name]
            caller = caller.f_back
    finally:
        del frame
    return default


def convert_irregular(daily_wage, ind_code, real_hours, avg_hours_by_ind=None):
    avg_hours_by_ind = (
        avg_hours_by_ind
        if avg_hours_by_ind is not None
        else _resolve_from_caller("avg_hours_by_ind")
    )
    if avg_hours_by_ind is None:
        raise ValueError(
            "avg_hours_by_ind is required to convert irregular wage values."
        )

    used_real = real_hours.notna() & (real_hours > 0)
    hours = real_hours.where(used_real, ind_code.map(avg_hours_by_ind))
    return daily_wage * (hours / 8) * 4


def attach_household_context(
    pool,
    household_size=None,
    n_employed_in_hh=None,
    head_info=None,
    hh=None,
    hh_wage=None,
):
    household_size = (
        household_size
        if household_size is not None
        else _resolve_from_caller("household_size")
    )
    n_employed_in_hh = (
        n_employed_in_hh
        if n_employed_in_hh is not None
        else _resolve_from_caller("n_employed_in_hh")
    )
    head_info = (
        head_info if head_info is not None else _resolve_from_caller("head_info")
    )
    hh = hh if hh is not None else _resolve_from_caller("hh")
    hh_wage = hh_wage if hh_wage is not None else _resolve_from_caller("hh_wage")

    pool = pool.copy()
    pool["household_size"] = pool["caseser"].map(household_size)
    pool["n_other_employed"] = pool["caseser"].map(n_employed_in_hh) - 1
    pool = pool.merge(head_info, on="caseser", how="left")
    pool = pool.merge(
        hh[["caseser", "malinlf", "feminlf", "occhd"]], on="caseser", how="left"
    )
    pool = pool.merge(
        hh_wage[["caseser", "household_wage_total", "wage_ratio"]],
        on="caseser",
        how="left",
    )
    return pool


def build_wage_target(pool):
    pool = pool.copy()
    valid_hours = pool["hrswk"].notna() & (pool["hrswk"] > 0)
    pool = pool[valid_hours].copy()

    if "immigr" in pool.columns:
        pool["immigr"] = pool["immigr"].fillna(3)
    if "disabl" in pool.columns:
        pool["disabl"] = pool["disabl"].fillna(0)
    if "mart_d" in pool.columns:
        n_before = len(pool)
        pool = pool.dropna(subset=["mart_d"])
        print(f"  dropped {n_before - len(pool)} rows missing mart_d")

    pool["sempinc"] = pool["sempinc"].fillna(0)
    pool["irrgwag"] = pool["irrgwag"].fillna(0)
    pool["totwag"] = pool["totwag"].fillna(0)
    pool["totwag_all"] = pool["totwag"] + pool["sempinc"] + pool["irrgwag"] * 22

    n_before = len(pool)
    pool = pool[pool["totwag_all"] < 10000].copy()
    print(
        f"  dropped {n_before - len(pool)} rows with totwag_all >= 10,000 EGP (outlier cap)"
    )

    pool["is_paid"] = (pool["totwag_all"] > 0).astype(int)
    pool["log1p_wage"] = np.log1p(pool["totwag_all"])
    return pool


def attach_governorate(pool, crosswalk=None):
    crosswalk = (
        crosswalk if crosswalk is not None else _resolve_from_caller("crosswalk")
    )
    pool = pool.merge(
        crosswalk[["reg_code", "OBJECTID", "NAME1_"]],
        left_on="reg",
        right_on="reg_code",
        how="left",
    )
    return pool


def assign_command_units(pool, seed, distribution_df=None):
    distribution_df = (
        distribution_df
        if distribution_df is not None
        else _resolve_from_caller("distribution_df")
    )
    rng = np.random.default_rng(seed)
    pool = pool.copy()
    pool["command_unit"] = None
    command_units = distribution_df.columns
    for gov, idx in pool.groupby("NAME1_").groups.items():
        if gov not in distribution_df.index:
            continue
        probs = distribution_df.loc[gov].to_numpy()
        pool.loc[idx, "command_unit"] = rng.choice(
            command_units, size=len(idx), p=probs
        )
    pool = pool[pool["command_unit"] != "other"]
    return pool[pool["command_unit"].notna()].copy()


def drop_random_zeros(frame, frac, seed):
    rng = np.random.default_rng(seed)
    zero_idx = frame.index[frame["is_paid"] == 0]
    n_drop = int(len(zero_idx) * frac)
    drop_idx = rng.choice(zero_idx, size=n_drop, replace=False)
    kept = frame.drop(index=drop_idx)
    return kept


def perturbate_wage(frame, seed, mu=0.0, sigma=None):
    sigma = (
        sigma
        if sigma is not None
        else _resolve_from_caller("WAGE_PERTURBATION_SIGMA", 0.0)
    )
    rng = np.random.default_rng(seed)
    wage = frame["totwag_all"]
    new_wage = wage * (1 + rng.normal(mu, sigma, size=wage.shape))
    frame["totwag_all"] = new_wage
    return frame


def lognormal_synthetic_augment(
    frame,
    feature_cols,
    cat_cols,
    n_synthetic,
    k=5,
    seed=42,
    geo_cols=None,
    excel_df=None,
):
    geo_cols = geo_cols if geo_cols is not None else _resolve_from_caller("GEO_COLS")
    excel_df = excel_df if excel_df is not None else _resolve_from_caller("excel_df")

    if geo_cols is None:
        geo_cols = ["command_unit"] + [
            c for c in (excel_df.columns if excel_df is not None else []) if c != "area"
        ]

    rng = np.random.default_rng(seed)
    paid = frame[frame["is_paid"] == 1]
    if len(paid) < 5:
        raise ValueError("too few real paid rows to fit a conditional log-normal")

    overall_resid_std = (
        paid["log1p_wage"]
        - paid.groupby("command_unit")["log1p_wage"].transform("median")
    ).std()
    cu_median_log_wage = paid.groupby("command_unit")["log1p_wage"].median()
    global_median_log_wage = paid["log1p_wage"].median()

    donor_pool = frame[frame["is_paid"] == 0].reset_index(drop=True)
    geo_cols_present = [c for c in geo_cols if c in donor_pool.columns]
    num_cols = [
        c for c in feature_cols if c not in cat_cols and c not in geo_cols_present
    ]

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    synth_rows = []
    for cu, group in donor_pool.groupby("command_unit"):
        if len(group) < 2:
            continue
        X_num = group[num_cols].fillna(np.nanmedian(group[num_cols])).to_numpy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_num)
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(group))).fit(X_scaled)
        _, neighbor_idx = nn.kneighbors(X_scaled)

        n_this_cu = max(1, round(n_synthetic * len(group) / len(donor_pool)))
        for _ in range(n_this_cu):
            i = rng.integers(0, len(group))
            j = rng.choice(neighbor_idx[i][1:]) if len(group) > 1 else i
            lam = rng.uniform(0, 1)
            row = group.iloc[i if lam < 0.5 else j].copy()
            row[num_cols] = group.iloc[i][num_cols].to_numpy() + lam * (
                group.iloc[j][num_cols].to_numpy() - group.iloc[i][num_cols].to_numpy()
            )
            mu = cu_median_log_wage.get(cu, global_median_log_wage)
            synth_log_wage = mu + rng.normal(0, overall_resid_std)
            row["log1p_wage"] = max(synth_log_wage, 0.0)
            row["totwag_all"] = np.expm1(row["log1p_wage"])
            row["is_paid"] = 1
            synth_rows.append(row)

    synth_df = pd.DataFrame(synth_rows)
    print(
        f"  lognormal_synthetic_augment: generated {len(synth_df)} synthetic paid rows "
        f"\nfrom {donor_pool['command_unit'].nunique()} command units"
        f"\ntotal amount of orginal rows: {len(frame)} ({len(synth_df) / len(frame) * 100:.1f}% synthetic data)"
    )
    return pd.concat([frame, synth_df], ignore_index=True)


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


def build_feature_frame(
    pool_with_geo,
    sex_code,
    base_cols=None,
    MALE_EXTRA=None,
    FEMALE_EXTRA=None,
    categorical_columns=None,
    excel_df=None,
):
    base_cols = (
        base_cols if base_cols is not None else _resolve_from_caller("base_cols")
    )
    MALE_EXTRA = (
        MALE_EXTRA if MALE_EXTRA is not None else _resolve_from_caller("MALE_EXTRA")
    )
    FEMALE_EXTRA = (
        FEMALE_EXTRA
        if FEMALE_EXTRA is not None
        else _resolve_from_caller("FEMALE_EXTRA")
    )
    categorical_columns = (
        categorical_columns
        if categorical_columns is not None
        else _resolve_from_caller("categorical_columns")
    )
    excel_df = excel_df if excel_df is not None else _resolve_from_caller("excel_df")

    cols = base_cols + (MALE_EXTRA if sex_code == 1 else FEMALE_EXTRA)
    frame = (
        pool_with_geo[pool_with_geo["sex"] == sex_code]
        .merge(excel_df, left_on="command_unit", right_on="area", how="inner")
        .drop(columns=["area"])
    )
    keep = [c for c in cols if c in frame.columns] + [
        "log1p_wage",
        "totwag_all",
        "is_paid",
        "command_unit",
    ]
    frame = frame[keep].dropna(subset=["log1p_wage"])
    cats = [c for c in categorical_columns if c in frame.columns]
    frame[cats] = frame[cats].fillna("missing").astype(str)
    return frame, cats


def apply_zero_handling(
    frame, cat_cols, feature_cols, mode, seed, DROP_RANDOM_FRAC=None
):
    if mode == "none":
        return frame
    if mode == "drop_random":
        drop_frac = (
            DROP_RANDOM_FRAC
            if DROP_RANDOM_FRAC is not None
            else _resolve_from_caller("DROP_RANDOM_FRAC")
        )
        return drop_random_zeros(frame, drop_frac, seed)
    if mode == "lognormal_synthetic":
        n_zero = (frame["is_paid"] == 0).sum()
        n_pos = (frame["is_paid"] == 1).sum()
        n_synthetic = max(0, n_zero - n_pos)
        return lognormal_synthetic_augment(
            frame, feature_cols, cat_cols, n_synthetic, seed=seed
        )
    raise ValueError(mode)


def fit_and_predict(
    train_frame,
    test_frame,
    feature_cols,
    cat_cols,
    iterations=500,
    learning_rate=0.05,
    depth=6,
    early_stopping_rounds=None,
    use_best_model=None,
    random_seed=None,
):
    random_seed = (
        random_seed if random_seed is not None else _resolve_from_caller("RNG_SEED", 42)
    )
    X_train = train_frame[feature_cols]
    y_train = train_frame["totwag_all"]
    X_test = test_frame[feature_cols]
    y_test = test_frame["totwag_all"]
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        random_seed=random_seed,
        verbose=False,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_test, y_test),
        use_best_model=use_best_model,
        early_stopping_rounds=early_stopping_rounds,
    )
    pred_wage = model.predict(X_test)
    return model, np.clip(pred_wage, 0, None)


def fit_and_predict_classifier(
    train_frame,
    test_frame,
    feature_cols,
    cat_cols,
    iterations=500,
    learning_rate=0.05,
    depth=6,
    early_stopping_rounds=None,
    use_best_model=None,
    random_seed=None,
):
    random_seed = (
        random_seed if random_seed is not None else _resolve_from_caller("RNG_SEED", 42)
    )
    X_train = train_frame[feature_cols]
    y_train = train_frame["is_paid"]
    X_test = test_frame[feature_cols]
    y_test = test_frame["is_paid"]
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_seed=random_seed,
        verbose=False,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_cols,
        eval_set=(X_test, y_test),
        use_best_model=use_best_model,
        early_stopping_rounds=early_stopping_rounds,
    )
    pred_paid = model.predict(X_test)
    return model, pred_paid


def run_once(
    seed,
    iterations=500,
    learning_rate=0.05,
    depth=6,
    early_stopping_rounds=None,
    use_best_model=None,
    sex_label=None,
):
    train_pool = _resolve_from_caller("train_pool")
    test_pool = _resolve_from_caller("test_pool")
    survey_columns = _resolve_from_caller("survey_columns")
    INCLUDE_RIBASIM = _resolve_from_caller("INCLUDE_RIBASIM")
    WAGE_PERTURBATION = _resolve_from_caller("WAGE_PERTURBATION")
    PREDICT_LOG_WAGE = _resolve_from_caller("PREDICT_LOG_WAGE")
    ZERO_HANDLING = _resolve_from_caller("ZERO_HANDLING")
    RNG_SEED = _resolve_from_caller("RNG_SEED")

    train_geo = assign_command_units(train_pool, seed)
    test_geo = assign_command_units(test_pool, seed)
    if sex_label is None:
        model_options = [(1, "male"), (2, "female")]
    elif sex_label == "male":
        model_options = [(1, "male")]
    elif sex_label == "female":
        model_options = [(2, "female")]

    results = {}
    preds = {}
    for sex_code, sex_name in model_options:
        train_frame, cats = build_feature_frame(train_geo, sex_code)
        test_frame, _ = build_feature_frame(test_geo, sex_code)
        feature_cols = [
            c
            for c in train_frame.columns
            if c not in ("log1p_wage", "totwag_all", "is_paid", "command_unit")
        ]
        if not INCLUDE_RIBASIM:
            feature_cols = [c for c in feature_cols if c in survey_columns]
            cats = [c for c in cats if c in survey_columns]
        if WAGE_PERTURBATION:
            train_frame = perturbate_wage(train_frame, seed=seed)
            test_frame = perturbate_wage(test_frame, seed=seed)

        if PREDICT_LOG_WAGE:
            test_frame["totwag_all"] = np.log1p(test_frame["totwag_all"])
            train_frame["totwag_all"] = np.log1p(train_frame["totwag_all"])
        train_frame = apply_zero_handling(
            train_frame, cats, feature_cols, ZERO_HANDLING, seed
        )
        test_frame = apply_zero_handling(
            test_frame, cats, feature_cols, ZERO_HANDLING, seed
        )

        model, pred_wage = fit_and_predict(
            train_frame,
            test_frame,
            feature_cols,
            cats,
            learning_rate=learning_rate,
            depth=depth,
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=use_best_model,
            random_seed=RNG_SEED,
        )
        m = evaluate(
            test_frame["totwag_all"].to_numpy(),
            pred_wage,
            test_frame["is_paid"].to_numpy(),
            label=sex_name,
        )
        results[sex_name] = m
        preds[sex_name] = (pred_wage, test_frame["totwag_all"].to_numpy())
        LAST_MODELS[sex_name] = {
            "model": model,
            "X_test": test_frame[feature_cols],
            "cat_features": cats,
            "y_test": test_frame["totwag_all"].to_numpy(),
            "y_pred": pred_wage,
        }

    if sex_label is None:
        gap = gender_gap(
            preds["male"][0], preds["female"][0], preds["male"][1], preds["female"][1]
        )
    else:
        gap = {}
    return results, gap, LAST_MODELS


def run_once_classifier(
    seed,
    iterations=500,
    learning_rate=0.05,
    depth=6,
    early_stopping_rounds=None,
    use_best_model=None,
    sex_label=None,
):
    train_pool = _resolve_from_caller("train_pool")
    test_pool = _resolve_from_caller("test_pool")
    survey_columns = _resolve_from_caller("survey_columns")
    INCLUDE_RIBASIM = _resolve_from_caller("INCLUDE_RIBASIM")
    RNG_SEED = _resolve_from_caller("RNG_SEED")

    train_geo = assign_command_units(train_pool, seed)
    test_geo = assign_command_units(test_pool, seed)
    if sex_label is None:
        model_options = [(1, "male"), (2, "female")]
    elif sex_label == "male":
        model_options = [(1, "male")]
    elif sex_label == "female":
        model_options = [(2, "female")]

    results = {}
    preds = {}
    for sex_code, sex_name in model_options:
        train_frame, cats = build_feature_frame(train_geo, sex_code)
        test_frame, _ = build_feature_frame(test_geo, sex_code)
        feature_cols = [
            c
            for c in train_frame.columns
            if c not in ("log1p_wage", "totwag_all", "is_paid", "command_unit")
        ]
        if not INCLUDE_RIBASIM:
            feature_cols = [c for c in feature_cols if c in survey_columns]
            cats = [c for c in cats if c in survey_columns]

        model, pred_paid = fit_and_predict_classifier(
            train_frame,
            test_frame,
            feature_cols,
            cats,
            learning_rate=learning_rate,
            depth=depth,
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=use_best_model,
            random_seed=RNG_SEED,
        )
        m = evaluate_participation(
            test_frame["is_paid"].to_numpy(), pred_paid, label=sex_name
        )
        results[sex_name] = m
        preds[sex_name] = (pred_paid, test_frame["is_paid"].to_numpy())
        LAST_MODELS_CLASSIFIER[sex_name] = {
            "model": model,
            "X_test": test_frame[feature_cols],
            "cat_features": cats,
            "y_test": test_frame["is_paid"].to_numpy(),
            "y_pred": pred_paid,
        }

    if sex_label is None:
        gap = gender_gap(
            preds["male"][0], preds["female"][0], preds["male"][1], preds["female"][1]
        )
    else:
        gap = {}
    return results, gap, LAST_MODELS_CLASSIFIER
