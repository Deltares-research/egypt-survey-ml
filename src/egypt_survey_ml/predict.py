"""Ensemble inference for the wage models.

Each ensemble member is a classifier (P(paid)) and a regressor (wage if paid) that were
trained together on one random command-unit draw, identified by its seed. Prediction
must reuse that same draw, so a member is a ``(seed, classifier, regressor)`` triple.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.metrics import r2_score

STAT_NAMES = [
    "p5",
    "p10",
    "p20",
    "p30",
    "p40",
    "p45",
    "p50",
    "p55",
    "p60",
    "p70",
    "p80",
    "p90",
    "p95",
    "mean",
    "std",
    "min",
    "max",
]


@dataclass(frozen=True)
class EnsembleMember:
    """A classifier/regressor pair and the seed of the command-unit draw it was trained on.

    ``classifier`` / ``regressor`` are either paths to ``.cbm`` files (loaded on use)
    or already fitted CatBoost models.
    """

    seed: int
    classifier: Any
    regressor: Any


def load_members(
    classifier_dir, regressor_dir, sex, load=False, seeds=None
) -> list[EnsembleMember]:
    """Build the members for ``sex`` from the ``model_metrics.csv`` files of both model folders.

    The csv files give the seed of every model; classifier and regressor of the same ``rep``
    must have been trained with the same seed. Model files are resolved relative to the
    given folders (``<dir>/<sex>/<file name>``), not from the absolute paths in the csv.
    With ``load=True`` the models are loaded now, so they can be reused across calls.
    """
    classifier_dir, regressor_dir = Path(classifier_dir), Path(regressor_dir)
    cls_df = pd.read_csv(classifier_dir / "model_metrics.csv")
    reg_df = pd.read_csv(regressor_dir / "model_metrics.csv")
    if sorted(cls_df["rep"]) != sorted(reg_df["rep"]):
        raise ValueError("classifier and regressor metrics list different reps")

    if seeds is not None:
        cls_df = cls_df[cls_df["seed"].isin(seeds)]
        reg_df = reg_df[reg_df["seed"].isin(seeds)]

    merged = cls_df[["rep", "seed", "model_file"]].merge(
        reg_df[["rep", "seed", f"{sex}_model_file"]],
        on="rep",
        suffixes=("_cls", "_reg"),
        validate="one_to_one",
    )
    mismatch = merged[merged["seed_cls"] != merged["seed_reg"]]
    if not mismatch.empty:
        raise ValueError(
            f"classifier and regressor seeds differ for reps {mismatch['rep'].tolist()}"
        )

    members = []
    for row in merged.sort_values("rep").itertuples():
        cls_path = classifier_dir / sex / Path(row.model_file).name
        reg_path = regressor_dir / sex / Path(getattr(row, f"{sex}_model_file")).name
        for path in (cls_path, reg_path):
            if not path.exists():
                raise FileNotFoundError(path)
        member = EnsembleMember(int(row.seed_cls), cls_path, reg_path)
        members.append(_loaded(member) if load else member)
    return members


def _as_model(model, model_cls):
    if isinstance(model, (str, Path)):
        loaded = model_cls()
        loaded.load_model(str(model))
        return loaded
    return model


def _loaded(member: EnsembleMember) -> EnsembleMember:
    return EnsembleMember(
        member.seed,
        _as_model(member.classifier, CatBoostClassifier),
        _as_model(member.regressor, CatBoostRegressor),
    )


def describe(values, weights=None):
    """[p5, p50, p95, mean, std, min, max] (see STAT_NAMES); the mean is weighted if weights are given."""
    values = np.asarray(values, dtype=float)
    mean = np.mean(values) if weights is None else np.average(values, weights=weights)
    return [
        np.percentile(values, 5),
        np.percentile(values, 10),
        np.percentile(values, 20),
        np.percentile(values, 30),
        np.percentile(values, 40),
        np.percentile(values, 45),
        np.percentile(values, 50),
        np.percentile(values, 55),
        np.percentile(values, 60),
        np.percentile(values, 70),
        np.percentile(values, 80),
        np.percentile(values, 90),
        np.percentile(values, 95),
        mean,
        np.std(values),
        np.min(values),
        np.max(values),
    ]


@dataclass
class EnsemblePrediction:
    """Predictions of all members. 2D arrays are (n_members, n_respondents).

    Respondents are in the same order for every member, so column ``j`` is the same
    person throughout; only ``command_unit`` differs between members.
    """

    sex: str
    seeds: np.ndarray  # (n_members,)
    cls_prob: np.ndarray  # P(paid)
    reg_pred: np.ndarray  # wage if paid (raw regressor output)
    command_unit: np.ndarray  # command unit drawn for each respondent, per member
    true_income: np.ndarray  # (n_respondents,)
    true_paid: np.ndarray  # (n_respondents,)
    weight: np.ndarray  # survey weights, (n_respondents,)
    # model input per member, stacked member after member; only set by
    # ``predict_ensemble(..., keep_inputs=True)``
    inputs: pd.DataFrame | None = None

    @property
    def expected_wage(self):
        """P(paid) * predicted wage, per member."""
        new_wage = self.reg_pred
        new_wage = np.where(self.cls_prob < 0.5, 0, new_wage)
        return self.cls_prob * self.reg_pred

    def to_frame(self) -> pd.DataFrame:
        """Long dataframe with the input data and predictions of every member.

        One row per (member, respondent). The input columns are what each member saw (the
        command-area covariates differ between members because of the command-unit draw),
        followed by ``cls_prob``, ``reg_pred``, ``expected_wage`` and ``final_wage``
        (aggregated over members, so identical for a respondent in all members).
        Requires ``predict_ensemble(..., keep_inputs=True)``.
        """
        if self.inputs is None:
            raise ValueError(
                "no input data stored; use predict_ensemble(..., keep_inputs=True)"
            )
        out = self.inputs.copy()
        out["cls_prob"] = self.cls_prob.ravel()
        out["reg_pred"] = self.reg_pred.ravel()
        out["expected_wage"] = self.expected_wage.ravel()
        out["final_wage"] = np.tile(self.final_wage(), len(self.seeds))
        return out

    def final_wage(self, aggregate="median"):
        """Expected wage aggregated over members, per respondent."""
        agg = {"median": np.median, "mean": np.mean}[aggregate]
        return agg(self.expected_wage, axis=0)
        # return self.expected_wage[0, :]

    def predicted_paid(self, threshold=0.5):
        """1 where the member-averaged P(paid) reaches ``threshold`` (classification only)."""
        return (self.cls_prob.mean(axis=0) >= threshold).astype(int)

    def thresholded_wage(self, threshold):
        """Per member: predicted wage, set to 0 where P(paid) < ``threshold``."""
        return np.where(self.cls_prob < threshold, 0, self.reg_pred)

    def r2_per_member(self, threshold=None):
        """R2 of every member against the true income.

        Uses P(paid) * wage, or the hard-thresholded wage if ``threshold`` is given.
        """
        pred = (
            self.expected_wage
            if threshold is None
            else self.thresholded_wage(threshold)
        )
        return np.array([r2_score(self.true_income, p) for p in pred])

    def mean_by_command_unit(self, values=None, units=None, weighted=True):
        """Mean predicted wage per command unit.

        Within each member the respondents are averaged per command unit (survey-weighted
        by default); those per-member means are then averaged over the members in which
        the unit is populated. Units without respondents in every member are NaN.

        ``values`` defaults to the expected wage; ``units`` (e.g. ``ds.excel_df["area"]``)
        makes sure every command unit appears in the result. ``n_draws`` is the number of
        members in which the unit had respondents.
        """
        values = self.expected_wage if values is None else np.asarray(values)
        n_members, n_resp = values.shape
        w = self.weight if weighted else np.ones(n_resp)
        long = pd.DataFrame(
            {
                "draw": np.repeat(np.arange(n_members), n_resp),
                "unit": self.command_unit.ravel(),
                "w": np.tile(w, n_members),
                "wy": (values * w).ravel(),
            }
        )
        sums = long.groupby(["draw", "unit"])[["w", "wy"]].sum()
        per_draw = (sums["wy"] / sums["w"]).unstack("unit")
        if units is not None:
            per_draw = per_draw.reindex(columns=pd.Index(units))
        out = pd.DataFrame(
            {"wage": per_draw.mean(axis=0), "n_draws": per_draw.notna().sum(axis=0)}
        )
        out.index.name = "command_unit"
        return out


def model_columns(*models):
    """(feature names, categorical feature names) the given fitted CatBoost models were trained on."""
    cols, cats = [], []
    for model in models:
        names = list(model.feature_names_)
        cols += names
        get_cat = getattr(model, "get_cat_feature_indices", None)
        cats += [names[i] for i in get_cat()] if get_cat else []
    return list(dict.fromkeys(cols)), list(dict.fromkeys(cats))


def predict_ensemble(
    dataset, members: Sequence[EnsembleMember], sex, excel_df=None, keep_inputs=False
) -> EnsemblePrediction:
    """Run every member on the test set of ``sex`` under its own command-unit draw.

    ``dataset`` only needs a ``realization(sex, seed, excel_df, columns, categorical)``
    method (see ``egypt_survey_ml.dataset.Dataset``). Each model uses the feature columns
    it was trained on, requested from the dataset on top of the feature mask, so models
    trained with different feature sets work without extra arguments as long as those
    columns exist in the survey or command-area data.
    ``excel_df`` swaps in other command-area covariates (e.g. another RIBASIM run).
    With ``keep_inputs=True`` the model input of every member is kept, so that
    ``EnsemblePrediction.to_frame()`` can return the input data with the predictions.
    """
    members = list(members)
    if not members:
        raise ValueError("no ensemble members given")

    cls_prob, reg_pred, units, static, inputs = [], [], [], None, []
    for member in members:
        clf = _as_model(member.classifier, CatBoostClassifier)
        reg = _as_model(member.regressor, CatBoostRegressor)
        columns, categorical = model_columns(clf, reg)
        real = dataset.realization(
            sex,
            member.seed,
            excel_df=excel_df,
            columns=columns,
            categorical=categorical,
        )
        for label, model in (("classifier", clf), ("regressor", reg)):
            missing = [c for c in model.feature_names_ if c not in real.frame.columns]
            if missing:
                raise ValueError(
                    f"Incompatible {sex} {label} (seed {member.seed}): features absent "
                    f"from the survey and command-area data: {missing}. "
                    "Check data_filename or use models trained on the available data."
                )
        cls_prob.append(clf.predict_proba(real.frame[clf.feature_names_])[:, 1])
        reg_pred.append(reg.predict(real.frame[reg.feature_names_]))
        units.append(real.command_unit)
        if keep_inputs:
            inputs.append(
                real.frame.reset_index(drop=True).assign(
                    seed=member.seed,
                    respondent=np.arange(len(real.frame)),
                    command_unit=real.command_unit,
                    weight=real.weight,
                )
            )

        this_static = (
            real.frame["totwag_all"].to_numpy(),
            real.frame["is_paid"].to_numpy(),
            real.weight,
        )
        if static is None:
            static = this_static
        elif not all(np.array_equal(a, b) for a, b in zip(static, this_static)):
            raise ValueError(
                f"respondents differ between draws (seed {member.seed}); "
                "predictions cannot be stacked"
            )

    true_income, true_paid, weight = static
    return EnsemblePrediction(
        sex=sex,
        seeds=np.array([m.seed for m in members]),
        cls_prob=np.array(cls_prob),
        reg_pred=np.array(reg_pred),
        command_unit=np.array(units),
        true_income=true_income,
        true_paid=true_paid,
        weight=weight,
        inputs=pd.concat(inputs, ignore_index=True) if keep_inputs else None,
    )
