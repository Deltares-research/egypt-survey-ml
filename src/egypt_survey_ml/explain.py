"""Feature attribution for the combined classifier x regressor wage model.

The final prediction of one ensemble member is ``E = p * r`` with ``p`` the classifier's
P(paid) and ``r`` the regressor's wage. SHAP values of the two models live on different
scales (log-odds vs EGP), so they are converted to EGP and combined per respondent such
that the contributions add up exactly to ``p * r - p0 * r0`` (``p0``, ``r0`` are the
models' baselines). Averaging over the ensemble members gives one attribution for the
combined model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from .predict import EnsembleMember, _as_model, model_columns


def catboost_shap(model, frame):
    """SHAP values of a fitted CatBoost model: (DataFrame respondents x features, baseline)."""
    X = frame[model.feature_names_]
    pool = Pool(X, cat_features=model.get_cat_feature_indices())
    sv = model.get_feature_importance(pool, type="ShapValues")
    return pd.DataFrame(sv[:, :-1], columns=model.feature_names_, index=frame.index), float(
        sv[0, -1]
    )


def combine_shap(shap_cls, base_cls, shap_reg, base_reg):
    """Attribution of ``P(paid) * wage`` in EGP, from the two models' SHAP values.

    ``shap_cls`` is in log-odds, ``shap_reg`` in EGP; ``base_*`` are the baselines.
    The classifier part is first rescaled to probability units (so it sums to
    ``p - p0``) and then combined with the regressor part with the product rule
    ``p*r - p0*r0 = (p - p0) * (r + r0) / 2 + (r - r0) * (p + p0) / 2``.
    The result sums to ``p * r - p0 * r0`` for every respondent.
    """
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    z = shap_cls.sum(axis=1) + base_cls
    p, p0 = sigmoid(z), sigmoid(base_cls)
    r = shap_reg.sum(axis=1) + base_reg

    dz = z - base_cls
    # where the log-odds do not move, fall back to the local slope of the sigmoid
    scale = np.where(np.abs(dz) > 1e-12, (p - p0) / np.where(dz == 0, 1, dz), p * (1 - p))
    shap_prob = shap_cls.mul(scale, axis=0)

    return shap_prob.mul((r + base_reg) / 2, axis=0).add(
        shap_reg.mul((p + p0) / 2, axis=0), fill_value=0
    ), float(p0 * base_reg)


@dataclass
class CombinedShap:
    """Ensemble-averaged attribution of the combined model, in EGP.

    ``values`` is (n_respondents x n_features); row ``j`` is the same respondent as in
    ``EnsemblePrediction``. ``base_value`` is the member-averaged baseline, so
    ``base_value + values.sum(axis=1)`` is the mean expected wage per respondent.
    """

    sex: str
    values: pd.DataFrame
    features: pd.DataFrame
    base_value: float
    weight: np.ndarray
    n_members: int

    def importance(self, weighted=False) -> pd.Series:
        """Mean absolute attribution per feature (EGP), largest first.

        With ``weighted=True`` respondents are weighted by their survey weight.
        """
        abs_vals = self.values.abs()
        if weighted:
            imp = abs_vals.mul(self.weight, axis=0).sum() / self.weight.sum()
        else:
            imp = abs_vals.mean()
        return imp.sort_values(ascending=False)


def combined_shap(
    dataset, members: Sequence[EnsembleMember], sex, excel_df=None
) -> CombinedShap:
    """Attribution of the combined model, averaged over all ensemble members.

    Every member is explained on the command-unit draw it was trained on (its seed), the
    same way as in :func:`egypt_survey_ml.predict.predict_ensemble`. Because the draws
    change the command-area covariates, the attribution and the value of those features
    are averaged over draws. ``features`` holds the (draw-averaged) numeric feature values
    for colouring plots; categorical features are NaN there.
    """
    members = list(members)
    if not members:
        raise ValueError("no ensemble members given")

    total, feature_total, base, weight = None, None, 0.0, None
    for member in members:
        clf = _as_model(member.classifier, CatBoostClassifier)
        reg = _as_model(member.regressor, CatBoostRegressor)
        columns, categorical = model_columns(clf, reg)
        real = dataset.realization(
            sex, member.seed, excel_df=excel_df, columns=columns, categorical=categorical
        )
        shap_cls, base_cls = catboost_shap(clf, real.frame)
        shap_reg, base_reg = catboost_shap(reg, real.frame)
        phi, phi_base = combine_shap(shap_cls, base_cls, shap_reg, base_reg)

        phi = phi.reset_index(drop=True)
        feature_values = real.frame[list(phi.columns)].reset_index(drop=True)
        feature_values = feature_values.apply(
            lambda col: col.astype(float)
            if pd.api.types.is_numeric_dtype(col)
            else np.nan
        )
        total = phi if total is None else total.add(phi, fill_value=0)
        feature_total = (
            feature_values
            if feature_total is None
            else feature_total.add(feature_values, fill_value=0)
        )
        base += phi_base
        weight = real.weight if weight is None else weight

    n = len(members)
    return CombinedShap(
        sex=sex,
        values=total / n,
        features=feature_total / n,
        base_value=base / n,
        weight=weight,
        n_members=n,
    )


def plot_importance(importance: pd.Series, top=20, ax=None, title=None):
    """Horizontal bar chart of the ``top`` most important features."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 0.35 * min(top, len(importance)) + 1.5))
    importance.head(top).iloc[::-1].plot.barh(ax=ax, color="#0072B2")
    ax.set_xlabel("Mean |attribution| to predicted wage (EGP)")
    ax.set_ylabel("")
    if title:
        ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    ax.figure.tight_layout()
    return ax


def plot_beeswarm(result: CombinedShap, top=20, title=None, show=True):
    """SHAP-style beeswarm of the combined model.

    One dot per respondent and feature: x is the attribution to the predicted wage (EGP;
    positive raises it), colour is the feature value (red = high, blue = low). Categorical
    features have no value scale and are drawn in grey. Features are ordered by mean
    |attribution| (unweighted, as in ``shap``); respondents are not survey-weighted.
    """
    import matplotlib.pyplot as plt
    import shap

    shap.summary_plot(
        result.values.to_numpy(),
        features=result.features.to_numpy(dtype=float),
        feature_names=list(result.values.columns),
        max_display=top,
        plot_type="dot",
        show=False,
    )
    ax = plt.gca()
    ax.set_xlabel("Attribution to predicted wage (EGP)")
    if title:
        ax.set_title(title)
    if show:
        plt.show()
    return ax
