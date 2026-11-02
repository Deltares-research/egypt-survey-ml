import numpy as np
import pandas as pd
import pytest
from catboost import CatBoostClassifier, CatBoostRegressor

from egypt_survey_ml.dataset import Realization
from egypt_survey_ml.explain import (
    CombinedShap,
    catboost_shap,
    combine_shap,
    combined_shap,
)
from egypt_survey_ml.predict import EnsembleMember, predict_ensemble


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def test_combine_shap_adds_up_to_product_of_predictions():
    rng = np.random.default_rng(0)
    cols = ["a", "b", "c"]
    shap_cls = pd.DataFrame(rng.normal(size=(50, 3)), columns=cols)
    shap_reg = pd.DataFrame(rng.normal(scale=100, size=(50, 3)), columns=cols)
    base_cls, base_reg = 0.4, 500.0

    phi, base = combine_shap(shap_cls, base_cls, shap_reg, base_reg)

    p = sigmoid(shap_cls.sum(axis=1) + base_cls)
    r = shap_reg.sum(axis=1) + base_reg
    np.testing.assert_allclose(phi.sum(axis=1) + base, p * r, atol=1e-9)


def test_combine_shap_handles_unmoved_log_odds_and_different_feature_sets():
    shap_cls = pd.DataFrame({"a": [0.0, 1.0], "b": [0.0, -1.0]})
    shap_reg = pd.DataFrame({"a": [10.0, 20.0], "c": [5.0, 5.0]})

    phi, base = combine_shap(shap_cls, 0.0, shap_reg, 100.0)

    assert set(phi.columns) == {"a", "b", "c"}
    assert np.isfinite(phi.to_numpy()).all()
    r = shap_reg.sum(axis=1) + 100.0
    p = sigmoid(shap_cls.sum(axis=1))
    np.testing.assert_allclose(phi.sum(axis=1) + base, p * r, atol=1e-9)


@pytest.fixture(scope="module")
def tiny_setup():
    rng = np.random.default_rng(1)
    n = 200
    frame = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "cat": rng.choice(["u", "v", "w"], size=n),
        }
    )
    frame["totwag_all"] = np.maximum(0, 100 + 80 * frame["x1"] + rng.normal(scale=10, size=n))
    frame["is_paid"] = (frame["totwag_all"] > 50).astype(int)

    def fit(model):
        model.fit(frame[["x1", "x2", "cat"]], target, cat_features=["cat"], verbose=False)
        return model

    target = frame["is_paid"]
    clf = fit(CatBoostClassifier(iterations=20, depth=3, random_seed=0, allow_writing_files=False))
    target = frame["totwag_all"]
    reg = fit(CatBoostRegressor(iterations=20, depth=3, random_seed=0, allow_writing_files=False))
    return frame, clf, reg


class FakeDataset:
    def __init__(self, frame):
        self.frame = frame

    def realization(self, sex, seed, excel_df=None, **kwargs):
        units = np.array(["A", "B"] * (len(self.frame) // 2))
        return Realization(seed, 2, self.frame, np.ones(len(self.frame)), units)


def test_combined_shap_matches_ensemble_prediction(tiny_setup):
    frame, clf, reg = tiny_setup
    ds = FakeDataset(frame)
    members = [EnsembleMember(1, clf, reg), EnsembleMember(2, clf, reg)]

    result = combined_shap(ds, members, "female")
    pred = predict_ensemble(ds, members, "female")

    assert isinstance(result, CombinedShap) and result.n_members == 2
    np.testing.assert_allclose(
        result.base_value + result.values.sum(axis=1).to_numpy(),
        pred.expected_wage.mean(axis=0),
        atol=1e-6,
    )
    imp = result.importance()
    assert imp.index[0] == "x1"  # the wage is driven by x1
    assert imp.is_monotonic_decreasing
    assert result.importance(weighted=True).index.tolist() == imp.index.tolist()


def test_combined_shap_keeps_numeric_feature_values_for_colouring(tiny_setup):
    frame, clf, reg = tiny_setup
    result = combined_shap(
        FakeDataset(frame), [EnsembleMember(1, clf, reg)], "female"
    )
    assert result.features.shape == result.values.shape
    np.testing.assert_allclose(result.features["x1"], frame["x1"])
    assert result.features["cat"].isna().all()  # categorical: no value scale


def test_plot_beeswarm_draws_one_row_per_feature(tiny_setup):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from egypt_survey_ml.explain import plot_beeswarm

    frame, clf, reg = tiny_setup
    result = combined_shap(
        FakeDataset(frame), [EnsembleMember(1, clf, reg)], "female"
    )
    ax = plot_beeswarm(result, top=3, title="t", show=False)
    assert len(ax.get_yticklabels()) == 3
    assert ax.get_yticklabels()[-1].get_text() == "x1"  # most important is drawn on top
    plt.close("all")


def test_catboost_shap_is_additive(tiny_setup):
    frame, clf, _ = tiny_setup
    values, base = catboost_shap(clf, frame)
    raw = clf.predict(frame[clf.feature_names_], prediction_type="RawFormulaVal")
    np.testing.assert_allclose(values.sum(axis=1) + base, raw, atol=1e-9)


def test_combined_shap_requires_members(tiny_setup):
    with pytest.raises(ValueError):
        combined_shap(FakeDataset(tiny_setup[0]), [], "female")
