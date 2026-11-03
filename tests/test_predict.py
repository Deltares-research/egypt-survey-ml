import numpy as np
import pandas as pd
import pytest

from egypt_survey_ml.dataset import Realization
from egypt_survey_ml.predict import (
    STAT_NAMES,
    EnsembleMember,
    EnsemblePrediction,
    describe,
    load_members,
    predict_ensemble,
)


def make_prediction(**overrides):
    # 2 members, 3 respondents. Respondent 2 is in unit "B" in member 0 and "C" in member 1.
    values = dict(
        sex="female",
        seeds=np.array([42, 43]),
        cls_prob=np.array([[0.9, 0.2, 0.6], [0.7, 0.4, 0.5]]),
        reg_pred=np.array([[100.0, 200.0, 300.0], [110.0, 210.0, 310.0]]),
        command_unit=np.array([["A", "A", "B"], ["A", "B", "C"]]),
        true_income=np.array([90.0, 0.0, 250.0]),
        true_paid=np.array([1, 0, 1]),
        weight=np.array([1.0, 3.0, 2.0]),
    )
    values.update(overrides)
    return EnsemblePrediction(**values)


def test_describe_weighted_mean_only_changes_mean():
    plain = describe([1.0, 2.0, 3.0, 10.0])
    weighted = describe([1.0, 2.0, 3.0, 10.0], weights=[1, 1, 1, 0])
    assert len(plain) == len(STAT_NAMES)
    assert weighted[3] == pytest.approx(2.0)
    assert plain[:3] == weighted[:3] and plain[4:] == weighted[4:]


def test_expected_wage_and_aggregation():
    pred = make_prediction()
    np.testing.assert_allclose(pred.expected_wage[0], [90.0, 40.0, 180.0])
    np.testing.assert_allclose(
        pred.final_wage(), np.median(pred.expected_wage, axis=0)
    )
    np.testing.assert_allclose(
        pred.final_wage("mean"), pred.expected_wage.mean(axis=0)
    )


def test_threshold_helpers():
    pred = make_prediction()
    # mean P(paid) = [0.8, 0.3, 0.55]
    np.testing.assert_array_equal(pred.predicted_paid(0.5), [1, 0, 1])
    np.testing.assert_array_equal(pred.predicted_paid(0.55), [1, 0, 1])
    np.testing.assert_array_equal(pred.predicted_paid(0.6), [1, 0, 0])
    np.testing.assert_allclose(
        pred.thresholded_wage(0.5), [[100.0, 0.0, 300.0], [110.0, 0.0, 310.0]]
    )


def test_r2_per_member_perfect_prediction():
    pred = make_prediction(
        cls_prob=np.ones((2, 3)), reg_pred=np.tile([90.0, 0.0, 250.0], (2, 1))
    )
    np.testing.assert_allclose(pred.r2_per_member(), [1.0, 1.0])
    np.testing.assert_allclose(pred.r2_per_member(threshold=0.5), [1.0, 1.0])


def test_mean_by_command_unit_weighted_with_nan_for_empty_units():
    pred = make_prediction()
    out = pred.mean_by_command_unit(units=["A", "B", "C", "D"])

    e = pred.expected_wage
    # draw 0: A = (1*90 + 3*40)/4, B = 180; draw 1: A = 1*77, B = 3*84, C = 2*155 (single respondent each)
    a = np.mean([(1 * e[0, 0] + 3 * e[0, 1]) / 4, e[1, 0]])
    b = np.mean([e[0, 2], e[1, 1]])
    c = e[1, 2]
    assert out.loc["A", "wage"] == pytest.approx(a)
    assert out.loc["B", "wage"] == pytest.approx(b)
    assert out.loc["C", "wage"] == pytest.approx(c)
    assert np.isnan(out.loc["D", "wage"])
    assert out["n_draws"].to_dict() == {"A": 2, "B": 2, "C": 1, "D": 0}


def test_mean_by_command_unit_unweighted_and_custom_values():
    pred = make_prediction()
    out = pred.mean_by_command_unit(values=np.ones((2, 3)), weighted=False)
    assert out["wage"].tolist() == [1.0, 1.0, 1.0]
    assert out.index.tolist() == ["A", "B", "C"]


def write_metrics(folder, rows, columns):
    folder.mkdir(parents=True)
    pd.DataFrame(rows, columns=columns).to_csv(folder / "model_metrics.csv", index=False)


@pytest.fixture
def model_dirs(tmp_path):
    cls_dir, reg_dir = tmp_path / "cls", tmp_path / "reg"
    for sex in ("male", "female"):
        for d in (cls_dir, reg_dir):
            (d / sex).mkdir(parents=True, exist_ok=True)
            for rep in range(3):
                (d / sex / f"model_{rep:03d}.cbm").touch()
    # csv files carry absolute paths of another machine; they must not be used
    cls_rows = [
        [r, 42 + r, 0.9, 0.9, f"/elsewhere/cls/female/model_{r:03d}.cbm"] for r in (2, 0, 1)
    ]
    reg_rows = [
        [r, 42 + r, 0.4, 0.4, f"/elsewhere/male/model_{r:03d}.cbm", f"/elsewhere/female/model_{r:03d}.cbm"]
        for r in range(3)
    ]
    pd.DataFrame(
        cls_rows, columns=["rep", "seed", "male_accuracy", "female_accuracy", "model_file"]
    ).to_csv(cls_dir / "model_metrics.csv", index=False)
    pd.DataFrame(
        reg_rows,
        columns=["rep", "seed", "male_r2", "female_r2", "male_model_file", "female_model_file"],
    ).to_csv(reg_dir / "model_metrics.csv", index=False)
    return cls_dir, reg_dir


def test_load_members_pairs_by_rep_and_resolves_paths(model_dirs):
    cls_dir, reg_dir = model_dirs
    members = load_members(cls_dir, reg_dir, "male")
    assert [m.seed for m in members] == [42, 43, 44]
    assert members[1].classifier == cls_dir / "male" / "model_001.cbm"
    assert members[1].regressor == reg_dir / "male" / "model_001.cbm"


def test_load_members_rejects_seed_mismatch(model_dirs):
    cls_dir, reg_dir = model_dirs
    metrics = pd.read_csv(reg_dir / "model_metrics.csv")
    metrics.loc[1, "seed"] = 999
    metrics.to_csv(reg_dir / "model_metrics.csv", index=False)
    with pytest.raises(ValueError, match="seeds differ"):
        load_members(cls_dir, reg_dir, "female")


def test_load_members_rejects_missing_model_file(model_dirs):
    cls_dir, reg_dir = model_dirs
    (reg_dir / "female" / "model_002.cbm").unlink()
    with pytest.raises(FileNotFoundError):
        load_members(cls_dir, reg_dir, "female")


class FakeClassifier:
    feature_names_ = ["x"]

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([1 - self.p * X["x"], self.p * X["x"]])


class FakeRegressor:
    feature_names_ = ["x", "y"]

    def predict(self, X):
        return X["x"].to_numpy() * 10 + X["y"].to_numpy()


class FakeDataset:
    def __init__(self, wages=(100.0, 0.0, 300.0)):
        self.wages = np.array(wages)
        self.calls = []
        self.zero_handling_requests = []
        self.governorate_requests = []

    def realization(
        self,
        sex,
        seed,
        excel_df=None,
        columns=(),
        categorical=(),
        zero_handling=None,
        governorates=None,
    ):
        self.calls.append((sex, seed, excel_df))
        self.governorate_requests.append(governorates)
        self.zero_handling_requests.append(zero_handling)
        self.requested = (list(columns), list(categorical))
        frame = pd.DataFrame(
            {
                "x": [1.0, 2.0, 3.0],
                "y": [seed, seed, seed],
                "totwag_all": self.wages,
                "is_paid": (self.wages > 0).astype(int),
            }
        )
        units = np.array(["A", "B", "A"]) if seed % 2 == 0 else np.array(["B", "B", "A"])
        return Realization(seed, 2, frame, np.array([1.0, 2.0, 3.0]), units)


def test_predict_ensemble_stacks_members_and_uses_member_seeds():
    ds = FakeDataset()
    members = [
        EnsembleMember(42, FakeClassifier(0.1), FakeRegressor()),
        EnsembleMember(43, FakeClassifier(0.2), FakeRegressor()),
    ]
    pred = predict_ensemble(ds, members, "female", excel_df="scenario")

    assert [c[1] for c in ds.calls] == [42, 43]
    assert all(c[2] == "scenario" for c in ds.calls)
    np.testing.assert_array_equal(pred.seeds, [42, 43])
    assert pred.cls_prob.shape == pred.reg_pred.shape == pred.command_unit.shape == (2, 3)
    np.testing.assert_allclose(pred.cls_prob[1], [0.2, 0.4, 0.6])
    np.testing.assert_allclose(pred.reg_pred[0], [52.0, 62.0, 72.0])
    np.testing.assert_array_equal(pred.true_paid, [1, 0, 1])
    np.testing.assert_array_equal(pred.weight, [1.0, 2.0, 3.0])
    assert pred.command_unit[0].tolist() == ["A", "B", "A"]
    assert pred.command_unit[1].tolist() == ["B", "B", "A"]


def test_predict_ensemble_rejects_draws_with_different_respondents():
    class Drifting(FakeDataset):
        def realization(self, sex, seed, excel_df=None, **kwargs):
            self.wages = self.wages + seed
            return super().realization(sex, seed, excel_df, **kwargs)

    members = [EnsembleMember(s, FakeClassifier(0.1), FakeRegressor()) for s in (1, 2)]
    with pytest.raises(ValueError, match="respondents differ"):
        predict_ensemble(Drifting(), members, "female")


def test_predict_ensemble_requires_members():
    with pytest.raises(ValueError):
        predict_ensemble(FakeDataset(), [], "female")


def test_predict_ensemble_passes_governorates_to_the_realization():
    ds = FakeDataset()
    member = EnsembleMember(42, FakeClassifier(0.1), FakeRegressor())
    predict_ensemble(ds, [member], "female")
    predict_ensemble(ds, [member], "female", governorates=["Cairo", "Giza"])
    assert ds.governorate_requests == [None, ["Cairo", "Giza"]]


def test_predict_ensemble_requests_the_columns_its_models_need():
    ds = FakeDataset()
    predict_ensemble(
        ds, [EnsembleMember(42, FakeClassifier(0.1), FakeRegressor())], "female"
    )
    assert ds.requested == (["x", "y"], [])
    # evaluation is on everyone, paid and unpaid, whatever the training zero handling
    assert ds.zero_handling_requests == ["none"]


@pytest.mark.parametrize("model_kind", ["classifier", "regressor"])
def test_predict_ensemble_reports_features_missing_from_data(model_kind):
    classifier = FakeClassifier(0.1)
    regressor = FakeRegressor()
    model = classifier if model_kind == "classifier" else regressor
    model.feature_names_ = ["x", "water_productivity"]
    member = EnsembleMember(42, classifier, regressor)
    with pytest.raises(
        ValueError,
        match=f"Incompatible female {model_kind}.*seed 42.*water_productivity",
    ):
        predict_ensemble(FakeDataset(), [member], "female")


def test_to_frame_returns_inputs_with_predictions():
    ds = FakeDataset()
    members = [
        EnsembleMember(42, FakeClassifier(0.1), FakeRegressor()),
        EnsembleMember(43, FakeClassifier(0.2), FakeRegressor()),
    ]
    pred = predict_ensemble(ds, members, "female", keep_inputs=True)
    df = pred.to_frame()

    assert len(df) == 6
    assert df["seed"].tolist() == [42, 42, 42, 43, 43, 43]
    assert df["respondent"].tolist() == [0, 1, 2, 0, 1, 2]
    assert df["y"].tolist() == [42, 42, 42, 43, 43, 43]  # inputs as seen by each member
    assert df["command_unit"].tolist() == ["A", "B", "A", "B", "B", "A"]
    np.testing.assert_allclose(df["cls_prob"], pred.cls_prob.ravel())
    np.testing.assert_allclose(df["expected_wage"], pred.expected_wage.ravel())
    np.testing.assert_allclose(df["final_wage"][:3], pred.final_wage())


def test_to_frame_requires_keep_inputs():
    pred = predict_ensemble(
        FakeDataset(), [EnsembleMember(1, FakeClassifier(0.1), FakeRegressor())], "female"
    )
    assert pred.inputs is None
    with pytest.raises(ValueError, match="keep_inputs"):
        pred.to_frame()
