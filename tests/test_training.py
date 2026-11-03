import warnings
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from egypt_survey_ml.predict import load_members
from egypt_survey_ml.training import HyperparameterWarning, ModelTrainer

FAST = dict(iterations=5, depth=2, early_stopping_rounds=None)


class FakeDataset:
    """Per-seed random frames; mimics the parts of ``Dataset`` that the trainer uses."""

    def __init__(self, columns=None):
        self.columns = columns or {"male": ["x", "z", "cat"], "female": ["x", "z", "cat"]}
        self.zero_handling_requests = []

    def feature_columns(self, sex):
        return self.columns[sex]

    def categorical_columns(self, sex):
        return [c for c in ["cat"] if c in self.columns[sex]]

    def realization(self, sex, seed, test_data=True, zero_handling=None):
        self.zero_handling_requests.append(zero_handling)
        rng = np.random.default_rng(seed + 1000 * (not test_data) + (sex == "male"))
        n = 60
        x = rng.normal(size=n)
        paid = (x + rng.normal(scale=0.5, size=n) > 0).astype(int)
        frame = pd.DataFrame(
            {
                "x": x,
                "z": rng.normal(size=n),
                "cat": rng.choice(["a", "b"], size=n),
                "totwag_all": paid * (500 + 100 * x),
                "is_paid": paid,
            }
        )
        return SimpleNamespace(frame=frame)


def write_hp(path, rows):
    pd.DataFrame(rows, columns=["seed", "lr", "depth"]).to_csv(path, index=False)
    return path


def trainer(ds=None, **paths):
    return ModelTrainer(FakeDataset() if ds is None else ds, **paths)


def hyperparameter_warnings(fn):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fn()
    return [str(w.message) for w in caught if issubclass(w.category, HyperparameterWarning)]


def test_no_hp_paths_trains_with_defaults_without_warnings(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        results = trainer().run_models(seeds=[3, 4], model_dir=tmp_path, **FAST)

    assert set(results) == {"regressor", "classifier"}
    assert results["regressor"].metrics["seed"].tolist() == [3, 4]
    assert results["classifier"].gaps.columns.tolist() == ["predicted_gap", "observed_gap"]
    # the saved layout is what predict.load_members expects
    members = load_members(tmp_path / "classifiers", tmp_path / "regressors", "female")
    assert [m.seed for m in members] == [3, 4]


def test_single_kind_is_saved_in_model_dir_itself(tmp_path):
    trainer().run_models(seeds=[1], kind="regressor", model_dir=tmp_path, **FAST)
    assert (tmp_path / "model_metrics.csv").exists()
    assert (tmp_path / "male" / "model_000.cbm").exists()
    assert not (tmp_path / "classifiers").exists()


def test_invalid_kind_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="kind"):
        trainer().run_models(seeds=[1], kind="both", model_dir=tmp_path, **FAST)


def test_tuned_hyperparameters_are_used_per_seed_and_sex(tmp_path):
    male = write_hp(tmp_path / "male.csv", [[1, 0.2, 3], [2, 0.3, 4]])
    t = trainer(male_hp_path=male)
    assert t._hyperparameters("regressor", "male", 2, 0.05, 6) == (0.3, 4)
    assert t._hyperparameters("regressor", "male", 9, 0.05, 6) == (0.05, 6)  # unknown seed
    assert t._hyperparameters("regressor", "female", 2, 0.05, 6) == (0.05, 6)  # no table
    assert t._hyperparameters("classifier", "male", 2, 0.05, 6) == (0.05, 6)


def test_hp_table_without_lr_or_depth_column_uses_defaults(tmp_path):
    path = tmp_path / "hp.csv"
    pd.DataFrame({"seed": [1], "depth": [3]}).to_csv(path, index=False)
    t = trainer(cls_hp_path=path)
    assert t._hyperparameters("classifier", "male", 1, 0.07, 6) == (0.07, 3)


def test_hp_table_requires_seed_column_and_unique_seeds(tmp_path):
    pd.DataFrame({"lr": [0.1]}).to_csv(tmp_path / "noseed.csv", index=False)
    with pytest.raises(ValueError, match="seed"):
        trainer(cls_hp_path=tmp_path / "noseed.csv")
    write_hp(tmp_path / "dup.csv", [[1, 0.1, 3], [1, 0.2, 3]])
    with pytest.raises(ValueError, match="more than once"):
        trainer(cls_hp_path=tmp_path / "dup.csv")


def test_complete_hp_files_give_no_warnings(tmp_path):
    rows = [[1, 0.2, 2], [2, 0.2, 2]]
    t = trainer(
        male_hp_path=write_hp(tmp_path / "m.csv", rows),
        female_hp_path=write_hp(tmp_path / "f.csv", rows),
        cls_hp_path=write_hp(tmp_path / "c.csv", rows),
    )
    out = tmp_path / "out"
    assert hyperparameter_warnings(lambda: t.run_models(seeds=[1, 2], model_dir=out, **FAST)) == []


def test_seeds_default_to_union_of_valid_tables(tmp_path):
    t = trainer(
        male_hp_path=write_hp(tmp_path / "m.csv", [[5, 0.2, 2]]),
        female_hp_path=write_hp(tmp_path / "f.csv", [[5, 0.2, 2], [7, 0.2, 2]]),
    )
    results = t.run_models(kind="regressor", model_dir=tmp_path / "out", **FAST)
    assert results["regressor"].metrics["seed"].tolist() == [5, 7]


def test_no_seeds_and_no_valid_tables_is_an_error(tmp_path):
    with pytest.raises(ValueError, match="no seeds"):
        trainer().run_models(model_dir=tmp_path, **FAST)


def test_warns_about_seeds_missing_in_valid_table(tmp_path):
    t = trainer(cls_hp_path=write_hp(tmp_path / "c.csv", [[1, 0.2, 2]]))
    msgs = hyperparameter_warnings(
        lambda: t.run_models(seeds=[1, 8], kind="classifier", model_dir=tmp_path / "o", **FAST)
    )
    assert len(msgs) == 1 and "[8]" in msgs[0] and "classifier" in msgs[0]


def test_warns_about_seed_in_only_some_tables(tmp_path):
    t = trainer(
        male_hp_path=write_hp(tmp_path / "m.csv", [[1, 0.2, 2], [2, 0.2, 2]]),
        female_hp_path=write_hp(tmp_path / "f.csv", [[1, 0.2, 2]]),
    )
    msgs = hyperparameter_warnings(
        lambda: t.run_models(seeds=[1, 2], kind="regressor", model_dir=tmp_path / "o", **FAST)
    )
    assert len(msgs) == 1
    assert "seed 2: in ['male'], missing in ['female']" in msgs[0]


def test_warns_if_only_some_hp_paths_are_provided(tmp_path):
    t = trainer(male_hp_path=write_hp(tmp_path / "m.csv", [[1, 0.2, 2]]))
    msgs = hyperparameter_warnings(
        lambda: t.run_models(seeds=[1], kind="regressor", model_dir=tmp_path / "o", **FAST)
    )
    assert len(msgs) == 1 and "['female']" in msgs[0]


def test_warns_if_provided_hp_file_does_not_exist(tmp_path):
    t = trainer(cls_hp_path=tmp_path / "missing.csv")
    msgs = hyperparameter_warnings(
        lambda: t.run_models(seeds=[1], kind="classifier", model_dir=tmp_path / "o", **FAST)
    )
    assert len(msgs) == 1 and "not found" in msgs[0]
    assert t.hp_parameters["classifier"] is None


def test_hp_paths_of_other_kind_do_not_trigger_warnings(tmp_path):
    t = trainer(cls_hp_path=write_hp(tmp_path / "c.csv", [[1, 0.2, 2]]))
    msgs = hyperparameter_warnings(
        lambda: t.run_models(seeds=[1, 2], kind="regressor", model_dir=tmp_path / "o", **FAST)
    )
    assert msgs == []


def test_catboost_random_seed_defaults_to_42_and_is_independent_of_the_draw_seed():
    t = trainer()
    for seed in (3, 11):
        _, _, models = t.run_once_classifier(None, seed=seed, sex_label="male", **FAST)
        assert models["male"]["model"].random_seed_ == 42
    _, _, models = t.run_once(None, seed=7, sex_label="female", random_seed=5, **FAST)
    assert models["female"]["model"].random_seed_ == 5


def test_run_models_passes_random_seed_to_every_model(tmp_path):
    t = trainer()
    t.run_models(seeds=[3, 4], model_dir=tmp_path, random_seed=9, **FAST)
    for store in (t.MODELS, t.MODELS_CLASSIFIER):
        assert all(m["model"].random_seed_ == 9 for m in store.values())


def test_run_once_returns_gap_for_both_sexes():
    t = trainer()
    res, gap, models = t.run_once(None, seed=1, **FAST)
    assert set(res) == set(models) == {"male", "female"}
    assert set(gap) == {"predicted_gap", "observed_gap"}

    res, gap, _ = t.run_once_classifier(None, seed=1, sex_label="male", **FAST)
    assert set(res) == {"male"} and gap == {}
    with pytest.raises(ValueError, match="sex_label"):
        t.run_once(None, seed=1, sex_label="other", **FAST)


def test_run_reports_where_the_hyperparameters_come_from_and_the_cost_per_model(
    tmp_path, capsys
):
    t = trainer(male_hp_path=write_hp(tmp_path / "m.csv", [[1, 0.2, 2]]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t.run_models(seeds=[1, 2], kind="regressor", model_dir=tmp_path / "o", **FAST)
    out = capsys.readouterr().out
    assert "male: tuned for 1/2 seeds, defaults for the rest" in out
    assert "female: defaults" in out
    assert "mean trees: male=" in out and "s/model" in out


def test_classifier_uses_all_rows_while_regressor_follows_the_config():
    ds = FakeDataset()
    t = trainer(ds)
    t.run_once(None, seed=1, **FAST)
    assert set(ds.zero_handling_requests) == {None}
    ds.zero_handling_requests.clear()
    t.run_once_classifier(None, seed=1, **FAST)
    assert set(ds.zero_handling_requests) == {"none"}


def test_models_use_the_feature_columns_of_their_sex(tmp_path):
    ds = FakeDataset({"male": ["x", "z", "cat"], "female": ["x"]})
    t = trainer(ds)
    t.run_models(seeds=[1], model_dir=tmp_path, **FAST)
    for store in (t.MODELS, t.MODELS_CLASSIFIER):
        assert store["male"]["model"].feature_names_ == ["x", "z", "cat"]
        assert store["male"]["cat_features"] == ["cat"]
        assert store["female"]["model"].feature_names_ == ["x"]
        assert store["female"]["cat_features"] == []
        assert list(store["female"]["X_test"].columns) == ["x"]

    train, test = ds.realization("female", 1, False).frame, ds.realization("female", 1).frame
    model, _ = t.fit_and_predict(train, test, "female", **FAST)
    assert model.feature_names_ == ["x"]
