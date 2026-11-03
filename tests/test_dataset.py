import warnings
from dataclasses import fields, replace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from egypt_survey_ml.config import DatasetConfig
from egypt_survey_ml.dataset import (
    Dataset,
    FeatureSelectionWarning,
    GovernorateWarning,
    prepare_dataset,
    select_features,
)
from egypt_survey_ml.paths import DataPaths
from egypt_survey_ml.predict import EnsembleMember, predict_ensemble


@pytest.fixture
def paths(tmp_path):
    return DataPaths(
        data_root=tmp_path,
        excel_path=tmp_path / "command_data.xlsx",
        crosswalk_csv=tmp_path / "crosswalk.csv",
        command_area_crosswalk_csv=tmp_path / "command_crosswalk.csv",
        lfs_dta=tmp_path / "individual.dta",
        hh_dta=tmp_path / "household.dta",
        result_dir=tmp_path,
        mask_xlsx=tmp_path / "default_mask.xlsx",
        pop_density_csv=tmp_path / "population.csv",
        source="test",
    )


@pytest.mark.parametrize("override", [False, True])
def test_prepare_dataset_loads_feature_mask(paths, override):
    mask_path = paths.result_dir / "custom_mask.xlsx" if override else paths.mask_xlsx
    mask = pd.DataFrame(
        {
            "variable": ["age", "educ", "excluded"],
            "correl": [1, 1, 0],
            "type": ["numeric", "categorical", "numeric"],
        }
    )
    mask.to_excel(mask_path, sheet_name="variable", index=False)
    captured = []
    read_excel = pd.read_excel

    def capture_mask(*args, **kwargs):
        frame = read_excel(*args, **kwargs)
        captured.append(frame)
        return frame

    class SurveyReached(Exception):
        pass

    kwargs = {"mask_path": str(mask_path)} if override else {}
    with (
        patch("egypt_survey_ml.dataset.pd.read_excel", side_effect=capture_mask),
        patch("egypt_survey_ml.dataset.load_survey", side_effect=SurveyReached) as load,
        pytest.raises(SurveyReached),
    ):
        prepare_dataset(paths=paths, **kwargs)

    effective_paths = replace(paths, mask_xlsx=mask_path)
    load.assert_called_once_with(effective_paths)
    assert paths.mask_xlsx.name == "default_mask.xlsx"
    pd.testing.assert_frame_equal(captured[0], mask)
    features = select_features(
        captured[0],
        pd.DataFrame(columns=["age", "educ", "excluded"]),
        DatasetConfig(),
    )
    assert features.base_cols == ["age", "educ"]
    assert features.categorical_columns == ["educ"]


def test_prepare_dataset_rejects_invalid_mask_before_loading_survey(paths):
    pd.DataFrame({"variable": ["age"]}).to_excel(
        paths.mask_xlsx, sheet_name="variable", index=False
    )
    with (
        patch("egypt_survey_ml.dataset.load_survey") as load,
        pytest.raises(ValueError, match="missing required columns.*correl.*type"),
    ):
        prepare_dataset(paths=paths)
    load.assert_not_called()


def test_prepare_dataset_missing_mask_is_not_silently_ignored(paths):
    with pytest.raises(FileNotFoundError):
        prepare_dataset(paths=paths, mask_path=paths.result_dir / "missing.xlsx")


MASK = pd.DataFrame(
    {
        "variable": ["age", "sex", "educ", "yeduc", "tenure", "female_violence_sexual"],
        "correl": [1, 1, 0, 0, 1, 1],
        "type": ["numerical", "binary", "categorical", "numerical", "numerical", "numerical"],
    }
)
POOL_COLUMNS = ["age", "sex", "educ", "yeduc", "tenure", "rel", "attsch", "occ", "hrswk"]


def select(**config):
    return select_features(MASK, pd.DataFrame(columns=POOL_COLUMNS), DatasetConfig(**config))


def test_mask_selection_adds_hours_and_extra_features():
    features = select()
    assert features.columns_for(1) == [
        "age", "sex", "tenure", "female_violence_sexual",
        "rel", "attsch", "occ", "hrswk", "hrswksc",
    ]
    assert features.columns_for(1) == features.columns_for(2)


def test_extra_and_excluded_columns_are_per_sex():
    features = select(
        male_extra_columns=["yeduc", "educ"],
        female_extra_columns=["yeduc"],
        male_exclude_columns=["tenure"],
        female_exclude_columns=["age", "hrswk", "occ"],
    )
    male, female = features.columns_for(1), features.columns_for(2)
    assert male[-2:] == ["yeduc", "educ"] and "tenure" not in male and "age" in male
    assert female[-1] == "yeduc" and "educ" not in female and "tenure" in female
    assert not {"age", "hrswk", "occ"} & set(female) and "hrswksc" in female
    assert {"age", "hrswk", "occ"} <= set(male)


def test_extra_columns_already_selected_are_not_duplicated():
    male = select(male_extra_columns=["age", "hrswk"]).columns_for(1)
    assert male.count("age") == 1 and male.count("hrswk") == 1


def test_exclusion_wins_over_extra_columns():
    male = select(male_extra_columns=["yeduc"], male_exclude_columns=["yeduc"]).columns_for(1)
    assert "yeduc" not in male


def test_extra_column_types_come_from_the_mask():
    features = select(male_extra_columns=["educ", "yeduc"])
    assert "educ" in features.categorical_columns
    assert "yeduc" not in features.categorical_columns


def test_unavailable_extra_columns_warn_and_are_not_included():
    available = set(POOL_COLUMNS) | {"water_productivity"}
    with pytest.warns(FeatureSelectionWarning, match="male_extra_columns.*not_in_data"):
        features = select_features(
            MASK,
            pd.DataFrame(columns=POOL_COLUMNS),
            DatasetConfig(
                male_extra_columns=["yeduc", "not_in_data", "water_productivity"],
                female_extra_columns=["yeduc"],
            ),
            available=available,
        )
    male = features.columns_for(1)
    assert "not_in_data" not in male and {"yeduc", "water_productivity"} <= set(male)
    assert "water_productivity" not in features.columns_for(2)


def test_excluding_a_column_that_is_not_selected_warns():
    with pytest.warns(FeatureSelectionWarning, match="female_exclude_columns.*educ"):
        features = select(female_exclude_columns=["educ", "age"])
    assert "age" not in features.columns_for(2)


def test_no_feature_warnings_without_extra_or_excluded_columns():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        select_features(
            MASK,
            pd.DataFrame(columns=POOL_COLUMNS),
            DatasetConfig(),
            available=set(POOL_COLUMNS),
        )


def make_dataset(features, test_pool, excel_df):
    values = {f.name: None for f in fields(Dataset)}
    values.update(features=features, test_pool=test_pool, excel_df=excel_df, geo_cols=[])
    return Dataset(**values)


def _dataset(**config):
    pool = pd.DataFrame(
        {
            "sex": [1, 2],
            "age": [30, 40],
            "yeduc": [5, 6],
            "occ": [1, 2],
            "pweight": [2.0, 3.0],
            "command_unit": ["A", "A"],
            "log1p_wage": [1.0, 2.0],
            "totwag_all": [10.0, 20.0],
            "is_paid": [1, 1],
        }
    )
    geo = pd.DataFrame(
        {"area": ["A"], "water_productivity": [1.5], "producer_price": [100.0]}
    )
    features = select_features(MASK, pool, DatasetConfig(include_hrswk=False, **config))
    return make_dataset(features, pool, geo), pool


def test_dataset_feature_columns_are_per_sex_and_exist_in_the_data():
    ds, _ = _dataset(
        male_extra_columns=["water_productivity"],
        female_extra_columns=["yeduc"],
        female_exclude_columns=["age"],
    )
    assert ds.feature_columns("male") == ["age", "sex", "occ", "water_productivity"]
    assert ds.feature_columns("female") == ["sex", "occ", "yeduc"]
    assert ds.categorical_columns("male") == ["sex", "occ"]
    assert ds.categorical_columns("female") == ["sex", "occ"]


def test_feature_frame_has_the_columns_of_its_sex():
    ds, pool = _dataset(
        male_extra_columns=["water_productivity"], female_exclude_columns=["age"]
    )
    male, cats, extra = ds.build_feature_frame(pool, 1)
    assert male.columns.tolist()[:4] == ["age", "sex", "occ", "water_productivity"]
    assert cats == ["sex", "occ"] and male["sex"].tolist() == ["1"]
    assert extra["pweight"].tolist() == [2.0]
    female, _, _ = ds.build_feature_frame(pool, 2)
    assert "age" not in female.columns and "water_productivity" not in female.columns


def test_feature_frame_skips_mask_columns_missing_from_the_data():
    ds, pool = _dataset()
    frame, _, _ = ds.build_feature_frame(pool, 2)
    assert "female_violence_sexual" not in frame.columns
    assert "tenure" not in frame.columns


def test_feature_frame_adds_columns_requested_by_models():
    ds, pool = _dataset()
    frame, cats, _ = ds.build_feature_frame(
        pool,
        2,
        extra_cols=["age", "water_productivity", "producer_price", "not_in_data"],
        extra_cats=["occ", "not_in_data"],
    )
    assert frame[["water_productivity", "producer_price"]].iloc[0].tolist() == [1.5, 100.0]
    assert frame.columns.tolist().count("age") == 1
    assert "not_in_data" not in frame.columns
    assert "occ" in cats and frame["occ"].tolist() == ["2"]


def _realization_dataset(zero_handling):
    pool = pd.DataFrame(
        {
            "sex": [1] * 5,
            "age": [30, 31, 32, 33, 34],
            "occ": [1] * 5,
            "pweight": [1.0, 2.0, 3.0, 4.0, 5.0],
            "command_unit": ["A"] * 5,
            "totwag_all": [0.0, 100.0, 0.0, 200.0, 9000.0],  # last one is above the male cap
        }
    )
    pool["is_paid"] = (pool["totwag_all"] > 0).astype(int)
    pool["log1p_wage"] = pool["totwag_all"]
    geo = pd.DataFrame({"area": ["A"]})
    features = select_features(MASK, pool, DatasetConfig(include_hrswk=False))
    ds = make_dataset(features, pool, geo)
    ds.config = DatasetConfig(zero_handling=zero_handling, include_hrswk=False)
    ds.assign_command_units = lambda pool, seed: pool
    return ds


def test_realization_zero_handling_can_be_overridden_per_call():
    ds = _realization_dataset("drop_random")  # the regressor's setting: drops zero-wage rows
    assert ds.realization("male", 1).frame["is_paid"].tolist() == [1, 1]

    # the participation classifier needs both classes, and the outlier cap still applies
    real = ds.realization("male", 1, zero_handling="none")
    assert real.frame["is_paid"].tolist() == [0, 1, 0, 1]
    assert len(real.frame) == len(real.weight) == len(real.command_unit) == 4
    assert real.weight.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_realization_without_zero_handling_applies_the_outlier_cap_to_the_frame():
    real = _realization_dataset("none").realization("male", 1)
    assert real.frame["totwag_all"].max() < 5000
    assert len(real.frame) == len(real.weight)


class _Model:
    feature_names_ = ["age"]

    def __init__(self, value):
        self.value = value

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.value), np.full(len(X), self.value)])

    def predict(self, X):
        return np.full(len(X), self.value)


def test_prediction_is_scored_on_paid_and_unpaid_even_if_training_drops_zero_wages():
    # a training config that drops the zero-wage rows must not leak into the evaluation
    ds = _realization_dataset("drop_random")
    member = EnsembleMember(1, _Model(0.9), _Model(150.0))
    pred = predict_ensemble(ds, [member], "male")
    assert pred.true_paid.tolist() == [0, 1, 0, 1]
    assert pred.true_income.tolist() == [0.0, 100.0, 0.0, 200.0]
    assert pred.weight.tolist() == [1.0, 2.0, 3.0, 4.0]


def _governorate_dataset():
    n = 60
    pool = pd.DataFrame(
        {
            "sex": [1] * n,
            "age": np.arange(n, dtype=float),
            "pweight": np.arange(1.0, n + 1),  # unique per respondent, to identify rows
            "NAME1_": ["Cairo", "Giza", "Aswan"] * (n // 3),
            "totwag_all": 100.0,
            "is_paid": 1,
            "log1p_wage": 1.0,
        }
    )
    units = ["u1", "u2", "u3"]
    dist = pd.DataFrame(
        0.0, index=pd.Index(["Cairo", "Giza", "Aswan"], name="NAME1_"), columns=units + ["other"]
    )
    dist[units] = 1 / 3
    dist.loc["Aswan", "other"] = 0.0
    features = select_features(MASK, pool, DatasetConfig(zero_handling="none"))
    ds = make_dataset(features, pool, pd.DataFrame({"area": units}))
    ds.train_pool = pool
    ds.config = DatasetConfig(zero_handling="none", include_hrswk=False)
    ds.distribution_df = dist
    return ds


def test_realization_can_be_limited_to_governorates():
    ds = _governorate_dataset()
    everyone = ds.realization("male", 7)
    assert len(everyone.frame) == 60

    some = ds.realization("male", 7, governorates=["Giza", "Aswan"])
    assert len(some.frame) == 40
    assert set(some.frame["age"] % 3) == {1.0, 2.0}  # Giza and Aswan rows only

    # the filter does not change the draw: the same respondents get the same command units
    unit_of = dict(zip(everyone.weight, everyone.command_unit))
    assert all(unit_of[w] == u for w, u in zip(some.weight, some.command_unit))
    assert ds.realization("male", 7, governorates=None).weight.tolist() == everyone.weight.tolist()


def test_realization_governorates_accepts_a_single_name_and_works_for_train_data():
    ds = _governorate_dataset()
    assert len(ds.realization("male", 1, governorates="Cairo").frame) == 20
    assert len(ds.realization("male", 1, test_data=False, governorates=("Cairo",)).frame) == 20


def test_governorates_missing_from_the_data_are_skipped_with_a_warning():
    ds = _governorate_dataset()
    with pytest.warns(GovernorateWarning, match="skipped.*Atlantis"):
        real = ds.realization("male", 1, governorates=["Cairo", "Atlantis"])
    assert len(real.frame) == 20  # the available governorate is still used

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning when every governorate exists
        ds.realization("male", 1, governorates=["Cairo", "Giza"])


def test_it_is_an_error_if_no_requested_governorate_has_respondents():
    ds = _governorate_dataset()
    with pytest.warns(GovernorateWarning), pytest.raises(
        ValueError, match="no respondents.*Atlantis.*available.*Cairo"
    ):
        ds.realization("male", 1, governorates=["Atlantis"])
    with pytest.raises(ValueError, match="no respondents"):
        ds.realization("male", 1, governorates=[])


def test_predict_ensemble_can_be_limited_to_governorates():
    ds = _governorate_dataset()
    member = EnsembleMember(1, _Model(0.9), _Model(150.0))
    pred = predict_ensemble(ds, [member], "male", governorates=["Cairo"])
    assert pred.expected_wage.shape == (1, 20)
    assert predict_ensemble(ds, [member], "male").expected_wage.shape == (1, 60)


