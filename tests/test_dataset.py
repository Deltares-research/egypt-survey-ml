from dataclasses import replace
from unittest.mock import patch

import pandas as pd
import pytest

from egypt_survey_ml.config import DatasetConfig
from egypt_survey_ml.dataset import build_feature_frame, prepare_dataset, select_features
from egypt_survey_ml.paths import DataPaths


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
    mask.to_excel(mask_path, sheet_name="Sheet1", index=False)
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
        paths.mask_xlsx, sheet_name="Sheet1", index=False
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


def test_mask_selection_adds_hours_and_extra_features():
    mask = pd.DataFrame({"variable": ["age"], "correl": [1], "type": ["numerical"]})
    features = select_features(
        mask, pd.DataFrame(columns=["age", "rel", "attsch", "occ"]), DatasetConfig()
    )
    assert features.columns_for(1) == ["age", "rel", "attsch", "occ", "hrswk", "hrswksc"]


def _frame_inputs():
    pool = pd.DataFrame(
        {
            "sex": [2],
            "age": [30],
            "occ": [1],
            "pweight": [2.0],
            "command_unit": ["A"],
            "log1p_wage": [1.0],
            "totwag_all": [10.0],
            "is_paid": [1],
        }
    )
    mask = pd.DataFrame(
        {
            "variable": ["age", "sex", "female_violence_sexual"],
            "correl": [1, 1, 1],  # the last one is in the mask but not in the data
            "type": ["numerical", "binary", "numerical"],
        }
    )
    geo = pd.DataFrame(
        {"area": ["A"], "water_productivity": [1.5], "producer_price": [100.0]}
    )
    return pool, mask, geo


def test_feature_frame_skips_mask_columns_missing_from_the_data():
    pool, mask, geo = _frame_inputs()
    features = select_features(mask, pool, DatasetConfig())
    frame, cats, extra = build_feature_frame(pool, 2, geo, features)
    assert "female_violence_sexual" not in frame.columns
    assert frame.columns[:2].tolist() == ["age", "sex"]
    assert "water_productivity" not in frame.columns
    assert cats[0] == "sex" and frame["sex"].tolist() == ["2"]
    assert extra["pweight"].tolist() == [2.0]


def test_feature_frame_adds_columns_requested_by_models():
    pool, mask, geo = _frame_inputs()
    features = select_features(mask, pool, DatasetConfig())
    frame, cats, _ = build_feature_frame(
        pool,
        2,
        geo,
        features,
        extra_cols=["age", "water_productivity", "producer_price", "not_in_data"],
        extra_cats=["occ", "not_in_data"],
    )
    assert frame[["water_productivity", "producer_price"]].iloc[0].tolist() == [1.5, 100.0]
    assert frame.columns.tolist().count("age") == 1
    assert "not_in_data" not in frame.columns
    assert "occ" in cats and frame["occ"].tolist() == ["1"]
