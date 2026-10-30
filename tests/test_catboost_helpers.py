import numpy as np
import pandas as pd

from egypt_survey_ml import apply_zero_handling
from egypt_survey_ml.catboost import convert_irregular, evaluate, gender_gap
from egypt_survey_ml.data_prep import (
    build_household_features,
    filter_rural_agriculture_population,
    split_train_test,
)


def test_convert_irregular_uses_average_hours_for_missing_values():
    avg_hours_by_ind = pd.Series({1: 8, 2: 10})
    daily_wage = pd.Series([10, 20])
    ind_code = pd.Series([1, 2])
    real_hours = pd.Series([np.nan, 6])

    result = convert_irregular(
        daily_wage,
        ind_code,
        real_hours,
        avg_hours_by_ind=avg_hours_by_ind,
    )

    np.testing.assert_allclose(result.to_numpy(), np.array([40.0, 60.0]))


def test_evaluate_and_gender_gap_smoke():
    metrics = evaluate(
        y_true_wage=np.array([100, 60, 80]),
        y_pred_wage=np.array([90, 70, 85]),
        is_paid_true=np.array([1, 1, 0]),
        label="demo",
    )

    assert metrics["label"] == "demo"
    assert metrics["n"] == 3
    assert "rmse" in metrics

    gap = gender_gap(
        pred_male=np.array([10, 12]),
        pred_female=np.array([8, 9]),
        actual_male=np.array([12, 13]),
        actual_female=np.array([7, 9]),
    )

    assert gap["predicted_gap"] == 2.5
    assert gap["observed_gap"] == 4.5


def test_apply_zero_handling_drop_random_removes_zero_rows():
    frame = pd.DataFrame(
        {
            "is_paid": [0, 0, 1, 1],
            "totwag_all": [0.0, 0.0, 10.0, 12.0],
        }
    )

    result = apply_zero_handling(
        frame,
        cat_cols=[],
        feature_cols=[],
        mode="drop_random",
        seed=0,
        DROP_RANDOM_FRAC=0.5,
    )

    assert len(result) == 3
    assert (result["is_paid"] == 1).sum() == 2


def test_build_household_features_and_filter_population():
    df = pd.DataFrame(
        {
            "caseser": [1, 1, 2, 2, 3, 4],
            "ind": [10, 10, 10, 10, 11, 10],
            "age": [25, 25, 30, 30, 20, 18],
            "rururb": [0, 0, 0, 0, 1, 0],
            "emps": [1, 1, 1, 1, 1, 1],
            "sex": [1, 2, 1, 2, 1, 1],
            "educ": [8, 9, 10, 12, 7, 11],
            "pnum": [1, 1, 1, 1, 1, 1],
            "rel": [1, 2, 1, 2, 1, 1],
            "hrswk": [8, np.nan, 5, 7, 6, 9],
            "irrgwaghd": [0, 0, 0, 0, 0, 0],
            "indhd": [10, 10, 10, 10, 10, 10],
            "empinchd": [0, 0, 0, 0, 0, 0],
            "sempinchd": [0, 0, 0, 0, 0, 0],
            "totwaghd": [0, 0, 0, 0, 0, 0],
            "irrgwagsp_1": [0, 0, 0, 0, 0, 0],
            "indsp_1": [10, 10, 10, 10, 10, 10],
            "empincsp_1": [0, 0, 0, 0, 0, 0],
            "sempincsp_1": [0, 0, 0, 0, 0, 0],
            "totwagsp_1": [0, 0, 0, 0, 0, 0],
            "irrwagehd": [0, 0, 0, 0, 0, 0],
            "irrwagesp_1": [0, 0, 0, 0, 0, 0],
        }
    )
    hh = pd.DataFrame(
        {
            "caseser": [1, 2, 3, 4],
            "rururb": [0, 0, 1, 0],
            "empinchd": [10, 5, 0, 0],
            "sempinchd": [0, 0, 0, 0],
            "totwaghd": [0, 0, 0, 0],
            "irrgwaghd": [0, 0, 0, 0],
            "indhd": [10, 10, 10, 10],
            "empincsp_1": [0, 0, 0, 0],
            "sempincsp_1": [0, 0, 0, 0],
            "totwagsp_1": [0, 0, 0, 0],
            "irrgwagsp_1": [0, 0, 0, 0],
        }
    )
    avg_hours_by_ind = pd.Series({10: 8})

    household_size, n_employed_in_hh, head_info, hh_wage = build_household_features(
        df, hh, avg_hours_by_ind=avg_hours_by_ind
    )
    assert len(household_size) == 4
    assert len(n_employed_in_hh) == 4
    assert set(["head_educ", "head_sex"]).issubset(head_info.columns)
    assert set(["household_wage_total", "wage_ratio"]).issubset(hh_wage.columns)

    agri_pool = filter_rural_agriculture_population(
        df, min_working_age=15, industry_code=10
    )
    assert len(agri_pool) == 5

    train_pool, test_pool, test_ids = split_train_test(
        agri_pool, seed=42, test_size=0.25
    )
    assert len(train_pool) + len(test_pool) == len(agri_pool)
    assert len(test_ids) == len(test_pool)
