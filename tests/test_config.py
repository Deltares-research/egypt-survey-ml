import pytest

from egypt_survey_ml.config import DatasetConfig


def test_defaults_and_replace():
    cfg = DatasetConfig()
    assert cfg.rng_seed == 42 and cfg.female_outlier_cap == 4000
    other = cfg.replace(include_ribasim=True, female_outlier_cap=3000)
    assert other.include_ribasim and other.female_outlier_cap == 3000
    assert not cfg.include_ribasim


def test_from_dict_rejects_unknown_keys():
    with pytest.raises(KeyError):
        DatasetConfig.from_dict({"not_a_setting": 1})


def test_invalid_zero_handling():
    with pytest.raises(ValueError):
        DatasetConfig(zero_handling="bogus")


def test_per_sex_column_lists_default_to_empty_and_roundtrip():
    cfg = DatasetConfig()
    assert cfg.male_extra_columns == cfg.female_exclude_columns == []
    cfg.male_extra_columns.append("x")
    assert DatasetConfig().male_extra_columns == []  # not shared between instances

    cfg = DatasetConfig(male_extra_columns=("a", "b"), female_exclude_columns=["c"])
    assert cfg.male_extra_columns == ["a", "b"]
    assert DatasetConfig.from_dict(cfg.to_dict()) == cfg


@pytest.mark.parametrize("name", ["male_extra_columns", "female_exclude_columns"])
def test_a_single_column_name_must_be_given_as_a_list(name):
    with pytest.raises(ValueError, match=name):
        DatasetConfig(**{name: "age"})
