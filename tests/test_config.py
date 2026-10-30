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
