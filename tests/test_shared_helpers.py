import math
from pathlib import Path

import pandas as pd
import pytest

from egypt_survey_ml.io_utils import write_excel_file
from egypt_survey_ml.production import get_hectares_fixed


def test_write_excel_file_writes_sheet(tmp_path):
    path = tmp_path / "out.xlsx"
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    write_excel_file(df, str(path), "demo")

    assert path.exists()
    written = pd.read_excel(path, sheet_name="demo")
    assert written.equals(df)


def test_get_hectares_fixed_returns_max_over_time_window():
    class FakeVar:
        def __init__(self):
            self.values = [1.0, 4.0, 2.5]

    class FakeStationGroup:
        values = ["1/Station A"]

    class FakeSelection:
        def __init__(self):
            self.sizes = {"time": 3}

        def sel(self, station, time):
            return self

        def max(self, dim):
            class MaxResult:
                @property
                def values(self):
                    return 4.0

            return MaxResult()

    class FakeDataset:
        def __init__(self):
            self.data_vars = {"P Cr12/maize": FakeVar()}
            self.station = FakeStationGroup()

        def __getitem__(self, key):
            return FakeSelection()

    ds = FakeDataset()
    value = get_hectares_fixed(
        hectare_ds=ds,
        area=None,
        area_id=1,
        crop_name="maize",
        crop_id=12,
        crop_start_ts="01-01",
        crop_end_ts="12-31",
        crop_start_ts_dt=None,
        crop_end_ts_dt=None,
        year=2024,
    )

    assert value == 4.0
