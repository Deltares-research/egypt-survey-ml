"""Shared production and crop-area helpers used in the survey notebooks."""

from __future__ import annotations

import numpy as np

try:
    from food_security import salinity_correction
except ImportError:  # pragma: no cover - optional dependency in some environments.
    salinity_correction = None


def get_hectares_fixed(
    hectare_ds,
    area,
    area_id,
    crop_name,
    crop_id,
    crop_start_ts,
    crop_end_ts,
    crop_start_ts_dt,
    crop_end_ts_dt,
    year,
):
    """Return the maximum hectare series for a crop in the requested window."""
    start_ts = f"{year}-{crop_start_ts}"
    if crop_start_ts_dt is not None and crop_end_ts_dt is not None:
        end_year = year + 1 if crop_start_ts_dt >= crop_end_ts_dt else year
    else:
        end_year = year
    end_ts = f"{end_year}-{crop_end_ts}"

    requested = f"P Cr{crop_id}/{crop_name}"
    variable = requested

    if requested not in hectare_ds.data_vars:
        prefix = f"P Cr{crop_id}/"
        matches = [
            name
            for name in hectare_ds.data_vars
            if name.startswith(prefix) and requested.startswith(name)
        ]
        if len(matches) != 1:
            raise KeyError(f"Cannot uniquely match {requested!r}: {matches}")
        variable = matches[0]

    stations = [str(s) for s in hectare_ds.station.values]
    matches = [s for s in stations if s.split("/", 1)[0].strip() == str(area_id)]

    if len(matches) != 1:
        raise KeyError(f"Expected one station for area ID {area_id}, found: {matches}")

    node = matches[0]
    selection = hectare_ds[variable].sel(station=node, time=slice(start_ts, end_ts))

    if selection.sizes.get("time", 0) == 0:
        return np.nan

    return selection.max(dim="time").values


def bind_salinity_correction():
    """Attach the shared hectare helper to the upstream salinity correction object."""
    if salinity_correction is not None:
        salinity_correction.get_hectares = get_hectares_fixed
    return salinity_correction
