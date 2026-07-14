"""
Builds the command-area to governorate spatial crosswalk (project plan,
Phase 1): the join table that lets RIBASIM's 61 (actually 60, verified)
command-area outputs be linked to governorate-level survey/wage data for the
first time. Nothing in the codebase did this before; `preprocessed_data.xlsx`
was built by aggregating rasters directly to governorate polygons, never
touching command areas at all.

Method: GIS overlay (geopandas.overlay, "intersection") between the two
layers, reprojected to a common projected CRS so area calculations are
correct (never compute area weights in lat/long, a correctness detail
that's easy to get silently wrong). Produces area-weighted shares
(areal-weighting interpolation, appropriate for land-based biophysical
quantities). Separate population-weighted shares require the WorldPop/age-sex
rasters, which live on Karisma's own OneDrive and aren't accessible from this
environment. That part of Phase 1 has to run on her machine (see
zonal_statistics.ipynb).

Re-runnable: overwrites its single output file each run. Validates inputs
(CRS, geometry validity, expected feature counts) before running the overlay,
rather than letting a bad input silently produce a wrong crosswalk.
"""

import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "Data"
COMMAND_AREA_SHP = DATA_ROOT / "gis_inputs" / "command_area" / "Final2_Command_Area.shp"
GOVERNORATES_SHP = DATA_ROOT / "gis_inputs" / "governorates" / "Governorates.shp"
OUTPUT_CSV = DATA_ROOT / "crosswalk" / "command_area_governorate_crosswalk.csv"

# Command area shapefile's native CRS (confirmed via its .prj: WGS_1984_UTM_Zone_36N).
# Reproject both layers to this rather than the governorate file's native
# EPSG:4326 (lat/long), since area calculations must happen in a projected CRS.
TARGET_CRS = "EPSG:32636"


def load_and_validate(path: Path, expected_min_features: int, layer_name: str) -> gpd.GeoDataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{layer_name} shapefile not found: {path}")
    gdf = gpd.read_file(path)
    if len(gdf) < expected_min_features:
        raise ValueError(
            f"{layer_name} has only {len(gdf)} features, expected at least "
            f"{expected_min_features}. Check this is the right file"
        )
    if gdf.crs is None:
        raise ValueError(f"{layer_name} has no CRS defined: can't safely reproject")

    n_invalid = (~gdf.geometry.is_valid).sum()
    if n_invalid:
        print(f"{layer_name}: fixing {n_invalid}/{len(gdf)} invalid geometries (buffer(0))")
        gdf["geometry"] = gdf.geometry.buffer(0)
        still_invalid = (~gdf.geometry.is_valid).sum()
        if still_invalid:
            raise ValueError(f"{layer_name}: {still_invalid} geometries still invalid after buffer(0) fix")

    return gdf.to_crs(TARGET_CRS)


def build_crosswalk() -> pd.DataFrame:
    command_areas = load_and_validate(COMMAND_AREA_SHP, expected_min_features=50, layer_name="Command areas")
    governorates = load_and_validate(GOVERNORATES_SHP, expected_min_features=25, layer_name="Governorates")

    command_areas = command_areas.rename(columns={"OBJECTID": "command_area_objectid", "Name": "command_area_name"})
    governorates = governorates.rename(columns={"OBJECTID": "gov_OBJECTID", "NAME1_": "gov_NAME1_"})

    command_areas["command_area_full_km2"] = command_areas.geometry.area / 1e6
    governorates["gov_full_km2"] = governorates.geometry.area / 1e6

    overlay = gpd.overlay(
        command_areas[["command_area_objectid", "command_area_name", "Feddan", "command_area_full_km2", "geometry"]],
        governorates[["gov_OBJECTID", "gov_NAME1_", "gov_full_km2", "geometry"]],
        how="intersection",
        keep_geom_type=False,
    )

    if overlay.empty:
        raise ValueError("Overlay produced zero intersections: check both layers actually cover the same area (CRS/extent mismatch?)")

    overlay["intersection_km2"] = overlay.geometry.area / 1e6
    # Drop slivers from imprecise boundary alignment between two independently-digitized layers
    overlay = overlay[overlay["intersection_km2"] > 0.01].copy()

    overlay["pct_of_command_area_area"] = overlay["intersection_km2"] / overlay["command_area_full_km2"]
    overlay["pct_of_governorate_area"] = overlay["intersection_km2"] / overlay["gov_full_km2"]

    result = overlay[[
        "command_area_objectid", "command_area_name", "gov_OBJECTID", "gov_NAME1_",
        "intersection_km2", "pct_of_command_area_area", "pct_of_governorate_area",
    ]].sort_values(["gov_OBJECTID", "command_area_objectid"]).reset_index(drop=True)

    return result, command_areas, governorates


def run_sanity_checks(result: pd.DataFrame, command_areas: gpd.GeoDataFrame, governorates: gpd.GeoDataFrame) -> None:
    print(f"\n{len(result)} command-area x governorate intersection rows "
          f"({result['command_area_objectid'].nunique()} command areas x "
          f"{result['gov_OBJECTID'].nunique()} governorates touched)")

    # Per the project plan's verification section: for each command area, its
    # area-share across all governorates it touches should sum to ~1.0.
    ca_sums = result.groupby("command_area_objectid")["pct_of_command_area_area"].sum()
    off = ca_sums[(ca_sums < 0.98) | (ca_sums > 1.02)]
    print(f"Command areas with area-share sum outside [0.98, 1.02]: {len(off)}/{len(ca_sums)}")
    if len(off):
        print("(expected for command areas that extend slightly beyond the governorate boundaries used here, or vice versa)")
        print(off.round(3).to_string())

    gov_sums = result.groupby("gov_OBJECTID")["pct_of_governorate_area"].sum()
    print(f"\nGovernorates with NO command-area overlap at all: "
          f"{governorates['gov_OBJECTID'].nunique() - gov_sums.index.nunique()}")
    missing_govs = set(governorates["gov_OBJECTID"]) - set(gov_sums.index)
    if missing_govs:
        names = governorates.set_index("gov_OBJECTID").loc[list(missing_govs), "gov_NAME1_"].tolist()
        print(f"  {names} (expected: low-irrigation desert governorates outside the RIBASIM command-area network)")


def main() -> None:
    result, command_areas, governorates = build_crosswalk()
    run_sanity_checks(result, command_areas, governorates)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(result)}-row crosswalk to {OUTPUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
