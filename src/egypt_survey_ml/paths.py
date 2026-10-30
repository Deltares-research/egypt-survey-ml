"""Locate the survey data, either in the repo (ERF_Data) or in the Deltares OneDrive."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

_ONEDRIVE = "OneDrive - Stichting Deltares"
_EGYPT_DIR = "Tiaravanni Hermawan's files - Egypt"
_ERF_DIR = "Tiaravanni Hermawan's files - Egypt_ERF_data"
_LFS_DIR = "Labor Force Survey, LFS 2022 - Egypt, Arab Rep., 2022"


@dataclass(frozen=True)
class DataPaths:
    data_root: Path
    excel_path: Path
    crosswalk_csv: Path
    command_area_crosswalk_csv: Path
    lfs_dta: Path
    hh_dta: Path
    result_dir: Path
    mask_xlsx: Path
    pop_density_csv: Path
    source: str

    def report(self) -> None:
        """Print which input files exist."""
        for p in [
            self.lfs_dta,
            self.hh_dta,
            self.crosswalk_csv,
            self.command_area_crosswalk_csv,
            self.mask_xlsx,
            self.pop_density_csv,
            self.excel_path,
        ]:
            print(("OK  " if p.exists() else "MISSING "), p)


def resolve_data_paths(
    data_filename: str = "data_correlation.xlsx", repo_root: Path | None = None
) -> DataPaths:
    """Prefer local data in ``ERF_Data/Data`` (or $EGYPT_DATA_ROOT), else the OneDrive copy."""
    repo_root = Path(repo_root) if repo_root is not None else REPO_ROOT
    local_root = Path(
        os.environ.get("EGYPT_DATA_ROOT", repo_root / "ERF_Data" / "Data")
    )

    if (local_root / data_filename).exists():
        data_root = local_root
        excel_path = data_root / data_filename
        crosswalk_dir = data_root / "crosswalk"
        result_dir = data_root.parent / "Result"
        source = f"local ({data_root})"
    else:
        onedrive = Path("~").expanduser() / _ONEDRIVE
        egypt_root = onedrive / _EGYPT_DIR
        erf_root = onedrive / _ERF_DIR
        if not (egypt_root.exists() and erf_root.exists()):
            raise FileNotFoundError("Could not find required OneDrive folders.")
        data_root = egypt_root / "04_Data/2026_data"
        excel_path = erf_root / data_filename
        crosswalk_dir = data_root.parent / "cleaned_data/crosswalk"
        result_dir = data_root.parent / "Result"
        source = f"OneDrive ({data_root})"

    return DataPaths(
        data_root=data_root,
        excel_path=excel_path,
        crosswalk_csv=crosswalk_dir / "reg_governorate_crosswalk.csv",
        command_area_crosswalk_csv=crosswalk_dir
        / "command_area_governorate_crosswalk.csv",
        lfs_dta=data_root / _LFS_DIR / "Egypt 2022-LFS IND-V1.dta",
        hh_dta=data_root / _LFS_DIR / "Egypt 2022-LFS HH-V1.dta",
        result_dir=result_dir,
        mask_xlsx=result_dir / "correlation.xlsx",
        pop_density_csv=result_dir / "population_density_matrix.csv",
        source=source,
    )
