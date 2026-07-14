"""
Builds a verified crosswalk between LFS2022's `reg` variable (UN M49 country
code 818 + 3-digit governorate code) and the `OBJECTID`/`NAME1_` governorate
scheme used in preprocessed_data.xlsx (the existing RIBASIM-linked spatial
table).

Replaces the hand-typed `reg_level_mapping` dict in LFS2022_DataUnderstanding.md,
which only covered 19 of 27 governorates and had spelling inconsistencies.
This script derives the mapping directly from the two source-of-truth files
(the LFS2022 .dta value labels and preprocessed_data.xlsx), so it can't drift
out of sync the way a hand-copied dict can.

Re-runnable: overwrites its single output file each run, no manual cleanup
needed. Raises with a clear message rather than silently mismatching if either
source changes shape.
"""

import sys
from pathlib import Path

import pandas as pd
import pyreadstat

DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "Data"
LFS_DTA = DATA_ROOT / "Labor Force Survey, LFS 2022 - Egypt, Arab Rep., 2022" / "Egypt 2022-LFS IND-V1.dta"
PREPROCESSED_XLSX = DATA_ROOT / "preprocessed_data.xlsx"
OUTPUT_CSV = DATA_ROOT / "crosswalk" / "reg_governorate_crosswalk.csv"


def load_lfs_reg_labels() -> dict[int, str]:
    if not LFS_DTA.exists():
        raise FileNotFoundError(f"LFS2022 individual file not found: {LFS_DTA}")
    _, meta = pyreadstat.read_dta(LFS_DTA, metadataonly=True)
    reg_labels = meta.variable_value_labels.get("reg")
    if not reg_labels:
        raise ValueError("No value labels found for 'reg' in the LFS2022 .dta metadata")
    # 999999 is "Not stated": not a real governorate, exclude explicitly.
    return {int(code): name.strip() for code, name in reg_labels.items() if int(code) != 999999}


def load_preprocessed_governorates() -> pd.DataFrame:
    if not PREPROCESSED_XLSX.exists():
        raise FileNotFoundError(f"preprocessed_data.xlsx not found: {PREPROCESSED_XLSX}")
    df = pd.read_excel(PREPROCESSED_XLSX, usecols=["OBJECTID", "ID_", "NAME1_"])
    if df.empty:
        raise ValueError("preprocessed_data.xlsx loaded but has no rows")
    # ID_ looks like 'AFREGY033': the trailing 3 digits are the governorate code.
    df["gov_code_suffix"] = df["ID_"].str.extract(r"(\d{3})$").astype(int)
    return df


def build_crosswalk() -> pd.DataFrame:
    reg_labels = load_lfs_reg_labels()
    gov_df = load_preprocessed_governorates()

    rows = []
    for reg_code, reg_name in sorted(reg_labels.items()):
        suffix = reg_code % 1000
        match = gov_df.loc[gov_df["gov_code_suffix"] == suffix]
        if len(match) == 1:
            rows.append({
                "reg_code": reg_code,
                "reg_name_lfs": reg_name,
                "OBJECTID": int(match.iloc[0]["OBJECTID"]),
                "NAME1_": match.iloc[0]["NAME1_"],
                "in_preprocessed_data": True,
            })
        elif len(match) == 0:
            rows.append({
                "reg_code": reg_code,
                "reg_name_lfs": reg_name,
                "OBJECTID": pd.NA,
                "NAME1_": pd.NA,
                "in_preprocessed_data": False,
            })
        else:
            raise ValueError(
                f"reg_code {reg_code} ({reg_name}) matched {len(match)} rows in "
                f"preprocessed_data.xlsx by governorate-code suffix {suffix:03d}. "
                "Expected exactly 0 or 1. Check for duplicate ID_ values."
            )

    crosswalk = pd.DataFrame(rows)
    n_matched = crosswalk["in_preprocessed_data"].sum()
    n_total = len(crosswalk)
    print(f"Matched {n_matched}/{n_total} LFS2022 governorates to preprocessed_data.xlsx.")
    unmatched = crosswalk.loc[~crosswalk["in_preprocessed_data"], "reg_name_lfs"].tolist()
    if unmatched:
        print(f"Unmatched (expected: these lack RIBASIM-linked biophysical covariates): {unmatched}")
    return crosswalk


def main() -> None:
    crosswalk = build_crosswalk()
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    crosswalk.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(crosswalk)}-row crosswalk to {OUTPUT_CSV}")


if __name__ == "__main__":
    sys.exit(main())
